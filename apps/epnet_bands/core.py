"""EpNet band-structure prediction for one uploaded structure (web app core; 2026-10-16, EpNet-sp; s/p/d model
5c-N from 2026-11-04).

structure file -> standardized primitive cell -> Setyawan-Curtarolo high-symmetry path -> GeneralEPM bands
-> gap / VBM / CBM (insulators) or a mesh Fermi level (metals) -> figure, CSV, POSCAR and a JSON summary.

The model is an empirical pseudopotential Hamiltonian trained on Materials Project PBE bands: non-magnetic, no
spin-orbit, no +U, valence bands only (semicore states are not modelled), 63 elements of the s, p and d blocks.
"""
from __future__ import annotations

import io
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from epnet.data.structures import Structure
from epnet.data.valence import Z_MODEL
from epnet.models.general_epm import GeneralEPM

HERE = Path(__file__).resolve().parent
DEFAULT_CKPT = HERE.parents[1] / "models/epnet_spd_5cN_s0.pt"   # the final model (seed 0)
MAX_SITES, WARN_SITES = 40, 24          # trained on <= 24-site cells; 25-40 is extrapolation in size, and slower
D_BLOCK = {"Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag",
           "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg"}
OPEN_SHELL_3D = {"Cr", "Mn", "Fe", "Co", "Ni"}            # trained non-magnetic
PLUS_U_METALS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}   # their oxides and fluorides are +U in MP: not in training
DOS_COVER = 7.0                          # eV above the VBM for which the DOS is complete (plot window: +6 eV)
MESH_BUDGET = 6000                       # DOS / E_F mesh: total k-points x sites (before the time-reversal halving)


@dataclass
class Prediction:
    formula: str
    spacegroup: str
    nsites: int
    n_electrons: int
    n_occupied: int
    metal: bool
    reference: str                       # "VBM" or "E_F"
    e_ref_raw: float                     # model energy of the reference (eV, internal zero)
    gap: float | None
    gap_direct: bool | None
    vbm_k: str | None
    cbm_k: str | None
    distance: np.ndarray                 # (n_k,) path distance, 1/A (2 pi included)
    kfrac: np.ndarray                    # (n_k, 3) fractional, of the standardized primitive cell
    labels: list                         # (n_k,) label or ""
    ticks: list                          # [(distance, label)]
    bands: np.ndarray                    # (n_k, n_bands), eV relative to the reference
    poscar: str
    dos_e: np.ndarray = None             # DOS energies (eV, relative to the reference)
    dos: np.ndarray = None               # states / eV / cell (both spins)
    dos_int: np.ndarray = None           # integrated electrons
    dos_top: float = 0.0                 # DOS complete below this energy (minimum of the highest computed band)
    dos_mesh: tuple = ()
    dos_sigma: float = 0.1
    warnings: list = field(default_factory=list)
    timing_s: float = 0.0
    model: dict = field(default_factory=dict)


_MODEL = {}


def load_model(ckpt: str | os.PathLike | None = None):
    path = Path(ckpt or os.environ.get("EPNET_CKPT", DEFAULT_CKPT))
    if str(path) not in _MODEL:
        if os.environ.get("EPNET_THREADS"):                 # CPU threads per prediction (default: torch's choice)
            torch.set_num_threads(int(os.environ["EPNET_THREADS"]))
        payload = torch.load(path, map_location="cpu", weights_only=False)
        model = GeneralEPM.from_checkpoint(payload).eval()
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _MODEL[str(path)] = (model.to(dev), dev, {"checkpoint": path.name, "elements": list(model.species_symbols),
                                                "ecut_ry": model.config.get("ecut_ry"),
                                                "ecut_by_element": model.config.get("ecut_by_element"), "device": str(dev)})
    return _MODEL[str(path)]


# structure formats read through ASE, by file-name ending (2026-11-04: an explicit list for a public server)
ASE_FORMATS = {".extxyz": "extxyz", ".xyz": "extxyz", ".pwi": "espresso-in", ".in": "espresso-in", "geometry.in": "aims",
               ".res": "res", ".gen": "gen", ".cell": "castep-cell", ".xsf": "xsf", ".cfg": "cfg"}


def read_structure(data: bytes, filename: str):
    """pymatgen Structure from CIF / POSCAR, or one of ASE_FORMATS (by extension)."""
    from pymatgen.core import Structure as PmgStructure
    from pymatgen.io.ase import AseAtomsAdaptor
    text = data.decode("utf-8", errors="replace")
    name = filename.lower()
    fmt = "cif" if name.endswith(".cif") else ("poscar" if "poscar" in name or "contcar" in name or name.endswith(".vasp") else None)
    errors = []
    for f in ([fmt] if fmt else []) + ["cif", "poscar"]:
        try:
            return PmgStructure.from_str(text, fmt=f)
        except Exception as ex:          # noqa: BLE001 - try the next reader
            errors.append(f"{f}: {ex}")
    ase_fmt = next((f for suffix, f in ASE_FORMATS.items() if name.endswith(suffix)), None)
    if ase_fmt:                          # only these structure formats, chosen by extension (no format guessing on uploads)
        import shutil
        import ase.io
        path = _tmp(data, filename)
        try:
            return AseAtomsAdaptor.get_structure(ase.io.read(path, format=ase_fmt))
        except Exception as ex:          # noqa: BLE001
            errors.append(f"ase ({ase_fmt}): {ex}")
        finally:
            shutil.rmtree(Path(path).parent, ignore_errors=True)
    raise ValueError("could not read the structure (" + "; ".join(errors) + "). Accepted: CIF, POSCAR/CONTCAR/.vasp, "
                     "and by extension " + ", ".join(ASE_FORMATS))


def _tmp(data: bytes, filename: str) -> str:
    import tempfile
    d = Path(tempfile.mkdtemp(prefix="epnet_"))
    p = d / (Path(filename).name or "structure")
    p.write_bytes(data)
    return str(p)


# fallback when Setyawan-Curtarolo has no path for the cell (e.g. a slightly distorted perovskite that is P1 at the
# 1e-3 tolerance with 90-degree angles: pymatgen returns kpath=None; Alexandria CsPbCl3 agm002143576, 2026-10-19).
# The triclinic-style path in the cell's own reciprocal basis always exists and matches the cell the bands are
# computed on; it covers Gamma and every face, edge and corner midpoint.
GENERIC_KPATH = {
    "kpoints": {"\\Gamma": (0, 0, 0), "X": (0.5, 0, 0), "Y": (0, 0.5, 0), "Z": (0, 0, 0.5), "V": (0.5, 0.5, 0),
                "U": (0.5, 0, 0.5), "T": (0, 0.5, 0.5), "R": (0.5, 0.5, 0.5)},
    "path": [["X", "\\Gamma", "Y"], ["Z", "\\Gamma", "V"], ["U", "\\Gamma", "T"], ["R", "\\Gamma"]],
}


def kpath(prim, line_density: float, info: dict | None = None):
    """Setyawan-Curtarolo path on the standardized primitive cell: fractional k, distances, labels, ticks.
    Falls back to GENERIC_KPATH (info["generic"] = True) when the convention has no path for the cell."""
    from pymatgen.symmetry.bandstructure import HighSymmKpath
    hk = HighSymmKpath(prim, path_type="setyawan_curtarolo")
    rec = prim.lattice.reciprocal_lattice.matrix          # rows, 2 pi included
    path = hk.kpath if hk.kpath is not None else GENERIC_KPATH
    if info is not None:
        info["generic"] = hk.kpath is None
    kp = path["kpoints"]
    kfrac, labels, dist, ticks = [], [], [], []
    d = 0.0
    for b, branch in enumerate(path["path"]):
        for s in range(len(branch) - 1):
            a, c = np.asarray(kp[branch[s]], float), np.asarray(kp[branch[s + 1]], float)
            seg = np.linalg.norm((c - a) @ rec)
            n = max(2, int(math.ceil(seg * line_density)) + 1)
            for i, t in enumerate(np.linspace(0.0, 1.0, n)):
                if s > 0 and i == 0:
                    continue                                 # shared vertex within a branch
                k = a + t * (c - a)
                if kfrac:
                    jump = (s == 0 and i == 0)               # new branch: no distance added
                    d += 0.0 if jump else float(np.linalg.norm((k - kfrac[-1]) @ rec))
                lab = branch[s] if i == 0 else (branch[s + 1] if i == n - 1 else "")
                kfrac.append(k); labels.append(lab); dist.append(d)
    for i, lab in enumerate(labels):
        if lab:
            pretty = lab.replace("\\Gamma", "Γ").replace("GAMMA", "Γ").replace("_1", "₁").replace("_2", "₂").replace("_0", "₀")
            if ticks and abs(ticks[-1][0] - dist[i]) < 1e-9 and ticks[-1][1] != pretty:
                ticks[-1] = (dist[i], ticks[-1][1] + "|" + pretty)
            elif not ticks or abs(ticks[-1][0] - dist[i]) > 1e-9:
                ticks.append((dist[i], pretty))
            labels[i] = pretty
    return np.asarray(kfrac), np.asarray(dist), labels, ticks


def _bands(model, dev, st: Structure, kfrac: np.ndarray, n_bands: int, chunk: int = 64) -> np.ndarray:
    out = []
    with torch.no_grad():
        for i in range(0, len(kfrac), chunk):
            out.append(model.predict_bands(st, kfrac[i:i + chunk], n_bands=n_bands).cpu().numpy().astype(float))
    return np.concatenate(out, 0)


# k-point parallelism for CPU servers (2026-11-04, user: speed up CPU mode): a persistent pool of P worker processes,
# each with its own model copy and cores/P threads, solves contiguous slices of the k list. Measured on 16 cores against
# one process with 16 threads (path + DOS mesh): Si 4.2x, Cu 3.9x, ZnO 3.1x, TiO2 2.6x, MoS2 1.9x, bands within 1 meV;
# the manuscript's timing set gave 2.3x (median) with 4 x 4 and up to 4.5x for 16-24-site cells. Below the FFT switch
# (3,000 plane waves) each slice uses the dense solver, above it the warm-started chain (epnet.screen.chain_bands).
_POOL: dict = {}


def _worker_init(threads: int, ckpt: str) -> None:
    os.environ["EPNET_THREADS"] = str(threads)      # load_model applies EPNET_THREADS: the worker's share, not the total
    os.environ["OMP_NUM_THREADS"] = str(threads)    # (2026-11-04: inheriting 16 made 4 workers run 64 threads, 100x slower)
    torch.set_num_threads(threads)
    load_model(ckpt)


def _worker_solve(args) -> np.ndarray:
    st, k, n_bands, ckpt, use_chain = args
    model, dev, _ = load_model(ckpt)
    if use_chain:
        from epnet.screen import chain_bands
        return chain_bands(model, st, k, n_bands, batch=1, tol=1e-3)[0]
    return _bands(model, dev, st, k, n_bands)


def start_pool(processes: int, threads_per: int, ckpt: str | None = None) -> None:
    """Start the worker pool (local CPU mode; call once before serving). processes <= 1: no pool."""
    if processes <= 1 or _POOL:
        return
    import torch.multiprocessing as mp
    path = str(ckpt or os.environ.get("EPNET_CKPT", DEFAULT_CKPT))
    _POOL.update(pool=mp.get_context("spawn").Pool(processes, initializer=_worker_init, initargs=(threads_per, path)),
                 p=processes, ckpt=path)
    import atexit
    atexit.register(_stop_pool)


def _stop_pool() -> None:
    pool = _POOL.pop("pool", None)
    if pool is not None:
        pool.terminate()
        pool.join()


def solve_bands(model, dev, st: Structure, kfrac: np.ndarray, n_bands: int) -> np.ndarray:
    """Lowest n_bands energies at kfrac: on the worker pool when one is running and the k list is long enough."""
    pool = _POOL.get("pool")
    if pool is None or len(kfrac) < 2 * _POOL["p"]:
        return _bands(model, dev, st, kfrac, n_bands)
    packed = model.pack(st)
    k0 = torch.as_tensor(np.asarray(kfrac[:1]), dtype=packed.lattice.dtype)
    use_chain = int(model.make_pw_cache(packed, k0).kinetic.shape[-1]) >= int(model.config.get("fft_ng_threshold") or 3000)
    edges = np.linspace(0, len(kfrac), _POOL["p"] + 1).round().astype(int)
    jobs = [(st, np.asarray(kfrac[a:b]), n_bands, _POOL["ckpt"], use_chain) for a, b in zip(edges[:-1], edges[1:]) if b > a]
    return np.concatenate(pool.map(_worker_solve, jobs, chunksize=1), axis=0)


def mesh_size(prim, kspacing: float) -> list[int]:
    """Gamma-centred mesh for KSPACING (1/A, 2 pi included), coarsened to the MESH_BUDGET of k-points x sites."""
    rec = prim.lattice.reciprocal_lattice.matrix
    n = [max(1, int(math.ceil(np.linalg.norm(b) / kspacing))) for b in rec]
    cap = int(np.clip(MESH_BUDGET / len(prim), 64, 3000))
    while np.prod(n) > cap:
        n = [max(1, x - 1) for x in n]
    return n


def time_reversal_mesh(n: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """The mesh with k and -k merged (no spin-orbit, no field): fractional k and weights (sum 1). Pairs are found on the
    integer grid, i and -i mod n (2026-11-04: the earlier float version kept both members of the pairs on the k = 1/2
    planes of even meshes, each with weight 2)."""
    n = np.asarray(n)
    idx = np.stack(np.meshgrid(*[np.arange(x) for x in n], indexing="ij"), -1).reshape(-1, 3)
    flat = lambda i: (i[:, 0] * n[1] + i[:, 1]) * n[2] + i[:, 2]          # noqa: E731
    me, partner = flat(idx), flat((-idx) % n)
    keep = me <= partner
    k = idx[keep] / n
    k = np.where(k > 0.5 + 1e-9, k - 1.0, k)
    w = np.where(me[keep] == partner[keep], 1.0, 2.0)
    return k, w / w.sum()


def irreducible_mesh(prim, n: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    """The same mesh reduced by the crystal's point group and time reversal (spglib; 2026-11-04, user: speed up the DOS
    mesh): the bands at symmetry-equivalent k are equal for this Hamiltonian (atom-centred, equivariant, no spin-orbit),
    so the DOS is unchanged and only the irreducible k are solved (MoS2 18x18x3: 539 k with time reversal only). None if spglib fails."""
    try:
        import spglib
        cell = (prim.lattice.matrix, prim.frac_coords, [site.specie.Z for site in prim])
        mapping, grid = spglib.get_ir_reciprocal_mesh(n, cell, is_shift=[0, 0, 0], is_time_reversal=True, symprec=1e-3)
    except Exception:                                        # noqa: BLE001 - fall back to time reversal only
        return None
    if mapping is None:
        return None
    uniq, counts = np.unique(mapping, return_counts=True)
    k = grid[uniq] / np.asarray(n, float)
    return k, counts / counts.sum()


def mesh_eigs(model, dev, st: Structure, prim, n_bands: int, kspacing: float = 0.12, symmetry: bool = True):
    """Bands on a Gamma-centred mesh (KSPACING in 1/A, 2 pi included), on its irreducible k (point group and time
    reversal; time reversal only if spglib fails or symmetry=False): energies, weights, mesh."""
    n = mesh_size(prim, kspacing)
    red = irreducible_mesh(prim, n) if symmetry else None
    k, w = red if red is not None else time_reversal_mesh(n)
    return solve_bands(model, dev, st, k, n_bands), w, n


def fermi_from_mesh(e: np.ndarray, w: np.ndarray, n_electrons: int, sigma: float = 0.1) -> float:
    """E_F filling the mesh bands with Fermi-Dirac smearing, 2 electrons per band and k-point."""
    lo, hi = e.min() - 1.0, e.max() + 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        ne = 2.0 * float((w[:, None] / (1.0 + np.exp(np.clip((e - mid) / sigma, -60, 60)))).sum())
        lo, hi = (mid, hi) if ne < n_electrons else (lo, mid)
    return 0.5 * (lo + hi)


def dos(e: np.ndarray, w: np.ndarray, ref: float, sigma: float, step: float = 0.01):
    """Total DOS (states / eV / cell, both spins) with Gaussian broadening, relative to ref; complete below the minimum
    of the highest computed band ('top'). Returns energies, DOS, integrated electrons, top."""
    top = float(e[:, -1].min() - ref)
    lo = float(e.min() - ref) - 4 * sigma
    edges = np.arange(lo, float(e.max() - ref) + 4 * sigma + step, step)
    hist, _ = np.histogram((e - ref).ravel(), bins=edges, weights=np.repeat(2.0 * w, e.shape[1]))   # 2 spins
    kx = np.arange(-int(4 * sigma / step), int(4 * sigma / step) + 1) * step
    kern = np.exp(-0.5 * (kx / sigma) ** 2); kern /= kern.sum() * step
    g = np.convolve(hist, kern, mode="same")
    grid = 0.5 * (edges[1:] + edges[:-1])
    integ = np.cumsum(g) * step
    keep = grid <= top
    return grid[keep], g[keep], integ[keep], top


def predict(data: bytes, filename: str, *, line_density: float = 25.0, n_empty: int = 8, ckpt=None,
            dos_sigma: float = 0.1, kspacing: float = 0.12) -> Prediction:
    t0 = time.time()
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    model, dev, info = load_model(ckpt)
    s = read_structure(data, filename)
    if not s.is_ordered:
        raise ValueError("the structure has partial occupancies; EpNet needs an ordered cell")
    elements = sorted({sp.symbol for sp in s.species})
    bad = [e for e in elements if e not in info["elements"]]
    if bad:
        raise ValueError(f"elements outside the trained set: {', '.join(bad)}. Trained: {' '.join(info['elements'])}")
    sga = SpacegroupAnalyzer(s, symprec=1e-3)
    prim = sga.get_primitive_standard_structure()
    warnings = []
    if len(prim) > MAX_SITES:
        raise ValueError(f"{len(prim)} sites in the primitive cell; the limit is {MAX_SITES}")
    if len(prim) > WARN_SITES:
        warnings.append(f"{len(prim)} sites: larger than any training cell (<= {WARN_SITES}); expect slower runs and less tested accuracy")
    open3d = [e for e in elements if e in OPEN_SHELL_3D]
    if open3d:
        warnings.append(f"contains {', '.join(open3d)}: computed non-magnetic (no spin polarization, as the training data); "
                        "the real compound may be magnetic")
    if set(elements) & PLUS_U_METALS and set(elements) & {"O", "F"}:
        warnings.append(f"oxide or fluoride of {', '.join(sorted(set(elements) & PLUS_U_METALS))}: Materials Project computes "
                        "these with PBE+U, so they were left out of training; this plain-PBE-like prediction is untested there")
    st = Structure(prim.lattice.matrix, [site.specie.symbol for site in prim], prim.frac_coords)
    n_e = int(sum(Z_MODEL[x] for x in st.species))
    n_occ = int(math.ceil(n_e / 2))
    n_d = sum(x in D_BLOCK for x in st.species)
    n_bands = n_occ + max(int(n_empty), 5 * n_d + 2 * len(st.species) + 4)   # 2026-11-04: with d atoms the empty d manifold
    # alone filled the old 8 empty bands (MoS2: DOS complete only to 2.6 eV); this covers the default plot window (MoS2 12.8 eV)
    path_info = {}
    kfrac, dist, labels, ticks = kpath(prim, line_density, path_info)
    if path_info.get("generic"):
        warnings.append("no standard (Setyawan-Curtarolo) path for this cell: generic path through the zone-face, edge "
                        "and corner midpoints of the primitive cell")
    e = solve_bands(model, dev, st, kfrac, n_bands)
    gap = direct = vbm_k = cbm_k = None
    metal = n_e % 2 == 1
    if not metal:
        vb, cb = e[:, n_occ - 1], e[:, n_occ]
        gap = float(cb.min() - vb.max())
        metal = gap <= 0.0
    # the DOS mesh (the costly part: MoS2 539 k against 179 on the path) needs only the bands that reach DOS_COVER above
    # the top valence band on the path (a VBM, or an upper bound on E_F for metals), plus one; the path keeps them all
    top = float(e[:, n_occ - 1].max())
    reach = [b for b in range(n_occ, n_bands) if e[:, b].min() - top >= DOS_COVER]
    n_mesh = max(n_occ + int(n_empty), reach[0] + 2) if reach else n_bands
    em, wm, mesh = mesh_eigs(model, dev, st, prim, min(n_mesh, n_bands), kspacing)
    if metal:
        ef = fermi_from_mesh(em, wm, n_e)
        ref, ref_name, gap = ef, "E_F", None
        warnings.append(f"metal: E_F from the {mesh[0]}x{mesh[1]}x{mesh[2]} DOS k-mesh (Fermi-Dirac smearing 0.1 eV)")
        if n_e % 2 == 1:
            warnings.append("odd valence-electron count: a metal by electron counting (non-magnetic; spin polarization is not modelled)")
    else:
        iv, ic = int(np.argmax(e[:, n_occ - 1])), int(np.argmin(e[:, n_occ]))
        ref, ref_name = float(e[iv, n_occ - 1]), "VBM"
        direct = bool(np.isclose(e[ic, n_occ] - e[ic, n_occ - 1], gap, atol=1e-3))
        name = lambda i: labels[i] or "k=(" + ", ".join(f"{x:.3f}" for x in kfrac[i]) + ")"  # noqa: E731
        vbm_k, cbm_k = name(iv), name(ic)
        warnings.append("gap from the high-symmetry path (as in Materials Project line-mode bands)")
    de, dg, di, dtop = dos(em, wm, ref, dos_sigma)
    from pymatgen.io.vasp import Poscar
    return Prediction(dos_e=de, dos=dg, dos_int=di, dos_top=dtop, dos_mesh=tuple(mesh), dos_sigma=dos_sigma, formula=prim.composition.reduced_formula, spacegroup=f"{sga.get_space_group_symbol()} ({sga.get_space_group_number()})",
                      nsites=len(prim), n_electrons=n_e, n_occupied=n_occ, metal=bool(metal), reference=ref_name, e_ref_raw=float(ref),
                      gap=gap, gap_direct=direct, vbm_k=vbm_k, cbm_k=cbm_k, distance=dist, kfrac=kfrac, labels=labels, ticks=ticks,
                      bands=e - ref, poscar=Poscar(prim).get_str(), warnings=warnings, timing_s=time.time() - t0, model=info)


def path_breaks(p: Prediction) -> np.ndarray:
    """Indices where a new path branch starts (distance does not advance but k jumps, e.g. A|L): plots insert a gap
    there instead of a vertical connecting line (2026-10-19)."""
    same_d = np.diff(np.asarray(p.distance, float)) <= 1e-12
    jump = np.abs(np.diff(np.asarray(p.kfrac, float), axis=0)).sum(1) > 1e-9
    return np.where(same_d & jump)[0] + 1


def to_csv(p: Prediction) -> str:
    head = ["k_index", "distance_invA", "k1", "k2", "k3", "label"] + [f"band_{i + 1:02d}" for i in range(p.bands.shape[1])]
    lines = [f"# EpNet prediction for {p.formula}, space group {p.spacegroup}; energies in eV relative to the {p.reference};"
             f" bands 1-{p.n_occupied} occupied ({p.n_electrons} valence electrons); k in fractional units of the standardized"
             f" primitive cell (structure.vasp); model {Path(p.model.get('checkpoint', '')).name}", ",".join(head)]
    for i in range(len(p.distance)):
        row = [str(i), f"{p.distance[i]:.6f}"] + [f"{x:.6f}" for x in p.kfrac[i]] + [p.labels[i]] + [f"{x:.5f}" for x in p.bands[i]]
        lines.append(",".join(row))
    return "\n".join(lines) + "\n"


def to_dos_csv(p: Prediction) -> str:
    lines = [f"# EpNet total DOS for {p.formula}: energies in eV relative to the {p.reference}; states/eV/cell (both spins);"
             f" Gaussian sigma {p.dos_sigma} eV on a {'x'.join(map(str, p.dos_mesh))} k-mesh; complete below {p.dos_top:.3f} eV",
             "energy_eV,dos_states_per_eV,integrated_electrons"]
    lines += [f"{a:.4f},{b:.6f},{c:.5f}" for a, b, c in zip(p.dos_e, p.dos, p.dos_int)]
    return "\n".join(lines) + "\n"


def summary(p: Prediction) -> dict:
    return {"formula": p.formula, "space_group": p.spacegroup, "nsites_primitive": p.nsites, "valence_electrons": p.n_electrons,
            "occupied_bands": p.n_occupied, "metal": p.metal, "energy_reference": p.reference,
            "band_gap_eV": None if p.gap is None else round(p.gap, 4), "gap_direct": p.gap_direct, "vbm_k": p.vbm_k, "cbm_k": p.cbm_k,
            "n_kpoints": int(len(p.distance)), "n_bands": int(p.bands.shape[1]), "warnings": p.warnings,
            "dos": {"k_mesh": list(p.dos_mesh), "gaussian_sigma_eV": p.dos_sigma, "complete_below_eV": round(p.dos_top, 3),
                    "units": "states/eV/cell, both spins"},
            "runtime_s": round(p.timing_s, 1), "model": p.model,
            "method": "EpNet empirical pseudopotential Hamiltonian (s, p and d channels; 63 elements) trained on Materials Project PBE bands (non-magnetic, no SOC, no +U)"}


def figure_png(p: Prediction, emin: float = -6.0, emax: float = 6.0) -> bytes:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax, axd) = plt.subplots(1, 2, figsize=(6.6, 4.0), sharey=True, gridspec_kw={"width_ratios": (3.2, 1), "wspace": 0.04})
    sel = (p.dos_e >= emin) & (p.dos_e <= emax)
    axd.fill_betweenx(p.dos_e[sel], 0, p.dos[sel], color="#1f8a8a", alpha=0.25, lw=0)
    axd.plot(p.dos[sel], p.dos_e[sel], color="#1f8a8a", lw=1.0)
    axd.axhline(0, color="#6b6a64", lw=0.7, ls=":")
    if p.dos_top < emax:
        axd.axhspan(p.dos_top, emax, color="#eeeeea", zorder=0)
    axd.set_xlabel("DOS (states/eV)"); axd.set_xlim(0, None); axd.tick_params(axis="y", left=False)
    cut = path_breaks(p)
    for b in range(p.bands.shape[1]):
        ax.plot(np.insert(np.asarray(p.distance, float), cut, np.nan), np.insert(p.bands[:, b], cut, np.nan),
                color="#1f8a8a" if (p.metal or b < p.n_occupied) else "#eb6834", lw=1.1)
    for x, _ in p.ticks:
        ax.axvline(x, color="#d9d8d2", lw=0.7, zorder=0)
    ax.axhline(0, color="#6b6a64", lw=0.7, ls=":")
    ax.set_xticks([t[0] for t in p.ticks], [t[1] for t in p.ticks])
    ax.set_xlim(p.distance[0], p.distance[-1]); ax.set_ylim(emin, emax)
    ax.set_ylabel(f"E − E$_{{{'VBM' if p.reference == 'VBM' else 'F'}}}$ (eV)")
    gap = "metal" if p.metal else f"gap {p.gap:.3f} eV ({'direct' if p.gap_direct else 'indirect'})"
    ax.set_title(f"{p.formula}  {p.spacegroup}  —  {gap}\nEpNet prediction", fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=200); plt.close(fig)
    return buf.getvalue()


if __name__ == "__main__":                       # python -m apps.epnet_bands.core FILE
    import sys
    f = Path(sys.argv[1]); p = predict(f.read_bytes(), f.name)
    print(json.dumps(summary(p), indent=1))
