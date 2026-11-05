"""GeneralEPM: one gate-free plane-wave EPM for many elements (plan G0-G3).

Per-atom local form factor on the unique reciprocal transfers ``G``::

    v_a(G) = v_Z(|G|)                         species base (hypernetwork + per-element residual)
           + sum_k pi_ak r^k_a(G)             K l<=2 residual readouts, softmax-routed
           - 4 pi q_a exp(-s^2 G^2 / 2) / G^2  neutral charge transfer (q_a > 0 for cations)

    V_NL(k) = sum_a sum_{l<=2} sum_r (D^Z_lr + dD_a,lr) |beta_alr(k+G)><beta_alr(k+G)|

Everything learned is a continuous function of element features and the local
environment: no gates, thresholds or family names. The router reads the local
valence-electron density ``n_v(r) = sum_j Z_m,j w(r_aj; r) / (4/3 pi r^3)``
(``zm-v1`` valences), normalized by 0.2 e/A^3 so that pure Si reproduces the
Si general_v1d router input exactly. ``general_from_v1d`` builds a GeneralEPM
that reproduces a v1d checkpoint (see ``tests/test_general_epm.py``).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from epnet.data.element_features import FEATURE_NAMES, periodic_table_features
from epnet.data.elements import SpeciesTable
from epnet.data.structures import Structure
from epnet.data.valence import D_ELEMENTS, Z_MODEL
from epnet.hamiltonian import reciprocal_vectors
from epnet.hamiltonian.fft_grid import flat_positions, grid_shape, potential_to_real
from epnet.hamiltonian.fft_solver import PWOperator, fft_chain_band_energies, fft_stored_band_energies
from epnet.hamiltonian.torch_pw import (
    gather_rows_cols,
    COULOMB_EV_A,
    KINETIC_EV_A2,
    batched_band_energies,
    chain_band_energies,
    fold_to_first_zone,
    g_cutoff_angstrom,
    kinetic_diagonal,
    scatter_hermitian_potential,
    stored_band_energies,
    unique_g_transfers,
    unique_g_transfers_encoded,
)
from epnet.models.baselines.radial import screened_empty_core_form_factor, silicon_cohen_form_factor
from epnet.models.element_encoder import PeriodicTableEmbedding
from epnet.models.equivariant_encoder import (
    EquivariantEncoder,
    SmoothCompactRadialBasis,
    build_periodic_neighbor_graph,
    norm_invariant,
    periodic_displacements,
)
from epnet.models.multipole_form_factor import MultipoleFormFactor
from epnet.physics.long_range import neutralize_effective_charges, reciprocal_coulomb_potential
from epnet.physics.qeq import coulomb_matrix, mulliken_parameters, qeq_charges

V1D_ROUTER_RADII = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
DEFAULT_ROUTER_RADII = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
PAD_KINETIC_EV = 1.0e4  # padded plane waves of a k-centred basis: decoupled, far above every band
# Si valence-electron density (8 e / 40 A^3): pure Si maps n_v/N_REF onto v1d's CN/V/0.05.
N_REF = 0.2
_RCOV = FEATURE_NAMES.index("covalent_radius_angstrom_div_2p5")
_CHI = FEATURE_NAMES.index("pauling_electronegativity_div_4")
L_SIZES = (1, 3, 5)


ELEMENT_FEATURE_SETS = ("v1", "spd-v1")
_S_OCC = FEATURE_NAMES.index("valence_s_div_2")
_P_OCC = FEATURE_NAMES.index("valence_p_div_6")


# atomic channels (AD): radial grid to AC_R_MAX A, q grids for the local-potential and projector transforms
AC_R_MAX, AC_NR, AC_NQ_LOC, AC_NQ_PROJ = 10.0, 2000, 1701, 1001


def _interp_uniform(table: Tensor, x: Tensor, grid: Tensor) -> Tensor:
    """Linear interpolation of ``table`` (..., n) given on the uniform ``grid`` at positions ``x`` (same leading shape);
    zero beyond the grid end. Differentiable in x and in table."""
    r0, h, n = grid[0], grid[1] - grid[0], grid.shape[0]
    pos = (x - r0) / h
    i0 = pos.floor().clamp(0, n - 2)
    f = pos - i0
    i0 = i0.long()
    a, b = torch.gather(table, -1, i0), torch.gather(table, -1, i0 + 1)
    return torch.where(pos <= n - 1, a * (1 - f) + b * f, torch.zeros_like(f))


def _interp_q(table: Tensor, q: Tensor, grid: Tensor) -> Tensor:
    """Per-atom linear interpolation: ``table`` (atom, n) on the uniform q ``grid``, at |k+G| = ``q`` (k, g);
    returns (k, g, atom), zero beyond the grid."""
    h, n = grid[1] - grid[0], grid.shape[0]
    pos = (q.reshape(-1) / h).to(table.dtype)
    i0 = pos.floor().clamp(0, n - 2)
    f = pos - i0
    i0 = i0.long()
    vals = table[:, i0] * (1 - f) + table[:, i0 + 1] * f
    vals = torch.where(pos <= n - 1, vals, torch.zeros_like(vals))
    return vals.T.reshape(*q.shape, table.shape[0])


def element_feature_table(symbols: Sequence[str], feature_set: str = "v1") -> Tensor:
    """Periodic-table features plus model valence Z_m / 12, one row per species.

    ``spd-v1`` (EpNet-spd) appends the model d count / 10, Z_m minus the valence s and p occupancy: 10 for Zn and
    Cu, 2 for Ti, 0 for Ga-Bi, whose filled d is semicore (dropped by the contract) although ``valence_d_div_10`` is
    1 for them. It is 0 for every s/p element, so a v1 model padded with a zero input column is unchanged."""
    if feature_set not in ELEMENT_FEATURE_SETS:
        raise ValueError(f"element_feature_set must be one of {ELEMENT_FEATURE_SETS}")
    base = periodic_table_features(list(symbols), dtype=torch.float32)
    zm = torch.tensor([[Z_MODEL[s] / 12.0] for s in symbols], dtype=torch.float32)
    columns = [base, zm]
    if feature_set == "spd-v1":
        n_d = torch.tensor([float(Z_MODEL[s]) for s in symbols]) - 2.0 * base[:, _S_OCC] - 6.0 * base[:, _P_OCC]
        columns.append((n_d.round() / 10.0)[:, None])
    return torch.cat(columns, dim=-1)


def cosine_basis(q: Tensor, n_basis: int, q_cutoff: float) -> Tensor:
    """Same raised-cosine basis as ``CosineRadialEP.basis``."""
    x = (q / q_cutoff).clamp(0, 1)
    envelope = 0.5 * (1.0 + torch.cos(math.pi * x))
    orders = torch.arange(n_basis, device=q.device, dtype=q.dtype)
    return envelope.unsqueeze(-1) * torch.cos(math.pi * x.unsqueeze(-1) * orders)


def fit_local_basis(q: np.ndarray, v_q: np.ndarray, n_basis: int, q_cut: float, n_tail: int, q_tail: float):
    """Least-squares coefficients (cosine basis to q_cut, tail basis to q_tail) of a form factor v(q) given on ``q``
    (which should reach q_tail); returns (coef (n_basis,), tail (n_tail,) or None)."""
    qt = torch.as_tensor(q, dtype=torch.float64)
    cols = [cosine_basis(qt, n_basis, q_cut)]
    if n_tail:
        cols.append(cosine_basis(qt, n_tail, q_tail))
    a = torch.cat(cols, dim=1).numpy()
    sol = np.linalg.lstsq(a.T @ a + 1e-10 * np.eye(a.shape[1]), a.T @ v_q, rcond=None)[0]
    return sol[:n_basis], (sol[n_basis:] if n_tail else None)


def eval_local_basis(q: np.ndarray, coef: np.ndarray, tail: np.ndarray | None, n_basis: int, q_cut: float, n_tail: int,
                     q_tail: float) -> np.ndarray:
    qt = torch.as_tensor(q, dtype=torch.float64)
    v = cosine_basis(qt, n_basis, q_cut).numpy() @ coef
    if n_tail:
        v = v + cosine_basis(qt, n_tail, q_tail).numpy() @ tail
    return v


def ae_local_table(symbol: str):
    """(r, V_loc(r)) of the element's all-electron-tail local potential (step 5; ``vloc`` in
    epnet/data/atomic_channels/<El>.npz), or None if it has none (it failed the 3d gate)."""
    path = Path(__file__).resolve().parents[1] / "data" / "atomic_channels" / f"{symbol}.npz"
    if not path.exists():
        return None
    z = np.load(path)
    return (z["r"], z["vloc"]) if "vloc" in z.files else None


def prior_q_max(ecuts: Sequence[float]) -> float:
    """q range (1/A) of the tabulated prior: the unique G transfers of the highest cutoff (2 g_cut) plus 4."""
    return 2.0 * g_cutoff_angstrom(max(ecuts)) + 4.0


def plateau_weights(distance: Tensor, radius: float) -> Tensor:
    """Weight 1 inside 0.8 R, raised cosine to 0 at R (as ``smooth_coordination``)."""
    x = distance / radius
    t = ((x - 0.8) / 0.2).clamp(0.0, 1.0)
    w = torch.where(x <= 0.8, torch.ones_like(distance), 0.5 * (1.0 + torch.cos(math.pi * t)))
    return torch.where((distance > 0) & (distance < radius), w, torch.zeros_like(w))


def d_harmonics(u: Tensor) -> Tensor:
    """The five real l = 2 harmonics of unit vectors ``u`` (..., 3), exactly as the l = 2 projectors use them
    (order xy, yz, zx, x^2-y^2, 3z^2-1; sum of squares 5)."""
    ux, uy, uz = u.unbind(dim=-1)
    d = torch.stack(
        (
            math.sqrt(3.0) * ux * uy,
            math.sqrt(3.0) * uy * uz,
            math.sqrt(3.0) * uz * ux,
            0.5 * math.sqrt(3.0) * (ux.square() - uy.square()),
            0.5 * (3.0 * uz.square() - 1.0),
        ),
        dim=-1,
    )
    return math.sqrt(5.0) * d


def environment_descriptors(
    packed: SimpleNamespace,
    z_model: Tensor,
    chi: Tensor,
    router_radii: Sequence[float],
    contrast_radius: float,
) -> dict[str, Tensor]:
    """Geometry-only per-atom descriptors, computed once per cell.

    ``router``: log(n_v(r)/N_REF) at each radius plus the electronegativity
    contrast of the ``contrast_radius`` shell. ``cn3``/``cn10``: plain plateau
    coordination counts for the nonlocal head.
    """
    radii = [float(r) for r in router_radii]
    rmax = max(radii + [float(contrast_radius), 10.0])
    edge_index, shifts = build_periodic_neighbor_graph(packed.positions, packed.lattice, rmax)
    distance = torch.linalg.vector_norm(periodic_displacements(packed.positions, edge_index, packed.lattice, shifts), dim=-1)
    center, neighbor = edge_index
    species = packed.species
    n = packed.positions.shape[0]

    def shell_sum(radius: float, values: Tensor) -> Tensor:
        out = packed.positions.new_zeros(n)
        return out.index_add_(0, center, plateau_weights(distance, radius) * values)

    zj = z_model[species[neighbor]].to(distance)
    columns = []
    for r in radii:
        n_v = shell_sum(r, zj) / (4.0 / 3.0 * math.pi * r**3)
        columns.append(torch.log((n_v / N_REF).clamp_min(1e-3)))
    chi_diff = (chi[species[center]] - chi[species[neighbor]]).abs().to(distance)
    weight = shell_sum(contrast_radius, torch.ones_like(distance))
    contrast = shell_sum(contrast_radius, chi_diff) / weight.clamp_min(1e-6)
    ones = torch.ones_like(distance)
    return {
        "router": torch.stack(columns + [contrast], dim=-1),
        "cn3": shell_sum(3.0, ones),
        "cn10": shell_sum(10.0, ones),
    }


def pack_general(structure: Structure, species_table: SpeciesTable, dtype: torch.dtype = torch.float32) -> SimpleNamespace:
    lattice = torch.as_tensor(np.asarray(structure.lattice), dtype=dtype)
    frac = torch.as_tensor(np.asarray(structure.fractional_positions), dtype=dtype)
    species = torch.as_tensor(species_table.encode(structure.species), dtype=torch.long)
    n_occ = structure.metadata.get("n_occupied_model")
    if n_occ is None:
        electrons = sum(Z_MODEL[s] for s in structure.species)
        n_occ = electrons // 2 if electrons % 2 == 0 else None
    return SimpleNamespace(
        lattice=lattice,
        fractional_positions=frac,
        positions=frac @ lattice,
        species=species,
        volume=torch.as_tensor(structure.volume, dtype=dtype),
        n_atoms=structure.n_atoms,
        n_occupied=None if n_occ is None else int(n_occ),
    )


_D_METALS = frozenset("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg".split())


def _d_orbital_radius(z) -> float:
    """<r> (A) of the all-electron d orbital in an atomic_channels npz (u = r R or R, decided by the normalization)."""
    r, u = z["r"], z["ua2"]
    n1, n2 = np.trapz(u ** 2, r), np.trapz(r ** 2 * u ** 2, r)
    val = np.trapz(r * u ** 2, r) / n1 if abs(n1 - 1.0) <= abs(n2 - 1.0) else np.trapz(r ** 3 * u ** 2, r) / n2
    return float(val) * 0.529177


class GeneralEPM(nn.Module):
    def __init__(
        self,
        species_symbols: Sequence[str],
        *,
        channels: int = 32,
        cutoff: float = 6.0,
        num_radial: int = 8,
        n_readouts: int = 4,
        scalar_cosine_basis: int = 12,
        scalar_cosine_cutoff: float = 8.0,
        sphere_n_basis: int = 12,
        sphere_q_cutoff: float = 8.0,
        sphere_tail_n_basis: int = 0,
        fft_ng_threshold: int | None = None,
        sphere_tail_q_cutoff: float = 17.0,
        router_radii: Sequence[float] = DEFAULT_ROUTER_RADII,
        router_hidden: int = 16,
        contrast_radius: float = 4.0,
        projector_l_max: int = 2,
        projector_width_constants: Sequence[float] = (1.665, 3.885),
        charge_transfer: bool = True,
        charge_width: float = 0.5,
        hyper_hidden: int = 32,
        ecut_ry: float = 8.0,
        subset_ng_min: int = 4096,
        k_batch: int | None = None,
        projector_offdiag_l: Sequence[int] = (),
        charge_coupling: bool = False,
        charge_model: str = "head",
        projector_q_width_max: float | None = None,
        k_centred_basis: bool = False,
        firstrow_lmax1: bool = False,
        projector_radii_firstrow: Sequence[float] | None = None,
        projector_radius_mode: str = "rcov",
        projector_orbital_multipliers: Sequence[float] | None = None,
        learn_projector_radii: bool = False,
        ecut_by_element: dict[str, float] | None = None,
        base_prior: bool = False,
        encoder_layers: int = 1,
        invariant_norm: str = "sqrt",
        invariant_scale: float = 0.05,
        projector_d_radius: str = "p",
        solver: str = "eigh",
        element_feature_set: str = "v1",
        d_overlap: bool = False,
        d_crystal_field: bool = False,
        crystal_field_cutoff: float = 4.0,
        d_prior_valence: str = "full",
        atomic_channels: Sequence[int] = (),
        atomic_ultrasoft: bool = True,
        atomic_free_d_slot: bool = True,
        d_species_learn_scale: float = 1.0,
        atomic_learn_l: Sequence[int] = (0, 1, 2),
        atomic_two_ref_d: bool = False,
        atomic_atom_shift: bool = False,
        atomic_two_ref_stable: bool = False,
        atomic_two_ref_safe: bool = False,
        d_charge_mode: str | None = None,
        d_coord_term: bool = False,
    ) -> None:
        super().__init__()
        self.species_symbols = tuple(str(s) for s in species_symbols)
        missing = [s for s in self.species_symbols if s not in Z_MODEL]
        if missing:
            raise ValueError(f"elements outside the zm-v1 contract: {missing}")
        if n_readouts < 1 or projector_l_max not in (0, 1, 2):
            raise ValueError("n_readouts must be >= 1 and projector_l_max in {0, 1, 2}")
        self.species_table = SpeciesTable.from_symbols(self.species_symbols)
        self.config = dict(
            channels=int(channels), cutoff=float(cutoff), num_radial=int(num_radial), n_readouts=int(n_readouts),
            scalar_cosine_basis=int(scalar_cosine_basis), scalar_cosine_cutoff=float(scalar_cosine_cutoff),
            sphere_n_basis=int(sphere_n_basis), sphere_q_cutoff=float(sphere_q_cutoff),
            sphere_tail_n_basis=int(sphere_tail_n_basis), sphere_tail_q_cutoff=float(sphere_tail_q_cutoff),
            fft_ng_threshold=None if fft_ng_threshold is None else int(fft_ng_threshold),
            router_radii=[float(r) for r in router_radii], router_hidden=int(router_hidden),
            contrast_radius=float(contrast_radius), projector_l_max=int(projector_l_max),
            projector_width_constants=[float(c) for c in projector_width_constants],
            charge_transfer=bool(charge_transfer), charge_width=float(charge_width), hyper_hidden=int(hyper_hidden),
            ecut_ry=float(ecut_ry), subset_ng_min=int(subset_ng_min), k_batch=k_batch,
            projector_offdiag_l=[int(l) for l in projector_offdiag_l],
            charge_coupling=bool(charge_coupling), charge_model=str(charge_model),
            projector_q_width_max=None if projector_q_width_max is None else float(projector_q_width_max),
            k_centred_basis=bool(k_centred_basis),
            firstrow_lmax1=bool(firstrow_lmax1),
            projector_radii_firstrow=None if projector_radii_firstrow is None else [float(x) for x in projector_radii_firstrow],
            projector_radius_mode=str(projector_radius_mode),
            projector_orbital_multipliers=None if projector_orbital_multipliers is None else [float(x) for x in projector_orbital_multipliers],
            learn_projector_radii=bool(learn_projector_radii),
            ecut_by_element=None if not ecut_by_element else {str(k): float(v) for k, v in ecut_by_element.items()},
            base_prior=bool(base_prior), encoder_layers=int(encoder_layers),
            invariant_norm=str(invariant_norm), invariant_scale=float(invariant_scale),
            projector_d_radius=str(projector_d_radius), solver=str(solver),
            element_feature_set=str(element_feature_set),
            d_overlap=bool(d_overlap), d_crystal_field=bool(d_crystal_field), crystal_field_cutoff=float(crystal_field_cutoff),
            d_prior_valence=str(d_prior_valence), atomic_channels=[int(l) for l in atomic_channels],
            atomic_ultrasoft=bool(atomic_ultrasoft), atomic_free_d_slot=bool(atomic_free_d_slot),
            d_species_learn_scale=float(d_species_learn_scale), atomic_learn_l=[int(l) for l in atomic_learn_l],
            atomic_two_ref_d=bool(atomic_two_ref_d), atomic_atom_shift=bool(atomic_atom_shift),
            atomic_two_ref_stable=bool(atomic_two_ref_stable), atomic_two_ref_safe=bool(atomic_two_ref_safe),
            d_charge_mode=None if d_charge_mode is None else str(d_charge_mode), d_coord_term=bool(d_coord_term),
        )
        if self.config["d_charge_mode"] not in (None, "nonneg"):
            raise ValueError("d_charge_mode is None or 'nonneg'")
        if self.config["atomic_two_ref_safe"] and not self.config["atomic_two_ref_d"]:
            raise ValueError("atomic_two_ref_safe needs atomic_two_ref_d")
        if self.config["atomic_two_ref_d"] and (2 not in self.config["atomic_channels"] or not self.config["atomic_ultrasoft"]):
            raise ValueError("atomic_two_ref_d needs the ultrasoft atomic d channel (atomic_channels with 2)")
        if self.config["atomic_atom_shift"] and not self.config["atomic_two_ref_d"]:
            raise ValueError("atomic_atom_shift is implemented for the two-reference d channel")
        if self.config["atomic_two_ref_d"] and self.config["projector_offdiag_l"]:
            raise ValueError("atomic_two_ref_d and projector_offdiag_l both couple projector slots; use one")
        if self.config["atomic_channels"] and (projector_l_max < 2 or self.config["d_prior_valence"] not in ("s", "ae")
                                               or any(l not in (0, 1, 2) for l in self.config["atomic_channels"])):
            raise ValueError("atomic_channels (subset of 0, 1, 2) need projector_l_max = 2 and d_prior_valence 's' or 'ae'")
        if self.config["atomic_channels"] and self.config["atomic_ultrasoft"] and self.config["d_overlap"]:
            raise ValueError("atomic_ultrasoft and d_overlap both define S; use one")
        if self.config["d_prior_valence"] not in ("full", "s", "ae"):
            raise ValueError("d_prior_valence is 'full' (Z_m, default), 's' (Z_m minus the model d count) or 'ae' "
                             "(all-electron neutral-atom potential beyond r_loc, epnet/data/atomic_channels)")
        if (self.config["d_overlap"] or self.config["d_crystal_field"]) and projector_l_max < 2:
            raise ValueError("d_overlap / d_crystal_field act on the l = 2 projectors: projector_l_max must be 2")
        if self.config["charge_coupling"] and not self.config["charge_transfer"]:
            raise ValueError("charge_coupling needs charge_transfer")
        if self.config["charge_model"] not in ("head", "qeq", "head+qeq") or (
            self.config["charge_model"] in ("qeq", "head+qeq") and not self.config["charge_coupling"]
        ):
            raise ValueError("charge_model is 'head', 'qeq' or 'head+qeq'; the QEq models need charge_coupling")
        if any(not 0 <= l < projector_l_max + 1 for l in self.config["projector_offdiag_l"]) or (
            self.config["projector_offdiag_l"] and len(self.config["projector_width_constants"]) < 2
        ):
            raise ValueError("projector_offdiag_l needs l <= projector_l_max and at least two radial widths")
        c = self.config
        features = element_feature_table(self.species_symbols, c["element_feature_set"])
        self.register_buffer("element_features", features)
        self.register_buffer("z_model", torch.tensor([Z_MODEL[s] for s in self.species_symbols], dtype=torch.float32))
        self.register_buffer("r_cov", features[:, _RCOV] * 2.5)
        self.register_buffer("chi", features[:, _CHI] * 4.0)
        self.register_buffer("width_constants", torch.tensor(c["projector_width_constants"], dtype=torch.float32))
        # first-row atoms (B, C, N, O, F): nodeless 2p; optional l <= 1 projectors and 2s/2p-sized radial functions
        self.register_buffer("first_row", torch.tensor([s in ("B", "C", "N", "O", "F") for s in self.species_symbols]), persistent=False)
        # d-block atoms (zm-v2 D_ELEMENTS): the only atoms the d overlap and crystal-field terms act on
        self.register_buffer("is_d", torch.tensor([s in D_ELEMENTS for s in self.species_symbols]), persistent=False)
        if c["projector_radii_firstrow"] is not None and len(c["projector_radii_firstrow"]) != len(c["projector_width_constants"]):
            raise ValueError("projector_radii_firstrow needs one radius per radial projector")
        # Element-specific projector radii: each channel l placed at multiples of the element's own valence
        # orbital radius (s for l=0, p for l=1 and for the d-like l=2 channel), from an all-electron LDA atom
        # (epnet/physics/atomic_radii.py -> epnet/data/valence_orbital_radii.json) instead of Si constants / r_cov.
        if c["projector_radius_mode"] not in ("rcov", "orbital"):
            raise ValueError("projector_radius_mode is 'rcov' or 'orbital'")
        n_rad = len(c["projector_width_constants"])
        if c["projector_radius_mode"] == "orbital":
            table = json.loads((Path(__file__).resolve().parents[1] / "data" / "valence_orbital_radii.json").read_text())["radii"]
            missing = [s for s in self.species_symbols if s not in table or "s" not in table[s]]
            if missing:
                raise ValueError(f"no valence orbital radii for {missing}")
            # H has no bound p state in the neutral LDA atom: its l >= 1 channels (empirical) use the 1s radius
            p_rad = lambda s: table[s].get("p", table[s]["s"])  # noqa: E731
            # l = 2 channel: the valence p radius (default), or with projector_d_radius="d" the element's lowest
            # empty (n-1)d radius where one is tabulated (Ca/Sr/Ba: 3d/4d/5d form their conduction-band edge)
            if c["projector_d_radius"] not in ("p", "d"):
                raise ValueError("projector_d_radius is 'p' or 'd'")
            d_rad = (lambda s: table[s].get("d", p_rad(s))) if c["projector_d_radius"] == "d" else p_rad  # noqa: E731
            orb = [[table[s]["s"], p_rad(s), d_rad(s)][: c["projector_l_max"] + 1] for s in self.species_symbols]
            self.register_buffer("orbital_radii", torch.tensor(orb, dtype=torch.float32))
            mult = c["projector_orbital_multipliers"] or [1.0, 0.5][:n_rad]
            if len(mult) != n_rad:
                raise ValueError("projector_orbital_multipliers needs one value per radial projector")
            self.register_buffer("orbital_multipliers", torch.tensor(mult, dtype=torch.float32))
        self.projector_log_mult, self.projector_log_offset = None, None
        if c["learn_projector_radii"]:
            if c["projector_radius_mode"] != "orbital":
                raise ValueError("learn_projector_radii needs projector_radius_mode='orbital'")
            self.projector_log_mult = nn.Parameter(torch.zeros(n_rad))
            self.projector_log_offset = nn.Parameter(torch.zeros(len(self.species_symbols), c["projector_l_max"] + 1))
        self.n_basis = c["sphere_n_basis"]
        self.n_proj_radial = len(c["projector_width_constants"])
        self.n_l = c["projector_l_max"] + 1

        # Species base: hypernetwork(features) + per-element residual, both over
        # [cosine coefficients | projector strengths D_{r,l}].
        # Optional high-q tail of the species base: a second cosine basis on [0, sphere_tail_q_cutoff], appended after the
        # projector strengths and zero at init (2026-10-16: first-row atoms at 20 Ry need v(|G-G'|) to 16.9 1/A while
        # the main basis is enveloped to zero at 8). Layout: [cosines | projector strengths | tail cosines].
        self.n_tail = int(c.get("sphere_tail_n_basis", 0))
        n_base = self.n_basis + self.n_proj_radial * self.n_l + self.n_tail
        self.base_hyper = nn.Sequential(nn.Linear(features.shape[1], c["hyper_hidden"]), nn.SiLU(), nn.Linear(c["hyper_hidden"], n_base))
        nn.init.zeros_(self.base_hyper[-1].weight)
        nn.init.zeros_(self.base_hyper[-1].bias)
        self.base_residual = nn.Parameter(torch.zeros(len(self.species_symbols), n_base))
        # base_prior: the physical start (screened empty core from Z_m and r_cov, both element features) is a fixed
        # buffer, so it exists for an unseen element too; the residual table starts at zero and is penalized
        # (base_residual_penalty), and what the prior misses must be learned by the shared hypernetwork.
        # Default (legacy): the start sits in the residual table, which can then absorb everything per element.
        self.register_buffer("base_prior", torch.zeros(len(self.species_symbols), n_base), persistent=bool(c["base_prior"]))
        self._init_base_collocation(self.base_prior if c["base_prior"] else self.base_residual, cohen_si=not c["base_prior"])
        self._init_prior_table()
        # guard B (Stage 2 follow-up): the learned species-level deviation (hypernetwork + residual) of d elements enters
        # x d_species_learn_scale, so with the species lr 0.3 its effective rate is ~0.3 x scale; the prior is unscaled
        self.register_buffer("species_learn_scale", None, persistent=False)
        if float(c.get("d_species_learn_scale", 1.0)) != 1.0:
            scale = torch.tensor([float(c["d_species_learn_scale"]) if s in D_ELEMENTS else 1.0 for s in self.species_symbols])
            self.register_buffer("species_learn_scale", scale[:, None], persistent=False)

        self.encoder = EquivariantEncoder(len(self.species_symbols), channels=c["channels"], cutoff=c["cutoff"], num_radial=c["num_radial"],
                                          interactions=c["encoder_layers"], invariant_norm=c.get("invariant_norm", "sqrt"),
                                          invariant_scale=c.get("invariant_scale", 0.05))
        self.encoder.species_embedding = PeriodicTableEmbedding(features, c["channels"])
        self.readouts = nn.ModuleList(
            [
                MultipoleFormFactor(
                    channels=c["channels"], num_radial=c["num_radial"], l_max=2,
                    scalar_cosine_basis=c["scalar_cosine_basis"], scalar_cosine_cutoff=c["scalar_cosine_cutoff"],
                )
                for _ in range(c["n_readouts"])
            ]
        )
        n_router_in = len(c["router_radii"]) + 1
        self.router = nn.Sequential(nn.Linear(n_router_in, c["router_hidden"]), nn.SiLU(), nn.Linear(c["router_hidden"], c["n_readouts"]))
        self.register_buffer("router_mean", torch.zeros(n_router_in))
        self.register_buffer("router_std", torch.ones(n_router_in))
        n_inv = 3 * c["channels"]
        self.charge_head = nn.Linear(c["channels"], 1, bias=False) if c["charge_transfer"] else None
        # Charge-coupled species base: v_Z(q) + q_a w_Z(q) and D_Z + q_a E_Z, with the charge response
        # [w_Z | E_Z] from element features (+ per-element residual). The charge head is zero-initialized,
        # so the starting Hamiltonian is unchanged; a small random w lets the coupling receive gradients.
        self.charge_hyper, self.charge_residual = None, None
        if c["charge_coupling"]:
            self.charge_hyper = nn.Sequential(nn.Linear(features.shape[1], c["hyper_hidden"]), nn.SiLU(), nn.Linear(c["hyper_hidden"], n_base))
            nn.init.normal_(self.charge_hyper[-1].weight, std=1e-3)
            nn.init.zeros_(self.charge_hyper[-1].bias)
            self.charge_residual = nn.Parameter(torch.zeros(len(self.species_symbols), n_base))
        if self.charge_head is not None:
            nn.init.zeros_(self.charge_head.weight)
        # QEq charges (charge_model="qeq"): chi_a = chi0_Z + dchi_Z + head(features_a) and J_Z = J0_Z exp(dlogJ_Z)
        # from Mulliken IE/EA (J scaled within [0.75, 1.34]), screened Ewald Coulomb kappa*C (cache.qeq_coulomb). The charges are physical
        # from step 0, so the Hamiltonian terms that use them start at zero instead: the coupling [w|E] and
        # the long-range Coulomb term (scale coulomb_scale) are zero-initialized.
        self.qeq_hyper, self.qeq_residual = None, None
        # "head+qeq" (2026-10-17, warm starts): the trained charge-head charges and their Coulomb term are kept exactly;
        # QEq charges (chi from Mulliken chi0 + learned per-element shifts, no head term) are added through a zero-init
        # gate, so step 0 equals the source model and training decides how much electronegativity equalization to use.
        if c["charge_model"] in ("qeq", "head+qeq"):
            chi0, j0 = mulliken_parameters(self.species_symbols)
            self.register_buffer("qeq_chi0", chi0)
            self.register_buffer("qeq_j0", j0)
            self.qeq_hyper = nn.Sequential(nn.Linear(features.shape[1], c["hyper_hidden"]), nn.SiLU(), nn.Linear(c["hyper_hidden"], 2))
            nn.init.zeros_(self.qeq_hyper[-1].weight)
            nn.init.zeros_(self.qeq_hyper[-1].bias)
            self.qeq_residual = nn.Parameter(torch.zeros(len(self.species_symbols), 2))
            self.qeq_kappa_raw = nn.Parameter(torch.tensor(math.log((0.2 - 0.02) / (0.25 - 0.2))))
            if c["charge_model"] == "qeq":
                self.coulomb_scale = nn.Parameter(torch.zeros(()))
            else:
                self.qeq_gate = nn.Parameter(torch.zeros(()))
            nn.init.zeros_(self.charge_hyper[-1].weight)
        self.projector_head = nn.Sequential(nn.Linear(n_inv + 2, 16), nn.SiLU(), nn.Linear(16, self.n_proj_radial * self.n_l))
        nn.init.zeros_(self.projector_head[-1].weight)
        nn.init.zeros_(self.projector_head[-1].bias)
        # Optional radial coupling D_{r r'} for the listed l (zero init = the diagonal model).
        coupled = c["projector_offdiag_l"]
        self.projector_offdiag = (
            nn.Parameter(torch.zeros(len(self.species_symbols), len(coupled), self.n_proj_radial * (self.n_proj_radial - 1) // 2))
            if coupled else None
        )
        # d overlap (EpNet-spd M4): S = 1 + sum_{a in d, r} q_{Z,r} sum_m |beta_{a,r,d,m}><beta_{a,r,d,m}|, H psi = E S psi.
        # q = softplus(raw) >= 0 (ultrasoft-like augmentation of the d channel, S stays positive definite); raw starts
        # at -5 (q ~ 0.007 A^3), a species-level quantity trained with the species base.
        self.d_overlap_raw = (nn.Parameter(torch.full((len(self.species_symbols), self.n_proj_radial), -5.0))
                              if c["d_overlap"] else None)
        # d crystal field (EpNet-spd M5): sigma-only angular-overlap term dD_{a,r} = sum_j e_r(a, j) y_j y_j^T on the d
        # block of d atoms, y_j = the projector's own d harmonics along the bond a -> j (unit norm). Equivariant by
        # construction; contains the l = 4 (cubic) splitting that l <= 2 encoder features cannot express.
        # e_r from bond length and the two species embeddings, zero-initialized.
        self.cf_basis, self.cf_net = None, None
        if c["d_crystal_field"]:
            self.cf_basis = SmoothCompactRadialBasis(c["num_radial"], c["crystal_field_cutoff"])
            self.cf_net = nn.Sequential(nn.Linear(c["num_radial"] + 2 * c["channels"], 32), nn.SiLU(), nn.Linear(32, self.n_proj_radial))
            nn.init.zeros_(self.cf_net[-1].weight)
            nn.init.zeros_(self.cf_net[-1].bias)
        # Atomic-orbital channels for d elements (EpNet-spd, docs/d_channel_plan.md, AD): the r = 0 projector slot of
        # each listed l becomes chi_l = (eps_l - T - V_loc) phi_l with strength pi^3 / <phi_l|chi_l> (Kleinman-Bylander /
        # Vanderbilt construction from the atom's pseudo-orbital phi_l, epnet/data/atomic_channels), rebuilt from the
        # current species local potential every forward pass. Learned per element and l: a level shift (eV) and an
        # extent scale exp(0.3 tanh x) of phi; both start at zero change.
        # atomic_ultrasoft (step 3c, default): phi keeps the all-electron tail (not renormalized) and its norm deficit q
        # is restored by S = 1 + |beta> q <beta| (Vanderbilt 1990); H psi = e S psi is solved through a Cholesky
        # transform. The augmentation feels the cell's local potential at the atom (see projector_strengths).
        self.ac_on = bool(c["atomic_channels"])
        self.ac_eps_shift, self.ac_extent = None, None
        if self.ac_on:
            self._init_atomic_channels()
        self._last: dict[str, Tensor] = {}
        self.atom_base_delta: Tensor | None = None  # set per cell by diagnostics (scripts/environment_map_cv.py)
        # diagnostic hook, set per cell (scripts/stage1_zero_shot.py): per-atom 1/0, 0 switches off that atom's
        # environment terms (routed readout form factors, projector head, charge; the other charges are re-neutralized)
        self.atom_env_keep: Tensor | None = None

    # ------------------------------------------------------------------ species base
    def _init_base_collocation(self, out: Tensor, cohen_si: bool = True) -> None:
        q = torch.linspace(0.0, self.config["sphere_q_cutoff"], 81)
        basis = cosine_basis(q, self.n_basis, self.config["sphere_q_cutoff"])
        gram = basis.T @ basis + 1e-10 * torch.eye(self.n_basis)
        with torch.no_grad():
            for i, s in enumerate(self.species_symbols):
                if self.config.get("d_prior_valence") == "ae" and s in D_ELEMENTS and ae_local_table(s) is not None:
                    continue   # the all-electron-tail local is the fixed table prior_table (_init_prior_table)
                if s == "Si" and cohen_si:   # Si-specific; not derivable from element features
                    target = silicon_cohen_form_factor(q)
                else:  # Ashcroft core radius ~0.53 r_cov (Na 0.88, K 1.08, Al 0.64 A)
                    z = Z_MODEL[s]
                    if self.config.get("d_prior_valence") in ("s", "ae") and s in D_ELEMENTS:   # 'ae' fallback
                        # localized-d test (2026-10-20): the local start sees only the s (and p) valence; the d
                        # electrons are left to the l = 2 projector (Zn 12 -> 2, Cu 11 -> 1, Ti 4 -> 2)
                        row = periodic_table_features([s], dtype=torch.float32)[0]
                        z = int(round(2.0 * float(row[_S_OCC]) + 6.0 * float(row[_P_OCC])))
                    target = screened_empty_core_form_factor(q, z, 0.53 * float(self.r_cov[i]))
                out[i, : self.n_basis] = torch.linalg.solve(gram, basis.T @ target)

    def _init_prior_table(self) -> None:
        """d_prior_valence = "ae" (step 5): a fixed tabulated form factor per species (zero for others), v_prior(q) =
        4 pi int r^2 V_loc j0(q r) dr on a uniform grid to prior_q_max, zero beyond; added to the learned form factor
        wherever it is evaluated. A basis fit cannot hold the neutral-atom tail (to ~6 A; 26 tail functions reach ~4.6 A)."""
        self.register_buffer("prior_q", None, persistent=False)
        self.register_buffer("prior_table", None, persistent=False)
        if self.config.get("d_prior_valence") != "ae":
            return
        from epnet.physics.atomic_channels import radial_form_factor

        ecuts = [self.config["ecut_ry"], 30.0] + list((self.config.get("ecut_by_element") or {}).values())
        q = np.linspace(0.0, prior_q_max(ecuts), 1601)
        table = np.zeros((len(self.species_symbols), q.size))
        for i, sym in enumerate(self.species_symbols):
            tab = ae_local_table(sym) if sym in D_ELEMENTS else None
            if tab is not None:
                table[i] = radial_form_factor(tab[1], tab[0], q)
        self.register_buffer("prior_q", torch.as_tensor(q, dtype=torch.float32), persistent=False)
        self.register_buffer("prior_table", torch.as_tensor(table, dtype=torch.float32), persistent=False)

    def prior_at(self, q: Tensor) -> Tensor:
        """Tabulated prior form factor at |q| (any shape): (S, *q.shape), linear interpolation, zero beyond the grid."""
        grid, table = self.prior_q.to(q), self.prior_table.to(q)
        h, n = grid[1] - grid[0], grid.shape[0]
        pos = q / h
        i0 = pos.floor().clamp(0, n - 2)
        f = pos - i0
        i0 = i0.long()
        flat_i, flat_f = i0.reshape(-1), f.reshape(-1)
        v = table[:, flat_i] * (1 - flat_f) + table[:, flat_i + 1] * flat_f
        v = torch.where((pos.reshape(-1) <= n - 1)[None, :], v, torch.zeros_like(v))
        return v.reshape(table.shape[0], *q.shape)

    def _species_values(self) -> Tensor:
        learned = self.base_hyper(self.element_features) + self.base_residual
        if self.species_learn_scale is not None:
            learned = learned * self.species_learn_scale.to(learned)
        return learned + self.base_prior

    def species_parameters(self) -> tuple[Tensor, Tensor]:
        """(S, n_basis) cosine coefficients and (S, n_radial, n_l) projector strengths."""
        values = self._species_values()
        end = self.n_basis + self.n_proj_radial * self.n_l
        return values[:, : self.n_basis], values[:, self.n_basis : end].reshape(-1, self.n_proj_radial, self.n_l)

    def species_tail(self) -> Tensor | None:
        """(S, n_tail) coefficients of the high-q tail basis, or None."""
        if not self.n_tail:
            return None
        values = self._species_values()
        return values[:, -self.n_tail :]

    def charge_response(self) -> tuple[Tensor, Tensor]:
        """(S, n_basis) form-factor response w_Z and (S, n_radial, n_l) projector response E_Z to atomic charge."""
        values = self.charge_hyper(self.element_features) + self.charge_residual
        end = self.n_basis + self.n_proj_radial * self.n_l
        return values[:, : self.n_basis], values[:, self.n_basis : end].reshape(-1, self.n_proj_radial, self.n_l)

    def qeq_kappa(self) -> Tensor:
        """Coulomb screening in (0.02, 0.25). With J scaled by >= 0.75 the effective kappa/J stays <= 1/3;
        kappa C + J is positive definite on all 515 first-row train/val cells up to kappa = 0.35 (not at 0.45)."""
        return 0.02 + 0.23 * torch.sigmoid(self.qeq_kappa_raw)

    def atomic_charges(self, features: dict[str, Tensor], cache: SimpleNamespace | None = None,
                       species: Tensor | None = None) -> Tensor | None:
        if self.charge_head is None:
            return None
        if self.qeq_hyper is not None:
            if cache is None or species is None:
                raise ValueError("QEq charges need the PW cache and the species")
            hybrid = self.config["charge_model"] == "head+qeq"
            shift = self.qeq_hyper(self.element_features) + self.qeq_residual  # (S, 2)
            chi = (self.qeq_chi0 + shift[:, 0])[species]
            if not hybrid:
                chi = chi + self.charge_head(features["scalar"]).squeeze(-1)
            hardness = (self.qeq_j0 * torch.exp(0.29 * torch.tanh(shift[:, 1])))[species]  # x [0.75, 1.34]
            charges = qeq_charges(cache.qeq_coulomb, chi, hardness, kappa=self.qeq_kappa())
            if hybrid:
                charges = neutralize_effective_charges(self.charge_head(features["scalar"]).squeeze(-1)) + self.qeq_gate * charges
        elif self.config.get("d_charge_mode") == "nonneg" and species is not None:
            charges = self._d_nonneg_charges(self.charge_head(features["scalar"]).squeeze(-1), species)
        else:
            charges = neutralize_effective_charges(self.charge_head(features["scalar"]).squeeze(-1))
        if self.atom_env_keep is not None:
            keep = self.atom_env_keep.to(charges)
            charges = charges * keep
            charges = charges - keep * charges.sum() / keep.sum().clamp_min(1.0)
        self._last["charges"] = charges
        return charges

    def _d_nonneg_charges(self, raw: Tensor, species: Tensor) -> Tensor:
        """Step 9c (d_charge_mode 'nonneg'): d-metal atoms get non-negative charges, 0.05 softplus(raw / 0.05) (about 0 for
        negative raw values, about raw for positive ones); the cell is neutralized over the other atoms only. A cell of
        d atoms only is neutral with zero charges."""
        d = torch.tensor([s in _D_METALS for s in self.species_symbols], dtype=torch.bool, device=raw.device)[species.to(raw.device)]
        if not bool(d.any()):
            return neutralize_effective_charges(raw)
        if bool(d.all()):
            return torch.zeros_like(raw)
        raw = torch.where(d, 0.05 * torch.nn.functional.softplus(raw / 0.05), raw)
        return neutralize_effective_charges(raw, weights=(~d).to(raw.dtype))

    def coordination_descriptors(self, packed: SimpleNamespace, cache: SimpleNamespace) -> Tensor:
        """Step 9d: per atom (atom, 3): smooth coordination by more electronegative neighbours (chi_j > chi_a + 0.3) / 6,
        their mean distance over r_cov(a) + r_cov(j) minus 1, and the density sum_j exp(-(r_aj - r0) / 0.3 A) / 6
        (r0 = r_cov(a) + r_cov(j); smooth cutoff at 1.3 r0), from the cache's neighbour graph (6 A)."""
        sp = packed.species
        ei = cache.edge_index.to(packed.positions.device)
        r = torch.linalg.vector_norm(periodic_displacements(packed.positions, ei, packed.lattice, cache.cell_shifts.to(packed.positions)), dim=-1)
        i, j = ei[0].long(), ei[1].long()
        chi, rc = self.chi.to(r)[sp], self.r_cov.to(r)[sp]
        lig = (chi[j] > chi[i] + 0.3).to(r)
        r0 = rc[i] + rc[j]
        w = torch.sigmoid((1.3 * r0 - r) / 0.1) * lig
        n = sp.shape[0]
        cn = torch.zeros(n, dtype=r.dtype, device=r.device).index_add(0, i, w)
        rel = torch.zeros_like(cn).index_add(0, i, w * r / r0) / cn.clamp_min(1e-6) - 1.0
        dens = torch.zeros_like(cn).index_add(0, i, w * torch.exp(-(r - r0) / 0.3))
        return torch.stack((cn / 6.0, rel * (cn > 1e-3).to(r), dens / 6.0), dim=-1)

    def coordination_shift(self, packed: SimpleNamespace, cache: SimpleNamespace) -> Tensor:
        """Step 9d: delta_env(a) = softplus(s0 + s1 z(Z_a)) g(x_a), z = the d-orbital radius standardized over the d
        elements; zero at start (g's output layer is zero-initialized)."""
        x = self.coordination_descriptors(packed, cache)
        s = torch.nn.functional.softplus(self.coord_s[0] + self.coord_s[1] * self.coord_rd_z.to(x)[packed.species])
        return s * self.coord_g(x).squeeze(-1)

    def base_residual_penalty(self) -> Tensor:
        """L2 on the per-element residual: pushes shared structure into the hypernetwork."""
        return self.base_residual.square().sum()

    def embedding_residual_penalty(self) -> Tensor:
        """L2 on the per-species residual of the encoder's species embedding (feature branch must carry it)."""
        emb = self.encoder.species_embedding
        return emb.species_residual.weight.square().sum() if hasattr(emb, "species_residual") else self.base_residual.new_zeros(())

    def set_router_normalization(self, mean: Tensor, std: Tensor) -> None:
        with torch.no_grad():
            self.router_mean.copy_(mean.to(self.router_mean))
            self.router_std.copy_(std.clamp_min(1e-3).to(self.router_std))

    # ------------------------------------------------------------------ caches
    def pack(self, structure: Structure) -> SimpleNamespace:
        return pack_general(structure, self.species_table, dtype=self.element_features.dtype)

    def _init_atomic_channels(self) -> None:
        from scipy.interpolate import CubicSpline
        from scipy.special import spherical_jn

        c = self.config
        n_s = len(self.species_symbols)
        r = np.linspace(AC_R_MAX / AC_NR, AC_R_MAX, AC_NR)
        u, u2, rho = np.zeros((n_s, 3, AC_NR)), np.zeros((n_s, 3, AC_NR)), np.zeros((n_s, 3, AC_NR))
        eps, mask, q_def = np.zeros((n_s, 3)), np.zeros((n_s, 3), dtype=bool), np.zeros((n_s, 3))
        ultrasoft = bool(c.get("atomic_ultrasoft", True))
        data_dir = Path(__file__).resolve().parents[1] / "data" / "atomic_channels"
        for i, sym in enumerate(self.species_symbols):
            if sym not in D_ELEMENTS:
                continue
            z = np.load(data_dir / f"{sym}.npz")
            for l in c["atomic_channels"]:
                if f"u{l}" not in z.files:       # unbound valence p: the slot stays a free Gaussian
                    continue
                ur = z[f"u{l}"]
                norm = float(np.trapz(ur * ur, z["r"]))
                if not ultrasoft:                # step 3b form: renormalized (inflates the all-electron tail)
                    ur = ur / math.sqrt(norm)
                spl = CubicSpline(z["r"], ur)
                u[i, l], u2[i, l], eps[i, l], mask[i, l] = spl(r), spl(r, 2), float(z[f"eps{l}"]), True
                if ultrasoft and 1.0 - norm > 1e-6:
                    ua = CubicSpline(z["r"], z[f"ua{l}"])(r)
                    q_def[i, l] = 1.0 - norm
                    rho[i, l] = (ua * ua - u[i, l] ** 2) / q_def[i, l]      # augmentation shape, int rho dr = 1
        q_loc_max = max(c["sphere_q_cutoff"], c["sphere_tail_q_cutoff"] if c.get("sphere_tail_n_basis") else 0.0)
        if c.get("d_prior_valence") == "ae":
            q_loc_max = max(q_loc_max, prior_q_max([c["ecut_ry"], 30.0] + list((c.get("ecut_by_element") or {}).values())))
        q_loc = np.linspace(0.0, q_loc_max, AC_NQ_LOC)
        w_q = np.full(AC_NQ_LOC, q_loc[1] - q_loc[0]); w_q[[0, -1]] *= 0.5
        j0 = (q_loc**2 * w_q)[None, :] * spherical_jn(0, np.outer(r, q_loc)) / (2.0 * math.pi**2)       # V(r) = J0 v(q)
        ecuts = [c["ecut_ry"], 30.0] + list((c.get("ecut_by_element") or {}).values())
        q_p = np.linspace(0.0, g_cutoff_angstrom(max(ecuts)) + 2.0, AC_NQ_PROJ)
        w_r = np.full(AC_NR, r[1] - r[0]); w_r[[0, -1]] *= 0.5
        jl = np.stack([(2.0 / math.pi) * (r * w_r)[None, :] * spherical_jn(l, np.outer(q_p, r)) for l in range(3)])  # rad = Jl chi_u
        q_rho = np.linspace(0.0, 2.0 * g_cutoff_angstrom(max(ecuts)) + 4.0, AC_NQ_PROJ)       # unique transfers reach 2 g_cut
        j_rho = w_r[None, :] * spherical_jn(0, np.outer(q_rho, r))                                 # rho(q) = int rho_u j0(q r) dr
        for name, val, dtype in (("ac_rho", rho, torch.float32), ("ac_q", q_def, torch.float32), ("ac_qrho", q_rho, torch.float32),
                                 ("ac_Jrho", j_rho, torch.float32),
                                 ("ac_r", r, torch.float32), ("ac_u", u, torch.float32), ("ac_u2", u2, torch.float32),
                                 ("ac_eps", eps, torch.float32), ("ac_mask", mask, torch.bool), ("ac_qloc", q_loc, torch.float32),
                                 ("ac_J0", j0, torch.float32), ("ac_qp", q_p, torch.float32), ("ac_Jl", jl, torch.float32),
                                 ("ac_wr", w_r, torch.float32)):
            self.register_buffer(name, torch.as_tensor(val, dtype=dtype), persistent=False)
        self.ac_eps_shift = nn.Parameter(torch.zeros(n_s, 3))
        self.ac_extent = nn.Parameter(torch.zeros(n_s, 3))
        # Step 6: two-reference d channel (Vanderbilt): the second pseudo-orbital (eps_d + 4 eV) in projector slot 1 of
        # l = 2, a 2x2 (D, Q) block per d atom; Q and M = <AE_i|AE_j> are fixed by the stored orbitals
        self.register_buffer("ac_two", None, persistent=False)
        if c.get("atomic_two_ref_d"):
            from epnet.physics.atomic_radii import BOHR_A
            two, u2b, u2b_dd = np.zeros(n_s, dtype=bool), np.zeros((n_s, AC_NR)), np.zeros((n_s, AC_NR))
            eps2b, rcut, Q2, M2 = np.zeros(n_s), np.full(n_s, AC_R_MAX), np.zeros((n_s, 2, 2)), np.zeros((n_s, 2, 2))
            for i, sym in enumerate(self.species_symbols):
                if sym not in D_ELEMENTS or not mask[i, 2]:
                    continue
                z = np.load(data_dir / f"{sym}.npz")
                if "u2b" not in z.files:
                    raise ValueError(f"{sym}: no second d reference (run scripts/build_two_reference.py)")
                rz = z["r"]
                rcut[i] = max(float(z["rc2"]) * BOHR_A, float(z["r_loc"])) + 0.3
                inside = rz <= rcut[i]
                phis, aes = [z["u2"], z["u2b"]], [z["ua2"], z["ua2b"]]
                stable = bool(c.get("atomic_two_ref_stable") or c.get("atomic_two_ref_safe"))
                if stable:   # Stage 3d fix: second reference scaled to <AE_2|AE_2> = 1 (the construction is invariant to it)
                    sc = 1.0 / math.sqrt(np.trapz(aes[1] ** 2 * inside, rz))
                    phis[1], aes[1] = phis[1] * sc, aes[1] * sc
                for a_ in range(2):
                    for b_ in range(2):
                        M2[i, a_, b_] = np.trapz(aes[a_] * aes[b_] * inside, rz)
                        Q2[i, a_, b_] = M2[i, a_, b_] - np.trapz(phis[a_] * phis[b_] * inside, rz)
                if stable:   # Q is nearly rank one with a tiny negative eigenvalue (<= 0.003): its PSD part keeps S >= 1
                    lam, vec = np.linalg.eigh(Q2[i])
                    Q2[i] = (vec * np.clip(lam, 0.0, None)) @ vec.T
                spl = CubicSpline(rz, np.where(rz <= rcut[i] + 1.0, phis[1], 0.0))
                u2b[i], u2b_dd[i], eps2b[i], two[i] = spl(r), spl(r, 2), float(z["eps2b"]), True
            # Step 6d (revised), atomic_two_ref_safe: per-element extent bound for l = 2 (0 = fixed), from the safeguard scan
            amp = np.full((n_s, 3), 0.3)
            if c.get("atomic_two_ref_safe"):
                for i, sym in enumerate(self.species_symbols):
                    if two[i]:
                        zz = np.load(data_dir / f"{sym}.npz")
                        amp[i, 2] = math.log1p(float(zz["d_extent_max"])) if "d_extent_max" in zz.files else 0.0
            self.register_buffer("ac_extent_amp", torch.as_tensor(amp, dtype=torch.float32), persistent=False)
            for name, val, dtype in (("ac_two", two, torch.bool), ("ac_u2b", u2b, torch.float32), ("ac_u2b_dd", u2b_dd, torch.float32),
                                     ("ac_eps2b", eps2b, torch.float32), ("ac_rcut", rcut, torch.float32),
                                     ("ac_Q2", Q2, torch.float32), ("ac_M2", M2, torch.float32)):
                self.register_buffer(name, torch.as_tensor(val, dtype=dtype), persistent=False)
            if c.get("atomic_atom_shift"):   # Step 6c: per-atom d level from the atom's invariants (zero at start)
                n_inv = 3 * c["channels"] + 2
                self.ac_shift_head = nn.Sequential(nn.Linear(n_inv, 16), nn.SiLU(), nn.Linear(16, 1))
                nn.init.zeros_(self.ac_shift_head[-1].weight)
                nn.init.zeros_(self.ac_shift_head[-1].bias)
            if c.get("d_coord_term"):   # Step 9d: element-scaled coordination term on the d level (zero at start)
                if not c.get("atomic_two_ref_d"):
                    raise ValueError("d_coord_term is implemented for the two-reference d channel")
                rd = np.zeros(len(self.species_symbols))
                for i, sym in enumerate(self.species_symbols):
                    if sym in _D_METALS and (data_dir / f"{sym}.npz").exists():
                        rd[i] = _d_orbital_radius(np.load(data_dir / f"{sym}.npz"))
                d_rows = rd > 0
                z = np.zeros_like(rd)
                if d_rows.any():
                    z[d_rows] = (rd[d_rows] - rd[d_rows].mean()) / max(float(rd[d_rows].std()), 1e-6)
                self.register_buffer("coord_rd_z", torch.as_tensor(z, dtype=torch.float32), persistent=False)
                self.coord_g = nn.Sequential(nn.Linear(3, 16), nn.SiLU(), nn.Linear(16, 1))
                nn.init.zeros_(self.coord_g[-1].weight)
                nn.init.zeros_(self.coord_g[-1].bias)
                self.coord_s = nn.Parameter(torch.tensor([math.log(math.e - 1.0), 0.0]))   # softplus(a + b z): 1 at start
        # channels whose level shift and extent are learned; the others stay at the atomic construction (the np channel
        # sits near the Kleinman-Bylander singularity with the all-electron local: a_p = 0.27-1.08, Stage 2 guard B)
        learn = torch.tensor([1.0 if l in c.get("atomic_learn_l", (0, 1, 2)) else 0.0 for l in range(3)])
        self.register_buffer("ac_learn_mask", learn, persistent=False)

    def atomic_tables(self, full: bool = False):
        """(S, 3, n_qp) crystal radial functions rad_l(q) of the atomic channels on ``ac_qp`` and their (S, 3) crystal
        H strengths pi^3 (a + eps q) / a^2, a = <phi|chi> (pi^3 / a without augmentation); zero where a species/l has
        no atomic channel. ``full``: a namespace that adds the S strengths pi^3 q / a^2, the augmentation shapes
        rho_l(q) on ``ac_qrho`` (S, 3, n) and the atom's own potential averaged over them, int V_own rho dr (S, 3)."""
        dt = self.ac_J0.dtype
        coeff, _ = self.species_parameters()
        v_q = cosine_basis(self.ac_qloc, self.n_basis, self.config["sphere_q_cutoff"]) @ coeff.to(dt).T      # (n_ql, S)
        tail = self.species_tail()
        if tail is not None:
            v_q = v_q + cosine_basis(self.ac_qloc, self.n_tail, self.config["sphere_tail_q_cutoff"]) @ tail.to(dt).T
        if self.prior_table is not None:
            v_q = v_q + self.prior_at(self.ac_qloc).to(dt).T
        v_r = (self.ac_J0 @ v_q).T                                         # (S, n_r): species local potential, eV
        amp = self.ac_extent_amp.to(dt) if getattr(self, "ac_extent_amp", None) is not None else 0.3
        scale = torch.exp(amp * torch.tanh(self.ac_extent.to(dt) * self.ac_learn_mask.to(dt)))         # (S, 3)
        x = self.ac_r[None, None, :] / scale[..., None]
        u = _interp_uniform(self.ac_u, x, self.ac_r) / scale.sqrt()[..., None]          # phi_s(r) = s^-1/2 phi(r / s)
        u2 = _interp_uniform(self.ac_u2, x, self.ac_r) / scale.pow(2.5)[..., None]
        ll = torch.arange(3, device=u.device, dtype=dt)
        eps = (self.ac_eps + self.ac_eps_shift.to(dt) * self.ac_learn_mask.to(dt))[..., None]
        chi = (eps - v_r[:, None, :]) * u - KINETIC_EV_A2 * (-u2 + (ll * (ll + 1))[None, :, None] * u / self.ac_r.square())
        overlap = (u * chi * self.ac_wr).sum(-1)                           # a = <phi|chi>
        mask = self.ac_mask
        a = torch.where(mask, overlap, torch.ones_like(overlap))
        q_def = self.ac_q.to(dt)
        d_crystal = torch.where(mask, math.pi**3 * (a + eps[..., 0] * q_def) / a.square(), torch.zeros_like(overlap))
        rad = torch.einsum("lqr,slr->slq", self.ac_Jl, chi) * mask[..., None].to(dt)
        if not full:
            return rad, d_crystal
        s_crystal = torch.where(mask, math.pi**3 * q_def / a.square(), torch.zeros_like(overlap))
        rho = _interp_uniform(self.ac_rho, x, self.ac_r) / scale[..., None]               # rho_s(r) = s^-1 rho(r / s)
        rho_q = torch.einsum("qr,slr->slq", self.ac_Jrho, rho)
        own = (v_r[:, None, :] * rho * self.ac_wr).sum(-1)
        two = None
        if self.ac_two is not None:
            # Vanderbilt two-reference d block: chi_i = (eps_i - T - V) phi_i inside r_cut, B_ij = <phi_i|chi_j>,
            # beta = B^-T chi, D_ij = B_ij + eps_j Q_ij (beta normalization: crystal strengths pi^3 D, pi^3 Q)
            s2 = scale[:, 2:3]
            x2 = self.ac_r[None, :] / s2
            phi2 = _interp_uniform(self.ac_u2b, x2, self.ac_r) / s2.sqrt()
            phi2_dd = _interp_uniform(self.ac_u2b_dd, x2, self.ac_r) / s2.pow(2.5)
            inside = (self.ac_r[None, :] <= self.ac_rcut.to(dt)[:, None]).to(dt)
            safe = bool(self.config.get("atomic_two_ref_safe"))
            d_shift = self.ac_eps_shift[:, 2].to(dt) * self.ac_learn_mask[2].to(dt)                     # (S,)
            eps2 = (self.ac_eps2b.to(dt) + (0.0 if safe else d_shift))[:, None]
            if safe:   # the level shift stays out of chi (and B): applied below as D += shift <AE_i|AE_j>
                ll2 = 6.0
                chi1 = ((self.ac_eps[:, 2:3].to(dt) - v_r) * u[:, 2] - KINETIC_EV_A2 * (-u2[:, 2] + ll2 * u[:, 2] / self.ac_r.square())) * inside
            else:
                chi1 = chi[:, 2] * inside
            chi2 = ((eps2 - v_r) * phi2 - KINETIC_EV_A2 * (-phi2_dd + 6.0 * phi2 / self.ac_r.square())) * inside
            phis, chis = torch.stack((u[:, 2], phi2), 1), torch.stack((chi1, chi2), 1)              # (S, 2, n_r)
            B = torch.einsum("sir,sjr->sij", phis * inside[:, None] * self.ac_wr, chis)
            tw = self.ac_two
            B = torch.where(tw[:, None, None], B, torch.eye(2, dtype=dt, device=B.device).expand_as(B))
            beta = torch.einsum("sji,sjr->sir", torch.linalg.inv(B), chis)                      # beta_i = sum_j B^-1_ji chi_j
            eps12 = torch.cat((self.ac_eps[:, 2:3].to(dt) if safe else eps[:, 2], eps2), dim=1)       # (S, 2)
            D = B + self.ac_Q2.to(dt) * eps12[:, None, :]
            if safe:
                D = D + d_shift[:, None, None] * self.ac_M2.to(dt)
            rad2 = torch.einsum("qr,sir->siq", self.ac_Jl[2], beta) * tw[:, None, None].to(dt)
            two = SimpleNamespace(mask=tw, rad=rad2, D=D * tw[:, None, None].to(dt), Q=self.ac_Q2.to(dt), M=self.ac_M2.to(dt))
        return SimpleNamespace(rad=rad, h=d_crystal, s=s_crystal, rho_q=rho_q, own=own, two=two)

    def projectors(self, packed: SimpleNamespace, k_frac: Tensor, g_idx: Tensor) -> Tensor:
        """Per-atom s/p/d Gaussian projectors, widths sigma_{a,r} = c_r / r_cov(Z_a); (k, g, atom*R*9).
        ``g_idx`` is (g, 3) shared by all k, or (k, g, 3) for a k-centred basis."""
        k_frac = fold_to_first_zone(k_frac, packed.lattice)
        reciprocal = 2.0 * torch.pi * torch.linalg.inv(packed.lattice).transpose(0, 1)
        g_cart = g_idx.to(reciprocal) @ reciprocal
        momentum = (k_frac.to(reciprocal) @ reciprocal)[:, None, :] + (g_cart if g_idx.ndim == 3 else g_cart[None])
        q = torch.linalg.vector_norm(momentum, dim=-1)
        u = momentum / q.clamp_min(1e-8).unsqueeze(-1)
        scale = self.width_constants.to(q)[None, :] / self.r_cov.to(q)[packed.species][:, None]  # (atom, R)
        if self.config.get("projector_radius_mode") == "orbital":
            sp_ = packed.species.to(q.device)
            radius = self.orbital_multipliers.to(q)[None, :, None] * self.orbital_radii.to(q)[sp_][:, None, :]  # (atom, R, l)
            if self.projector_log_mult is not None:
                radius = radius * torch.exp(self.projector_log_mult.to(q.device))[None, :, None] \
                    * torch.exp(0.3 * torch.tanh(self.projector_log_offset.to(q.device)))[sp_][:, None, :]
            scale_l = 1.0 / radius
        elif self.config.get("projector_radii_firstrow") is not None:
            # l=1 radial function x exp(-x^2/2) peaks at r = 1/scale: place first-row peaks at the given radii (A)
            fr = self.first_row.to(q.device)[packed.species.to(q.device)]
            fixed = 1.0 / torch.tensor(self.config["projector_radii_firstrow"], dtype=q.dtype, device=q.device)
            scale = torch.where(fr[:, None], fixed[None, :].expand_as(scale), scale)
        if self.config.get("projector_q_width_max") is not None:
            # 1/r_cov widths tuned on Si reach the 12 Ry basis edge for first-row atoms (F 6.8 1/A);
            # trained potentials then exploit the truncation and are not basis-converged
            scale = scale.clamp(max=self.config["projector_q_width_max"])
        if self.config.get("projector_radius_mode") != "orbital":
            scale_l = scale[:, :, None].expand(-1, -1, self.n_l)
        xs = [q[:, :, None, None] / scale_l[None, None, :, :, l] for l in range(self.n_l)]  # per l: (k, g, atom, R)
        ux, uy, uz = u.unbind(dim=-1)
        ang = [torch.ones_like(ux)[..., None], math.sqrt(3.0) * u]
        rad = [torch.exp(-0.5 * xs[0].square()), xs[1] * torch.exp(-0.5 * xs[1].square())]
        if self.n_l > 2:
            d = torch.stack(
                (
                    math.sqrt(3.0) * ux * uy,
                    math.sqrt(3.0) * uy * uz,
                    math.sqrt(3.0) * uz * ux,
                    0.5 * math.sqrt(3.0) * (ux.square() - uy.square()),
                    0.5 * (3.0 * uz.square() - 1.0),
                ),
                dim=-1,
            )
            ang.append(math.sqrt(5.0) * d)
            rad.append(xs[2].square() * torch.exp(-0.5 * xs[2].square()))
        if self.ac_on:   # atomic channels of d atoms: slot 0 takes the constructed chi_l(|k+G|)
            tables, _ = self.atomic_tables()
            sp_ = packed.species.to(q.device)
            for l in self.config["atomic_channels"]:
                m = self.ac_mask[sp_, l]
                if bool(m.any()):
                    vals = _interp_q(tables[sp_, l], q, self.ac_qp).to(rad[l].dtype)        # (k, g, atom)
                    slot0 = torch.where(m[None, None, :], vals, rad[l][..., 0])
                    rad[l] = torch.cat((slot0[..., None], rad[l][..., 1:]), dim=-1)
            if self.ac_two is not None and self.n_proj_radial > 1:
                two = self.atomic_tables(full=True).two
                m2 = two.mask[sp_]
                if bool(m2.any()):
                    b1 = _interp_q(two.rad[sp_, 0], q, self.ac_qp).to(rad[2].dtype)
                    b2 = _interp_q(two.rad[sp_, 1], q, self.ac_qp).to(rad[2].dtype)
                    s0 = torch.where(m2[None, None, :], b1, rad[2][..., 0])
                    s1 = torch.where(m2[None, None, :], b2, rad[2][..., 1])
                    rad[2] = torch.cat((s0[..., None], s1[..., None], rad[2][..., 2:]), dim=-1)
        phase = torch.exp(-1j * torch.einsum("kgc,nc->kgn", momentum, packed.positions.to(momentum)))
        parts = [
            phase[..., None, None] * rad[l][..., None] * ang[l][:, :, None, None, :]
            for l in range(self.n_l)
        ]
        # (k, g, atom, R, sum_l 2l+1) -> (k, g, channel)
        return torch.cat(parts, dim=-1).flatten(start_dim=2) / math.sqrt(float(packed.volume))

    def cell_ecut(self, packed: SimpleNamespace) -> float:
        """Cutoff (Ry) of one cell: the highest its elements need (``ecut_by_element``; others ``ecut_ry``).
        First-row 2p orbitals need 16-20 Ry while the other s/p elements converge at 12 Ry."""
        table = self.config.get("ecut_by_element")
        if not table:
            return self.config["ecut_ry"]
        present = {self.species_symbols[int(i)] for i in packed.species.unique()}
        return max(table.get(s, self.config["ecut_ry"]) for s in present)

    def make_pw_cache(self, packed: SimpleNamespace, k_frac: Tensor, ecut_ry: float | None = None) -> SimpleNamespace:
        """``ecut_ry`` overrides the cell cutoff (basis-convergence checks and penalties)."""
        k_frac = fold_to_first_zone(k_frac, packed.lattice)  # E(k) = E(k+G); the G sphere is centred at 0
        cutoff = g_cutoff_angstrom(self.cell_ecut(packed) if ecut_ry is None else ecut_ry)
        recip = 2.0 * torch.pi * torch.linalg.inv(packed.lattice.detach()).transpose(0, 1)
        k_centred = bool(self.config.get("k_centred_basis"))
        k_cart = k_frac.detach().to(recip) @ recip
        # k-centred basis {G : |k+G| < gcut} (the standard choice): the potential is built once on the
        # sphere |G| < gcut + max|k| and each k gathers its own sub-basis. The G-centred sphere
        # {|G| < gcut} truncates differently at every k and in every cell, which hard (first-row)
        # potentials turn into geometry-dependent band errors (~0.3-0.45 eV median).
        reach = cutoff + (float(torch.linalg.vector_norm(k_cart, dim=-1).max()) + 1e-6 if k_centred and len(k_cart) else 0.0)
        g_idx_np, _ = reciprocal_vectors(packed.lattice.detach().cpu().numpy(), reach)
        g_idx = torch.as_tensor(g_idx_np, dtype=torch.long, device=k_frac.device)
        unique_cart, inverse = unique_g_transfers_encoded(g_idx, recip)  # 1-D sort; unique(dim=0) was ~95% of prep
        k_index = k_valid = g_idx_k = None
        if k_centred:
            g_cart = g_idx.to(recip) @ recip
            inside = torch.linalg.vector_norm(k_cart[:, None, :] + g_cart[None], dim=-1) <= cutoff + 1e-9
            n_max = int(inside.sum(dim=1).max())
            k_index = torch.argsort((~inside).to(torch.int8), dim=1, stable=True)[:, :n_max]  # valid G first
            k_valid = torch.gather(inside, 1, k_index)
            g_idx_k = g_idx[k_index]
            momentum = k_cart[:, None, :] + g_cart[k_index]
            kinetic = torch.where(k_valid, KINETIC_EV_A2 * momentum.square().sum(-1), torch.full_like(momentum[..., 0], PAD_KINETIC_EV))
        else:
            kinetic = kinetic_diagonal(g_idx, recip, k_frac)
        edge_index, cell_shifts = build_periodic_neighbor_graph(
            packed.positions.detach(), packed.lattice.detach(), self.config["cutoff"]
        )
        desc = environment_descriptors(
            packed, self.z_model, self.chi, self.config["router_radii"], self.config["contrast_radius"]
        )
        return SimpleNamespace(
            g_idx=g_idx,
            unique_cart=unique_cart.detach(),
            inverse=inverse.detach(),
            kinetic=kinetic.detach(),
            volume=float(packed.volume),
            n_g=int(g_idx.shape[0] if k_index is None else k_index.shape[1]),
            k_index=k_index,
            k_valid=k_valid,
            g_idx_k=g_idx_k,
            edge_index=edge_index.detach(),
            cell_shifts=cell_shifts.detach(),
            router_inputs=desc["router"].detach(),
            cn3=desc["cn3"].detach(),
            cn10=desc["cn10"].detach(),
            projectors=self.projectors(packed, k_frac, g_idx if g_idx_k is None else g_idx_k).detach(),
            k_frac=k_frac.detach(),
            qeq_coulomb=(
                coulomb_matrix(packed.positions, packed.lattice, 0.5 * self.r_cov[packed.species]).to(k_frac.device)
                if self.qeq_hyper is not None else None
            ),
        )

    # ------------------------------------------------------------------ forward
    def mixture_weights(self, cache: SimpleNamespace) -> Tensor:
        if len(self.readouts) == 1:
            return cache.router_inputs.new_ones(cache.router_inputs.shape[0], 1)
        x = (cache.router_inputs - self.router_mean) / self.router_std
        return torch.softmax(self.router(x), dim=-1)

    def local_form_factor(self, packed: SimpleNamespace, cache: SimpleNamespace, features: dict[str, Tensor],
                          charges: Tensor | None = None) -> Tensor:
        """Sum over atoms of phase * v_a on the unique transfers, before the 1/Omega scatter."""
        g = cache.unique_cart
        q = torch.linalg.vector_norm(g, dim=-1)
        coeff, _ = self.species_parameters()
        if charges is None:
            charges = self.atomic_charges(features, species=packed.species)
        atom_coeff = coeff[packed.species]
        if self.atom_base_delta is not None:  # diagnostic hook: per-atom species-base change (atom, n_base)
            atom_coeff = atom_coeff + self.atom_base_delta[:, : self.n_basis]
        if self.charge_hyper is not None:
            atom_coeff = atom_coeff + charges[:, None] * self.charge_response()[0][packed.species]
        phase = torch.exp(-1j * torch.einsum("gc,nc->ng", g.to(packed.positions), packed.positions))
        base = torch.einsum("nb,gb->ng", atom_coeff, cosine_basis(q, self.n_basis, self.config["sphere_q_cutoff"]))
        tail = self.species_tail()
        if tail is not None:
            base = base + torch.einsum("nb,gb->ng", tail[packed.species], cosine_basis(q, self.n_tail, self.config["sphere_tail_q_cutoff"]))
        if self.prior_table is not None:
            base = base + self.prior_at(q)[packed.species].to(base)
        total = (base.to(phase.dtype) * phase).sum(dim=0)
        weights = self.mixture_weights(cache)
        if self.atom_env_keep is not None:
            weights = weights * self.atom_env_keep.to(weights)[:, None]
        self._last["mixture_weights"] = weights
        for k, readout in enumerate(self.readouts):
            _, per_atom = readout(packed.positions, g, features, return_per_atom=True)
            total = total + (per_atom * weights[:, k].to(per_atom.real.dtype).unsqueeze(-1)).sum(dim=0).to(total)
        if charges is not None:
            coulomb = reciprocal_coulomb_potential(
                packed.positions, charges, g, volume=1.0, gaussian_width=self.config["charge_width"]
            )
            scale = self.coulomb_scale if self.config["charge_model"] == "qeq" else 1.0
            total = total - (scale * coulomb * COULOMB_EV_A).to(total)
        return total.masked_fill(q < 1e-8, 0)

    def projector_strengths(self, packed: SimpleNamespace, cache: SimpleNamespace, features: dict[str, Tensor],
                            charges: Tensor | None = None, local: Tensor | None = None) -> Tensor:
        """Per-atom D_{a,r,l} = D^Z + q_a E^Z (if charge-coupled) + dD_a, shape (atom, R, n_l). ``local``: the cell's
        local form factor on the unique transfers (computed if needed by the ultrasoft augmentation)."""
        _, d_species = self.species_parameters()
        invariants = torch.cat(
            (
                features["scalar"],
                norm_invariant(features["vector"].square().sum(-1), self.config.get("invariant_norm", "sqrt"), self.config.get("invariant_scale", 0.05)),
                norm_invariant(features["tensor"].square().sum((-1, -2)), self.config.get("invariant_norm", "sqrt"), self.config.get("invariant_scale", 0.05)),
                (cache.cn3 / 4.0)[:, None],
                (cache.cn10 / 150.0)[:, None],
            ),
            dim=-1,
        )
        delta = self.projector_head(invariants).reshape(-1, self.n_proj_radial, self.n_l)
        if self.atom_env_keep is not None:
            delta = delta * self.atom_env_keep.to(delta)[:, None, None]
        self._last["projector_delta"] = delta
        strengths = d_species[packed.species] + delta
        if self.atom_base_delta is not None:
            strengths = strengths + self.atom_base_delta[:, self.n_basis : self.n_basis + self.n_proj_radial * self.n_l].reshape(-1, self.n_proj_radial, self.n_l)
        if self.charge_hyper is not None:
            if charges is None:
                charges = self.atomic_charges(features, species=packed.species)
            strengths = strengths + charges[:, None, None] * self.charge_response()[1][packed.species]
        if self.config.get("firstrow_lmax1") and self.n_l > 2:
            keep = torch.ones_like(strengths)
            keep[self.first_row.to(strengths.device)[packed.species.to(strengths.device)], :, 2] = 0.0   # no d projector on first-row atoms
            strengths = strengths * keep
        if self.ac_on:   # atomic channels: slot 0 strength is derived, not learned
            tab = self.atomic_tables(full=True)
            sp_ = packed.species.to(strengths.device)
            m = self.ac_mask[sp_]
            d_atom = tab.h[sp_]
            if bool((self.ac_q[sp_] > 0).any()):
                # augmentation in the cell's local potential (USPP D += int V Q): eps -> eps + dV_al in the q term, with
                # dV_al = (1/Omega) sum_{g != 0} V~(g) rho_l(|g|) e^{i g.tau_a} - int V_own rho_l (V(G = 0) is dropped,
                # so in a large box dV = -v(0)/Omega and every level shifts alike, as without S)
                if local is None:
                    if charges is None:
                        charges = self.atomic_charges(features, cache, packed.species)
                    local = self.local_form_factor(packed, cache, features, charges)
                g = cache.unique_cart.to(strengths.device)
                phase = torch.exp(1j * (g.to(packed.positions) @ packed.positions.T))                   # (u, atom)
                vg = (local[:, None] * phase).to(torch.complex128 if strengths.dtype == torch.float64 else torch.complex64)
                gn = torch.linalg.vector_norm(g, dim=-1)[None, :]
                dv = []
                for l in range(3):
                    rq = _interp_q(tab.rho_q[sp_, l], gn, self.ac_qrho)[0]                            # (u, atom)
                    dv.append((vg * rq.to(vg.real.dtype)).real.sum(0) / float(cache.volume) - tab.own[sp_, l].to(strengths))
                hook = getattr(self, "_dv_hook", None)   # diagnostics only (Step 9a ablations); unset in training and scoring
                if hook is not None:
                    dv = hook(dv, sp_)
                d_atom = d_atom + tab.s[sp_] * torch.stack(dv, dim=-1).to(d_atom)
            slot0 = torch.where(m, d_atom.to(strengths), strengths[:, 0, :])
            self._last["ac_overlap"] = torch.where(m, tab.s[sp_].to(strengths), torch.zeros_like(slot0))   # S strengths
            rest = strengths[:, 1:, :]
            if not self.config.get("atomic_free_d_slot", True):
                # Stage 2 follow-up: no free (Gaussian) l = 2 correction on atoms with an atomic d channel, so the fit
                # cannot bypass the constructed channel (CuBr AD20 expelled the d state)
                keep_d = (~m[:, 2]).to(strengths)[:, None, None]
                rest = torch.cat((rest[..., :2], rest[..., 2:3] * keep_d, rest[..., 3:]), dim=-1)
            strengths = torch.cat((slot0[:, None, :], rest), dim=1)
            self._last["ac_pair"] = None
            if tab.two is not None and self.n_proj_radial > 1:
                m2 = tab.two.mask[sp_]
                if bool(m2.any()):
                    D_a = tab.two.D[sp_].to(strengths)
                    if bool((self.ac_q[sp_] > 0).any()):                         # augmentation: D_ij += dV_a Q_ij
                        D_a = D_a + dv[2].to(strengths)[:, None, None] * tab.two.Q[sp_].to(strengths)
                    if getattr(self, "ac_shift_head", None) is not None:            # Step 6c: D_ij += delta_a M_ij
                        delta = self.ac_shift_head(invariants).squeeze(-1) * m2.to(strengths)
                        self._last["ac_atom_shift"] = delta
                        D_a = D_a + delta[:, None, None] * tab.two.M[sp_].to(strengths)
                    if getattr(self, "coord_g", None) is not None:                 # Step 9d: D_ij += s(Z) g(x_a) M_ij
                        denv = self.coordination_shift(packed, cache).to(strengths) * m2.to(strengths)
                        self._last["ac_coord_shift"] = denv
                        D_a = D_a + denv[:, None, None] * tab.two.M[sp_].to(strengths)
                    D_a, Q_a = math.pi**3 * D_a, math.pi**3 * tab.two.Q[sp_].to(strengths)
                    w = m2.to(strengths)
                    s0 = torch.cat((strengths[:, 0, :2], torch.where(m2, D_a[:, 0, 0], strengths[:, 0, 2])[:, None]), dim=1)
                    s1 = torch.cat((strengths[:, 1, :2], torch.where(m2, D_a[:, 1, 1], strengths[:, 1, 2])[:, None]), dim=1)
                    strengths = torch.cat((s0[:, None], s1[:, None], strengths[:, 2:]), dim=1)
                    ov = self._last["ac_overlap"]
                    self._last["ac_overlap"] = torch.cat((ov[:, :2], torch.where(m2, Q_a[:, 0, 0], ov[:, 2])[:, None]), dim=1)
                    self._last["ac_pair"] = (m2, D_a[:, 0, 1] * w, Q_a[:, 0, 1] * w, Q_a[:, 1, 1] * w)
        return strengths

    def _nonlocal_coupled(self, packed: SimpleNamespace, strengths: Tensor, basis: Tensor) -> Tensor:
        """B M B^dagger with M_{a,l} an R x R matrix: diagonal D_{a,r,l}, off-diagonal per species for coupled l."""
        n_atom, n_r = strengths.shape[0], self.n_proj_radial
        full = torch.diag_embed(strengths.transpose(1, 2))  # (atom, n_l, R, R)
        iu = torch.triu_indices(n_r, n_r, 1, device=strengths.device)
        off = torch.zeros_like(full)
        values = self.projector_offdiag[packed.species]  # (atom, n_coupled, R(R-1)/2)
        for j, l in enumerate(self.config["projector_offdiag_l"]):
            off[:, l, iu[0], iu[1]] = values[:, j]
        full = full + off + off.transpose(-1, -2)
        per_m = torch.cat([full[:, l : l + 1].expand(-1, L_SIZES[l], -1, -1) for l in range(self.n_l)], dim=1)
        b = basis.reshape(*basis.shape[:2], n_atom, n_r, -1)  # (k, g, atom, R, m)
        coupled = torch.einsum("kgarm,amrs->kgasm", b, per_m.to(b))
        return coupled.flatten(start_dim=2) @ basis.conj().transpose(-1, -2)

    def crystal_field(self, packed: SimpleNamespace, cache: SimpleNamespace, features: dict[str, Tensor]) -> Tensor:
        """(atom, R, 5, 5) sigma angular-overlap d-block term, sum over bonds a -> j of e_r(a, j) y_j y_j^T; zero for
        atoms without d. Uses the encoder graph (6 A) restricted to ``crystal_field_cutoff``."""
        disp = periodic_displacements(packed.positions, cache.edge_index, packed.lattice, cache.cell_shifts)
        dist = torch.linalg.vector_norm(disp, dim=-1)
        center, neighbor = cache.edge_index.to(disp.device)
        keep = (dist > 1e-6) & (dist < self.config["crystal_field_cutoff"]) & self.is_d.to(disp.device)[packed.species[center]]
        center, neighbor, disp, dist = center[keep], neighbor[keep], disp[keep], dist[keep]
        y = d_harmonics(disp / dist[:, None]) / math.sqrt(5.0)
        e = self.cf_net(torch.cat((self.cf_basis(dist), features["scalar"][center], features["scalar"][neighbor]), dim=-1))
        self._last["crystal_field_e"] = e
        out = disp.new_zeros(packed.positions.shape[0], self.n_proj_radial, 5, 5)
        return out.index_add_(0, center, e[:, :, None, None] * (y[:, None, :, None] * y[:, None, None, :]))

    def d_terms(self, packed: SimpleNamespace, cache: SimpleNamespace, features: dict[str, Tensor]):
        """(d-atom indices, crystal field (ad, R, 5, 5) or None, overlap q (ad, R) or None), or None when the cell has no
        d atom or neither term is configured: such cells take exactly the s/p code path."""
        if self.d_overlap_raw is None and self.cf_net is None:
            return None
        d_atoms = torch.nonzero(self.is_d.to(packed.species.device)[packed.species]).flatten()
        if not d_atoms.numel():
            return None
        cf = self.crystal_field(packed, cache, features)[d_atoms] if self.cf_net is not None else None
        q = torch.nn.functional.softplus(self.d_overlap_raw)[packed.species[d_atoms]] if self.d_overlap_raw is not None else None
        return d_atoms, cf, q

    def _apply_d_terms(self, total: Tensor, b: Tensor, kinetic: Tensor, terms) -> Tensor:
        """Add the crystal field to the potential part of H and, with the overlap, return the potential part of
        L^-1 H L^-dagger (S = L L^dagger), whose ordinary eigenvalues are the generalized ones of (H, S).
        ``b`` are this chunk's projectors (k, g, atom*R*9), padded rows already zeroed."""
        d_atoms, cf, q = terms
        n_k, n_g = b.shape[:2]
        bd = b.reshape(n_k, n_g, -1, self.n_proj_radial, 9)[:, :, d_atoms, :, 4:]    # (k, g, ad, R, 5) d block
        flat_h = bd.reshape(n_k, n_g, -1).conj().transpose(-1, -2)
        if cf is not None:
            w = torch.einsum("kgarm,armn->kgarn", bd, cf.to(bd))
            total = total + (w.reshape(n_k, n_g, -1) @ flat_h).to(total)
        if q is not None:
            eye = torch.eye(n_g, dtype=total.dtype, device=total.device)
            s_mat = eye + ((bd * q[None, None, :, :, None].to(bd.real.dtype)).reshape(n_k, n_g, -1) @ flat_h).to(total)
            diag = torch.diag_embed(kinetic.to(total.dtype))
            chol = torch.linalg.cholesky(s_mat)
            x = torch.linalg.solve_triangular(chol, total + diag, upper=False)          # L^-1 H
            y = torch.linalg.solve_triangular(chol, x.mH, upper=False)                  # L^-1 H L^-dagger (H Hermitian)
            total = 0.5 * (y + y.mH) - diag
        return total

    def _augmentation_s(self, b: Tensor, aug) -> Tensor:
        """S = 1 + B q B^dagger over the augmented projector columns (ultrasoft atomic channels); padded rows of ``b``
        are zero, so S is the identity there. ``aug`` = (idx, val) or (idx, val, pair): pair = (cols0, cols1, q01)
        adds the off-diagonal blocks b0 q01 b1^dagger + h.c. (two-reference d channel)."""
        idx, val = aug[0], aug[1]
        bs = b[..., idx]
        eye = torch.eye(b.shape[1], dtype=b.dtype, device=b.device)
        s_mat = eye + (bs * val.to(bs.real.dtype)) @ bs.mH
        if len(aug) > 2 and aug[2] is not None:
            c0, c1, q01 = aug[2]
            off = (b[..., c0] * q01.to(b.real.dtype)) @ b[..., c1].mH
            s_mat = s_mat + off + off.mH
        return s_mat

    @staticmethod
    def _overlap_block(aug) -> tuple[Tensor, Tensor]:
        """The same S as ``_augmentation_s`` in low-rank form for the FFT path: (cols, q) with S = 1 + B_cols q B_cols^dagger,
        cols the augmented columns (and the pair columns), q (n, n) real symmetric (diagonal ``val``, pair couplings q01)."""
        idx, val = aug[0], aug[1]
        pair = aug[2] if len(aug) > 2 else None
        cols = idx if pair is None else torch.unique(torch.cat((idx, pair[0], pair[1])))
        where = torch.full((int(cols.max()) + 1,), -1, dtype=torch.long, device=cols.device)
        where[cols] = torch.arange(cols.numel(), device=cols.device)
        q = val.new_zeros(cols.numel(), cols.numel()).index_put((where[idx], where[idx]), val)
        if pair is not None:
            c0, c1, q01 = pair
            q = q.index_put((where[c0], where[c1]), q01.to(q)).index_put((where[c1], where[c0]), q01.to(q))
        return cols, q

    def _apply_augmentation(self, total: Tensor, b: Tensor, kinetic: Tensor, aug: tuple[Tensor, Tensor]) -> Tensor:
        """Potential part of L^-1 H L^-dagger (S = L L^dagger), whose ordinary eigenvalues are those of H psi = e S psi."""
        s_mat = self._augmentation_s(b, aug).to(total)
        diag = torch.diag_embed(kinetic.to(total.dtype))
        chol, info = torch.linalg.cholesky_ex(s_mat)
        if bool((info > 0).any()):   # S is positive definite by construction; float32 can fail when it is badly conditioned
            wide = torch.complex128 if total.is_complex() else torch.float64
            chol = torch.linalg.cholesky(s_mat.to(wide))
            x = torch.linalg.solve_triangular(chol, (total + diag).to(wide), upper=False)
            y = torch.linalg.solve_triangular(chol, x.mH, upper=False)
            return (0.5 * (y + y.mH)).to(total.dtype) - diag
        x = torch.linalg.solve_triangular(chol, total + diag, upper=False)
        y = torch.linalg.solve_triangular(chol, x.mH, upper=False)
        return 0.5 * (y + y.mH) - diag

    def band_path(self, packed: SimpleNamespace, k_frac: Tensor, n_bands: int, cache: SimpleNamespace | None = None,
                  return_hamiltonians: bool = False):
        """Lowest ``n_bands`` band energies (k, n_bands). With ``return_hamiltonians`` (diagnostics, dense path only):
        a list of (H, S or None) dense pairs per k chunk ((k_chunk, g, g); S with ultrasoft atomic channels) and the
        cache, instead of energies."""
        if cache is None:
            cache = self.make_pw_cache(packed, k_frac)
        features = self.encoder(
            packed.positions, packed.species, packed.lattice, edge_index=cache.edge_index, cell_shifts=cache.cell_shifts
        )
        charges = self.atomic_charges(features, cache, packed.species)
        local = self.local_form_factor(packed, cache, features, charges)
        # solver "chefsi_fft" (training path, warm store): H is applied through the FFT grid and never built densely
        # and, with fft_ng_threshold, for every cell with n_G >= threshold in every path (training, convergence penalty,
        # evaluation): dense H is slower above n_G ~4,500 and does not fit above ~8,000 on a 48 GB GPU (2026-10-18)
        thr = self.config.get("fft_ng_threshold")
        big = thr is not None and int(cache.kinetic.shape[1]) >= int(thr)
        use_fft = self.projector_offdiag is None and (big or (
            self.config.get("solver") == "chefsi_fft" and torch.is_grad_enabled() and getattr(cache, "warm", None) is not None))
        fft_store = getattr(cache, "warm", None)
        if use_fft and fft_store is None:
            fft_store = {}                       # no warm store (evaluation, convergence penalty): a cold start each call
        d_terms = self.d_terms(packed, cache, features)
        if d_terms is not None and use_fft:
            raise NotImplementedError("the d overlap / crystal-field terms are implemented on the dense path only")
        if return_hamiltonians and use_fft:
            raise NotImplementedError("return_hamiltonians needs the dense path")
        if use_fft:
            v_r, fft_pos, fft_shape = self._fft_local(packed, cache, local)
        else:
            potential = scatter_hermitian_potential(local, cache.inverse, cache.volume)
        strengths = self.projector_strengths(packed, cache, features, charges, local=local)
        channels = torch.cat(
            [strengths[..., l : l + 1].expand(*strengths.shape[:-1], L_SIZES[l]) for l in range(self.n_l)], dim=-1
        ).flatten()
        aug = pair_h = None
        if self.ac_on and self.config.get("atomic_ultrasoft", True) and bool((self.ac_q[packed.species] > 0).any()):
            s_atom = self._last["ac_overlap"]                                                    # (atom, n_l), slot 0
            s_full = torch.cat((s_atom[:, None, :], s_atom.new_zeros(s_atom.shape[0], self.n_proj_radial - 1, self.n_l)), dim=1)
            vec = torch.cat([s_full[..., l : l + 1].expand(*s_full.shape[:-1], L_SIZES[l]) for l in range(self.n_l)], dim=-1).flatten()
            pair_h = None
            if self._last.get("ac_pair") is not None:   # two-reference d: slot 1 of l = 2 and the slot 0-1 couplings
                m2, d01, q01, q11 = self._last["ac_pair"]
                s1 = torch.cat((s_full[:, 1, :2], torch.where(m2, q11, s_full[:, 1, 2])[:, None]), dim=1)
                s_full = torch.cat((s_full[:, :1], s1[:, None], s_full[:, 2:]), dim=1)
                vec = torch.cat([s_full[..., l : l + 1].expand(*s_full.shape[:-1], L_SIZES[l]) for l in range(self.n_l)], dim=-1).flatten()
                atoms = torch.nonzero(m2).flatten()
                per = self.n_proj_radial * 9
                mm = torch.arange(5, device=atoms.device)
                c0 = (atoms[:, None] * per + 0 * 9 + 4 + mm[None, :]).flatten()
                c1 = (atoms[:, None] * per + 1 * 9 + 4 + mm[None, :]).flatten()
                pair = (c0, c1, q01[atoms].repeat_interleave(5))
                pair_h = (c0, c1, d01[atoms].repeat_interleave(5))
            else:
                pair = None
            idx = torch.nonzero(vec.detach()).flatten()
            if idx.numel():
                aug = (idx, vec[idx], pair)
                if use_fft and not self.config.get("atomic_two_ref_safe"):
                    raise NotImplementedError("on the FFT path the ultrasoft atomic channels need the safe two-reference "
                                              "d channel (atomic_two_ref_safe); the other forms are dense only")
        # FFT path with ultrasoft channels: S = 1 + B_cols q B_cols^dagger as a low-rank block (Löwdin in PWOperator)
        overlap = self._overlap_block(aug) if use_fft and aug is not None else None
        basis = cache.projectors
        if (self.projector_log_mult is not None or self.ac_on) and torch.is_grad_enabled():   # learnable shapes: rebuild with gradients
            basis = self.projectors(packed, cache.k_frac, cache.g_idx if getattr(cache, "g_idx_k", None) is None else cache.g_idx_k)
        # k_batch chunks the k-points here, so the (k, g, g) nonlocal term and Hamiltonian
        # are never built for the whole path at once (12 Ry cells reach ~2,800 plane waves)
        n_k = int(cache.kinetic.shape[0])
        step = n_k if self.config["k_batch"] is None else max(int(self.config["k_batch"]), 1)
        chunks = []
        k_index = getattr(cache, "k_index", None)
        for start in range(0, n_k, step):
            sl = slice(start, start + step)
            b = basis[sl]
            chain = None if torch.is_grad_enabled() else getattr(cache, "chain", None)   # evaluation along a path (screen.py)
            if use_fft:
                op = PWOperator(cache.kinetic[sl], cache.k_valid[sl] if k_index is not None else None, fft_pos[sl], fft_shape, v_r,
                                b, channels, pair=pair_h if overlap is not None else None, overlap=overlap)
                if chain is not None:
                    chunks.append(fft_chain_band_energies(op, n_bands=n_bands, chain=chain,
                                                          k_index=k_index[sl] if k_index is not None else None,
                                                          valid=cache.k_valid[sl] if k_index is not None else None))
                    continue
                chunks.append(fft_stored_band_energies(op, n_bands=n_bands, store=fft_store, key=start))
                continue
            pot = potential
            if k_index is not None:  # k-centred basis: gather each k's sub-basis; padding decouples
                valid = cache.k_valid[sl]
                b = b * valid[..., None].to(b.real.dtype)
                idx = k_index[sl]
                pot = gather_rows_cols(potential, idx) * (valid[:, :, None] & valid[:, None, :]).to(potential.real.dtype)
            if self.projector_offdiag is None:
                nonlocal_h = (b * channels.to(b)[None, None, :]) @ b.conj().transpose(-1, -2)
                if aug is not None and pair_h is not None:   # two-reference d: D_01 couplings between the l = 2 slots
                    c0, c1, d01 = pair_h
                    off = (b[..., c0] * d01.to(b.real.dtype)) @ b[..., c1].mH
                    nonlocal_h = nonlocal_h + off + off.mH
            else:
                nonlocal_h = self._nonlocal_coupled(packed, strengths, b)
            total = pot + nonlocal_h.to(pot)
            if d_terms is not None:
                total = self._apply_d_terms(total, b, cache.kinetic[sl], d_terms)
            if return_hamiltonians:
                chunks.append((total + torch.diag_embed(cache.kinetic[sl].to(total.dtype)),
                               self._augmentation_s(b, aug).to(total) if aug is not None else None))
                continue
            if aug is not None:
                total = self._apply_augmentation(total, b, cache.kinetic[sl], aug)
            if chain is not None:
                chunks.append(chain_band_energies(cache.kinetic[sl], total, n_bands=n_bands, chain=chain,
                                                  k_index=k_index[sl] if k_index is not None else None,
                                                  valid=cache.k_valid[sl] if k_index is not None else None))
                continue
            warm = getattr(cache, "warm", None)
            if warm is not None and self.config.get("solver") == "chefsi" and torch.is_grad_enabled():
                # training path: warm-started CheFSI for the lowest bands, store kept per cell (train_general.cell_loss)
                chunks.append(stored_band_energies(cache.kinetic[sl], total, n_bands=n_bands,
                                                   store=warm, key=start))
                continue
            chunks.append(batched_band_energies(
                cache.kinetic[sl],
                total,
                n_bands=n_bands,
                k_batch=self.config["k_batch"],
                subset_ng_min=self.config["subset_ng_min"],
            ))
        if return_hamiltonians:
            return chunks, cache
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=0)

    def _fft_local(self, packed: SimpleNamespace, cache: SimpleNamespace, local: Tensor) -> tuple[Tensor, Tensor, tuple]:
        """Real-space local potential V(r) on an alias-free grid, from the same unique-transfer values as the dense
        path (V(M) = local(M) / Omega, M = 0 removed as the dense zero diagonal), and each k's sphere positions."""
        recip = 2.0 * torch.pi * torch.linalg.inv(packed.lattice.detach()).transpose(0, 1)
        m_int = torch.round(cache.unique_cart.to(recip) @ torch.linalg.inv(recip)).long()
        shape = grid_shape(cache.g_idx.abs().max(dim=0).values.tolist())
        v = torch.where((m_int == 0).all(dim=1), torch.zeros_like(local), local / cache.volume)
        v_r = potential_to_real(v, m_int, shape).real
        g_k = cache.g_idx_k if getattr(cache, "g_idx_k", None) is not None else cache.g_idx[None].expand(cache.kinetic.shape[0], -1, -1)
        return v_r, flat_positions(g_k, shape), shape

    def predict_bands(self, structure: Structure, kpoints: np.ndarray, n_bands: int) -> Tensor:
        packed = self.pack(structure)
        dev = self.element_features.device                  # a model on GPU (pack builds CPU tensors)
        for name, value in vars(packed).items():
            if isinstance(value, Tensor):
                setattr(packed, name, value.to(dev))
        k = torch.as_tensor(np.asarray(kpoints), dtype=packed.lattice.dtype, device=dev)
        return self.band_path(packed, k, n_bands=n_bands)

    # ------------------------------------------------------------------ checkpoints
    def checkpoint_payload(self) -> dict[str, Any]:
        return {"model": "GeneralEPM", "species_symbols": list(self.species_symbols), "config": dict(self.config), "state_dict": self.state_dict()}

    @classmethod
    def from_checkpoint(cls, payload: dict[str, Any]) -> "GeneralEPM":
        if payload.get("model") != "GeneralEPM":
            raise ValueError("not a GeneralEPM checkpoint")
        model = cls(payload["species_symbols"], **payload["config"])
        model.load_state_dict(payload["state_dict"])
        return model


def general_from_v1d(gated: nn.Module) -> GeneralEPM:
    """Build a Si-only GeneralEPM whose bands equal a general_v1d ``GatedSiEPM``.

    Requires the v1d configuration: free monopole, residual experts routed by
    ``log CN(r)/V`` at ``V1D_ROUTER_RADII``, species s/p projector with an
    ungated per-atom head, and no env/gate/long-range/coordination channels.
    """
    if tuple(gated.species_symbols) != ("Si",):
        raise ValueError("general_from_v1d expects a Si-only GatedSiEPM")
    blocked = [
        name for name in ("env_isotropic", "residual_pack_gate", "env_pack_gate", "long_range", "environment_ionic",
                          "sp_environment", "coordination_features", "quad_scale_env")
        if getattr(gated, name)
    ]
    if blocked or gated.monopole_mode != "free" or not gated.sp_gated_head or gated.sphere_density is not None:
        raise ValueError(f"not a v1d configuration: {blocked or gated.monopole_mode}")
    readouts = [gated.readout, *(gated.readout_extra or [])]
    scales = gated.sp_projector.radial_scales.detach()
    # v1d projector widths are fixed Si numbers; GeneralEPM uses c_r / r_cov(Z).
    r_cov_si = float(element_feature_table(("Si",))[0, _RCOV] * 2.5)
    model = GeneralEPM(
        ("Si",),
        channels=gated.channels,
        cutoff=gated.encoder.cutoff,
        num_radial=gated.residual_num_radial,
        n_readouts=len(readouts),
        scalar_cosine_basis=gated.residual_scalar_cosine_basis,
        scalar_cosine_cutoff=gated.residual_scalar_cosine_cutoff,
        sphere_n_basis=gated.sphere.n_basis,
        sphere_q_cutoff=gated.sphere.q_cutoff,
        router_radii=V1D_ROUTER_RADII,
        projector_l_max=1,
        projector_width_constants=[float(s) * r_cov_si for s in scales],
        charge_transfer=False,
        ecut_ry=gated.ecut_ry,
        subset_ng_min=gated.subset_ng_min,
        k_batch=gated.k_batch,
    )
    with torch.no_grad():
        model.base_residual.zero_()
        model.base_residual[0, : model.n_basis] = gated.sphere.coefficients[0]
        model.base_residual[0, model.n_basis :] = gated.sp_projector.coefficients[0].reshape(-1)
        enc = model.encoder
        enc.radial_basis.load_state_dict(gated.encoder.radial_basis.state_dict())
        enc.radial_projection.load_state_dict(gated.encoder.radial_projection.state_dict())
        enc.gating.load_state_dict(gated.encoder.gating.state_dict())
        emb = enc.species_embedding
        emb.feature_projection.weight.zero_()
        emb.feature_projection.bias.zero_()
        emb.species_residual.weight.copy_(gated.encoder.species_embedding.weight)
        for mine, theirs in zip(model.readouts, readouts):
            mine.load_state_dict(theirs.state_dict())
        if len(readouts) > 1:
            first, last = gated.expert_router[0], gated.expert_router[2]
            model.router[0].weight.zero_()
            model.router[0].weight[:, : first.weight.shape[1]] = first.weight
            model.router[0].bias.copy_(first.bias)
            model.router[2].load_state_dict(last.state_dict())
            n = gated.router_mean.numel()
            model.router_mean[:n] = gated.router_mean
            model.router_std[:n] = gated.router_std
        model.projector_head[0].load_state_dict(gated.sp_head[0].state_dict())
        model.projector_head[2].load_state_dict(gated.sp_head[2].state_dict())
    return model

