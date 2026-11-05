#!/usr/bin/env python3
"""Download Materials Project line-mode band structures as EpNet NPZs.

Home disk on this machine is nearly full, so the default output directory is
``/tmp/epnet_mp_bands``. The job is resumable: rerunning skips finished MP IDs.

Labels are MP GGA / GGA+U quasiparticle-unrelaxed DFT bands, not GW.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_OUT = Path("/tmp/epnet_mp_bands")
SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "chemsys",
    "elements",
    "nelements",
    "nsites",
    "band_gap",
    "is_metal",
    "is_gap_direct",
    "is_magnetic",
    "is_stable",
    "energy_above_hull",
    "volume",
    "density",
    "symmetry",
    "theoretical",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Directory for inventory, NPZs, and logs")
    p.add_argument("--inventory-only", action="store_true", help="Write the ID catalog and exit")
    p.add_argument("--workers", type=int, default=4, help="Parallel band-structure downloads")
    p.add_argument("--limit", type=int, default=0, help="Download at most this many new materials (0 = all)")
    p.add_argument("--gapped-only", action="store_true", help="Skip metals (summary is_metal or band_gap==0)")
    p.add_argument("--max-ehull", type=float, default=None, help="Keep only energy_above_hull <= this (eV/atom)")
    p.add_argument("--max-nsites", type=int, default=None, help="Skip cells larger than this")
    p.add_argument("--sleep", type=float, default=0.0, help="Optional pause between submissions (seconds)")
    return p.parse_args()


def _client():
    from mp_api.client import MPRester

    key = os.environ.get("MP_API_KEY") or os.environ.get("MAPI_KEY")
    if not key:
        raise SystemExit("Set MP_API_KEY in the environment (do not put the key in the repo).")
    return MPRester(key, mute_progress_bars=True)


def _as_id(value) -> str:
    return str(value)


def _doc_to_row(doc) -> dict:
    sym = getattr(doc, "symmetry", None)
    return {
        "material_id": _as_id(doc.material_id),
        "formula": getattr(doc, "formula_pretty", None),
        "chemsys": getattr(doc, "chemsys", None),
        "elements": [str(e) for e in (getattr(doc, "elements", None) or [])],
        "nelements": getattr(doc, "nelements", None),
        "nsites": getattr(doc, "nsites", None),
        "band_gap": getattr(doc, "band_gap", None),
        "is_metal": getattr(doc, "is_metal", None),
        "is_gap_direct": getattr(doc, "is_gap_direct", None),
        "is_magnetic": getattr(doc, "is_magnetic", None),
        "is_stable": getattr(doc, "is_stable", None),
        "energy_above_hull": getattr(doc, "energy_above_hull", None),
        "volume": getattr(doc, "volume", None),
        "density": getattr(doc, "density", None),
        "theoretical": getattr(doc, "theoretical", None),
        "spacegroup": getattr(sym, "symbol", None) if sym is not None else None,
        "spacegroup_number": getattr(sym, "number", None) if sym is not None else None,
        "crystal_system": str(getattr(sym, "crystal_system", "") or "") or None,
    }


def fetch_inventory(out: Path) -> list[dict]:
    inv_path = out / "inventory.json"
    if inv_path.exists():
        rows = json.loads(inv_path.read_text())["materials"]
        print(f"Loaded cached inventory: {len(rows)} materials from {inv_path}")
        return rows

    print("Querying Materials Project summary for materials with band structures...")
    with _client() as mpr:
        try:
            docs = mpr.materials.summary.search(has_props=["bandstructure"], fields=SUMMARY_FIELDS)
        except Exception:
            docs = mpr.materials.summary.search(has_bandstructure=True, fields=SUMMARY_FIELDS)

    rows = [_doc_to_row(d) for d in docs]
    rows.sort(key=lambda r: r["material_id"])
    payload = {
        "source": "Materials Project summary",
        "query": "has_props=bandstructure",
        "n_materials": len(rows),
        "note": "Line-mode DFT bands are downloaded separately. These are PBE/PBE+U, not GW.",
        "materials": rows,
    }
    _atomic_json(inv_path, payload)
    print(f"Wrote {len(rows)} materials to {inv_path}")
    return rows


def _atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def _keep(row: dict, args: argparse.Namespace) -> bool:
    if args.gapped_only:
        gap = row.get("band_gap")
        if row.get("is_metal") or gap is None or gap <= 0.0:
            return False
    if args.max_ehull is not None:
        ehull = row.get("energy_above_hull")
        if ehull is None or ehull > args.max_ehull:
            return False
    if args.max_nsites is not None:
        nsites = row.get("nsites")
        if nsites is None or nsites > args.max_nsites:
            return False
    return True


def _path_ticks(bs, kpoints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels_dict = getattr(bs, "labels_dict", None) or {}
    ticks = []
    labels = []
    for label, kpt in labels_dict.items():
        frac = np.asarray(getattr(kpt, "frac_coords", kpt), dtype=float)
        dist = np.linalg.norm(kpoints - frac, axis=1)
        idx = int(np.argmin(dist))
        if dist[idx] < 1e-5:
            ticks.append(idx)
            labels.append(str(label))
    order = np.argsort(ticks) if ticks else np.array([], dtype=int)
    tick_idx = np.asarray(ticks, dtype=np.int64)[order] if ticks else np.zeros(0, dtype=np.int64)
    tick_lab = np.asarray([labels[i] for i in order], dtype=object) if ticks else np.asarray([], dtype=object)
    return tick_idx, tick_lab


def _path_distance(lattice: np.ndarray, k_frac: np.ndarray) -> np.ndarray:
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    cart = k_frac @ recip
    if len(cart) == 0:
        return np.zeros(0, dtype=np.float32)
    step = np.linalg.norm(np.diff(cart, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(step)]).astype(np.float32)


def _n_occupied(bs, energies: np.ndarray, fermi: float, is_metal: bool) -> int | None:
    if is_metal:
        return None
    try:
        vbm = bs.get_vbm()
        bands = vbm.get("band_index") or {}
        if bands:
            return int(max(max(idx) for idx in bands.values() if idx) + 1)
    except Exception:
        pass
    occupied = np.any(energies <= fermi + 1e-4, axis=0)
    if not np.any(occupied):
        return None
    return int(np.max(np.flatnonzero(occupied)) + 1)


def bs_to_npz(bs, row: dict, dest: Path) -> dict:
    from pymatgen.electronic_structure.core import Spin

    struct = bs.structure
    lattice = np.asarray(struct.lattice.matrix, dtype=np.float64)
    species = np.asarray([str(sp) for sp in struct.species])
    frac = np.asarray(struct.frac_coords, dtype=np.float64)
    kpoints = np.asarray([k.frac_coords for k in bs.kpoints], dtype=np.float64)
    e_up = np.asarray(bs.bands[Spin.up], dtype=np.float32).T
    e_down = np.asarray(bs.bands[Spin.down], dtype=np.float32).T if Spin.down in bs.bands else None
    energies = e_up if e_down is None else np.stack([e_up, e_down], axis=-1).mean(axis=-1)
    fermi = float(bs.efermi)
    try:
        gap_info = bs.get_band_gap()
        gap = float(gap_info.get("energy", row.get("band_gap") or 0.0))
        direct = bool(gap_info.get("direct", row.get("is_gap_direct")))
    except Exception:
        gap = float(row.get("band_gap") or 0.0)
        direct = bool(row.get("is_gap_direct"))
    try:
        is_metal = bool(bs.is_metal())
    except Exception:
        is_metal = bool(row.get("is_metal"))
    ticks, labels = _path_ticks(bs, kpoints)
    n_occ = _n_occupied(bs, energies, fermi, is_metal)
    payload = {
        "lattice": lattice,
        "species": species,
        "fractional_positions": frac,
        "kpoints": kpoints,
        "energies": energies.astype(np.float32),
        "kpoint_weights": np.full(len(kpoints), 1.0 / max(len(kpoints), 1), dtype=np.float32),
        "path_distance": _path_distance(lattice, kpoints),
        "tick_indices": ticks,
        "path_labels": labels,
        "fermi_energy": np.asarray(fermi, dtype=np.float32),
        "label_source": np.asarray("mp-gga"),
    }
    if e_down is not None:
        payload["energies_up"] = e_up
        payload["energies_down"] = e_down
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".writing.npz")
    np.savez_compressed(tmp, **payload)
    tmp.replace(dest)
    return {
        "id": row["material_id"],
        "path": str(dest),
        "metadata": {
            **row,
            "label_source": "mp-gga",
            "n_bands": int(energies.shape[1]),
            "n_k": int(energies.shape[0]),
            "n_occupied": n_occ,
            "spin_polarized": e_down is not None,
            "gapped": (not is_metal) and gap > 0.0,
            "mp_gap_eV": gap,
            "mp_direct": direct,
            "mp_metallic": is_metal,
        },
    }


def download_one(mpr, row: dict, dest: Path) -> dict:
    bs = mpr.get_bandstructure_by_material_id(row["material_id"], line_mode=True)
    if bs is None:
        raise RuntimeError("no line-mode band structure")
    return bs_to_npz(bs, row, dest)


def npz_path(out: Path, mpid: str) -> Path:
    shard = mpid.split("-")[-1]
    shard = shard[-3:].rjust(3, "0") if shard else "000"
    return out / "npz" / shard / f"{mpid}.npz"


def write_status(out: Path, done: int, failed: int, total: int, extra: str = "") -> None:
    payload = {"done": done, "failed": failed, "total": total, "extra": extra, "updated": time.time()}
    _atomic_json(out / "download_status.json", payload)


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = fetch_inventory(out)
    selected = [r for r in rows if _keep(r, args)]
    print(f"Selected {len(selected)} / {len(rows)} materials")
    if args.inventory_only:
        return

    pending = []
    done = 0
    for row in selected:
        dest = npz_path(out, row["material_id"])
        if dest.exists() and dest.stat().st_size > 0:
            done += 1
        else:
            pending.append((row, dest))
    if args.limit:
        pending = pending[: args.limit]
    print(f"Already on disk: {done}. To download now: {len(pending)}")
    write_status(out, done, 0, len(selected), "starting")

    fail_log = out / "failures.jsonl"
    n_fail = 0
    with _client() as mpr:
        if args.workers <= 1:
            for i, (row, dest) in enumerate(pending, 1):
                try:
                    download_one(mpr, row, dest)
                    done += 1
                except Exception as exc:
                    n_fail += 1
                    with fail_log.open("a") as fh:
                        fh.write(json.dumps({"id": row["material_id"], "error": str(exc)}) + "\n")
                if i % 25 == 0 or i == len(pending):
                    print(f"  {i}/{len(pending)} this run | {done} ok | {n_fail} failed")
                    write_status(out, done, n_fail, len(selected))
                if args.sleep:
                    time.sleep(args.sleep)
        else:
            # MPRester is not guaranteed thread-safe; one client per worker.
            def _job(item):
                row, dest = item
                if args.sleep:
                    time.sleep(args.sleep)
                with _client() as local:
                    return download_one(local, row, dest)

            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(_job, item): item[0]["material_id"] for item in pending}
                for i, fut in enumerate(as_completed(futures), 1):
                    mpid = futures[fut]
                    try:
                        fut.result()
                        done += 1
                    except Exception as exc:
                        n_fail += 1
                        with fail_log.open("a") as fh:
                            fh.write(
                                json.dumps({"id": mpid, "error": str(exc), "trace": traceback.format_exc()[-400:]})
                                + "\n"
                            )
                    if i % 25 == 0 or i == len(pending):
                        print(f"  {i}/{len(pending)} this run | {done} ok | {n_fail} failed")
                        write_status(out, done, n_fail, len(selected))

    write_status(out, done, n_fail, len(selected), "finished_run")
    print(f"Done. {done} NPZs present, {n_fail} failures this run. Output: {out}")


if __name__ == "__main__":
    main()
