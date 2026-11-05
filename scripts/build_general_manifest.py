#!/usr/bin/env python3
"""Build a general manifest: zm-v1 valence contract + preregistered splits.

Pool: nonmagnetic MP cells with at most ``--max-sites`` atoms whose elements are all in
the chosen scope: ``sp`` (default, EpNet-sp: main group, ``SP_ELEMENTS``), ``all``
(the original general-v1 scope with Cu, Ag, Au, Zn, Cd, Hg) or ``spd`` (EpNet-spd: main
group + d block, contract zm-v2). The split salt stays ``general-v1`` so a cell keeps the
split it had in general_v1.json. ``--inherit`` copies the entries of an earlier manifest
verbatim (split, flags and manual moves included), so e.g. every EpNet-sp cell keeps its
general_sp_v3 split; only cells absent from it are built and assigned here.
Every entry records the drop-boundary gap and ``n_deep_ambiguous`` kept bands that
deep-band losses must skip (see ``epnet.data.valence.drop_boundary``).
Writes the manifest and a summary with rejection and split counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epnet.data.general_manifest import assign_split, build_entry, write_general_manifest  # noqa: E402
from epnet.data.valence import CONTRACT_VERSION, D10_ELEMENTS, IN_SCOPE, SP_ELEMENTS, d_tier  # noqa: E402

SCOPES = {"sp": SP_ELEMENTS, "all": SP_ELEMENTS | D10_ELEMENTS, "spd": IN_SCOPE}
DEFAULT_OUTPUT = {"sp": "data/manifests/general_sp_v1.json", "all": "data/manifests/general_v1.json",
                  "spd": "data/manifests/general_spd_v1.json"}
DEFAULT_HOLDOUT_ELEMENTS = {"sp": ("Ge", "Se", "Sr", "In"), "all": ("Ge", "Se", "Sr", "In", "Cd"), "spd": ()}


def npz_relpath(material_id: str) -> str:
    number = int(material_id.split("-")[1])
    return f"npz/{number % 1000:03d}/{material_id}.npz"


def _work(args: tuple[dict, str, str]) -> dict:
    material, npz, rel = args
    path = Path(npz)
    if not path.exists():
        return {"id": material["material_id"], "rejected": "npz_missing"}
    return build_entry(material, path, rel)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mp-root", type=Path, default=ROOT / "data/mp_bands")
    p.add_argument("--scope", choices=sorted(SCOPES), default="sp")
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--max-sites", type=int, default=16)
    p.add_argument("--salt", default="general-v1")
    p.add_argument("--holdout-elements", nargs="+", default=None)
    p.add_argument("--no-holdout-elements", action="store_true",
                   help="no element holdout (every element trains); other splits are assigned as usual")
    p.add_argument("--sealed-frac", type=float, default=0.05)
    p.add_argument("--chemsys-frac", type=float, default=0.10)
    p.add_argument("--prototype-frac", type=float, default=0.05)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="first N pool materials only (smoke test)")
    p.add_argument("--inherit", type=Path, default=None,
                   help="earlier general manifest whose entries are copied verbatim (their splits and flags kept)")
    a = p.parse_args()
    scope = SCOPES[a.scope]
    a.output = a.output or ROOT / DEFAULT_OUTPUT[a.scope]
    a.holdout_elements = [] if a.no_holdout_elements else (a.holdout_elements or list(DEFAULT_HOLDOUT_ELEMENTS[a.scope]))

    inventory_path = a.mp_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text())["materials"]
    pool = [
        m for m in inventory
        if not m["is_magnetic"] and set(m["elements"]) <= scope and int(m["nsites"]) <= a.max_sites
    ]
    pool.sort(key=lambda m: m["material_id"])
    if a.limit:
        pool = pool[: a.limit]
    inherited: list[dict] = []
    if a.inherit:
        inherited = json.loads(a.inherit.read_text())["samples"]
        if a.inherit.resolve().parent != a.output.resolve().parent:
            raise SystemExit("--inherit: entry paths are relative to the manifest directory; write the output next to it")
        known = {e["id"] for e in inherited}
        pool = [m for m in pool if m["material_id"] not in known]
    by_id = {m["material_id"]: m for m in pool}
    out_dir = a.output.resolve().parent
    rel_root = Path(__import__("os").path.relpath(a.mp_root.resolve(), out_dir))
    jobs = [(m, str(a.mp_root / npz_relpath(m["material_id"])), str(rel_root / npz_relpath(m["material_id"]))) for m in pool]
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        built = list(ex.map(_work, jobs, chunksize=32))

    holdout = frozenset(a.holdout_elements)
    unknown = holdout - scope
    if unknown:
        raise SystemExit(f"holdout elements outside scope: {sorted(unknown)}")
    entries, rejected = [], Counter()
    rejected_by_element: dict[str, Counter] = {}
    for entry in inherited:  # verbatim; s/p entries predate the d_tier field
        entry["metadata"].setdefault("d_tier", d_tier(entry["metadata"]["elements"]))
        entry["metadata"]["inherited_from"] = a.inherit.name
        entries.append(entry)
    for entry in built:
        if "rejected" in entry:
            reason = entry["rejected"].split(":")[0]
            rejected[reason] += 1
            for el in by_id[entry["id"]]["elements"]:
                rejected_by_element.setdefault(el, Counter())[reason] += 1
            continue
        meta = entry["metadata"]
        split, reason = assign_split(
            {"id": entry["id"], "elements": meta["elements"], "chemsys": meta["chemsys"], "prototype": meta["prototype"]},
            salt=a.salt,
            holdout_elements=holdout,
            sealed_frac=a.sealed_frac,
            chemsys_frac=a.chemsys_frac,
            prototype_frac=a.prototype_frac,
            val_frac=a.val_frac,
        )
        meta["split"], meta["holdout_reason"] = split, reason
        entries.append(entry)

    splits = Counter(e["metadata"]["split"] for e in entries)
    gapped = Counter(e["metadata"]["split"] for e in entries if e["metadata"]["gapped"])
    train_elements = Counter(el for e in entries if e["metadata"]["split"] == "train" for el in e["metadata"]["elements"])
    summary = {
        "z_model_version": CONTRACT_VERSION,
        "n_pool": len(pool),
        "n_entries": len(entries),
        "rejected": dict(sorted(rejected.items())),
        "splits": {k: {"cells": splits[k], "gapped": gapped[k]} for k in sorted(splits)},
        "n_drop_nonzero": sum(e["metadata"]["band_offset"] > 0 for e in entries),
        "shallowest_dropped_top_minus_edge_eV": max(
            (e["metadata"]["dropped_top_minus_edge_eV"] for e in entries if e["metadata"]["band_offset"] > 0), default=None
        ),
        "drop_splits_multiplet": sum(
            e["metadata"]["band_offset"] > 0 and e["metadata"]["drop_boundary_gap_eV"] <= 0.05 for e in entries
        ),
        "cells_with_deep_ambiguous": sum(e["metadata"]["n_deep_ambiguous"] > 0 for e in entries),
        "train_cells_per_element": dict(sorted(train_elements.items(), key=lambda kv: kv[1])),
        "n_inherited": len(inherited),
        "tiers": {
            str(t): {"cells": sum(e["metadata"]["d_tier"] == t for e in entries),
                     "gapped": sum(e["metadata"]["d_tier"] == t and e["metadata"]["gapped"] for e in entries),
                     "splits": dict(Counter(e["metadata"]["split"] for e in entries if e["metadata"]["d_tier"] == t))}
            for t in sorted({e["metadata"]["d_tier"] for e in entries})
        },
        # per element (built cells only): a POTCAR-table error shows up as many paw_count_mismatch rejections
        "rejected_by_element": {el: dict(c) for el, c in sorted(rejected_by_element.items())},
    }
    provenance = {
        "script": "scripts/build_general_manifest.py",
        "inventory": str(inventory_path),
        "inventory_sha256": hashlib.sha256(inventory_path.read_bytes()).hexdigest(),
        "filters": {"magnetic": False, "scope": a.scope, "elements": sorted(scope), "max_sites": a.max_sites,
                    "plus_u": "rejected (MP_U_ELEMENTS with O or F)"},
        "inherited": None if not a.inherit else {"manifest": str(a.inherit), "n": len(inherited),
                                                 "sha256": hashlib.sha256(a.inherit.read_bytes()).hexdigest()},
        "splits": {
            "salt": a.salt,
            "holdout_elements": sorted(holdout),
            "sealed_frac": a.sealed_frac,
            "chemsys_frac": a.chemsys_frac,
            "prototype_frac": a.prototype_frac,
            "val_frac": a.val_frac,
        },
        "summary": summary,
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_general_manifest(a.output, entries, provenance)
    a.output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("train_cells_per_element", "rejected_by_element")}, indent=1))
    print("wrote", a.output)


if __name__ == "__main__":
    main()
