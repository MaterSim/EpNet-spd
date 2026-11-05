#!/usr/bin/env python3
"""EpNet-spd Alexandria external test, step 2 (2026-10-31; EpNet-sp's dedupe_mp.py for the spd scope): merge the filtered
chunks, drop Alexandria duplicates, apply the spd data scope, and match every entry against the in-scope MP band-structure
pool (same reduced formula + pymatgen StructureMatcher, default tolerances).
Scope (as the spd MP data): non-magnetic (|total_mag| < 0.05 muB per cell); no +U chemistry (oxides / fluorides of Co, Cr,
Fe, Mn, Mo, Ni, V, W); n_G at the cell's own cutoff <= 8,700 (5c-L / 5c-N's training cap).
Tags:
  new         no MP twin in the in-scope pool (the external test proper)
  mp_train    twin is a 5c-N training cell (the model saw its MP bands: code-to-code noise floor)
  mp_heldout  twin is an spd manifest cell not trained on (val, chemsys / prototype holdout, inner and select carve-outs)
  mp_sealed   twin is a sealed cell: never scored here (the sealed set is scored once, at the close-out)
  mp_unused   twin is in the MP pool but not in the spd manifest (size / n_G caps, label checks)
Writes $EPNET_ALEXANDRIA_DIR/candidates.json.   Usage: dedupe_mp.py [--workers N]"""
import argparse
import json
import os
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Lattice, Structure

ROOT = Path(__file__).resolve().parents[2]
WORK = Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[2] / "data/alexandria"))
SM = StructureMatcher()
MP_BY_FORMULA, MP_PATH = {}, {}


def alex_struct(e):
    return Structure(Lattice(e["lattice"]), e["species"], e["frac"])


def mp_struct(path):
    z = np.load(path, allow_pickle=True)
    return Structure(Lattice(z["lattice"]), [str(x) for x in z["species"]], z["fractional_positions"])


def match(args):
    i, e = args
    s = alex_struct(e)
    for mid in MP_BY_FORMULA.get(Composition(e["formula"]).reduced_formula, []):
        try:
            if SM.fit(s, mp_struct(MP_PATH[mid])):
                return i, mid
        except Exception:                 # noqa: BLE001 - a failed comparison is a non-match
            pass
    return i, None


def main():
    p = argparse.ArgumentParser(); p.add_argument("--workers", type=int, default=8); a = p.parse_args()
    entries, counts = [], defaultdict(int)
    for f in sorted((WORK / "filtered").glob("chunk_*.json")):
        j = json.loads(f.read_text())
        entries += j["entries"]
        for k, v in j["counts"].items():
            counts[k] += v
    seen, uniq = set(), []
    for e in entries:
        if e["mat_id"] not in seen:
            seen.add(e["mat_id"]); uniq.append(e)
    counts["unique"] = len(uniq)
    uniq = [e for e in uniq if abs(e["total_mag"] or 0.0) < 0.05]; counts["non_magnetic"] = len(uniq)
    uniq = [e for e in uniq if not e["plus_u"]]; counts["no_plus_u"] = len(uniq)
    uniq = [e for e in uniq if e["n_g_own"] <= 8700]; counts["n_g_own_le_8700"] = len(uniq)
    met = json.loads((ROOT / "results/stage5/5cN/s0/metrics.json").read_text())
    el, train = set(met["config"]["elements"]), set(met["train_ids"])
    split = {e["id"]: e["metadata"]["split"] for e in json.loads((ROOT / "data/manifests/general_spd_v1.json").read_text())["samples"]}
    inv = json.loads((ROOT / "data/mp_bands/inventory.json").read_text())["materials"]
    paths = {p.stem: p for p in (ROOT / "data/mp_bands/npz").glob("*/*.npz")}
    for x in inv:
        if set(x["elements"]) <= el and not x["is_magnetic"] and x["material_id"] in paths:
            MP_BY_FORMULA.setdefault(Composition(x["formula"]).reduced_formula, []).append(x["material_id"])
            MP_PATH[x["material_id"]] = paths[x["material_id"]]
    todo = [(i, e) for i, e in enumerate(uniq) if Composition(e["formula"]).reduced_formula in MP_BY_FORMULA]
    print(f"{dict(counts)}; {len(todo)} share a reduced formula with the MP pool ({len(MP_PATH)} MP cells)", flush=True)
    twin = {}
    with Pool(a.workers) as pool:
        for n, (i, mid) in enumerate(pool.imap_unordered(match, todo, chunksize=64)):
            twin[i] = mid
            if n % 5000 == 0:
                print(n, flush=True)
    tags = defaultdict(int)
    for i, e in enumerate(uniq):
        mid = twin.get(i)
        e["mp_twin"], e["mp_split"] = mid, split.get(mid)
        e["tag"] = ("new" if mid is None else "mp_train" if mid in train else "mp_sealed" if split.get(mid) == "sealed"
                    else "mp_heldout" if mid in split else "mp_unused")
        tags[e["tag"]] += 1
    print(dict(counts), dict(tags), flush=True)
    (WORK / "candidates.json").write_text(json.dumps({"filter_counts": dict(counts), "tags": dict(tags), "entries": uniq}))


if __name__ == "__main__":
    main()
