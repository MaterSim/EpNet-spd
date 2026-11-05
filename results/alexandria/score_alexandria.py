#!/usr/bin/env python3
"""EpNet-spd Alexandria external test, step 3 (2026-10-31; EpNet-sp's score_alexandria.py with the spd model): model gaps
for the sampled candidates. Per cell, on the standardized primitive cell: bands along the Setyawan-Curtarolo path (line
density 20) and on the web app's Gamma-centred mesh (KSPACING 0.12 1/A, the app's budget), with the app's code
(apps/epnet_bands/core.py). Model gap = min CB - max VB over path + mesh (0 for metals and odd electron counts); compared
with Alexandria's band_gap_ind / band_gap_dir. The FFT solver threshold is the model's own (5c-N: 3,000 plane waves).
Usage: score_alexandria.py --sample S.json --ckpt C.pt --part k --nparts n --out f.json"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from apps.epnet_bands.core import _bands, kpath, load_model, mesh_eigs  # noqa: E402
from epnet.data.structures import Structure  # noqa: E402
from epnet.data.valence import Z_MODEL  # noqa: E402

TIERS = (("t4", "Cr Mn Fe Co Ni"), ("t3", "Mo Tc Ru Rh Pd W Re Os Ir Pt"), ("t2", "Sc Y Ti Zr Hf V Nb Ta"), ("d10", "Zn Cd Cu Ag Au Hg"))


def tier(elements) -> str:
    """The hardest d element's tier (T4 > T3 > T2 > d10), as the spd data inventory bins cells; s/p otherwise."""
    els = set(elements)
    return next((t for t, v in TIERS if els & set(v.split())), "sp")


def gaps(e, n_occ):
    vb, cb = e[:, n_occ - 1], e[:, n_occ]
    return float(cb.min() - vb.max()), float((cb - vb).min())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--part", type=int, default=0); p.add_argument("--nparts", type=int, default=1)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--limit", type=int, default=None, help="score only the first N cells of this part (timing checks)")
    a = p.parse_args()
    from pymatgen.core import Lattice, Structure as PS
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    model, dev, info = load_model(a.ckpt)
    entries = json.loads(a.sample.read_text())["entries"][a.part::a.nparts][: a.limit]
    out = json.loads(a.out.read_text())["cells"] if a.out.exists() else {}      # resume
    t0 = time.time()
    for n, e in enumerate(entries):
        if e["mat_id"] in out:
            continue
        tc = time.time()
        try:
            prim = SpacegroupAnalyzer(PS(Lattice(e["lattice"]), e["species"], e["frac"]), symprec=1e-3).get_primitive_standard_structure()
            st = Structure(prim.lattice.matrix, [s.specie.symbol for s in prim], prim.frac_coords)
            n_e = int(sum(Z_MODEL[x] for x in st.species)); n_occ = int(math.ceil(n_e / 2))
            kf, _, _, _ = kpath(prim, 20.0)
            ep = _bands(model, dev, st, kf, n_occ + 4)
            t_path = time.time() - tc
            em, _, mesh = mesh_eigs(model, dev, st, prim, n_occ + 4)
            gp, dp = gaps(ep, n_occ); gm, dm = gaps(em, n_occ); gu, du = gaps(np.concatenate([ep, em]), n_occ)
            odd = n_e % 2 == 1
            out[e["mat_id"]] = {"tag": e.get("tag"), "tier": tier(e["elements"]), "formula": e["formula"], "elements": e["elements"],
                                "nsites": len(prim), "n_g_own": e.get("n_g_own"), "e_above_hull": e["e_above_hull"],
                                "ref_ind": e["band_gap_ind"], "ref_dir": e["band_gap_dir"], "odd": odd,
                                "gap_path": 0.0 if odd else max(gp, 0.0), "gap_mesh": 0.0 if odd else max(gm, 0.0),
                                "gap": 0.0 if odd else max(gu, 0.0), "gap_dir": 0.0 if odd else max(du, 0.0),
                                "mesh": list(map(int, mesh)), "n_path": int(len(kf)), "n_mesh_k": int(len(em)),
                                "seconds": round(time.time() - tc, 1), "seconds_path": round(t_path, 1)}
        except Exception as ex:              # noqa: BLE001 - record and continue
            out[e["mat_id"]] = {"tag": e.get("tag"), "tier": tier(e["elements"]), "formula": e["formula"], "error": f"{type(ex).__name__}: {ex}"[:300]}
        if n % 25 == 0:
            a.out.write_text(json.dumps({"checkpoint": a.ckpt, "cells": out}))
            print(f"{n}/{len(entries)} {time.time() - t0:.0f}s", flush=True)
    a.out.write_text(json.dumps({"checkpoint": a.ckpt, "cells": out}))


if __name__ == "__main__":
    main()
