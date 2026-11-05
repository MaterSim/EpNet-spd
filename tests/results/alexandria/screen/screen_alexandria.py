#!/usr/bin/env python3
"""Alexandria screen, step 2 (2026-11-01): EpNet band gaps for every in-scope entry, the fast configuration validated in
results/alexandria/fast_eval/ (thin_paths.md, compare_thinned.json):
  standardized primitive cell (symprec 1e-3, as score_alexandria.py); the Setyawan-Curtarolo path at line density 20
  thinned to every 4th point with every vertex kept (thin_paths.every(.., 4): 35 k median; gap within 4 meV of the full
  path for 90% of cells, 1.2% beyond 0.05 eV; the first ~28k cells used every 3rd point, 43 k; per cell "thin");
  no mesh; bands from epnet.screen.chain_bands (CheFSI warm-started along the path), batch 4 below 3,000 plane waves (dense path) and 1 at or above (FFT path; tune_large.json), tolerance 5e-3 eV.
  --min-ng / --max-ng restrict a run to a size range (CPU workers for small cells write results/part*_cpu.json).
Per cell: gap (min CB - max VB over the path, 0 for metals and odd electron counts), direct gap, the k of the VBM and
CBM (fractional, with the nearest vertex label when on one), and timing. The reference gaps of Alexandria are carried for
the comparison. Resumable. Usage: screen_alexandria.py --ckpt C.pt --part i [--thin 4 --batch 4 --batch-fft 1 --tol 5e-3 --max-pass 20]"""
import argparse
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "results/alexandria/fast_eval"))
from apps.epnet_bands.core import kpath, load_model  # noqa: E402
from epnet.data.structures import Structure  # noqa: E402
from epnet.data.valence import Z_MODEL  # noqa: E402
from epnet.screen import chain_bands  # noqa: E402
from thin_paths import every  # noqa: E402

WORK = (Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[3] / "data/alexandria")) / "screen")
TIERS = (("t4", "Cr Mn Fe Co Ni"), ("t3", "Mo Tc Ru Rh Pd W Re Os Ir Pt"), ("t2", "Sc Y Ti Zr Hf V Nb Ta"), ("d10", "Zn Cd Cu Ag Au Hg"))


def tier(elements):
    els = set(elements)
    return next((t for t, v in TIERS if els & set(v.split())), "sp")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--batch-fft", type=int, default=1, help="batch for cells with n_g_own >= 3,000 (FFT path)")
    p.add_argument("--tol", type=float, default=5e-3)
    p.add_argument("--max-pass", type=int, default=20)
    p.add_argument("--min-ng", type=int, default=0)
    p.add_argument("--max-ng", type=int, default=10 ** 9)
    p.add_argument("--suffix", default="")
    p.add_argument("--redo-errors", action="store_true", help="recompute only the cells recorded with an error")
    p.add_argument("--thin", type=int, default=4, help="every n-th path point, vertices kept (3 for the first ~28k cells, 4 from 2026-11-01 18:33: thin_paths.md m4, 1.2x faster)")
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    from pymatgen.core import Lattice, Structure as PS
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    src = WORK / f"parts/part{a.part:03d}.json"
    dst = WORK / f"results/part{a.part:03d}{a.suffix}.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    entries = [e for e in json.loads(src.read_text())["entries"] if a.min_ng <= (e["n_g_own"] or 0) < a.max_ng][: a.limit]
    out = json.loads(dst.read_text())["cells"] if dst.exists() else {}
    bad = {k for k, v in out.items() if "error" in v} if a.redo_errors else set()
    if a.redo_errors:                    # recompute only the failed cells (e.g. on CPU: fp32 cuSOLVER eigh did not converge)
        entries = [e for e in entries if e["mat_id"] in bad]
        out = {k: v for k, v in out.items() if k not in bad}
    model, dev, _ = load_model(a.ckpt)
    meta = {"checkpoint": a.ckpt, "batch": a.batch, "batch_fft": a.batch_fft, "tol": a.tol, "max_pass": a.max_pass,
            "ng_range": [a.min_ng, a.max_ng], "device": str(dev), "path": f"SC line density 20, every {a.thin}"}
    if a.redo_errors and dst.exists():   # keep the part's own settings; record the rerun beside them
        meta = {k: v for k, v in json.loads(dst.read_text()).items() if k != "cells"}
        meta["redo"] = {"device": str(dev), "cells": sorted(bad), "thin": a.thin}
    t0 = time.time()
    for n, e in enumerate(entries):
        if e["mat_id"] in out:
            continue
        tc = time.time()
        rec = {"redo": True} if a.redo_errors else {}
        rec |= {"tag": e["tag"], "formula": e["formula"], "tier": tier(e["elements"]), "e_hull": e["e_above_hull"],
               "ref_ind": e["band_gap_ind"], "ref_dir": e["band_gap_dir"], "n_g_own": e["n_g_own"]}
        try:
            prim = SpacegroupAnalyzer(PS(Lattice(e["lattice"]), e["species"], e["frac"]), symprec=1e-3).get_primitive_standard_structure()
            st = Structure(prim.lattice.matrix, [s.specie.symbol for s in prim], prim.frac_coords)
            n_e = int(sum(Z_MODEL[x] for x in st.species)); n_occ = int(math.ceil(n_e / 2))
            kf, _, labels, _ = kpath(prim, 20.0)
            idx = every([bool(x) for x in labels], a.thin)
            kf, labels = kf[idx], [labels[i] for i in idx]
            b = a.batch_fft if (e["n_g_own"] or 0) >= 3000 else a.batch
            ev, stats = chain_bands(model, st, kf, n_occ + 4, batch=b, tol=a.tol, max_pass=a.max_pass)
            vb, cb = ev[:, n_occ - 1], ev[:, n_occ]
            iv, ic = int(vb.argmax()), int(cb.argmin())
            odd = n_e % 2 == 1
            gap = 0.0 if odd else max(float(cb[ic] - vb[iv]), 0.0)
            rec.update({"nsites": len(prim), "n_k": len(kf), "thin": a.thin, "odd": odd, "gap": gap,
                        "gap_dir": 0.0 if odd else max(float((cb - vb).min()), 0.0),
                        "vbm_k": np.round(kf[iv], 4).tolist(), "cbm_k": np.round(kf[ic], 4).tolist(),
                        "vbm_label": labels[iv], "cbm_label": labels[ic], "direct": iv == ic or float(cb[iv] - vb[iv]) - gap < 1e-3,
                        "solver": stats, "seconds": round(time.time() - tc, 2)})
        except Exception as ex:                          # noqa: BLE001 - record and continue
            rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
        out[e["mat_id"]] = rec
        if n % 50 == 0:
            dst.write_text(json.dumps({**meta, "cells": out}))
            print(f"{n}/{len(entries)} {time.time() - t0:.0f}s", flush=True)
    dst.write_text(json.dumps({**meta, "cells": out}))
    print(f"done {len(out)} cells in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
