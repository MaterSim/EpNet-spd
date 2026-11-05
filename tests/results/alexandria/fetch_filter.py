#!/usr/bin/env python3
"""EpNet-spd Alexandria external test, step 1 (2026-10-31; following EpNet-sp, ../EpNet-sp/results/general_sp/alexandria):
stream one Alexandria PBE (2025.07.02) chunk and keep the entries in the spd model's broad scope, with the fields the
later rules need. Kept here: elements within the model's 63 (5c-N config); e_above_hull < 0.1 eV/atom (EpNet-sp's rule);
<= 24 sites. Stored, not yet filtered (the sample rule decides): total_mag; plus_u (an oxide or fluoride of Co, Cr, Fe,
Mn, Mo, Ni, V or W: MP's +U rule, which Alexandria follows: those entries carry MP-style +U energy corrections and the
spd MP data excludes them); n_G at 8 Ry and at the cell's own cutoff (the model's ecut_by_element).
Labels: band_gap_ind, band_gap_dir, dos_ef (Alexandria has no band structures).
    fetch_filter.py <chunk index>   -> $EPNET_ALEXANDRIA_DIR/filtered/chunk_XXXXX.json"""
import bz2
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from epnet.hamiltonian import reciprocal_vectors  # noqa: E402
from epnet.hamiltonian.torch_pw import g_cutoff_angstrom  # noqa: E402

URL = "https://alexandria.icams.rub.de/data/pbe/2025.07.02/alexandria_{:05d}.json.bz2"
WORK = Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[2] / "data/alexandria"))
cfg = json.loads((ROOT / "results/stage5/5cN/s0/metrics.json").read_text())["config"]
ELEMENTS, MAX_SITES = set(cfg["elements"]), int(cfg["max_sites"])
ECUT = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in cfg["ecut_by_element"]}
U_ELEMENTS, U_ANIONS = {"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"}, {"O", "F"}


def main(k: int) -> None:
    raw, out = WORK / f"raw/alexandria_{k:05d}.json.bz2", WORK / f"filtered/chunk_{k:05d}.json"
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True); raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        subprocess.run(["curl", "-s", "--fail", "--retry", "3", "--max-time", "1800", "-o", str(raw), URL.format(k)], check=True)
    entries = json.load(bz2.open(raw))["entries"]
    keep, counts = [], {"total": len(entries), "elements": 0, "hull": 0, "sites": 0}
    for e in entries:
        d, s = e["data"], e["structure"]
        els = set(d["elements"])
        if not els <= ELEMENTS:
            continue
        counts["elements"] += 1
        if d.get("e_above_hull") is None or d["e_above_hull"] >= 0.1:
            continue
        counts["hull"] += 1
        if len(s["sites"]) > MAX_SITES:
            continue
        counts["sites"] += 1
        lat = np.array(s["lattice"]["matrix"], float)
        ecut = max([ECUT.get(x, float(cfg["ecut_ry"])) for x in els])
        keep.append({"mat_id": d["mat_id"], "formula": d["formula"], "elements": sorted(els), "nsites": len(s["sites"]),
                     "spg": d.get("spg"), "e_above_hull": d["e_above_hull"], "band_gap_ind": d.get("band_gap_ind"),
                     "band_gap_dir": d.get("band_gap_dir"), "dos_ef": d.get("dos_ef"), "total_mag": d.get("total_mag"),
                     "plus_u": bool(els & U_ELEMENTS and els & U_ANIONS), "correction": e.get("correction"),
                     "n_g_ecut8": int(len(reciprocal_vectors(lat, g_cutoff_angstrom(8.0))[0])),
                     "n_g_own": int(len(reciprocal_vectors(lat, g_cutoff_angstrom(ecut))[0])), "ecut": ecut,
                     "lattice": lat.tolist(), "species": [x["species"][0]["element"] for x in s["sites"]],
                     "frac": [x["abc"] for x in s["sites"]]})
    out.write_text(json.dumps({"chunk": k, "counts": counts, "entries": keep}))
    raw.unlink()
    print(k, counts, len(keep), flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]))
