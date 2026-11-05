#!/usr/bin/env python3
"""Alexandria screen, step 1 (2026-11-01; user: all in-scope Alexandria PBE entries, speed first): split the candidate
pool (candidates.json from dedupe_mp.py: 615,564 entries within the 63 elements, e_hull < 0.1 eV/atom, <= 24 sites,
non-magnetic, no +U, n_G <= 8,700) into part files for the GPU array. The 835 twins of sealed MP cells are left out (the
sealed set is scored once, at the close-out). Entries are shuffled (seed 0) before a round-robin split, so every part
carries the same mix of sizes. CPU, ~10 GB memory: python prepare_screen.py [--nparts 400]"""
import argparse
import json
import os
import random
from pathlib import Path

WORK = Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[3] / "data/alexandria"))
KEEP = ("mat_id", "formula", "elements", "nsites", "spg", "e_above_hull", "band_gap_ind", "band_gap_dir", "n_g_own",
        "lattice", "species", "frac", "mp_twin", "mp_split", "tag")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--nparts", type=int, default=400)
    a = p.parse_args()
    c = json.loads((WORK / "candidates.json").read_text())
    entries = [{k: e.get(k) for k in KEEP} for e in c["entries"] if e["tag"] != "mp_sealed"]
    random.Random(0).shuffle(entries)
    out = WORK / "screen/parts"
    out.mkdir(parents=True, exist_ok=True)
    for i in range(a.nparts):
        (out / f"part{i:03d}.json").write_text(json.dumps({"part": i, "nparts": a.nparts, "entries": entries[i::a.nparts]}))
    tags = {}
    for e in entries:
        tags[e["tag"]] = tags.get(e["tag"], 0) + 1
    (WORK / "screen/manifest.json").write_text(json.dumps({"source": str(WORK / "candidates.json"), "filter_counts": c["filter_counts"],
                                                         "n": len(entries), "tags": tags, "nparts": a.nparts,
                                                         "excluded": {"mp_sealed": c["tags"].get("mp_sealed", 0)}}, indent=1))
    print(len(entries), tags)


if __name__ == "__main__":
    main()
