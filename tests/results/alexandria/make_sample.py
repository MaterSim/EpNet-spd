#!/usr/bin/env python3
"""EpNet-spd Alexandria external test, step 2b (2026-10-31; user's choices; fixed before any model score is seen):
the scored sample, from candidates.json (dedupe_mp.py).
  new         per tier (the hardest d element: s/p, d10, T2, T3, T4), 1,000 random entries with no MP twin (seed 0;
              all if fewer)                                                -> primary external test
  mp_heldout  1,000 random entries whose MP twin is an spd manifest cell not trained on  -> cross-check with MP held-out
  mp_unused   500 random entries whose MP twin is in the MP pool but not in the manifest
  mp_train    500 random entries whose MP twin is a 5c-N training cell     -> noise floor (Alexandria vs MP)
  mp_sealed   none: the sealed set is scored once, at the close-out
Model: 5c-N seed 0 (its smoothed-selection checkpoint).
Primary metrics (preregistered, as EpNet-sp's): on `new`, per tier and overall, the median abs(model gap - band_gap_ind)
over cells with band_gap_ind > 0 (Alexandria gapped), and the metal/insulator agreement (gap > 0 on each side).
References (reported, not gates): 5c-N seed 0's MP held-out |gap| medians (d10 0.109, T2 0.136, T3 0.100, T4 0.110)
and the MP-vs-Alexandria floor on the training twins; EpNet-sp G4's Alexandria s/p result (0.096, 98.8%)."""
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_alexandria import tier  # noqa: E402

WORK = Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[2] / "data/alexandria"))
c = json.loads((WORK / "candidates.json").read_text())
rng = random.Random(0)
groups = {}
for e in c["entries"]:
    key = (e["tag"], tier(e["elements"])) if e["tag"] == "new" else (e["tag"], None)
    groups.setdefault(key, []).append(e)
take = {("new", t): 1000 for t in ("sp", "d10", "t2", "t3", "t4")}
take |= {("mp_heldout", None): 1000, ("mp_unused", None): 500, ("mp_train", None): 500}
sample = []
for key, n in take.items():
    pool = sorted(groups.get(key, []), key=lambda e: e["mat_id"])
    sample += pool if len(pool) <= n else rng.sample(pool, n)
print({f"{k[0]}{':' + k[1] if k[1] else ''}": len(v) for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))})
print("sample:", {f"{k[0]}{':' + k[1] if k[1] else ''}": sum(e["tag"] == k[0] and (k[1] is None or tier(e["elements"]) == k[1]) for e in sample)
                  for k in take}, "total", len(sample))
(WORK / "sample.json").write_text(json.dumps({"rule": __doc__, "entries": sample}))
