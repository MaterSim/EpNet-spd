#!/usr/bin/env python3
"""EpNet-spd Alexandria external test, step 4 (2026-10-31; EpNet-sp's report.py with tiers): gap metrics from
results/alexandria/<MODEL>/score_part*.json (rule in make_sample.py), per tier and overall; the code-to-code floor (MP
band gap vs Alexandria band_gap_ind on the MP twins).   Usage: report.py [MODEL]"""
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
A = ROOT / "results/alexandria"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "5cN_s0"
TIERS = ("sp", "d10", "t2", "t3", "t4")
REF_MP = {"d10": 0.109, "t2": 0.136, "t3": 0.100, "t4": 0.110}          # 5c-N s0 MP chemsys held-out |gap| medians
cells = {}
for p in sorted((A / MODEL).glob("score_part*.json")):
    cells.update(json.loads(p.read_text())["cells"])
sample = {e["mat_id"]: e for e in json.loads((Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[2] / "data/alexandria")) / "sample.json").read_text())["entries"]}
inv = {x["material_id"]: x for x in json.loads((ROOT / "data/mp_bands/inventory.json").read_text())["materials"]}
man = json.loads((ROOT / "data/manifests/general_spd_v1.json").read_text())["samples"]
met = json.loads((ROOT / "results/stage5/5cN/s0/metrics.json").read_text())
train_ids = set(met["train_ids"])
train_chemsys = {e["metadata"]["chemsys"] for e in man if e["id"] in train_ids}
ok = {k: v for k, v in cells.items() if "error" not in v}
lines = [f"# Alexandria external test, {MODEL}", "", f"scored {len(ok)} of {len(sample)} ({len(cells) - len(ok)} errors)"]


def stats(ids, key="gap"):
    g = [i for i in ids if (ok[i]["ref_ind"] or 0) > 0]
    err = np.array([ok[i][key] - ok[i]["ref_ind"] for i in g])
    ref_ins = np.array([(ok[i]["ref_ind"] or 0) > 0 for i in ids]); mod_ins = np.array([ok[i][key] > 0 for i in ids])
    if not len(ids):
        return None
    s = {"n": len(ids), "n_gapped": len(g), "class_agree": float(np.mean(ref_ins == mod_ins)),
         "false_metal": int(np.sum(ref_ins & ~mod_ins)), "false_gap": int(np.sum(~ref_ins & mod_ins))}
    if len(err):
        s |= {"abs_med": float(np.median(np.abs(err))), "signed_med": float(np.median(err)), "le_0.1": float(np.mean(np.abs(err) <= 0.1)),
              "le_0.3": float(np.mean(np.abs(err) <= 0.3)), "p90": float(np.percentile(np.abs(err), 90))}
    return s


def show(name, s):
    if not s:
        return
    gap = (f"abs(gap err) median {s['abs_med']:.3f} (signed {s['signed_med']:+.3f}; <=0.1 {s['le_0.1']:.0%}, <=0.3 {s['le_0.3']:.0%}, "
           f"p90 {s['p90']:.2f}); " if "abs_med" in s else "")
    lines.append(f"- {name}: n {s['n']} (gapped {s['n_gapped']}): {gap}metal/insulator agree {s['class_agree']:.1%} "
                 f"(false metal {s['false_metal']}, false gap {s['false_gap']})")


out = {}
new = [i for i in ok if ok[i]["tag"] == "new"]
lines += ["", "## New structures (no MP twin), per tier (primary)"]
for t in TIERS:
    out[f"new_{t}"] = stats([i for i in new if ok[i]["tier"] == t])
    show(f"{t}" + (f" (MP held-out ref {REF_MP[t]:.3f})" if t in REF_MP else " (G4 Alexandria s/p ref 0.096)"), out[f"new_{t}"])
out["new_all"] = stats(new); show("all tiers", out["new_all"])
lines += ["", "## Twin groups"]
for tag in ("mp_heldout", "mp_unused", "mp_train"):
    out[tag] = stats([i for i in ok if ok[i]["tag"] == tag]); show(tag, out[tag])
lines += ["", "## New structures, breakdowns"]
show("path-only gap", stats(new, "gap_path"))
for t in TIERS[1:]:
    seen = [i for i in new if ok[i]["tier"] == t and "-".join(sorted(ok[i]["elements"])) in train_chemsys]
    unseen = [i for i in new if ok[i]["tier"] == t and "-".join(sorted(ok[i]["elements"])) not in train_chemsys]
    show(f"{t}, chemsys with training cells", stats(seen)); show(f"{t}, chemsys unseen in training", stats(unseen))
for lo, hi in ((0, 0.001), (0.001, 0.05), (0.05, 0.1)):
    show(f"e_hull [{lo}, {hi})", stats([i for i in new if lo <= ok[i]["e_above_hull"] < hi]))
for lo, hi in ((1, 8), (9, 16), (17, 24)):
    show(f"{lo}-{hi} sites", stats([i for i in new if lo <= ok[i]["nsites"] <= hi]))
for n in (2, 3, 4):
    show(f"{n} elements" if n < 4 else ">= 4 elements", stats([i for i in new if (len(ok[i]["elements"]) == n if n < 4 else len(ok[i]["elements"]) >= 4)]))
lines += ["", "## Code-to-code floor (MP band_gap vs Alexandria band_gap_ind on twins)"]
for tag in ("mp_train", "mp_heldout"):
    ids = [i for i in ok if ok[i]["tag"] == tag and sample[i]["mp_twin"] in inv]
    if not ids:
        continue
    mp = np.array([inv[sample[i]["mp_twin"]]["band_gap"] for i in ids]); al = np.array([ok[i]["ref_ind"] or 0 for i in ids])
    mo = np.array([ok[i]["gap"] for i in ids]); g = (mp > 0) & (al > 0)
    if g.any():
        lines.append(f"- {tag}: n {len(ids)} (both gapped {g.sum()}): abs(MP - Alexandria) median {np.median(np.abs(mp - al)[g]):.3f}, "
                     f"p90 {np.percentile(np.abs(mp - al)[g], 90):.2f}; abs(model - MP) median {np.median(np.abs(mo - mp)[g]):.3f}; "
                     f"abs(model - Alexandria) median {np.median(np.abs(mo - al)[g]):.3f}")
lines += ["", "## New structures, per element (abs gap error median, >= 20 gapped)"]
rows = []
for el in sorted({e for i in new for e in ok[i]["elements"]}):
    s = stats([i for i in new if el in ok[i]["elements"]])
    if s and s["n_gapped"] >= 20:
        rows.append((s["abs_med"], el, s["n_gapped"]))
lines.append("  " + "  ".join(f"{el} {m:.3f} ({n})" for m, el, n in sorted(rows, reverse=True)))
worst = sorted(((abs(ok[i]["gap"] - ok[i]["ref_ind"]), ok[i]["formula"], i, ok[i]["ref_ind"], ok[i]["gap"], ok[i]["tier"]) for i in new
                if (ok[i]["ref_ind"] or 0) > 0), reverse=True)[:15]
lines += ["", "## Worst new gapped cells"] + [f"- {w[0]:.2f} {w[1]} ({w[5]}) {w[2]}: Alexandria {w[3]:.2f}, model {w[4]:.2f}" for w in worst]
(A / MODEL / "report.md").write_text("\n".join(lines) + "\n")
(A / MODEL / "summary.json").write_text(json.dumps(out, indent=1))
print("\n".join(lines))
