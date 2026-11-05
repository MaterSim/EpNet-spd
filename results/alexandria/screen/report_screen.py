#!/usr/bin/env python3
"""Alexandria screen, step 3 (2026-11-02): summary of the 614,729-entry screen (screen_alexandria.py, 5c-N s0).
  A. Coverage, errors, cost (GPU-seconds per cell by path thinning).
  B. Against Alexandria's PBE gaps, per group (new / MP twins) and tier: median abs gap error over Alexandria-gapped
     cells, metal/insulator agreement (gap > 0 on both sides), mean abs error over all cells (metals as 0).
  C. Fast screen vs the full scorer (path + mesh, every path point; results/alexandria/5cN_s0) on the 6,872 cells both
     computed: abs(gap_screen - gap_full) and agreement.
  D. What the screen finds: gap distribution of the new structures (EpNet and Alexandria).
Writes report.md / summary.json here, parity_hist.json (EpNet vs Alexandria gaps of the new structures, 2D counts, for
Fig. 5a) and screen_gaps.csv.gz (one row per entry) beside the results on /scratch.
Light (reads JSON): login node or srun -c 2."""
import csv
import glob
import gzip
import json
import os
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
WORK = (Path(os.environ.get("EPNET_ALEXANDRIA_DIR", Path(__file__).resolve().parents[3] / "data/alexandria")) / "screen")
TIERS = ("sp", "d10", "t2", "t3", "t4")
GROUPS = ("new", "mp_heldout", "mp_unused", "mp_train")


def read(cells):
    ok = [c for c in cells if "gap" in c]
    if not ok:
        return None
    gapped = [c for c in ok if c["ref_ind"] > 0]
    err = np.array([abs(c["gap"] - c["ref_ind"]) for c in gapped])
    return {"n": len(ok), "n_gapped": len(gapped), "median_abs": float(np.median(err)) if len(err) else None,
            "le01": float(np.mean(err <= 0.1)) if len(err) else None,
            "agree": float(np.mean([(c["gap"] > 0) == (c["ref_ind"] > 0) for c in ok])),
            "mae_all": float(np.mean([abs(c["gap"] - c["ref_ind"]) for c in ok]))}


def fmt(s):
    if s is None:
        return "- | - | - | -"
    return f"{s['n']:,} | {s['median_abs']:.3f} ({s['n_gapped']:,}) | {100 * s['agree']:.1f}% | {s['mae_all']:.3f}"


def main():
    cells = {}
    for f in sorted(glob.glob(str(WORK / "results/part[0-9][0-9][0-9].json"))):
        cells.update(json.loads(Path(f).read_text())["cells"])
    total = json.loads((WORK / "manifest.json").read_text())["n"]
    ok = {k: c for k, c in cells.items() if "gap" in c}
    errors = {k: c["error"] for k, c in cells.items() if "error" in c}
    L = ["# Alexandria screen (EpNet 5c-N s0)", "",
         f"Entries {total:,} (sealed MP twins excluded); scored {len(ok):,}; errors {len(errors)}; "
         f"rerun on CPU {sum(bool(c.get('redo')) for c in cells.values())}.", ""]
    # A. cost
    L += ["## A. Cost", "", "| path thinning | cells | mean s per cell (one GPU) | GPU-h |", "|---|---|---|---|"]
    cost = {}
    for thin in (3, 4):
        s = [c["seconds"] for c in ok.values() if (c.get("thin") or 3) == thin and "redo" not in c]
        cost[thin] = {"n": len(s), "mean_s": float(np.mean(s)), "gpu_h": float(np.sum(s) / 3600)}
        L.append(f"| every {thin}{'rd' if thin == 3 else 'th'} point | {len(s):,} | {np.mean(s):.2f} | {np.sum(s) / 3600:.0f} |")
    # B. against Alexandria
    L += ["", "## B. Against Alexandria PBE gaps", "",
          "n | median abs gap error over Alexandria-gapped cells (count) | metal/insulator agreement | MAE over all cells "
          "(metals as 0), eV", "", "| group | tier | n | median abs (gapped) | agreement | MAE all |", "|---|---|---|---|---|---|"]
    summary = {"n": total, "scored": len(ok), "errors": errors, "cost": cost, "groups": {}}
    for g in GROUPS:
        cg = [c for c in ok.values() if c["tag"] == g]
        summary["groups"][g] = {"all": read(cg)}
        L.append(f"| {g} | all | {fmt(read(cg))} |")
        if g == "new":
            for t in TIERS:
                s = read([c for c in cg if c["tier"] == t]); summary["groups"][g][t] = s
                L.append(f"| {g} | {t} | {fmt(s)} |")
    # C. screen vs full scorer
    full = {}
    for f in glob.glob(str(HERE.parent / "5cN_s0/score_part*.json")):
        full.update({k: v for k, v in json.loads(Path(f).read_text())["cells"].items() if "error" not in v})
    both = [k for k in full if k in ok]
    d = np.array([abs(ok[k]["gap"] - full[k]["gap"]) for k in both])
    dp = np.array([abs(ok[k]["gap"] - full[k]["gap_path"]) for k in both])
    agree = np.mean([(ok[k]["gap"] > 0) == (full[k]["gap"] > 0) for k in both])
    t5 = [k for k in both if full[k].get("tag") == "new"]            # the 5,000 new test structures of Fig. 5
    d5 = np.array([abs(ok[k]["gap"] - full[k]["gap"]) for k in t5])
    summary["vs_full_test5000"] = {"n": len(t5), "median": float(np.median(d5)), "p90": float(np.percentile(d5, 90)),
                                   "share_gt005": float(np.mean(d5 > 0.05))}
    summary["vs_full"] = {"n": len(both), "median": float(np.median(d)), "p90": float(np.percentile(d, 90)),
                          "share_gt005": float(np.mean(d > 0.05)), "agree": float(agree),
                          "vs_full_path_only_median": float(np.median(dp)), "vs_full_path_only_p90": float(np.percentile(dp, 90))}
    s = summary["vs_full"]
    L += ["", "## C. Fast screen vs the full scorer (path + mesh, every path point)", "",
          f"{s['n']:,} cells: abs(gap_screen - gap_full) median {s['median']:.4f}, p90 {s['p90']:.3f} eV, "
          f"{100 * s['share_gt005']:.1f}% > 0.05 eV; metal/insulator agreement {100 * s['agree']:.1f}%. "
          f"Against the full scorer's path-only gap: median {s['vs_full_path_only_median']:.4f}, p90 "
          f"{s['vs_full_path_only_p90']:.3f} eV (the rest is the mesh). On the 5,000 new test structures: median "
          f"{summary['vs_full_test5000']['median']:.4f}, p90 {summary['vs_full_test5000']['p90']:.4f} eV, "
          f"{100 * summary['vs_full_test5000']['share_gt005']:.1f}% > 0.05 eV."]
    # D. distribution
    new = [c for c in ok.values() if c["tag"] == "new"]
    edges = np.linspace(0.0, 8.5, 171)
    h2, _, _ = np.histogram2d([min(c["ref_ind"] or 0.0, 8.499) for c in new], [min(c["gap"], 8.499) for c in new], bins=[edges, edges])
    (HERE / "parity_hist.json").write_text(json.dumps({"n": len(new), "edges": edges.round(4).tolist(),
                                                       "counts_ref_by_epnet": h2.astype(int).tolist()}))
    secs = [c["seconds"] for c in ok.values() if "redo" not in c]
    summary["seconds"] = {"median": float(np.median(secs)), "mean": float(np.mean(secs)), "gpu_h": float(np.sum(secs) / 3600),
                          "wall": "2026-11-01T17:34 to 2026-11-02T12:03 (18.5 h) on <= 16 GPUs (A40, A100, H100)"}
    bins = [0, 1e-9, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 100]
    names = ["metal (0)", "(0, 0.5)", "[0.5, 1)", "[1, 1.5)", "[1.5, 2)", "[2, 3)", "[3, 4)", "[4, 6)", ">= 6"]
    he = np.histogram([c["gap"] for c in new], bins=bins)[0]
    ha = np.histogram([c["ref_ind"] for c in new], bins=bins)[0]
    L += ["", f"## D. Gap distribution, new structures ({len(new):,})", "", "| gap (eV) | EpNet | Alexandria |", "|---|---|---|"]
    L += [f"| {n} | {a:,} | {b:,} |" for n, a, b in zip(names, he, ha)]
    sel = [c for c in new if 1.0 <= c["gap"] <= 2.0 and c.get("direct") and (c["e_hull"] or 0) < 0.025]
    L += ["", f"Direct-gap candidates, EpNet gap 1.0-2.0 eV and e_hull < 0.025 eV/atom: {len(sel):,}."]
    summary["distribution"] = {"bins": names, "epnet": he.tolist(), "alexandria": ha.tolist(), "direct_1_2_stable": len(sel)}
    if errors:
        L += ["", "## Errors", ""] + [f"- {k}: {v[:150]}" for k, v in errors.items()]
    (HERE / "report.md").write_text("\n".join(L) + "\n")
    (HERE / "summary.json").write_text(json.dumps(summary, indent=1))
    with gzip.open(WORK / "screen_gaps.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mat_id", "formula", "tag", "tier", "e_hull", "nsites", "n_g_own", "gap", "gap_dir", "direct",
                    "vbm_label", "cbm_label", "alexandria_gap_ind", "alexandria_gap_dir", "thin"])
        for k, c in sorted(ok.items()):
            w.writerow([k, c["formula"], c["tag"], c["tier"], c["e_hull"], c.get("nsites"), c["n_g_own"], round(c["gap"], 4),
                        round(c["gap_dir"], 4), int(bool(c.get("direct"))), c.get("vbm_label", ""), c.get("cbm_label", ""),
                        c["ref_ind"], c["ref_dir"], c.get("thin", 3)])
    print("\n".join(L))


if __name__ == "__main__":
    main()
