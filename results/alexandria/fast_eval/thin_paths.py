#!/usr/bin/env python3
"""Fast-screen validation, k-point sets (2026-11-01; offline, from path_reference.py's per-k band edges): the path gap
from thinned or adaptively refined subsets of the full Setyawan-Curtarolo path (line density 20), scored against the
full path and against Alexandria's band_gap_ind, on the reference cells (all Alexandria-gapped new cells and 1,000
metals). Variants:
  every m-th point of each segment, vertices always kept (m = 2, 3, 4, 6: line density 10, 6.7, 5, 3.3); vertices only;
  adaptive m: the every-m-th set plus, around the two highest valence and two lowest conduction points of that set,
  every full-density point within m - 1 steps (one refinement pass, as a two-pass screen would compute it).
Gap = min CB - max VB over the set, 0 for metals and odd electron counts. Usage: thin_paths.py (reads ref/ref_part*.json)"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def every(vertex, m):
    n = len(vertex)
    verts = sorted(set([0, n - 1] + [i for i, v in enumerate(vertex) if v]))
    keep = set(verts)
    for a, b in zip(verts[:-1], verts[1:]):
        keep.update(range(a, b, m))
    return np.array(sorted(keep))


def refine(idx, vb, cb, m, n):
    extra = set(idx.tolist())
    for arr, pick in ((vb, np.argsort(-vb[idx])[:2]), (cb, np.argsort(cb[idx])[:2])):
        for j in pick:
            c = int(idx[j])
            extra.update(range(max(0, c - m + 1), min(n, c + m)))
    return np.array(sorted(extra))


def gap(vb, cb, idx, odd):
    return 0.0 if odd else max(float(cb[idx].min() - vb[idx].max()), 0.0)


def main():
    cells = {}
    for f in sorted(glob.glob(str(HERE / "ref/ref_part*.json"))):
        cells.update({k: v for k, v in json.loads(Path(f).read_text())["cells"].items() if "error" not in v})
    variants = {"full": None, "m2": ("every", 2), "m3": ("every", 3), "m4": ("every", 4), "m6": ("every", 6),
                "vertices": ("every", 10 ** 6), "adaptive4": ("adaptive", 4), "adaptive6": ("adaptive", 6)}
    res = {v: {"nk": [], "dfull": [], "err": [], "agree": []} for v in variants}
    for c in cells.values():
        vb, cb, vertex, odd, ref = np.array(c["vb"]), np.array(c["cb"]), c["vertex"], c["odd"], c["ref_ind"]
        n = len(vb)
        g_full = gap(vb, cb, np.arange(n), odd)
        for v, spec in variants.items():
            if spec is None:
                idx = np.arange(n)
            elif spec[0] == "every":
                idx = every(vertex, spec[1])
            else:
                idx = refine(every(vertex, spec[1]), vb, cb, spec[1], n)
            g = gap(vb, cb, idx, odd)
            r = res[v]
            r["nk"].append(len(idx)); r["dfull"].append(abs(g - g_full)); r["agree"].append((g > 0) == (ref > 0))
            if ref > 0:
                r["err"].append(abs(g - ref))
    L = [f"# Thinned and adaptive path k-sets ({len(cells)} reference cells; "
         f"{sum(c['ref_ind'] > 0 for c in cells.values())} Alexandria-gapped)", "",
         "| k-set | k per cell (median / mean) | abs(gap - full-path gap): median / p90 / max | share > 0.05 eV | "
         "abs(gap - Alexandria), gapped: median | metal/insulator agreement |", "|---|---|---|---|---|---|"]
    summary = {}
    for v, r in res.items():
        nk, d, e = np.array(r["nk"]), np.array(r["dfull"]), np.array(r["err"])
        summary[v] = {"nk_median": float(np.median(nk)), "nk_mean": float(nk.mean()), "dfull_median": float(np.median(d)),
                      "dfull_p90": float(np.percentile(d, 90)), "dfull_max": float(d.max()), "share_gt005": float((d > 0.05).mean()),
                      "err_median": float(np.median(e)), "agree": float(np.mean(r["agree"]))}
        s = summary[v]
        L.append(f"| {v} | {s['nk_median']:.0f} / {s['nk_mean']:.0f} | {s['dfull_median']:.4f} / {s['dfull_p90']:.4f} / "
                 f"{s['dfull_max']:.3f} | {100 * s['share_gt005']:.1f}% | {s['err_median']:.3f} | {100 * s['agree']:.1f}% |")
    (HERE / "thin_paths.md").write_text("\n".join(L) + "\n")
    (HERE / "thin_paths.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    main()
