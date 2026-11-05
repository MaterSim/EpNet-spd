#!/usr/bin/env python3
"""Stage 5a of docs/d_channel_plan.md: scoring and readings for cells <= 8 sites (all tiers + s/p in one run).

Cell groups (one list for both seeds, results/stage5/eval_cells.json, built from seed 0's metrics):
- chem:  manifest holdout_chemsys cells <= 8 sites (d and s/p)      -> R1, R2, R3, R5
- proto: holdout_prototype <= 8 sites; val: val <= 8 sites           -> reported (proto s/p also in R3)
- inner: the trainer's scored inner-holdout subset                   -> reported
- s4:    the Stage 4 unseen cells of 4c-S s0 (<= 3 sites), its labels -> R4
- train: a fixed sample (seed 0) of 150 train cells per tier and 150 s/p -> R5 (held-out <= 2x train)
Scoring is on the path thinned to about --max-k k-points (default 64; amendment of 2026-10-24).
Modes: --cells (build the list); --run SEED --chunk I --nchunks N (score part of it with results/stage5/5a/s<SEED>);
--g4 --chunk I --nchunks N (G4 on the s/p cells of chem and proto); --report (readings; results/stage5/report_5a.md).

Stage 5c-L (--arm 5cL; preregistration of 2026-10-24, 5c-L draft of 2026-10-25, R2 gate of 2026-10-28): the same modes
on the <= 24-site holdouts (results/stage5/eval_cells_5cL.json, from 5c-L's metrics.json), plus the group t2gap (every
gapped T2 training cell, R7) and chem5a (5a's chemsys cells missing from the 5c set, R5). Cells are split into pass A
(the gated readings: chem, chem5a, s4, train, t2gap, s/p proto) and pass B (reported only: val, d proto, inner), scored
with --pass A|B; d character only below CHAR_MAX_NG plane waves (coverage reported). G4's s/p baseline (--g4 --arm 5cL)
goes to results/stage5/g4_24/ with the FFT path from 3,000 plane waves, as 5c-L.

Close-out (user, 2026-11-02; results/stage5/closeout/README.md): --closeout-cells writes results/stage5/closeout/cells.json
(groups sealed, big_sp, big_d); --run SEED --arm 5cN --cells-file that --out-prefix closeout_chunk_ scores them as above.
Cells are looked up in general_spd_v1.json, then general_sp_pool60.json and bigcell_spd_v1.json (the big cells).
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

TIERS = {"d10": ("Zn", "Cd", "Cu", "Ag", "Au", "Hg"), "t2": ("Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta"),
         "t3": ("Mo", "Tc", "Ru", "Rh", "Pd", "W", "Re", "Os", "Ir", "Pt"), "t4": ("Cr", "Mn", "Fe", "Co", "Ni")}
D = {x for v in TIERS.values() for x in v}
OUT = ROOT / "results/stage5"
G4 = ROOT / "models/stages/epnet_sp_g4_s0.pt"
S4 = ROOT / "results/stage4/4c-S/s0"
WATCH = ("TiO2", "NiC", "LiNiN", "MnN", "FeN", "ReC2", "CuCl", "CuBr")
SEEDS = (0, 1)
ARM = "5a"   # run directory under results/stage5 (--arm; 5ag = the gap-focused continuation); the cell list is 5a's
# Stage 5a amendment (user, 2026-10-24): score on the band path thinned to about MAX_K k-points (vertices, VBM, CBM
# kept; the trainer's thin_row) and the d character on at most MAX_K_CHAR of those; None = the full path
MAX_K, MAX_K_CHAR = None, 32
CHAR_MAX_NG = None   # 5c-L: d character (dense Hamiltonians) only for cells below 6,000 plane waves (preregistered)
CHAIN = False        # close-out big cells (--chain): warm-started CheFSI along the thinned path, k batch 1 (epnet.screen)
FFT_NG = None        # override of the checkpoint's fft_ng_threshold (G4 on the 5c-L cells: 3,000, as 5c-L)
CELLS = {"5cL": "eval_cells_5cL.json", "5cN": "eval_cells_5cL.json"}   # arm -> cell list (default eval_cells.json); 5c-N: the same cells
G4_SP = "Al As B Ba Be Bi Br C Ca Cl Cs F Ga Ge H I In K Li Mg N Na O P Pb Rb S Sb Si Sn Te Tl Sr Se".split()


def cells_path() -> Path:
    return OUT / CELLS.get(ARM, "eval_cells.json")


def _load(ckpt: str):
    import torch
    from epnet.models.general_epm import GeneralEPM
    payload = torch.load(ckpt, map_location="cpu", weights_only=False)
    if FFT_NG is not None:
        payload["config"] = {**payload["config"], "fft_ng_threshold": FFT_NG}
    return GeneralEPM.from_checkpoint(payload).eval()


def _prepare_thin(model, entry: dict) -> dict:
    """The full-path row of one cell thinned to MAX_K. The projectors of the full-path cache are not computed: they are
    ~97% of it, prepare_cell discards them and the scorer rebuilds them for the kept k (a full <= 24-site path needs
    several GB per cell, and 14 preparing workers ran out of memory). The row is otherwise unchanged."""
    import torch
    from epnet.train_general import prepare_cell, thin_row
    model.projectors = lambda *args, **kw: torch.empty(0)   # instance attribute shadows the method while preparing
    try:
        row = prepare_cell(model, entry, ROOT / "data/manifests", k_stride=1, full_path=True)
    finally:
        del model.projectors
    return thin_row(row, MAX_K)


def _want_char(row: dict) -> bool:
    return CHAR_MAX_NG is None or float(row["n_g"]) < CHAR_MAX_NG


def _char_k(row: dict) -> np.ndarray:
    k = row["k"].numpy()
    return k[:: max(1, -(-len(k) // MAX_K_CHAR))] if MAX_K is not None else k


def build_cells() -> list[dict]:
    from epnet.data.general_manifest import load_general_manifest
    from train_general_sp import pick
    from epnet.train_general import ordered_subset

    met = json.loads((OUT / "5a/s0/metrics.json").read_text())
    cfg = met["config"]
    raw = load_general_manifest(Path(cfg["manifest"]))
    by_id = {e["id"]: e for e in raw["samples"]}
    a = SimpleNamespace(elements=cfg["elements"], require_any=None, max_sites=8, max_ng8=cfg["max_ng8"], seed=0)
    rows = [("chem", e) for e in pick(raw, "holdout_chemsys", a, None)] + [("proto", e) for e in pick(raw, "holdout_prototype", a, None)]
    rows += [("val", e) for e in pick(raw, "val", a, None)] + [("inner", by_id[i]) for i in met["inner_ids"]]
    rows += [("s4:" + r["group"], by_id[r["id"]]) for r in json.loads((S4 / "unseen.json").read_text())]
    train = [by_id[i] for i in met["train_ids"]]
    for t, els in list(TIERS.items()) + [("sp", None)]:
        pool = [e for e in train if (set(e["metadata"]["elements"]) & set(els) if els else not set(e["metadata"]["elements"]) & D)]
        rows += [("train:" + t, e) for e in ordered_subset(pool, 150, 0)]
    return [{"group": g, "id": e["id"]} for g, e in rows]


def build_cells_5c(seed: int) -> list[dict]:
    """5c-L list from 5c-L's metrics.json (train and inner ids are the same for both seeds): chem / proto / val <= 24
    sites, inner, the Stage 4 cells, 150 train cells per tier and s/p, every gapped T2 training cell (t2gap, R7), and
    5a's chemsys cells not in the 5c chem set (chem5a, R5). pass A = the gated readings, B = reported only."""
    from epnet.data.general_manifest import load_general_manifest
    from train_general_sp import pick
    from epnet.train_general import ordered_subset

    met = json.loads((OUT / f"5cL/s{seed}/metrics.json").read_text())
    cfg = met["config"]
    raw = load_general_manifest(Path(cfg["manifest"]))
    by_id = {e["id"]: e for e in raw["samples"]}
    a = SimpleNamespace(elements=cfg["elements"], require_any=None, max_sites=cfg["max_sites"], max_ng8=cfg["max_ng8"], seed=0)
    rows = [("chem", e) for e in pick(raw, "holdout_chemsys", a, None)] + [("proto", e) for e in pick(raw, "holdout_prototype", a, None)]
    rows += [("val", e) for e in pick(raw, "val", a, None)] + [("inner", by_id[i]) for i in met["inner_ids"]]
    rows += [("s4:" + r["group"], by_id[r["id"]]) for r in json.loads((S4 / "unseen.json").read_text())]
    chem = {e["id"] for g, e in rows if g == "chem"}
    rows += [("chem5a", by_id[c["id"]]) for c in json.loads((OUT / "eval_cells.json").read_text()) if c["group"] == "chem" and c["id"] not in chem]
    train = [by_id[i] for i in met["train_ids"]]
    for t, els in list(TIERS.items()) + [("sp", None)]:
        pool = [e for e in train if (set(e["metadata"]["elements"]) & set(els) if els else not set(e["metadata"]["elements"]) & D)]
        rows += [("train:" + t, e) for e in ordered_subset(pool, 150, 0)]
    rows += [("t2gap", e) for e in train if e["metadata"]["gapped"] and set(e["metadata"]["elements"]) & set(TIERS["t2"])]
    sp = lambda e: not set(e["metadata"]["elements"]) & D   # noqa: E731
    gated = lambda g, e: g in ("chem", "chem5a", "t2gap") or g.startswith(("s4:", "train:")) or (g == "proto" and sp(e))  # noqa: E731
    return [{"group": g, "id": e["id"], "pass": "A" if gated(g, e) else "B"} for g, e in rows]


def score_one(job: tuple[str, str, dict, bool]) -> dict:
    import torch
    torch.set_num_threads(1)
    from epnet.data.general_manifest import load_general_sample
    from epnet.diagnostics.band_character import band_d_character
    from epnet.train_general import score_cell

    ckpt, group, entry, want_character = job
    model = _load(ckpt)
    root = ROOT / "data/manifests"
    md = entry["metadata"]
    out = {"id": entry["id"], "formula": md["formula"], "group": group, "gapped": md["gapped"], "elements": md["elements"],
           "nsites": md["nsites"]}
    try:
        row = _prepare_thin(model, entry)
        with torch.no_grad():
            s = score_cell(model, row, torch.device("cpu"), metal_window="ef")
        out["window_mae_eV"] = s["window_mae_eV"]
        if md["gapped"]:
            out["gap_error_eV"] = s["gap_error_eV"]
        st = load_general_sample(entry, root).structure
        n_d = 5 * sum(x in D for x in st.species)
        out["n_g"] = float(row["n_g"])
        if want_character and n_d and _want_char(row):
            e, c = band_d_character(model, st, _char_k(row), n_bands=row["n_occ"] + n_d)
            sel = np.sort(np.argsort(-c.mean(0))[:n_d])
            out.update(d_character=float(c[:, sel].mean()), d_expelled=bool(c[:, sel].mean() < 0.5))
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


_PREP_MODEL = None


def _prep_init(ckpt: str) -> None:
    global _PREP_MODEL
    import torch
    torch.set_num_threads(1)
    _PREP_MODEL = _load(ckpt)


def _prep_one(entry: dict):
    """CPU worker: the full-path row of one cell thinned to MAX_K (or the error text)."""
    try:
        return _prepare_thin(_PREP_MODEL, entry)
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"[:300]


def score_on_device(ckpt: str, jobs: list[tuple[str, str, dict, bool]], device: str, prep_workers: int) -> list[dict]:
    """GPU scoring (Stage 5a amendment): rows prepared by CPU workers ahead of the GPU, the model loaded once; the same
    numbers as score_one (window MAE, gap error, d character). A cell that fails on the device is rescored on the CPU."""
    import torch
    from epnet.data.general_manifest import load_general_sample
    from epnet.diagnostics.band_character import band_d_character
    from epnet.train_general import score_cell

    dev = torch.device(device)
    model = _load(ckpt).to(dev)
    if CHAIN:
        model.config["k_batch"] = 1
    root = ROOT / "data/manifests"
    out = []
    ahead = 2 * prep_workers   # rows prepared ahead of the GPU (bounded: <= 24-site rows are large)
    with ProcessPoolExecutor(prep_workers, initializer=_prep_init, initargs=(ckpt,)) as ex:
        futs = [ex.submit(_prep_one, job[2]) for job in jobs[:ahead]]
        for i, job in enumerate(jobs):
            if i + ahead < len(jobs):
                futs.append(ex.submit(_prep_one, jobs[i + ahead][2]))
            fut = futs[i]
            futs[i] = None
            _, group, entry, want_character = job
            md = entry["metadata"]
            r = {"id": entry["id"], "formula": md["formula"], "group": group, "gapped": md["gapped"],
                 "elements": md["elements"], "nsites": md["nsites"], "device": device}
            row = fut.result()
            try:
                if isinstance(row, str):
                    raise RuntimeError(row)
                with torch.no_grad():
                    sc = score_cell(model, row, dev, metal_window="ef",
                                    chain={"tol": 5e-3, "max_pass": 20} if CHAIN else None)
                r["window_mae_eV"] = sc["window_mae_eV"]
                if md["gapped"]:
                    r["gap_error_eV"] = sc["gap_error_eV"]
                st = load_general_sample(entry, root).structure
                n_d = 5 * sum(x in D for x in st.species)
                r["n_g"] = float(row["n_g"])
                if want_character and n_d and _want_char(row):
                    e, c = band_d_character(model, st, _char_k(row), n_bands=row["n_occ"] + n_d)
                    sel = np.sort(np.argsort(-c.mean(0))[:n_d])
                    r.update(d_character=float(c[:, sel].mean()), d_expelled=bool(c[:, sel].mean() < 0.5))
            except Exception as exc:                     # e.g. out of memory: the CPU path for this cell
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                r = score_one(job) | {"device": "cpu-fallback", "device_error": f"{type(exc).__name__}: {exc}"[:200]}
            out.append(r)
    return out


MANIFESTS = ("general_spd_v1.json", "bigcell_sp_v1.json", "bigcell_spd_v1.json")   # close-out: big cells after the main one


def entries_by_id(ids: set[str]) -> dict:
    from epnet.data.general_manifest import load_general_manifest
    out = {}
    for name in MANIFESTS:
        path = ROOT / "data/manifests" / name
        if path.exists() and ids - out.keys():
            out.update({e["id"]: e for e in load_general_manifest(path)["samples"] if e["id"] in ids and e["id"] not in out})
    return out


def score_chunk(ckpt: str, cells: list[dict], chunk: int, nchunks: int, workers: int, device: str = "cpu") -> list[dict]:
    by_id = entries_by_id({c["id"] for c in cells})
    cost = lambda c: by_id[c["id"]]["metadata"]["n_k"] * by_id[c["id"]]["metadata"]["n_g_ecut8"] ** 3   # noqa: E731
    order = sorted(cells, key=cost, reverse=True)[chunk::nchunks]          # balanced: round-robin over the cost order
    jobs = [(ckpt, c["group"], by_id[c["id"]], c["group"] in ("chem", "chem5a") and bool(set(by_id[c["id"]]["metadata"]["elements"]) & D))
            for c in order]
    if device != "cpu":
        return score_on_device(ckpt, jobs, device, workers)
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(score_one, jobs, chunksize=1))


def tier_of(r: dict) -> list[str]:
    els = set(r["elements"])
    tiers = [t for t, v in TIERS.items() if els & set(v)]
    return tiers or ["sp"]


def load_run(seed: int) -> list[dict]:
    d = OUT / ARM / f"s{seed}"
    rows = []
    for p in sorted(d.glob("eval_chunk_*.json")):
        rows += json.loads(p.read_text())
    return rows


def report() -> None:
    cells = json.loads((OUT / "eval_cells.json").read_text())
    g4 = {r["id"]: r for p in sorted((OUT / "g4").glob("chunk_*.json")) for r in json.loads(p.read_text()) if "error" not in r}
    # R4 references scored the same way as 5a (thinned paths, amendment of 2026-10-24): 4c-S s0 on its Stage 4 cells,
    # results/stage5_pre/thin_check.json (the full-path values are in results/stage4/4c-S/s0/unseen.json)
    s4_ref = {t: v["thin_median"] for t, v in json.loads((ROOT / "results/stage5_pre/thin_check.json").read_text())["tiers"].items()}
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")   # noqa: E731
    from epnet.data.general_manifest import load_general_manifest
    # R2s (user, 2026-10-26): R2 on the stable / metastable compounds only (energy above hull <= 0.1 eV/atom); reported
    ehull = {e["id"]: e["metadata"].get("energy_above_hull") for e in load_general_manifest(ROOT / "data/manifests/general_spd_v1.json")["samples"]}
    per = {}
    for seed in SEEDS:
        rows = [r for r in load_run(seed) if "error" not in r]
        if not rows:
            continue
        x = {"n_scored": len(rows), "n_errors": len(load_run(seed)) - len(rows), "n_cells": len(cells),
             "finished": (OUT / ARM / f"s{seed}" / "metrics.json").exists()}
        for t in TIERS:
            chem = [r for r in rows if r["group"] == "chem" and t in tier_of(r)]
            x[f"{t}_chem_median"] = med([r["window_mae_eV"] for r in chem])
            x[f"{t}_chem_n"] = len(chem)
            x[f"{t}_gap_median"] = med([abs(r["gap_error_eV"]) for r in chem if r["gapped"] and "gap_error_eV" in r])
            x[f"{t}_gap_n"] = sum(r["gapped"] and "gap_error_eV" in r for r in chem)
            stable = [r for r in chem if r["gapped"] and "gap_error_eV" in r and ehull.get(r["id"]) is not None and ehull[r["id"]] <= 0.1]
            x[f"{t}_gap_stable_median"] = med([abs(r["gap_error_eV"]) for r in stable])
            x[f"{t}_gap_stable_n"] = len(stable)
            x[f"{t}_train_median"] = med([r["window_mae_eV"] for r in rows if r["group"] == f"train:{t}"])
            x[f"{t}_s4_median"] = med([r["window_mae_eV"] for r in rows if r["group"].startswith(f"s4:{t}:")])
            for g in ("proto", "val", "inner"):
                x[f"{t}_{g}_median"] = med([r["window_mae_eV"] for r in rows if r["group"] == g and t in tier_of(r)])
            x[f"{t}_per_element"] = {el: [med([r["window_mae_eV"] for r in chem if el in r["elements"]]),
                                          sum(el in r["elements"] for r in chem)] for el in TIERS[t]}
        dchem = [r for r in rows if r["group"] == "chem" and "d_character" in r]
        x["expelled_frac"] = sum(r["d_expelled"] for r in dchem) / max(len(dchem), 1)
        sp = [r for r in rows if r["group"] in ("chem", "proto") and tier_of(r) == ["sp"] and r["id"] in g4]
        x["sp_n"] = len(sp)
        x["sp_window_change"] = med([r["window_mae_eV"] - g4[r["id"]]["window_mae_eV"] for r in sp])
        x["sp_gap_change"] = med([abs(r["gap_error_eV"]) - abs(g4[r["id"]]["gap_error_eV"]) for r in sp
                                  if r["gapped"] and "gap_error_eV" in r and "gap_error_eV" in g4[r["id"]]])
        x["sp_train_median"] = med([r["window_mae_eV"] for r in rows if r["group"] == "train:sp"])
        x["watch"] = {w: [round(r["window_mae_eV"], 3) for r in rows if r["formula"] == w and r["group"] in ("chem", "proto", "val", "inner")]
                      for w in WATCH}
        per[seed] = x
    m = lambda k: float(np.mean([v[k] for v in per.values()])) if per else float("nan")   # noqa: E731
    two = len(per) == 2
    readings = {
        "R1": {t: m(f"{t}_chem_median") for t in TIERS} | {"supported": two and all(m(f"{t}_chem_median") < 0.10 for t in TIERS)},
        "R2": {t: m(f"{t}_gap_median") for t in TIERS} | {"supported": two and all(m(f"{t}_gap_median") < 0.10 for t in ("d10", "t2", "t3")),
                                                           "note": "T4 has 3 gapped chemsys cells: reported, not gated"},
        "R2s": {t: m(f"{t}_gap_stable_median") for t in TIERS} | {"n": {t: m(f"{t}_gap_stable_n") for t in TIERS},
                "note": "R2 on chemsys cells with E_hull <= 0.1 eV/atom; reported, not gated (user, 2026-10-26)"},
        "R3": {"window": m("sp_window_change"), "gap": m("sp_gap_change"),
               "supported": two and m("sp_window_change") <= 0.005 and m("sp_gap_change") <= 0.005},
        "R4": {t: m(f"{t}_s4_median") for t in TIERS} | {"reference": s4_ref,
               "supported": two and all(m(f"{t}_s4_median") <= s4_ref[t] + 0.01 for t in TIERS)},
        "R5": {"heldout_over_train": {t: m(f"{t}_chem_median") / m(f"{t}_train_median") for t in TIERS},
               "expelled_frac": [v["expelled_frac"] for v in per.values()], "runs_finished": sum(v["finished"] for v in per.values()),
               "supported": two and all(m(f"{t}_chem_median") <= 2 * m(f"{t}_train_median") for t in TIERS)
               and all(v["expelled_frac"] <= 0.10 for v in per.values()) and all(v["finished"] for v in per.values())},
    }
    lines = [f"# Stage 5a ({ARM}): cells <= 8 sites, all tiers + s/p", "",
             "Scored on band paths thinned to about 64 k (vertices, VBM, CBM kept), the baselines (G4, 4c-S) too. On the "
             "Stage 4 cells this reads 0.0005-0.003 eV above full-path scoring, so the absolute readings (R1, R2) are "
             "slightly conservative; the paired ones (R3, R4) compare like with like.", "",
             "| seed | tier | chemsys median (n) | |gap| median (n) | train median | Stage 4 cells (ref) | proto / val / inner |",
             "|---|---|---|---|---|---|---|"]
    for seed, x in per.items():
        for t in TIERS:
            lines.append(f"| {seed} | {t} | {x[f'{t}_chem_median']:.3f} ({x[f'{t}_chem_n']}) | {x[f'{t}_gap_median']:.3f} ({x[f'{t}_gap_n']}) | "
                         f"{x[f'{t}_train_median']:.3f} | {x[f'{t}_s4_median']:.3f} ({s4_ref[t]:.3f}) | "
                         f"{x[f'{t}_proto_median']:.3f} / {x[f'{t}_val_median']:.3f} / {x[f'{t}_inner_median']:.3f} |")
        lines.append(f"| {seed} | s/p | window change vs G4 {x['sp_window_change']:+.4f}, |gap| change {x['sp_gap_change']:+.4f} ({x['sp_n']} cells) | | "
                     f"{x['sp_train_median']:.3f} | | expelled {x['expelled_frac']:.3f}; errors {x['n_errors']} |")
    lines += ["", "Readings:", *[f"- {k}: {json.dumps(v)}" for k, v in readings.items()]]
    (OUT / f"report_{ARM}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"summary_{ARM}.json").write_text(json.dumps({"readings": readings, "runs": per}, indent=1) + "\n")
    print("\n".join(lines))


def report_5c() -> None:
    """Stage 5c-L readings (mean of the two seeds; tiers d10, T2, T3, T4; preregistered 2026-10-24 / 2026-10-25):
    R1 chemsys window < 0.10 eV per tier; R2 chemsys |gap| < 0.15 eV for d10 / T2 / T3, T4 reported (gate of 2026-10-28),
    R2s reported; R3 s/p chemsys + prototype cells within 0.005 eV of G4 (window and |gap|); R4 chemsys <= 2x train per
    tier; R5 Stage 4 cells within 4c-S + 0.01 and 5a's chemsys cells within 5a-L's final + 0.01 per tier; R6 d-expelled
    <= 10% (coverage reported) and both runs finished; R7 T2 training |gap| median <= 0.06 eV (every gapped T2 training cell)."""
    from epnet.data.general_manifest import load_general_manifest
    cells = json.loads(cells_path().read_text())
    g4 = {r["id"]: r for p in sorted((OUT / "g4_24").glob("chunk_*.json")) for r in json.loads(p.read_text()) if "error" not in r}
    s4_ref = {t: v["thin_median"] for t, v in json.loads((ROOT / "results/stage5_pre/thin_check.json").read_text())["tiers"].items()}
    ref5a = json.loads((OUT / "summary_5aL.json").read_text())["readings"]["R1"]
    ids5a = {c["id"] for c in json.loads((OUT / "eval_cells.json").read_text()) if c["group"] == "chem"}
    ehull = {e["id"]: e["metadata"].get("energy_above_hull") for e in load_general_manifest(ROOT / "data/manifests/general_spd_v1.json")["samples"]}
    med = lambda xs: float(np.median(xs)) if len(xs) else float("nan")   # noqa: E731
    gaps = lambda rs: [abs(r["gap_error_eV"]) for r in rs if r["gapped"] and "gap_error_eV" in r]   # noqa: E731
    per = {}
    for seed in SEEDS:
        every = load_run(seed)
        rows = [r for r in every if "error" not in r]
        if not rows:
            continue
        x = {"n_scored": len(rows), "n_errors": len(every) - len(rows), "n_cells": len(cells),
             "finished": (OUT / ARM / f"s{seed}" / "metrics.json").exists()}
        chem = [r for r in rows if r["group"] == "chem"]
        for t in TIERS:
            c = [r for r in chem if t in tier_of(r)]
            x[f"{t}_chem_median"], x[f"{t}_chem_n"] = med([r["window_mae_eV"] for r in c]), len(c)
            x[f"{t}_gap_median"], x[f"{t}_gap_n"] = med(gaps(c)), len(gaps(c))
            x[f"{t}_gap_le8_median"] = med(gaps([r for r in c if r["nsites"] <= 8]))
            x[f"{t}_gap_gt8_median"] = med(gaps([r for r in c if r["nsites"] > 8]))
            stable = [r for r in c if ehull.get(r["id"]) is not None and ehull[r["id"]] <= 0.1]
            x[f"{t}_gap_stable_median"], x[f"{t}_gap_stable_n"] = med(gaps(stable)), len(gaps(stable))
            x[f"{t}_train_median"] = med([r["window_mae_eV"] for r in rows if r["group"] == f"train:{t}"])
            x[f"{t}_s4_median"] = med([r["window_mae_eV"] for r in rows if r["group"].startswith(f"s4:{t}:")])
            c5 = [r for r in rows if r["group"] in ("chem", "chem5a") and r["id"] in ids5a and t in tier_of(r)]
            x[f"{t}_chem5a_median"], x[f"{t}_chem5a_n"] = med([r["window_mae_eV"] for r in c5]), len(c5)
            for g in ("proto", "val", "inner"):
                x[f"{t}_{g}_median"] = med([r["window_mae_eV"] for r in rows if r["group"] == g and t in tier_of(r)])
            x[f"{t}_per_element"] = {el: [med([r["window_mae_eV"] for r in c if el in r["elements"]]),
                                          med(gaps([r for r in c if el in r["elements"]])),
                                          sum(el in r["elements"] for r in c)] for el in TIERS[t]}
        t2g = [r for r in rows if r["group"] == "t2gap"]
        x["t2_train_gap_median"], x["t2_train_gap_n"] = med(gaps(t2g)), len(gaps(t2g))
        x["t2_train_gap_le8_median"] = med(gaps([r for r in t2g if r["nsites"] <= 8]))
        dchem = [r for r in chem if set(r["elements"]) & D]
        withc = [r for r in dchem if "d_character" in r]
        x["expelled_frac"] = sum(r["d_expelled"] for r in withc) / max(len(withc), 1)
        x["char_coverage"] = len(withc) / max(len(dchem), 1)
        sp = [r for r in rows if r["group"] in ("chem", "proto") and tier_of(r) == ["sp"]]
        paired = [r for r in sp if r["id"] in g4]
        x["sp_n"], x["sp_missing_g4"] = len(paired), len(sp) - len(paired)
        x["sp_window_change"] = med([r["window_mae_eV"] - g4[r["id"]]["window_mae_eV"] for r in paired])
        x["sp_gap_change"] = med([abs(r["gap_error_eV"]) - abs(g4[r["id"]]["gap_error_eV"]) for r in paired
                                  if r["gapped"] and "gap_error_eV" in r and "gap_error_eV" in g4[r["id"]]])
        x["sp_train_median"] = med([r["window_mae_eV"] for r in rows if r["group"] == "train:sp"])
        x["watch"] = {w: [round(r["window_mae_eV"], 3) for r in rows if r["formula"] == w and r["group"] in ("chem", "proto", "val", "inner")]
                      for w in WATCH}
        per[seed] = x
    m = lambda k: float(np.mean([v[k] for v in per.values()])) if per else float("nan")   # noqa: E731
    two = len(per) == 2
    readings = {
        "R1": {t: m(f"{t}_chem_median") for t in TIERS} | {"supported": two and all(m(f"{t}_chem_median") < 0.10 for t in TIERS)},
        "R2": {t: m(f"{t}_gap_median") for t in TIERS} | {"gate": 0.15, "supported": two and all(m(f"{t}_gap_median") < 0.15 for t in ("d10", "t2", "t3")),
               "by_size_le8_gt8": {t: [m(f"{t}_gap_le8_median"), m(f"{t}_gap_gt8_median")] for t in TIERS},
               "note": "gate 0.15 eV (user, 2026-10-28); T4 reported, not gated"},
        "R2s": {t: m(f"{t}_gap_stable_median") for t in TIERS} | {"n": {t: m(f"{t}_gap_stable_n") for t in TIERS},
                "note": "R2 on chemsys cells with E_hull <= 0.1 eV/atom; reported, not gated"},
        "R3": {"window": m("sp_window_change"), "gap": m("sp_gap_change"), "n": m("sp_n"), "missing_g4": m("sp_missing_g4"),
               "supported": two and m("sp_window_change") <= 0.005 and m("sp_gap_change") <= 0.005},
        "R4": {"heldout_over_train": {t: m(f"{t}_chem_median") / m(f"{t}_train_median") for t in TIERS},
               "supported": two and all(m(f"{t}_chem_median") <= 2 * m(f"{t}_train_median") for t in TIERS)},
        "R5": {"stage4": {t: m(f"{t}_s4_median") for t in TIERS}, "stage4_reference_4cS": s4_ref,
               "chem_5a_cells": {t: m(f"{t}_chem5a_median") for t in TIERS}, "chem_5a_reference_5aL": {t: ref5a[t] for t in TIERS},
               "supported": two and all(m(f"{t}_s4_median") <= s4_ref[t] + 0.01 for t in TIERS)
               and all(m(f"{t}_chem5a_median") <= ref5a[t] + 0.01 for t in TIERS)},
        "R6": {"expelled_frac": [v["expelled_frac"] for v in per.values()], "char_coverage": [v["char_coverage"] for v in per.values()],
               "runs_finished": sum(v["finished"] for v in per.values()),
               "supported": two and all(v["expelled_frac"] <= 0.10 for v in per.values()) and all(v["finished"] for v in per.values())},
        "R7": {"t2_train_gap_median": m("t2_train_gap_median"), "n": m("t2_train_gap_n"), "le8_sites": m("t2_train_gap_le8_median"),
               "supported": two and m("t2_train_gap_median") <= 0.06},
    }
    lines = [f"# Stage 5c-L ({ARM}): cells <= 24 sites, all tiers + s/p", "",
             "Scored on band paths thinned to about 64 k (vertices, VBM, CBM kept), G4's s/p baseline too (FFT path from 3,000 "
             "plane waves for both). d character on cells below 3,000 plane waves, the dense path (coverage in R6).", "",
             "| seed | tier | chemsys median (n) | |gap| median (n) | |gap| <= 8 / > 8 sites | train | Stage 4 cells (ref) | 5a cells (5a-L) | proto / val / inner |",
             "|---|---|---|---|---|---|---|---|---|"]
    for seed, x in per.items():
        for t in TIERS:
            lines.append(f"| {seed} | {t} | {x[f'{t}_chem_median']:.3f} ({x[f'{t}_chem_n']}) | {x[f'{t}_gap_median']:.3f} ({x[f'{t}_gap_n']}) | "
                         f"{x[f'{t}_gap_le8_median']:.3f} / {x[f'{t}_gap_gt8_median']:.3f} | {x[f'{t}_train_median']:.3f} | "
                         f"{x[f'{t}_s4_median']:.3f} ({s4_ref[t]:.3f}) | {x[f'{t}_chem5a_median']:.3f} ({ref5a[t]:.3f}) | "
                         f"{x[f'{t}_proto_median']:.3f} / {x[f'{t}_val_median']:.3f} / {x[f'{t}_inner_median']:.3f} |")
        lines.append(f"| {seed} | s/p | window vs G4 {x['sp_window_change']:+.4f}, |gap| vs G4 {x['sp_gap_change']:+.4f} ({x['sp_n']} cells) | | | "
                     f"{x['sp_train_median']:.3f} | | | expelled {x['expelled_frac']:.3f} (coverage {x['char_coverage']:.2f}); "
                     f"T2 train |gap| {x['t2_train_gap_median']:.3f}; errors {x['n_errors']} |")
    lines += ["", "Readings:", *[f"- {k}: {json.dumps(v)}" for k, v in readings.items()]]
    (OUT / f"report_{ARM}.md").write_text("\n".join(lines) + "\n")
    (OUT / f"summary_{ARM}.json").write_text(json.dumps({"readings": readings, "runs": per}, indent=1) + "\n")
    print("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells", action="store_true")
    p.add_argument("--run", type=int, default=None)
    p.add_argument("--g4", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--chunk", type=int, default=0)
    p.add_argument("--nchunks", type=int, default=1)
    p.add_argument("--workers", type=int, default=32, help="CPU scoring processes, or row-prep processes with --device cuda")
    p.add_argument("--device", default="cpu", help="cpu (process pool) or cuda (one model on the GPU, rows prepared by CPU workers)")
    p.add_argument("--limit", type=int, default=None, help="score only the first N cells of the chunk (checks)")
    p.add_argument("--arm", default="5a", help="run directory under results/stage5 (5a, 5ag)")
    p.add_argument("--max-k", type=int, default=64, help="score on the path thinned to about this many k (0: the full path)")
    p.add_argument("--pass", dest="pass_", default=None, help="5cL: score only the cells of pass A (gated) or B (reported)")
    p.add_argument("--seed-cells", type=int, default=0, help="5cL --cells: the seed whose metrics.json gives the ids")
    p.add_argument("--closeout-cells", action="store_true", help="write results/stage5/closeout/cells.json (sealed, big_sp, big_d)")
    p.add_argument("--cells-file", default=None, help="score this cell list instead of the arm's")
    p.add_argument("--out-prefix", default="eval_chunk_", help="output file prefix in the run directory")
    p.add_argument("--chain", action="store_true", help="warm-chain solver (big cells; validated in results/alexandria/fast_eval)")
    a = p.parse_args()
    global MAX_K, ARM, CHAR_MAX_NG, FFT_NG, CHAIN
    MAX_K, ARM, CHAIN = a.max_k or None, a.arm, a.chain
    five_c = ARM in ("5cL", "5cN")   # 5c-N (2026-10-30): the same cells, scoring and readings as 5c-L
    if five_c:
        # d character only on the dense path: the 5c-L model solves cells from 3,000 plane waves on the FFT path, which
        # cannot return the Hamiltonians (6,000 made every larger d cell fail into the CPU fallback and drop out of the
        # readings with an error; 2026-10-30). Coverage is reported (R6).
        CHAR_MAX_NG = 3000
    if a.closeout_cells:
        from collections import Counter
        spd = json.loads((ROOT / "data/manifests/general_spd_v1.json").read_text())["samples"]
        cells = [{"group": "sealed", "id": e["id"]} for e in spd if e["metadata"].get("split") == "sealed"]
        sp_ids = json.loads((ROOT.parent / "EpNet-sp/results/general_sp/g4/bigcell_heldout_ids.json").read_text())
        cells += [{"group": "big_sp", "id": i} for i in sp_ids]
        cells += [{"group": "big_d", "id": e["id"]} for e in json.loads((ROOT / "data/manifests/bigcell_spd_v1.json").read_text())["samples"]]
        missing = {c["id"] for c in cells} - entries_by_id({c["id"] for c in cells}).keys()
        assert not missing, f"cells without a manifest entry: {sorted(missing)[:5]}"
        (OUT / "closeout").mkdir(exist_ok=True)
        (OUT / "closeout/cells.json").write_text(json.dumps(cells) + "\n")
        print(len(cells), Counter(c["group"] for c in cells))
    elif a.cells:
        cells = build_cells_5c(a.seed_cells) if five_c else build_cells()
        cells_path().write_text(json.dumps(cells) + "\n")
        from collections import Counter
        print(len(cells), Counter(c["group"].split(":")[0] for c in cells), Counter(c.get("pass") for c in cells))
    elif a.run is not None:
        d = OUT / ARM / f"s{a.run}"
        ckpt = d / "best_smoothed.pt" if (d / "best_smoothed.pt").exists() else d / "best.pt"
        cells = json.loads((Path(a.cells_file) if a.cells_file else cells_path()).read_text())
        if a.pass_:
            cells = [c for c in cells if c["pass"] == a.pass_]
        rows = score_chunk(str(ckpt), cells, a.chunk, a.nchunks, a.workers, a.device)
        tag = f"{a.pass_ or ''}{a.chunk:02d}"
        (d / f"{a.out_prefix}{tag}.json").write_text(json.dumps(rows) + "\n")
        print(f"{ARM} s{a.run} chunk {tag}: {len(rows)} cells, {sum('error' in r for r in rows)} errors ({ckpt.name})")
    elif a.g4:
        # s/p cells of the chemsys and prototype holdouts: independent of the 5a / 5c-L runs, so G4 can go first
        from epnet.data.general_manifest import load_general_manifest
        from train_general_sp import pick
        raw = load_general_manifest(ROOT / "data/manifests/general_spd_v1.json")
        if five_c:   # <= 24 sites, 5c-L's plane-wave limit and FFT threshold
            FFT_NG = 3000
            sel = SimpleNamespace(elements=G4_SP, require_any=None, max_sites=24, max_ng8=100000, seed=0)
        else:
            sel = SimpleNamespace(elements=G4_SP, require_any=None, max_sites=8, max_ng8=1500, seed=0)
        cells = [{"group": g, "id": e["id"]} for g, split in (("chem", "holdout_chemsys"), ("proto", "holdout_prototype"))
                 for e in pick(raw, split, sel, None)]
        rows = score_chunk(str(G4), cells, a.chunk, a.nchunks, a.workers, a.device)
        gdir = OUT / ("g4_24" if five_c else "g4")
        gdir.mkdir(parents=True, exist_ok=True)
        (gdir / f"chunk_{a.chunk:02d}.json").write_text(json.dumps(rows) + "\n")
        print(f"G4 chunk {a.chunk}: {len(rows)} cells, {sum('error' in r for r in rows)} errors")
    elif a.report:
        report_5c() if five_c else report()


if __name__ == "__main__":
    main()
