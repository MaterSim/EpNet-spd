"""Shared training of one ``GeneralEPM`` over many cells with the G1-sp recipe.

The per-cell loss is the one that passed the G1-sp gate (28/28 cells at 0.1 eV):
window MAE (VBM-aligned, or E_F-aligned for metals), a signed gap term that is off
for the first ``gap_off_fraction`` of training, and a plain deep-band term over all
occupied bands (the masked variant was tested and rejected). Per-cell plane-wave
caches do not depend on the parameters, so they are built once, without the projector
tensors (about 97% of their size), which are rebuilt on the training device each step.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np
import torch
from torch import Tensor

from epnet.data.general_manifest import load_general_sample
from epnet.data.kpath import capped_path_indices, strided_path_indices
from epnet.eval.metrics import band_gap, fermi_level_window_mae, fermi_window_mae, full_band_mae, vbm_window_mae
from epnet.hamiltonian.torch_pw import (fermi_aligned_window_loss, fermi_level_window_loss, unique_g_transfers_encoded,
                                        vbm_aligned_window_loss)
from epnet.models.general_epm import GeneralEPM

PASS_MAE = 0.1
# Metal window: "proxy" centres on the mid-point of max band n_occ-1 and min band n_occ (G1-G3 v2); "ef" on the
# stored true E_F with the best rigid shift (adopted 2026-10-12 for the next version; the proxy lies a median 1.3 eV
# from E_F for MP metals, outside the +-2 eV window for 31% of them).
METAL_WINDOWS = ("proxy", "ef")


def n_predicted_bands(target: np.ndarray, n_occ: int, gapped: bool, margin: float = 2.5) -> int:
    """Enough bands to cover the scored window above the gap (or E_F); as in G1."""
    vbm = target[:, n_occ - 1].max()
    edge = 0.5 * (vbm + target[:, n_occ].min()) if gapped else vbm
    top = np.flatnonzero(target.min(axis=0) <= edge + margin)
    return int(min(target.shape[1], max(n_occ + 1, (top.max() + 2) if top.size else n_occ + 1)))


def n_eval_bands(target: np.ndarray, n_occ: int, gapped: bool, n_bands: int, above: float = 3.0) -> int:
    """Bands needed to cover CBM (or the path E_F proxy) + ``above`` eV, never fewer than the training bands."""
    ref = target[:, n_occ].min() if gapped else 0.5 * (target[:, n_occ - 1].max() + target[:, n_occ].min())
    top = np.flatnonzero(target.min(axis=0) <= ref + above)
    return int(min(target.shape[1], max(n_bands, (top.max() + 1) if top.size else n_bands)))


def ordered_subset(entries: list[dict], n: int | None, seed: int) -> list[dict]:
    """Deterministic, seed-dependent subset (hash order), so a smoke set is reproducible."""
    key = lambda e: hashlib.sha256(f"{seed}:{e['id']}".encode()).hexdigest()  # noqa: E731
    ranked = sorted(entries, key=key)
    return ranked if not n else ranked[:n]


def _to(ns: SimpleNamespace, device: torch.device) -> SimpleNamespace:
    return SimpleNamespace(**{k: (v.to(device) if torch.is_tensor(v) else v) for k, v in vars(ns).items()})


def row_nbytes(row: dict) -> int:
    return sum(v.numel() * v.element_size() for ns in (row["packed"], row["cache"]) for v in vars(ns).values() if torch.is_tensor(v))


def move_row(row: dict, device: torch.device) -> dict:
    """Keep a prepared cell resident on ``device`` (avoids a host-to-device copy of its cache every step)."""
    return {**row, "packed": _to(row["packed"], device), "cache": _to(row["cache"], device), "target": row["target"].to(device),
            "k": row["k"].to(device)}


def _fermi(meta: dict) -> float | None:
    return float(meta["fermi_energy"]) if meta.get("fermi_energy") is not None else None


D10_ELEMENTS = frozenset({"Zn", "Cd", "Cu", "Ag", "Au", "Hg"})


def prepare_cell(model: GeneralEPM, entry: dict, root: Path, *, k_stride: int, full_path: bool, k_select: str = "stride",
                 max_k: int | None = None, cb_manifold: bool = False) -> dict[str, Any]:
    """Pack one cell and build its PW cache on the host (CPU). ``max_k`` raises the stride of long
    paths so about ``max_k`` points remain (plus vertices, VBM, CBM); the eigensolve is ~90% of a step.
    ``cb_manifold`` (Stage 5a-w): a gapped cell with N atoms of non-d10 d elements keeps the lowest 5 N conduction
    bands (the empty d manifold) and records ``window_top_band`` for the loss window."""
    sample = load_general_sample(entry, root)
    meta = entry["metadata"]
    n_occ, gapped = int(meta["n_occupied_model"]), bool(meta["gapped"])
    target_full = np.asarray(sample.bands.energies, dtype=float)
    n_bands = n_predicted_bands(target_full, n_occ, gapped)
    window_top = None
    if cb_manifold and gapped:
        from epnet.data.valence import D_ELEMENTS
        n_open = sum(s in D_ELEMENTS and s not in D10_ELEMENTS for s in sample.structure.species)
        if n_open:
            window_top = int(min(target_full.shape[1], n_occ + 5 * n_open))
            n_bands = max(n_bands, window_top)
    target_np = target_full[:, :n_bands]
    k_all = torch.as_tensor(np.asarray(sample.bands.kpoints), dtype=torch.float32)
    tgt_all = torch.as_tensor(target_np, dtype=torch.float32)
    if full_path:
        idx = torch.arange(len(k_all))
    elif meta.get("k_kind") == "mesh" and max_k and len(k_all) > max_k:
        # uniform-mesh labels (e.g. projects/epc_mgb2): path vertex logic would keep most points, so take a fixed,
        # id-seeded random subset instead
        seed = int(hashlib.sha256(f"mesh:{entry['id']}".encode()).hexdigest()[:8], 16)
        idx = torch.as_tensor(np.sort(np.random.default_rng(seed).choice(len(k_all), max_k, replace=False)))
    else:
        if max_k and k_select == "capped":   # conservative hard-ish cap: protected points kept, stride points thinned
            idx = capped_path_indices(k_all, max_k, energies=tgt_all if gapped else None, n_occupied=n_occ if gapped else None)
        else:
            if max_k:
                k_stride = max(k_stride, math.ceil(len(k_all) / max_k))
            idx = strided_path_indices(k_all, k_stride, energies=tgt_all if gapped else None, n_occupied=n_occ if gapped else None)
    packed = model.pack(sample.structure)
    with torch.no_grad():
        cache = model.make_pw_cache(packed, k_all[idx])
    # projectors are ~97% of the cache and depend only on fixed widths: rebuild them per step (see _cache_on)
    cache.projectors = None
    return {
        "id": entry["id"], "formula": meta["formula"], "chemsys": meta["chemsys"], "elements": meta["elements"],
        "packed": packed, "cache": cache, "target": tgt_all[idx], "n_occ": n_occ, "gapped": gapped,
        "n_bands": n_bands, "n_g": cache.n_g, "n_k": int(len(idx)), "k": k_all[idx], "ecut": model.cell_ecut(packed),
        "window_top_band": window_top,
        "fermi": _fermi(meta),  # same absolute scale as the target bands
        # scoring rows keep the bands up to CBM (E_F) + 3 eV for the wide-window metric (full_band_mae)
        "target_eval": torch.as_tensor(target_full[:, :n_eval_bands(target_full, n_occ, gapped, n_bands)], dtype=torch.float32)[idx]
        if full_path else None,
    }


# model-config keys that change a prepared cell (basis, graph, router inputs, QEq matrix); the rest
# (channels, readouts, encoder depth, projector radii, ...) do not, so ablations share one disk cache
CACHE_KEYS = ("ecut_ry", "ecut_by_element", "k_centred_basis", "cutoff", "router_radii", "contrast_radius", "charge_model",
              "charge_coupling")
CACHE_VERSION = "cells-v2"  # v2: transfer map not stored


def cache_key(model: GeneralEPM, entry: dict, *, k_stride: int, full_path: bool, max_k: int | None, k_select: str = "stride",
              cb_manifold: bool = False, legacy_k_rows: bool = False) -> str:
    spec = {"v": CACHE_VERSION, "id": entry["id"], "species": list(model.species_symbols),
            "cfg": {k: model.config.get(k) for k in CACHE_KEYS}, "k_stride": int(k_stride), "full": bool(full_path),
            "max_k": None if full_path else max_k}
    if k_select != "stride" and not full_path:   # older keys stay valid for the default selection
        spec["k_select"] = k_select
        if not legacy_k_rows:   # 2026-10-29: the selection is now applied; older keys hold rows built with "stride"
            spec["k_select_built"] = True
    if cb_manifold:                               # older keys stay valid without it
        spec["cb_manifold"] = True
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:20]


def prepare_cell_cached(model: GeneralEPM, entry: dict, root: Path, cache_dir: Path | None, *, k_stride: int, full_path: bool,
                        max_k: int | None = None, k_select: str = "stride", cb_manifold: bool = False,
                        legacy_k_rows: bool = False) -> dict[str, Any]:
    """``prepare_cell`` through a persistent disk cache (one file per cell and settings; atomic write).
    ``legacy_k_rows``: the rows of Stages 3-5c-L, built with the "stride" selection whatever ``k_select`` (the k-cap bug,
    fixed 2026-10-29; docs/d_channel_plan.md, "Early checks for the post-5c-L fixes"), under their old keys."""
    if cache_dir is None:
        return prepare_cell(model, entry, root, k_stride=k_stride, full_path=full_path, max_k=max_k, k_select=k_select,
                            cb_manifold=cb_manifold)
    key = cache_key(model, entry, k_stride=k_stride, full_path=full_path, max_k=max_k, k_select=k_select, cb_manifold=cb_manifold,
                    legacy_k_rows=legacy_k_rows)
    path = Path(cache_dir) / f"{entry['id']}_{key}.pt"
    if path.exists():
        try:
            row = _widen(torch.load(path, map_location="cpu", weights_only=False))
            row.setdefault("fermi", _fermi(entry["metadata"]))  # rows cached before the key existed
            return row
        except Exception:  # a truncated file from a killed writer: rebuild it
            pass
    row = prepare_cell(model, entry, root, k_stride=k_stride, full_path=full_path, max_k=max_k,
                       k_select="stride" if legacy_k_rows else k_select, cb_manifold=cb_manifold)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}")
    torch.save(_narrow(row), tmp)
    os.replace(tmp, path)
    return row


# integer index tensors stored narrow on disk (G-vector components fit int16, indices int32), widened on load
_NARROW = {"g_idx": torch.int16, "g_idx_k": torch.int16, "k_index": torch.int32, "inverse": torch.int32}


def _narrow(row: dict) -> dict:
    cache = SimpleNamespace(**vars(row["cache"]))
    # the (n_g, n_g) transfer map and the unique transfers follow from g_idx and the lattice: rebuilt on the
    # device when used (_cache_on), never stored (they are ~90% of a large cell's cache)
    cache.inverse = cache.unique_cart = None
    for name, dtype in _NARROW.items():
        v = getattr(cache, name, None)
        if torch.is_tensor(v) and v.dtype == torch.int64:
            info = torch.iinfo(dtype)
            if v.numel() and (int(v.min()) < info.min or int(v.max()) > info.max):
                continue  # out of range: keep int64
            setattr(cache, name, v.to(dtype))
    return {**row, "cache": cache}


def _widen(row: dict) -> dict:
    for name in _NARROW:
        v = getattr(row["cache"], name, None)
        if torch.is_tensor(v) and v.dtype in (torch.int16, torch.int32):
            setattr(row["cache"], name, v.long())
    return row


PER_K_CACHE = ("kinetic", "k_index", "k_valid", "g_idx_k", "k_frac")


def thin_row(row: dict, max_k: int | None) -> dict:
    """A view of a full-path row on about ``max_k`` k-points (path vertices, VBM and CBM kept), for cheap
    intermediate evaluations; the final scoring uses the full row."""
    n_k = int(row["k"].shape[0])
    if not max_k or n_k <= max_k:
        return row
    tgt = row["target"]
    idx = strided_path_indices(row["k"].cpu(), math.ceil(n_k / max_k), energies=tgt.cpu() if row["gapped"] else None,
                               n_occupied=row["n_occ"] if row["gapped"] else None).to(row["k"].device)
    cache = SimpleNamespace(**vars(row["cache"]))
    for name in PER_K_CACHE:
        value = getattr(cache, name, None)
        if torch.is_tensor(value):
            setattr(cache, name, value.index_select(0, idx.to(value.device)))
    cache.projectors = None
    out = {**row, "cache": cache, "target": tgt.index_select(0, idx.to(tgt.device)), "k": row["k"].index_select(0, idx),
           "n_k": int(idx.numel())}
    if row.get("target_eval") is not None:
        out["target_eval"] = row["target_eval"].index_select(0, idx.to(row["target_eval"].device))
    return out


def _cache_on(model: GeneralEPM, row: dict, device: torch.device) -> tuple[SimpleNamespace, SimpleNamespace]:
    """Packed structure and PW cache on ``device``, with the projectors rebuilt there."""
    packed, cache = _to(row["packed"], device), _to(row["cache"], device)
    if getattr(cache, "inverse", None) is None:  # disk-cached row: rebuild the transfer map here
        recip = 2.0 * torch.pi * torch.linalg.inv(packed.lattice.detach()).transpose(0, 1)
        cache.unique_cart, cache.inverse = unique_g_transfers_encoded(cache.g_idx, recip)
    if cache.projectors is None:
        with torch.no_grad():
            g = cache.g_idx if getattr(cache, "g_idx_k", None) is None else cache.g_idx_k
            cache.projectors = model.projectors(packed, row["k"].to(device), g).detach()
    return packed, cache


def router_normalization(rows: Iterable[dict]) -> tuple[Tensor, Tensor]:
    x = torch.cat([r["cache"].router_inputs for r in rows], dim=0)
    return x.mean(dim=0), x.std(dim=0)


def cell_loss(model: GeneralEPM, row: dict, device: torch.device, *, gap_on: bool, gap_weight: float,
              deep_weight: float, window_anchor: str = "mid", metal_window: str = "proxy") -> tuple[Tensor, Tensor]:
    """G1-sp recipe loss for one cell; returns (loss, prediction). ``metal_window`` (METAL_WINDOWS): where metals are
    windowed and aligned; with "ef" the deep-band term uses the window's rigid shift."""
    packed, cache = _cache_on(model, row, device)
    if model.config.get("solver") in ("chefsi", "chefsi_fft"):  # warm-start store for the lowest-band solver, kept with the cell
        cache.warm = row.setdefault("warm", {})
    tgt, n_occ = row["target"].to(device), row["n_occ"]
    pred = model.band_path(packed, None, n_bands=row["n_bands"], cache=cache)
    if row["gapped"] and row.get("window_top_band"):   # Stage 5a-w: the window reaches the top of the empty d manifold
        top = int(row["window_top_band"])
        h = max(2.0, float(tgt[:, top - 1].amax() - tgt[:, n_occ].amin()) + 0.01)
        loss = vbm_aligned_window_loss(pred, tgt, n_occ, anchor="edges", half_width=h)
        if gap_on and gap_weight > 0:
            pgap = pred[:, n_occ].amin() - pred[:, n_occ - 1].amax()
            tgap = tgt[:, n_occ].amin() - tgt[:, n_occ - 1].amax()
            loss = loss + gap_weight * (pgap - tgap).abs()
    elif row["gapped"]:
        loss = vbm_aligned_window_loss(pred, tgt, n_occ, anchor=window_anchor)
        if gap_on and gap_weight > 0:
            pgap = pred[:, n_occ].amin() - pred[:, n_occ - 1].amax()  # signed: a closed gap still has a gradient
            tgap = tgt[:, n_occ].amin() - tgt[:, n_occ - 1].amax()
            loss = loss + gap_weight * (pgap - tgap).abs()
    elif metal_window == "ef":
        loss, shift = fermi_level_window_loss(pred, tgt, row["fermi"])
    else:
        loss = fermi_aligned_window_loss(pred, tgt, n_occ)
    if deep_weight > 0:
        if not row["gapped"] and metal_window == "ef":
            deep = (pred[:, :n_occ] + shift - tgt[:, :n_occ]).abs().mean()
        else:
            ref = (lambda x: x[:, n_occ - 1].amax()) if row["gapped"] else (lambda x: 0.5 * (x[:, n_occ - 1].amax() + x[:, n_occ].amin()))
            deep = ((pred[:, :n_occ] - ref(pred)) - (tgt[:, :n_occ] - ref(tgt))).abs().mean()
        loss = loss + deep_weight * deep
    return loss, pred


def convergence_loss(model: GeneralEPM, row: dict, device: torch.device, *, ecut_hi: float, n_k: int = 4,
                     all_bands: bool = False) -> Tensor:
    """Mean |E_occ(ecut_hi) - E_occ(model cutoff)| over a few k (rigid shift removed): penalizes potentials
    whose bands depend on the plane-wave truncation (first-row potentials learned at 12 Ry drift ~1 eV by 16 Ry)."""
    packed = _to(row["packed"], device)
    k = row["k"].to(device)
    k = k[torch.linspace(0, len(k) - 1, min(n_k, len(k)), device=device).round().long()]
    with torch.no_grad():
        lo, hi = model.make_pw_cache(packed, k), model.make_pw_cache(packed, k, ecut_ry=ecut_hi)
    n = int(row["n_bands"]) if all_bands else int(row["n_occ"])   # all_bands: the loss-window conduction bands too
    diff = model.band_path(packed, None, n_bands=n, cache=hi) - model.band_path(packed, None, n_bands=n, cache=lo)
    return (diff - diff.median().detach()).abs().mean()


@torch.no_grad()
def score_cell(model: GeneralEPM, row: dict, device: torch.device, metal_window: str = "proxy",
               warm_store: dict | None = None, chain: dict | None = None) -> dict[str, Any]:
    """Full-path window MAE and gap for one prepared (full_path=True) cell. ``window_mae_eV`` (and the pass flag)
    follow ``metal_window``; ``ef_window_mae_eV`` (E_F-centred for metals, the window MAE for gapped cells) is always
    reported, and metals also report ``proxy_window_mae_eV``."""
    if metal_window not in METAL_WINDOWS:
        raise ValueError(f"unknown metal window {metal_window!r}")
    packed, cache = _cache_on(model, row, device)
    if warm_store is not None:   # FFT-path cells start from the store's eigenvectors (--eval-warm)
        cache.warm = warm_store
    if chain is not None:        # evaluation along the path with warm-started CheFSI (epnet.screen; opt-in, 2026-11-02)
        cache.chain = chain
    full_tgt = row.get("target_eval")
    n_pred = row["n_bands"] if full_tgt is None else max(row["n_bands"], int(full_tgt.shape[1]))
    pred_all = model.band_path(packed, None, n_bands=n_pred, cache=cache).cpu().numpy().astype(float)
    pred = pred_all[:, : row["n_bands"]]
    tgt, n_occ = row["target"].cpu().numpy().astype(float), row["n_occ"]
    out = {"id": row["id"], "formula": row["formula"], "gapped": row["gapped"]}
    fermi = row.get("fermi")
    if full_tgt is not None:
        out["full_mae_eV"] = float(full_band_mae(pred_all, full_tgt.cpu().numpy().astype(float), n_occ, row["gapped"],
                                                 fermi_energy=fermi if metal_window == "ef" else None))
    if row["gapped"]:
        out["window_mae_eV"] = float(vbm_window_mae(pred, tgt, n_occ))
        out["pred_gap_eV"] = float(pred[:, n_occ].min() - pred[:, n_occ - 1].max())
        out["target_gap_eV"] = float(band_gap(tgt, n_occ))
        out["gap_error_eV"] = out["pred_gap_eV"] - out["target_gap_eV"]
    else:
        out["proxy_window_mae_eV"] = float(fermi_window_mae(pred, tgt, n_occ))
        out["ef_window_mae_eV"] = float(fermi_level_window_mae(pred, tgt, fermi)) if fermi is not None else float("nan")
        out["window_mae_eV"] = out["ef_window_mae_eV"] if metal_window == "ef" else out["proxy_window_mae_eV"]
    out.setdefault("ef_window_mae_eV", out["window_mae_eV"])
    out["pass"] = bool(out["window_mae_eV"] <= PASS_MAE and (not row["gapped"] or out["pred_gap_eV"] > 0))
    return out


def summarize(scores: list[dict]) -> dict[str, Any]:
    """Aggregate scores; cells with an empty window (NaN, e.g. flat molecular O2 classed metallic) are counted apart."""
    if not scores:
        return {"n": 0}
    finite = [s for s in scores if np.isfinite(s["window_mae_eV"])]
    mae = [s["window_mae_eV"] for s in finite]
    gaps = [abs(s["gap_error_eV"]) for s in finite if s["gapped"]]
    if not mae:
        return {"n": len(scores), "n_undefined": len(scores)}
    full = [s["full_mae_eV"] for s in finite if np.isfinite(s.get("full_mae_eV", np.nan))]
    out = {
        "n": len(finite),
        "n_undefined": len(scores) - len(finite),
        "median_window_mae_eV": float(np.median(mae)),
        "mean_window_mae_eV": float(np.mean(mae)),
        "p90_window_mae_eV": float(np.percentile(mae, 90)),
        "pass_0.1": int(sum(m <= 0.1 for m in mae)),
        "pass_0.05": int(sum(m <= 0.05 for m in mae)),
        "frac_pass_0.1": float(np.mean([m <= 0.1 for m in mae])),
        "median_abs_gap_error_eV": float(np.median(gaps)) if gaps else None,
        "mean_abs_gap_error_eV": float(np.mean(gaps)) if gaps else None,
        "p90_abs_gap_error_eV": float(np.percentile(gaps, 90)) if gaps else None,
    }
    metals = [s for s in finite if not s["gapped"]]
    if metals:  # metal windows side by side: the configured one, the E_F-centred one, and the legacy proxy
        ef_m = [s.get("ef_window_mae_eV", np.nan) for s in metals]
        px_m = [s.get("proxy_window_mae_eV", np.nan) for s in metals]
        out.update({"n_metal": len(metals), "median_metal_window_mae_eV": float(np.median([s["window_mae_eV"] for s in metals])),
                    "median_metal_ef_window_mae_eV": float(np.nanmedian(ef_m)) if np.isfinite(ef_m).any() else None,
                    "median_metal_proxy_window_mae_eV": float(np.nanmedian(px_m)) if np.isfinite(px_m).any() else None})
    ef_all = [s.get("ef_window_mae_eV", np.nan) for s in finite]
    if np.isfinite(ef_all).any():
        out["median_ef_window_mae_eV"] = float(np.nanmedian(ef_all))
    if full:  # wide window (VBM - 6 to CBM + 3 eV): scores more than the two edge bands of wide-gap cells
        out.update({"median_full_mae_eV": float(np.median(full)), "mean_full_mae_eV": float(np.mean(full)),
                    "p90_full_mae_eV": float(np.percentile(full, 90))})
    return out


def sampling_weights(rows: list[dict]) -> np.ndarray:
    """Inverse-sqrt chemical-system frequency, so large chemsys families do not dominate."""
    counts = Counter(r["chemsys"] for r in rows)
    w = np.array([1.0 / math.sqrt(counts[r["chemsys"]]) for r in rows])
    return w / w.sum()


__all__ = ["METAL_WINDOWS", "PASS_MAE", "cell_loss", "move_row", "n_predicted_bands", "ordered_subset", "prepare_cell", "router_normalization", "row_nbytes",
           "sampling_weights", "score_cell", "summarize", "n_eval_bands",
           "prepare_cell_cached", "thin_row", "cache_key"]
