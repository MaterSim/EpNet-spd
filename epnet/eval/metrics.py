"""Numerically robust experiment metrics."""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np


def band_mae(prediction, target, *, weights=None, energy_window=None, reference=0.0) -> float:
    """Weighted band MAE, optionally restricted to a target-energy window."""
    pred, ref = np.asarray(prediction, float), np.asarray(target, float)
    if pred.shape != ref.shape or pred.ndim != 2:
        raise ValueError("bands must be matching (kpoint, band) arrays")
    mask = np.ones(ref.shape, dtype=bool)
    if energy_window is not None:
        low, high = energy_window
        mask = (ref - reference >= low) & (ref - reference <= high)
    if not mask.any():
        raise ValueError("energy window selects no bands")
    w = np.ones(pred.shape[0]) if weights is None else np.asarray(weights, float)
    if w.shape != (pred.shape[0],) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("weights must be nonnegative with positive sum")
    expanded = np.broadcast_to(w[:, None], pred.shape)
    return float(np.sum(np.abs(pred - ref)[mask] * expanded[mask]) / expanded[mask].sum())


def band_gap(energies, n_occupied: int) -> float:
    """Indirect fundamental gap, clipped at zero for band overlap."""
    values = np.asarray(energies, float)
    if values.ndim != 2 or not 0 < n_occupied < values.shape[1]:
        raise ValueError("invalid band array or occupied-band count")
    return float(max(0.0, values[:, n_occupied].min() - values[:, n_occupied - 1].max()))


def fermi_proxy(energies, n_occupied: int) -> float:
    """Mid-point of max occupied and min unoccupied eigenvalues (path E_F proxy)."""
    values = np.asarray(energies, float)
    return float(0.5 * (values[:, n_occupied - 1].max() + values[:, n_occupied].min()))


def vbm_window_mae(prediction, target, n_occupied: int, *, half_width: float = 2.0) -> float:
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    aligned = pred - pred[:, n_occupied - 1].max() + tgt[:, n_occupied - 1].max()
    mid = 0.5 * (tgt[:, n_occupied - 1].max() + tgt[:, n_occupied].min())
    window = np.abs(tgt - mid) <= half_width
    window[:, n_occupied - 1] = True
    window[:, n_occupied] = True
    return float(np.mean(np.abs(aligned - tgt)[window])) if window.any() else float("nan")


def full_band_mae(prediction, target, n_occupied: int, gapped: bool, *, below: float = 6.0, above: float = 3.0,
                  fermi_energy: float | None = None) -> float:
    """Wide-window band MAE: every band with target energy in [VBM - below, CBM + above] (gapped, VBM-aligned)
    or [E_F - below, E_F + above] (metals, path E_F proxy). The +-2 eV mid-gap window keeps only the two edge
    bands of a wide-gap cell; this one scores the upper valence and lower conduction manifolds too.
    ``fermi_energy`` (metals): centre on the true E_F and use the best rigid shift, as in ``fermi_level_window_mae``."""
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    if gapped:
        aligned = pred - pred[:, n_occupied - 1].max() + tgt[:, n_occupied - 1].max()
        low, high = tgt[:, n_occupied - 1].max() - below, tgt[:, n_occupied].min() + above
    elif fermi_energy is not None:
        window = (tgt >= fermi_energy - below) & (tgt <= fermi_energy + above)
        if not window.any():
            return float("nan")
        d = (tgt - pred)[window]
        return float(np.mean(np.abs(d - np.median(d))))
    else:
        aligned = pred - fermi_proxy(pred, n_occupied) + fermi_proxy(tgt, n_occupied)
        ef = fermi_proxy(tgt, n_occupied)
        low, high = ef - below, ef + above
    window = (tgt >= low) & (tgt <= high)
    return float(np.mean(np.abs(aligned - tgt)[window])) if window.any() else float("nan")


def fermi_level_window_mae(prediction, target, fermi_energy: float, *, half_width: float = 2.0) -> float:
    """Metal MAE over target states within +-half_width eV of the true E_F, after the best rigid shift (the median of
    target - prediction over those states; absolute EPM energies are arbitrary). ``fermi_window_mae`` centres on the
    proxy instead, which lies a median 1.3 eV from E_F for MP metals (MgB2: +5.2 eV), so it misses the Fermi surface."""
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    window = np.abs(tgt - float(fermi_energy)) <= half_width
    if not window.any():
        return float("nan")
    d = (tgt - pred)[window]
    return float(np.mean(np.abs(d - np.median(d))))


def fermi_window_mae(prediction, target, n_occupied: int, *, half_width: float = 2.0) -> float:
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    aligned = pred - fermi_proxy(pred, n_occupied) + fermi_proxy(tgt, n_occupied)
    window = np.abs(tgt - fermi_proxy(tgt, n_occupied)) <= half_width
    return float(np.mean(np.abs(aligned - tgt)[window])) if window.any() else float("nan")


def mixed_window_mae(prediction, target, n_occupied: int, gapped: bool, *, half_width: float = 2.0) -> float:
    """VBM window for gapped cells, Fermi window for metals."""
    if gapped:
        return vbm_window_mae(prediction, target, n_occupied, half_width=half_width)
    return fermi_window_mae(prediction, target, n_occupied, half_width=half_width)


def training_window_mae(
    prediction,
    target,
    n_occupied: int,
    gapped: bool,
    *,
    half_width: float = 2.0,
    edge_weight: float = 5.0,
) -> float:
    """NumPy equivalent of the aligned window term used during training."""
    pred, tgt = np.asarray(prediction, float), np.asarray(target, float)
    if pred.shape != tgt.shape or pred.ndim != 2 or not 0 < n_occupied < pred.shape[1]:
        raise ValueError("invalid band arrays or occupied-band count")
    pred_vbm = pred[:, n_occupied - 1].max()
    tgt_vbm = tgt[:, n_occupied - 1].max()
    tgt_cbm = tgt[:, n_occupied].min()
    if gapped:
        aligned = pred - pred_vbm + tgt_vbm
        reference = 0.5 * (tgt_vbm + tgt_cbm)
    else:
        pred_ef = 0.5 * (pred_vbm + pred[:, n_occupied].min())
        tgt_ef = 0.5 * (tgt_vbm + tgt_cbm)
        aligned = pred - pred_ef + tgt_ef
        reference = tgt_ef
    mask = np.abs(tgt - reference) <= half_width
    if gapped:
        mask[:, n_occupied - 1] = True
        mask[:, n_occupied] = True
    weight = np.ones_like(tgt)
    if gapped:
        weight[:, n_occupied - 1] = edge_weight
        weight[:, n_occupied] = edge_weight
    weight *= mask
    denominator = max(float(weight.sum()), 1.0)
    return float((np.abs(aligned - tgt) * weight).sum() / denominator)


def gap_error(prediction, target, n_occupied: int, *, absolute: bool = True) -> float:
    error = band_gap(prediction, n_occupied) - band_gap(target, n_occupied)
    return abs(error) if absolute else error


def subspace_distance(predicted_vectors, target_vectors, *, normalize=True) -> float:
    """Gauge/degeneracy-invariant Frobenius distance between projectors."""
    u, v = np.asarray(predicted_vectors), np.asarray(target_vectors)
    if u.shape != v.shape or u.ndim < 2:
        raise ValueError("eigenvector arrays must have matching (...,basis,state) shapes")
    overlap = np.matmul(np.swapaxes(u.conj(), -1, -2), v)
    rank = u.shape[-1]
    distance2 = np.maximum(0.0, 2.0 * rank - 2.0 * np.sum(np.abs(overlap) ** 2, axis=(-2, -1)))
    distance = np.sqrt(distance2 / (2.0 * rank) if normalize else distance2)
    return float(np.mean(distance))


def equivariance_error(reference, transformed, *, scale_floor=1e-12) -> dict[str, float]:
    """Absolute and relative max residual after applying the expected transform."""
    ref, value = np.asarray(reference), np.asarray(transformed)
    if ref.shape != value.shape:
        raise ValueError("arrays must have matching shapes")
    absolute = float(np.max(np.abs(value - ref), initial=0.0))
    scale = max(float(np.max(np.abs(ref), initial=0.0)), float(scale_floor))
    return {"absolute": absolute, "relative": absolute / scale}


def benchmark(function: Callable[[], object], *, warmup=1, repeats=10) -> dict[str, float]:
    """Measure wall time with warmup and report robust summary statistics."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    for _ in range(warmup):
        function()
    elapsed = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        elapsed.append(time.perf_counter() - start)
    values = np.asarray(elapsed)
    return {
        "median_seconds": float(np.median(values)),
        "mean_seconds": float(values.mean()),
        "min_seconds": float(values.min()),
        "repeats": repeats,
    }
