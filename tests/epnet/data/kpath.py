"""Correctness-preserving sampling helpers for line-mode band paths."""

from __future__ import annotations

import torch
from torch import Tensor


def strided_path_indices(
    kpoints: Tensor,
    stride: int,
    *,
    energies: Tensor | None = None,
    n_occupied: int | None = None,
    direction_tolerance: float = 1e-6,
) -> Tensor:
    """Return strided indices plus endpoints, geometric vertices, VBM, and CBM.

    Line-mode paths often contain segment junctions that do not land on a
    global ``range(0, n_k, stride)`` grid.  Dropping those points can change
    both the plotted path and the indirect gap used by the loss.
    """
    if kpoints.ndim != 2 or kpoints.shape[1] != 3 or kpoints.shape[0] < 1:
        raise ValueError("kpoints must have shape (n_k, 3) with n_k >= 1")
    step = max(int(stride), 1)
    n_k = int(kpoints.shape[0])
    keep = torch.zeros(n_k, dtype=torch.bool, device=kpoints.device)
    keep[::step] = True
    keep[0] = True
    keep[-1] = True

    if n_k > 2:
        delta = kpoints[1:] - kpoints[:-1]
        norm = torch.linalg.vector_norm(delta, dim=-1)
        prev, following = norm[:-1], norm[1:]
        denom = (prev * following).clamp_min(torch.finfo(kpoints.dtype).eps)
        cosine = (delta[:-1] * delta[1:]).sum(dim=-1) / denom
        vertex = (prev <= 1e-10) | (following <= 1e-10) | (cosine < 1.0 - direction_tolerance)
        keep[1:-1] |= vertex

    if energies is not None:
        if energies.ndim != 2 or energies.shape[0] != n_k:
            raise ValueError("energies must have shape (n_k, n_bands)")
        if n_occupied is None or not 0 < int(n_occupied) < energies.shape[1]:
            raise ValueError("n_occupied must identify valid VBM and CBM bands")
        n_occ = int(n_occupied)
        keep[int(torch.argmax(energies[:, n_occ - 1]))] = True
        keep[int(torch.argmin(energies[:, n_occ]))] = True

    return torch.nonzero(keep, as_tuple=False).flatten()


def virtual_band_window_margin(
    energies: Tensor,
    n_occupied: int,
    n_bands: int,
    *,
    half_width: float = 2.0,
) -> float:
    """Return the energy margin above the requested target window.

    A negative value means the first omitted band enters the target window.
    ``inf`` means no labeled bands were omitted.
    """
    if energies.ndim != 2 or not 0 < int(n_occupied) < energies.shape[1]:
        raise ValueError("invalid band array or occupied-band count")
    if not int(n_occupied) < int(n_bands) <= energies.shape[1]:
        raise ValueError("n_bands must include VBM and CBM and not exceed labels")
    if int(n_bands) == energies.shape[1]:
        return float("inf")
    n_occ = int(n_occupied)
    vbm = energies[:, n_occ - 1].max()
    cbm = energies[:, n_occ].min()
    midgap = 0.5 * (vbm + cbm)
    first_omitted = energies[:, int(n_bands)].min()
    return float(first_omitted - (midgap + float(half_width)))


def capped_path_indices(
    kpoints: Tensor,
    max_k: int,
    *,
    energies: Tensor | None = None,
    n_occupied: int | None = None,
    direction_tolerance: float = 1e-6,
) -> Tensor:
    """Conservative k cap for training paths (2026-10-17). ``strided_path_indices`` adds endpoints, every vertex and the
    VBM/CBM on top of a ~max_k stride grid, so MP line paths (many junctions, each stored twice) reached 40-55 k at
    max_k = 32. Here the protected points are always kept: endpoints, direction-change vertices, VBM and CBM; of an
    exactly repeated k (segment junction stored twice: same k, same bands) only the first copy is kept; the remaining
    budget max_k - len(protected) is filled with evenly spaced interior points. If the protected set alone exceeds max_k
    it is kept whole (the cap is soft; nothing important is dropped)."""
    n = int(kpoints.shape[0])
    if n <= max_k:
        return torch.arange(n, device=kpoints.device)
    delta = kpoints[1:] - kpoints[:-1]
    norm = torch.linalg.vector_norm(delta, dim=-1)
    duplicate = torch.zeros(n, dtype=torch.bool, device=kpoints.device)
    duplicate[1:] = norm <= 1e-10                       # k_i equals k_{i-1}: drop the later copy
    protected = torch.zeros(n, dtype=torch.bool, device=kpoints.device)
    protected[0] = protected[-1] = True
    if n > 2:
        prev, following = norm[:-1], norm[1:]
        denom = (prev * following).clamp_min(torch.finfo(kpoints.dtype).eps)
        cosine = (delta[:-1] * delta[1:]).sum(dim=-1) / denom
        protected[1:-1] |= (prev <= 1e-10) | (following <= 1e-10) | (cosine < 1.0 - direction_tolerance)
    protected &= ~duplicate
    if energies is not None:
        if energies.ndim != 2 or energies.shape[0] != n or n_occupied is None or not 0 < int(n_occupied) < energies.shape[1]:
            raise ValueError("energies (n_k, n_bands) and a valid n_occupied are needed to protect the band edges")
        protected[int(torch.argmax(energies[:, int(n_occupied) - 1]))] = True
        protected[int(torch.argmin(energies[:, int(n_occupied)]))] = True
    keep = protected.clone()
    room = int(max_k) - int(protected.sum())
    free = torch.nonzero(~protected & ~duplicate, as_tuple=False).reshape(-1)
    if room > 0 and len(free):
        pick = torch.linspace(0, len(free) - 1, min(room, len(free)), device=kpoints.device).round().long()
        keep[free[pick]] = True
    return torch.nonzero(keep, as_tuple=False).reshape(-1)
