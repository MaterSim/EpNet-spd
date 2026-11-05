"""FFT-grid building blocks for plane-wave Hamiltonians (shared with EpNet-EPC common/epc, 2026-10-18).

Ported from EpNet-EPC ``GridStates`` (local="fft"), where it is validated against the dense path to float32 rounding
(1-5e-6 relative on CaC6 and Mg7AlB16 at Gamma and finite q).

Conventions (``torch.fft`` with norm="forward"):
  u(r)   = sum_G c(G) e^{i G.r}            = ifftn(scatter(c), norm="forward")      (no 1/N)
  V(r)   = sum_M V(M) e^{i M.r}             = ifftn(scatter(V), norm="forward")
  (V u)(G) = (1/N) sum_r V(r) u(r) e^{-i G.r} = fftn(V(r) u(r), norm="forward")[G] = sum_G' V(G - G') c(G')

Exactness: with c supported on |G_d| <= s_d and V on |M_d| <= h_d, the circular convolution on an N_d grid has no
aliasing when N_d > s_d + max(s_d, ...) ... in general N_d >= 2 s_d + h_d + 1 (output on the sphere) and
N_d >= 2 h_d + 1 (V itself fits). For the local potential of one k-sphere, h = 2 s, so N_d >= 4 s_d + 1.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def fft_len(n: int) -> int:
    """Smallest m >= n of the form 2^a 3^b 5^c (fast FFT sizes)."""
    m = max(int(n), 1)
    while True:
        r = m
        for f in (2, 3, 5):
            while r % f == 0:
                r //= f
        if r == 1:
            return m
        m += 1


def grid_shape(sphere_half: Sequence[int], potential_half: Sequence[int] | None = None,
               shift: Sequence[int] | None = None) -> tuple[int, int, int]:
    """Alias-free grid for products V * u with u on |G_d| <= s_d and V on |M_d| <= h_d (default h = 2 s), optionally
    with a constant integer shift of the output sphere (EPC k -> k + q pairs)."""
    s = [int(x) for x in sphere_half]
    h = [2 * x for x in s] if potential_half is None else [int(x) for x in potential_half]
    sh = [0, 0, 0] if shift is None else [abs(int(x)) for x in shift]
    return tuple(fft_len(max(2 * s[d] + h[d] + sh[d] + 1, 2 * h[d] + 1 + sh[d])) for d in range(3))


def flat_positions(g_int: Tensor, shape: Sequence[int]) -> Tensor:
    """Flattened grid index of integer reciprocal vectors (n, 3) on a grid of ``shape`` (periodic wrap)."""
    n = torch.as_tensor(tuple(shape), device=g_int.device)
    g = torch.remainder(g_int.to(torch.long), n)
    return (g[..., 0] * int(shape[1]) + g[..., 1]) * int(shape[2]) + g[..., 2]


def to_real_space(coeff: Tensor, pos: Tensor, shape: Sequence[int]) -> Tensor:
    """(..., n_g, n_b) plane-wave coefficients on sphere positions ``pos`` (n_g,) -> (..., n_b, N_r) real-space values."""
    n_r = int(shape[0]) * int(shape[1]) * int(shape[2])
    lead = coeff.shape[:-2]
    grid = coeff.new_zeros(*lead, coeff.shape[-1], n_r)
    grid[..., pos] = coeff.transpose(-1, -2)
    return torch.fft.ifftn(grid.reshape(*lead, coeff.shape[-1], *shape), dim=(-3, -2, -1), norm="forward").reshape(
        *lead, coeff.shape[-1], n_r)


def to_sphere(values: Tensor, pos: Tensor, shape: Sequence[int]) -> Tensor:
    """(..., n_b, N_r) real-space values -> (..., n_g, n_b) coefficients on the sphere positions ``pos``."""
    lead = values.shape[:-2]
    g = torch.fft.fftn(values.reshape(*lead, values.shape[-2], *shape), dim=(-3, -2, -1), norm="forward")
    return g.reshape(*lead, values.shape[-2], -1)[..., pos].transpose(-1, -2)


def potential_to_real(v: Tensor, m_int: Tensor, shape: Sequence[int]) -> Tensor:
    """V(r) on the grid (N_r,) from Fourier components v (n_m,) at integer transfers m_int (n_m, 3)."""
    n_r = int(shape[0]) * int(shape[1]) * int(shape[2])
    grid = v.new_zeros(n_r)
    grid[flat_positions(m_int, shape)] = v
    return torch.fft.ifftn(grid.reshape(*shape), norm="forward").reshape(n_r)


def apply_local(coeff: Tensor, v_r: Tensor, pos: Tensor, shape: Sequence[int]) -> Tensor:
    """(V psi)(G) = sum_G' V(G - G') psi(G') for psi (..., n_g, n_b) on one sphere, via the grid (alias-free if
    ``shape`` comes from ``grid_shape``)."""
    return to_sphere(to_real_space(coeff, pos, shape) * v_r, pos, shape)


__all__ = ["apply_local", "fft_len", "flat_positions", "grid_shape", "potential_to_real", "to_real_space", "to_sphere"]
