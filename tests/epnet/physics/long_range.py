"""Differentiable reciprocal-space electrostatic utilities.

The conventions in this module are deliberately explicit: reciprocal vectors
are Cartesian, the Fourier phase is ``exp(-i G.r)``, and the zero reciprocal
mode is fixed to a caller-selected gauge (zero by default).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


def neutralize_effective_charges(
    charges: Tensor,
    weights: Tensor | None = None,
    *,
    dim: int = -1,
    target_charge: float | Tensor = 0.0,
) -> Tensor:
    """Project effective charges onto a specified total-charge constraint.

    ``weights`` controls how the correction is distributed over atoms and is
    normalized internally. With no weights, every atom receives the same
    correction. The operation is differentiable with respect to both inputs.
    """
    if not torch.is_floating_point(charges):
        raise TypeError("charges must have a floating-point dtype")
    if charges.shape[dim] == 0:
        raise ValueError("cannot neutralize an empty charge axis")

    if weights is None:
        normalized_weights = torch.ones_like(charges) / charges.shape[dim]
    else:
        weights = torch.as_tensor(weights, dtype=charges.dtype, device=charges.device)
        weights = torch.broadcast_to(weights, charges.shape)
        denominator = weights.sum(dim=dim, keepdim=True)
        if torch.any(denominator == 0):
            raise ValueError("weights must have a nonzero sum")
        normalized_weights = weights / denominator

    target = torch.as_tensor(target_charge, dtype=charges.dtype, device=charges.device)
    residual = charges.sum(dim=dim, keepdim=True) - target.unsqueeze(dim)
    return charges - normalized_weights * residual


def reciprocal_coulomb_potential(
    positions: Tensor,
    charges: Tensor,
    g_vectors: Tensor,
    volume: float | Tensor,
    *,
    gaussian_width: float | Tensor = 0.0,
    ewald_alpha: float | Tensor | None = None,
    g0_value: float | complex | Tensor = 0.0,
    zero_tolerance: float = 0.0,
) -> Tensor:
    r"""Return the periodic reciprocal Coulomb potential ``V(G)``.

    For nonzero ``G`` the convention is

    .. math::
       V(G) = {4\pi\over\Omega |G|^2}
              e^{-\sigma^2 |G|^2/2}
              e^{-|G|^2/(4\alpha^2)}
              \sum_i q_i e^{-iG\cdot r_i}.

    The final exponential is included only when ``ewald_alpha`` is supplied.
    ``g0_value`` fixes the otherwise undefined additive-potential gauge.
    Inputs may have leading batch dimensions; ``g_vectors`` has shape
    ``(n_g, 3)``, positions ``(..., n_atom, 3)``, and charges
    ``(..., n_atom)``.
    """
    if positions.ndim < 2 or positions.shape[-1] != 3:
        raise ValueError("positions must have shape (..., n_atom, 3)")
    if g_vectors.ndim != 2 or g_vectors.shape[-1] != 3:
        raise ValueError("g_vectors must have shape (n_g, 3)")
    if charges.shape[-1] != positions.shape[-2]:
        raise ValueError("charges and positions must have the same atom count")
    if not (torch.is_floating_point(positions) and torch.is_floating_point(g_vectors)):
        raise TypeError("positions and g_vectors must have floating-point dtypes")

    dtype = torch.promote_types(positions.dtype, g_vectors.dtype)
    positions = positions.to(dtype=dtype)
    g_vectors = g_vectors.to(device=positions.device, dtype=dtype)
    charges = charges.to(device=positions.device, dtype=dtype)
    volume_tensor = torch.as_tensor(volume, device=positions.device, dtype=dtype)
    if torch.any(volume_tensor <= 0):
        raise ValueError("volume must be positive")

    phases = torch.matmul(positions, g_vectors.transpose(-1, -2))
    structure_factor = (
        charges.unsqueeze(-1) * torch.exp(-1j * phases)
    ).sum(dim=-2)

    g_squared = (g_vectors * g_vectors).sum(dim=-1)
    is_g0 = g_squared <= zero_tolerance * zero_tolerance
    safe_g_squared = torch.where(is_g0, torch.ones_like(g_squared), g_squared)

    sigma = torch.as_tensor(gaussian_width, device=positions.device, dtype=dtype)
    smoothing = torch.exp(-0.5 * sigma.square() * g_squared)
    if ewald_alpha is not None:
        alpha = torch.as_tensor(ewald_alpha, device=positions.device, dtype=dtype)
        if torch.any(alpha <= 0):
            raise ValueError("ewald_alpha must be positive")
        smoothing = smoothing * torch.exp(-g_squared / (4.0 * alpha.square()))

    kernel = (4.0 * math.pi / volume_tensor.unsqueeze(-1)) * smoothing / safe_g_squared
    potential = structure_factor * kernel
    gauge = torch.as_tensor(g0_value, device=potential.device, dtype=potential.dtype)
    return torch.where(is_g0, gauge, potential)


def reciprocal_ewald_potential(
    positions: Tensor,
    charges: Tensor,
    g_vectors: Tensor,
    volume: float | Tensor,
    ewald_alpha: float | Tensor,
    **kwargs: object,
) -> Tensor:
    """Convenience wrapper requiring the reciprocal Ewald screening scale."""
    return reciprocal_coulomb_potential(
        positions,
        charges,
        g_vectors,
        volume,
        ewald_alpha=ewald_alpha,
        **kwargs,
    )


# Short, discoverable alias used by some model code.
neutralized_effective_charges = neutralize_effective_charges


__all__ = [
    "neutralize_effective_charges",
    "neutralized_effective_charges",
    "reciprocal_coulomb_potential",
    "reciprocal_ewald_potential",
]
