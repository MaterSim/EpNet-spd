"""Compact separable nonlocal-projector building blocks.

These routines assemble algebraic plane-wave operators. They do not prescribe
or calibrate pseudopotential radial projectors or spin-orbit strengths.
"""

from __future__ import annotations

import torch
from torch import Tensor


def separable_nonlocal_operator(projectors: Tensor, coefficients: Tensor) -> Tensor:
    r"""Assemble ``B D B^\dagger`` from compact projector data.

    ``projectors`` has shape ``(..., n_pw, n_channel)``. ``coefficients`` may
    hold one diagonal value per channel, shape ``(..., n_channel)``, or a full
    channel-coupling matrix, shape ``(..., n_channel, n_channel)``.
    """
    if projectors.ndim < 2:
        raise ValueError("projectors must have shape (..., n_pw, n_channel)")
    n_channel = projectors.shape[-1]
    if coefficients.shape[-1] != n_channel:
        raise ValueError("coefficient and projector channel counts differ")
    coefficients = coefficients.to(
        device=projectors.device,
        dtype=torch.promote_types(projectors.dtype, coefficients.dtype),
    )
    projectors = projectors.to(dtype=coefficients.dtype)

    if coefficients.ndim == projectors.ndim - 1:
        weighted = projectors * coefficients.unsqueeze(-2)
    elif coefficients.ndim == projectors.ndim:
        if coefficients.shape[-2] != n_channel:
            raise ValueError("full coefficients must be square in channel space")
        weighted = torch.matmul(projectors, coefficients)
    else:
        raise ValueError(
            "coefficients must be diagonal (..., channel) or full "
            "(..., channel, channel)"
        )
    return torch.matmul(weighted, projectors.conj().transpose(-1, -2))


def apply_separable_nonlocal(
    wavefunctions: Tensor,
    projectors: Tensor,
    coefficients: Tensor,
) -> Tensor:
    """Apply a separable operator without materializing its dense matrix."""
    if wavefunctions.shape[-2] != projectors.shape[-2]:
        raise ValueError("wavefunction and projector plane-wave counts differ")
    dtype = torch.promote_types(
        torch.promote_types(projectors.dtype, coefficients.dtype),
        wavefunctions.dtype,
    )
    projectors = projectors.to(device=wavefunctions.device, dtype=dtype)
    coefficients = coefficients.to(device=wavefunctions.device, dtype=dtype)
    wavefunctions = wavefunctions.to(dtype=dtype)
    overlaps = torch.matmul(projectors.conj().transpose(-1, -2), wavefunctions)
    if coefficients.ndim == projectors.ndim - 1:
        weighted_overlaps = coefficients.unsqueeze(-1) * overlaps
    elif coefficients.ndim == projectors.ndim:
        weighted_overlaps = torch.matmul(coefficients, overlaps)
    else:
        raise ValueError("invalid coefficient rank")
    return torch.matmul(projectors, weighted_overlaps)


def spin_orbit_channel_coefficients(
    scalar_coefficients: Tensor,
    soc_vectors: Tensor | None = None,
) -> Tensor:
    r"""Lift channel coefficients to general Hermitian spin blocks.

    For each channel this returns
    ``c I + lambda_x sigma_x + lambda_y sigma_y + lambda_z sigma_z``.
    Real scalar coefficients and real three-component ``soc_vectors`` therefore
    produce Hermitian 2x2 blocks. The vectors are a general trainable API; their
    physical normalization and calibration remain the caller's responsibility.
    """
    if scalar_coefficients.ndim < 1:
        raise ValueError("scalar_coefficients must include a channel axis")
    if soc_vectors is None:
        soc_vectors = torch.zeros(
            (*scalar_coefficients.shape, 3),
            dtype=scalar_coefficients.dtype,
            device=scalar_coefficients.device,
        )
    if soc_vectors.shape != (*scalar_coefficients.shape, 3):
        raise ValueError("soc_vectors must have shape (..., n_channel, 3)")

    dtype = (
        torch.complex128
        if scalar_coefficients.dtype in (torch.float64, torch.complex128)
        else torch.complex64
    )
    scalar = scalar_coefficients.to(dtype=dtype)
    soc = soc_vectors.to(device=scalar.device, dtype=dtype)
    x, y, z = soc.unbind(dim=-1)

    result = torch.empty((*scalar.shape, 2, 2), dtype=dtype, device=scalar.device)
    result[..., 0, 0] = scalar + z
    result[..., 1, 1] = scalar - z
    result[..., 0, 1] = x - 1j * y
    result[..., 1, 0] = x + 1j * y
    return result


def separable_spin_nonlocal_operator(
    projectors: Tensor,
    channel_coefficients: Tensor,
    *,
    flatten: bool = True,
) -> Tensor:
    r"""Assemble channel-diagonal spin blocks into a plane-wave operator.

    The unflattened result has indices ``(..., pw, spin, pw', spin')``.
    ``flatten=True`` returns the conventional ``(..., 2*n_pw, 2*n_pw)`` matrix.
    """
    n_channel = projectors.shape[-1]
    if channel_coefficients.shape[-3:] != (n_channel, 2, 2):
        raise ValueError(
            "channel_coefficients must have shape (..., n_channel, 2, 2)"
        )
    dtype = torch.promote_types(projectors.dtype, channel_coefficients.dtype)
    projectors = projectors.to(dtype=dtype)
    channel_coefficients = channel_coefficients.to(
        device=projectors.device, dtype=dtype
    )
    operator = torch.einsum(
        "...pc,...cst,...qc->...psqt",
        projectors,
        channel_coefficients,
        projectors.conj(),
    )
    if not flatten:
        return operator
    n_pw = projectors.shape[-2]
    return operator.reshape(*operator.shape[:-4], 2 * n_pw, 2 * n_pw)


# Naming aliases emphasizing the matrix returned by the assembly routines.
separable_nonlocal_matrix = separable_nonlocal_operator
separable_spin_nonlocal_matrix = separable_spin_nonlocal_operator


__all__ = [
    "apply_separable_nonlocal",
    "separable_nonlocal_matrix",
    "separable_nonlocal_operator",
    "separable_spin_nonlocal_matrix",
    "separable_spin_nonlocal_operator",
    "spin_orbit_channel_coefficients",
]
