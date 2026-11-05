"""Real Cartesian spherical harmonics and complex reciprocal form factors."""

from __future__ import annotations

from typing import Dict, Tuple
import math

import torch
from torch import Tensor, nn

from epnet.models.equivariant_encoder import SmoothCompactRadialBasis


class MultipoleFormFactor(nn.Module):
    """Trainable invariant-channel readout followed by exact harmonic assembly.

    Encoder channels are linearly reduced within each irrep, which preserves
    equivariance. A compact radial expansion gives each multipole order a
    learnable ``|G|`` dependence.  The public :meth:`forward` API consumes the
    feature dictionary returned by ``EquivariantEncoder``.
    """

    def __init__(
        self,
        channels: int,
        num_radial: int = 8,
        reciprocal_cutoff: float = 20.0,
        l_max: int = 2,
        zero_gauge: bool = True,
        include_monopole: bool = True,
        trainable_scalar_bias: bool = True,
        scalar_cosine_basis: int = 0,
        scalar_cosine_cutoff: float = 8.0,
    ) -> None:
        super().__init__()
        if l_max not in (0, 1, 2):
            raise ValueError("l_max must be 0, 1, or 2")
        self.channels = int(channels)
        self.num_radial = int(num_radial)
        self.l_max = int(l_max)
        self.zero_gauge = bool(zero_gauge)
        self.include_monopole = bool(include_monopole)
        self.trainable_scalar_bias = bool(trainable_scalar_bias)
        if scalar_cosine_basis < 0 or scalar_cosine_cutoff <= 0:
            raise ValueError("scalar cosine basis size must be nonnegative and cutoff positive")
        if scalar_cosine_basis and not include_monopole:
            raise ValueError("scalar cosine expansion requires the monopole channel")
        self.scalar_cosine_basis = int(scalar_cosine_basis)
        self.scalar_cosine_cutoff = float(scalar_cosine_cutoff)
        self.radial_basis = SmoothCompactRadialBasis(num_radial, reciprocal_cutoff)
        self.channel_readout = nn.ModuleList(
            [nn.Linear(channels, num_radial, bias=False) for _ in range(l_max + 1)]
        )
        # Only the scalar irrep admits an invariant bias.  Retain zero, frozen
        # l>0 entries for checkpoint compatibility, but never add them to the
        # vector/tensor channels: a Cartesian bias would select a lab-frame
        # direction and break rotational covariance.
        self.radial_bias = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(num_radial),
                    requires_grad=(order == 0 and self.trainable_scalar_bias),
                )
                for order in range(l_max + 1)
            ]
        )
        self.scalar_cosine = None
        if self.scalar_cosine_basis:
            self.scalar_cosine = nn.Linear(channels, self.scalar_cosine_basis, bias=trainable_scalar_bias)
            nn.init.zeros_(self.scalar_cosine.weight)
            if self.scalar_cosine.bias is not None:
                nn.init.zeros_(self.scalar_cosine.bias)

    def cosine_basis(self, q: Tensor) -> Tensor:
        """Same radial functions as the species sphere, without extra sphere parameters."""
        x = (q / self.scalar_cosine_cutoff).clamp(0, 1)
        envelope = 0.5 * (1 + torch.cos(math.pi * x))
        orders = torch.arange(self.scalar_cosine_basis, device=q.device, dtype=q.dtype)
        return envelope[:, None] * torch.cos(math.pi * x[:, None] * orders)

    def forward(
        self,
        positions: Tensor,
        reciprocal_vectors: Tensor,
        features: Dict[str, Tensor],
        *,
        return_per_atom: bool = False,
        quadrupole_scale: Tensor | None = None,
    ) -> Tensor | Tuple[Tensor, Tensor]:
        q = torch.linalg.vector_norm(reciprocal_vectors, dim=-1)
        radial = self.radial_basis(q)
        hat = reciprocal_vectors / q.clamp_min(1e-8).unsqueeze(-1)
        n_atoms = positions.shape[0]
        n_g = reciprocal_vectors.shape[0]
        per_atom = reciprocal_vectors.new_zeros((n_atoms, n_g), dtype=torch.complex64)

        if self.include_monopole and "scalar" in features and self.l_max >= 0:
            amp = self.channel_readout[0](features["scalar"]) + self.radial_bias[0]
            mono = torch.einsum("nr,gr->ng", amp, radial)
            if self.scalar_cosine is not None:
                extra = self.scalar_cosine(features["scalar"])
                mono = mono + torch.einsum("nr,gr->ng", extra, self.cosine_basis(q))
            per_atom = per_atom + mono.to(per_atom)

        if self.l_max >= 1 and "vector" in features:
            vec = features["vector"]
            amp = torch.einsum("nci,rc->nri", vec, self.channel_readout[1].weight)
            dipole = torch.einsum("nri,gi,gr->ng", amp, hat, radial)
            # With exp(-i G.r), the Fourier transform of a real l-pole carries
            # (-i)^l.  The odd-order phase is required for V(-G)=V(G)*.
            per_atom = per_atom + (-1j * dipole).to(per_atom)

        if self.l_max >= 2 and "tensor" in features:
            ten = features["tensor"]
            amp = torch.einsum("ncij,rc->nrij", ten, self.channel_readout[2].weight)
            quad = torch.einsum("nrij,gi,gj,gr->ng", amp, hat, hat, radial)
            if quadrupole_scale is not None:
                # Per-atom multiplier on the l=2 term (environment-dependent
                # quadrupole scale); ``None`` keeps the original readout.
                quad = quad * quadrupole_scale.to(quad).unsqueeze(-1)
            per_atom = per_atom - quad.to(per_atom)

        # Structure phase e^{-i G · r} with r cartesian, G cartesian (Å⁻¹).
        phase = torch.exp(-1j * torch.einsum("ga,na->ng", reciprocal_vectors.to(positions), positions))
        per_atom = per_atom * phase.to(per_atom)
        total = per_atom.sum(dim=0)
        if self.zero_gauge:
            zero = q < 1e-8
            total = total.masked_fill(zero, 0)
            per_atom = per_atom.masked_fill(zero[None, :], 0)
        if return_per_atom:
            return total, per_atom
        return total
