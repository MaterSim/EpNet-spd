"""Environment-dependent neutral ionic correction in reciprocal space."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from epnet.hamiltonian.torch_pw import COULOMB_EV_A
from epnet.models.equivariant_encoder import smooth_coordination
from epnet.physics.long_range import neutralize_effective_charges, reciprocal_coulomb_potential


class EnvironmentIonicCorrection(nn.Module):
    """Species-conditioned charges from fixed local coordination features.

    Each species owns one bias and one coefficient per coordination radius.
    The raw atom charges are projected to exact cell neutrality before a
    Gaussian-damped Coulomb form factor is assembled. Zero initialization is
    Hamiltonian-neutral and the default three radii give four parameters per
    species.
    """

    def __init__(
        self,
        n_species: int,
        coordination_cutoffs: tuple[float, ...] = (3.0, 5.0, 8.0),
        gaussian_width: float = 0.5,
    ) -> None:
        super().__init__()
        if n_species < 1:
            raise ValueError("n_species must be positive")
        if not coordination_cutoffs or any(value <= 0 for value in coordination_cutoffs):
            raise ValueError("coordination cutoffs must be positive")
        if gaussian_width < 0:
            raise ValueError("gaussian_width must be nonnegative")
        self.n_species = int(n_species)
        self.coordination_cutoffs = tuple(float(value) for value in coordination_cutoffs)
        self.gaussian_width = float(gaussian_width)
        self.coefficients = nn.Parameter(
            torch.zeros(self.n_species, 1 + len(self.coordination_cutoffs))
        )

    def charge_features(
        self,
        packed: Any,
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        coordination = smooth_coordination(
            packed.positions,
            packed.lattice,
            self.coordination_cutoffs,
            edge_index=edge_index,
            cell_shifts=cell_shifts,
        )
        return torch.cat((torch.ones_like(coordination[:, :1]), torch.log1p(coordination)), dim=-1)

    def charges(
        self,
        packed: Any,
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        features = self.charge_features(
            packed, edge_index=edge_index, cell_shifts=cell_shifts
        )
        weights = self.coefficients[packed.species.to(dtype=torch.long)]
        raw = (weights * features).sum(dim=-1)
        return neutralize_effective_charges(raw)

    def forward(
        self,
        packed: Any,
        g_cart: Tensor,
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        charges = self.charges(
            packed, edge_index=edge_index, cell_shifts=cell_shifts
        )
        values = reciprocal_coulomb_potential(
            packed.positions,
            charges,
            g_cart,
            volume=1.0,
            gaussian_width=self.gaussian_width,
        )
        return values * values.new_tensor(COULOMB_EV_A)


__all__ = ["EnvironmentIonicCorrection"]
