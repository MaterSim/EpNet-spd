"""Periodic-table-conditioned species representations."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PeriodicTableEmbedding(nn.Module):
    """Combine a transferable feature projection with species residuals.

    The feature projection and zero-initialized nonlinear branch are shared by
    all elements. The residual embedding can capture chemistry not explained
    by the fixed features, while leave-one-element-out runs can keep the held
    row at zero and still obtain a feature-derived representation.
    """

    def __init__(
        self,
        element_features: Tensor,
        channels: int,
        *,
        hidden: int = 32,
    ) -> None:
        super().__init__()
        if element_features.ndim != 2 or not element_features.shape[0]:
            raise ValueError("element_features must have shape (species, features)")
        if channels < 1 or hidden < 1:
            raise ValueError("channels and hidden must be positive")
        if not torch.isfinite(element_features).all():
            raise ValueError("element features must be finite")
        features = element_features.detach().clone().to(dtype=torch.float32)
        self.register_buffer("element_features", features)
        self.num_species = int(features.shape[0])
        self.feature_width = int(features.shape[1])
        self.channels = int(channels)
        self.feature_projection = nn.Linear(self.feature_width, channels)
        self.feature_residual = nn.Sequential(
            nn.Linear(self.feature_width, hidden),
            nn.SiLU(),
            nn.Linear(hidden, channels),
        )
        self.species_residual = nn.Embedding(self.num_species, channels)
        nn.init.zeros_(self.feature_residual[-1].weight)
        nn.init.zeros_(self.feature_residual[-1].bias)
        nn.init.zeros_(self.species_residual.weight)

    def embedding_table(self) -> Tensor:
        features = self.element_features
        return (
            self.feature_projection(features)
            + self.feature_residual(features)
            + self.species_residual.weight
        )

    def forward(self, species: Tensor) -> Tensor:
        if species.dtype != torch.long:
            species = species.long()
        if species.numel() and (
            int(species.min()) < 0 or int(species.max()) >= self.num_species
        ):
            raise IndexError("species index lies outside the element feature table")
        return self.embedding_table().index_select(0, species.reshape(-1)).reshape(
            species.shape + (self.channels,)
        )


__all__ = ["PeriodicTableEmbedding"]
