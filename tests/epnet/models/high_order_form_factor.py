"""Shared atom-centered neighbor-angular form factors through configurable l."""

from __future__ import annotations

import torch
from e3nn import o3
from torch import Tensor, nn

from epnet.models.equivariant_encoder import (
    SmoothCompactRadialBasis,
    periodic_displacements,
)


class HighOrderEnvironmentFormFactor(nn.Module):
    r"""An exactly rotation-invariant atom-centered form factor through ``l_max``.

    Learned edge weights multiply real spherical harmonics of neighbor directions.
    The resulting atom-centered multipoles are contracted with harmonics of ``G``
    and structure phases.  The factor ``(-i)^l`` guarantees
    ``V(-G) = V(G)^*`` for real learned parameters.
    """

    def __init__(
        self,
        n_species: int,
        channels: int = 64,
        cutoff: float = 5.0,
        real_num_radial: int = 8,
        reciprocal_num_radial: int = 64,
        reciprocal_cutoff: float = 20.0,
        l_max: int = 4,
        aggregation: str = "atom_phase",
    ) -> None:
        super().__init__()
        if n_species < 1 or channels < 1:
            raise ValueError("n_species and channels must be positive")
        if l_max < 0 or real_num_radial < 1 or reciprocal_num_radial < 1:
            raise ValueError("l_max must be non-negative and radial ranks positive")
        if aggregation not in {"atom_phase", "global"}:
            raise ValueError("aggregation must be atom_phase or global")
        self.n_species = int(n_species)
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.real_num_radial = int(real_num_radial)
        self.reciprocal_num_radial = int(reciprocal_num_radial)
        self.reciprocal_cutoff = float(reciprocal_cutoff)
        self.l_max = int(l_max)
        self.aggregation = str(aggregation)
        self.irreps = o3.Irreps.spherical_harmonics(self.l_max)

        self.species_embedding = nn.Embedding(self.n_species, self.channels)
        self.real_basis = SmoothCompactRadialBasis(self.real_num_radial, self.cutoff)
        edge_input = 2 * self.channels + self.real_num_radial
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input, 2 * self.channels),
            nn.SiLU(),
            nn.Linear(2 * self.channels, self.channels),
        )
        gate_input = self.channels * (self.l_max + 2)
        self.gate_mlp = nn.Sequential(
            nn.Linear(gate_input, 2 * self.channels),
            nn.SiLU(),
            nn.Linear(2 * self.channels, self.channels * (self.l_max + 1)),
        )
        self.reciprocal_basis = SmoothCompactRadialBasis(
            self.reciprocal_num_radial, self.reciprocal_cutoff
        )
        self.readout = nn.ModuleList(
            [
                nn.Linear(self.channels, self.reciprocal_num_radial, bias=False)
                for _ in range(self.l_max + 1)
            ]
        )
        self.scalar_bias = nn.Parameter(torch.zeros(self.reciprocal_num_radial))
        self.zero_output()

    def zero_output(self) -> None:
        """Make the attached correction exactly Hamiltonian-neutral."""
        with torch.no_grad():
            for layer in self.readout:
                layer.weight.zero_()
            self.scalar_bias.zero_()

    @staticmethod
    def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
        result = values.new_zeros((size,) + values.shape[1:])
        result.index_add_(0, index, values)
        return result

    def _moments(
        self,
        positions: Tensor,
        species: Tensor,
        lattice: Tensor,
        edge_index: Tensor,
        cell_shifts: Tensor,
    ) -> list[Tensor]:
        edge_index = edge_index.to(device=positions.device, dtype=torch.long)
        displacement = periodic_displacements(
            positions, edge_index, lattice, cell_shifts
        )
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        valid = (distance > torch.finfo(distance.dtype).eps) & (distance < self.cutoff)
        keep = torch.nonzero(valid, as_tuple=False).reshape(-1)
        edge_index = edge_index.index_select(1, keep)
        displacement = displacement.index_select(0, keep)
        distance = distance.index_select(0, keep)
        center, neighbor = edge_index
        direction = displacement / distance.unsqueeze(-1)

        embedding = self.species_embedding(species.to(torch.long))
        edge_input = torch.cat(
            (
                embedding.index_select(0, center),
                embedding.index_select(0, neighbor),
                self.real_basis(distance),
            ),
            dim=-1,
        )
        edge_weight = self.edge_mlp(edge_input)
        harmonics = o3.spherical_harmonics(
            list(range(self.l_max + 1)),
            direction,
            normalize=True,
            normalization="component",
        ).to(edge_weight)
        n_atoms = positions.shape[0]
        counts = positions.new_zeros(n_atoms)
        counts.index_add_(0, center, torch.ones_like(distance))
        normalization = torch.sqrt(counts.clamp_min(1.0)).reshape(-1, 1, 1)

        moments = []
        for order, component_slice in enumerate(self.irreps.slices()):
            contribution = edge_weight.unsqueeze(-1) * harmonics[
                :, component_slice
            ].unsqueeze(1)
            moment = self._scatter_sum(contribution, center, n_atoms) / normalization
            if order == 0:
                moment = moment + embedding.unsqueeze(-1)
            moments.append(moment)

        invariants = [embedding]
        invariants.extend(
            torch.sqrt(moment.square().mean(dim=-1) + 1e-12) for moment in moments
        )
        gates = torch.sigmoid(self.gate_mlp(torch.cat(invariants, dim=-1))).reshape(
            n_atoms, self.l_max + 1, self.channels
        )
        return [
            moment * gates[:, order, :, None]
            for order, moment in enumerate(moments)
        ]

    def forward(
        self,
        positions: Tensor,
        species: Tensor,
        lattice: Tensor,
        reciprocal_vectors: Tensor,
        *,
        edge_index: Tensor,
        cell_shifts: Tensor,
    ) -> Tensor:
        moments = self._moments(
            positions, species, lattice, edge_index, cell_shifts
        )
        q = torch.linalg.vector_norm(reciprocal_vectors, dim=-1)
        direction = reciprocal_vectors / q.clamp_min(1e-8).unsqueeze(-1)
        if bool((q < 1e-8).any()):
            fallback = torch.zeros_like(direction)
            fallback[:, 0] = 1
            direction = torch.where((q < 1e-8).unsqueeze(-1), fallback, direction)
        q_harmonics = o3.spherical_harmonics(
            list(range(self.l_max + 1)),
            direction,
            normalize=True,
            normalization="component",
        ).to(positions)
        radial = self.reciprocal_basis(q)
        complex_dtype = torch.complex128 if positions.dtype == torch.float64 else torch.complex64
        if self.aggregation == "global":
            moments = [moment.mean(dim=0, keepdim=True) for moment in moments]
            phase = None
        else:
            phase = torch.exp(
                -1j
                * torch.einsum(
                    "ga,na->ng", reciprocal_vectors.to(positions), positions
                )
            ).to(complex_dtype)
        per_atom = torch.zeros(
            moments[0].shape[0], reciprocal_vectors.shape[0],
            dtype=complex_dtype, device=positions.device
        )
        for order, (moment, component_slice) in enumerate(
            zip(moments, self.irreps.slices())
        ):
            amplitude = torch.einsum(
                "ncm,rc->nrm", moment, self.readout[order].weight
            )
            if order == 0:
                amplitude = amplitude + self.scalar_bias[None, :, None]
            angular = torch.einsum(
                "nrm,gm,gr->ng",
                amplitude,
                q_harmonics[:, component_slice],
                radial,
            )
            per_atom = per_atom + complex((-1j) ** order) * angular.to(complex_dtype)
        total = per_atom.sum(dim=0) if phase is None else (per_atom * phase).sum(dim=0)
        return total.masked_fill(q < 1e-8, 0)
