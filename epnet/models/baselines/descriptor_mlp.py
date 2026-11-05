"""A small periodic radial-descriptor MLP baseline."""

from __future__ import annotations

import itertools
import math

import torch
from torch import Tensor, nn


def periodic_radial_descriptor(
    lattice: Tensor,
    fractional_positions: Tensor,
    species_index: Tensor,
    *,
    n_species: int,
    n_radial: int = 16,
    cutoff: float = 5.0,
) -> Tensor:
    """Species-resolved Gaussian pair histogram for each central atom.

    Images in the neighboring 27 cells are considered, which is exact whenever
    the cutoff does not reach beyond that image shell.
    """
    if lattice.shape != (3, 3) or fractional_positions.ndim != 2 or fractional_positions.shape[1] != 3:
        raise ValueError("invalid lattice or position shape")
    if len(fractional_positions) != len(species_index) or cutoff <= 0:
        raise ValueError("species must match positions and cutoff must be positive")
    device, dtype = fractional_positions.device, fractional_positions.dtype
    shifts = torch.tensor(list(itertools.product((-1, 0, 1), repeat=3)), device=device, dtype=dtype)
    delta = fractional_positions[None, :, None, :] + shifts[None, None, :, :] - fractional_positions[:, None, None, :]
    distance = torch.linalg.vector_norm(delta @ lattice, dim=-1)
    atom_i = torch.arange(len(species_index), device=device)[:, None, None]
    atom_j = torch.arange(len(species_index), device=device)[None, :, None]
    zero_shift = torch.all(shifts == 0, dim=1)[None, None, :]
    valid = (distance > 1e-8) & (distance < cutoff) & ~((atom_i == atom_j) & zero_shift)
    centers = torch.linspace(0.0, cutoff, n_radial, device=device, dtype=dtype)
    width = cutoff / max(n_radial - 1, 1)
    radial = torch.exp(-0.5 * ((distance[..., None] - centers) / width) ** 2)
    radial = radial * (0.5 * (torch.cos(math.pi * distance / cutoff) + 1.0) * valid)[..., None]
    one_hot = torch.nn.functional.one_hot(species_index.long(), n_species).to(dtype)
    descriptor = torch.einsum("ijrn,js->isn", radial, one_hot)
    return descriptor.flatten(1)


class DescriptorMLP(nn.Module):
    """Predict atom-centered cosine form-factor coefficients from environments."""

    def __init__(
        self,
        n_species: int,
        *,
        n_radial: int = 16,
        n_output: int = 12,
        hidden: tuple[int, ...] = (64, 64),
        cutoff: float = 5.0,
    ):
        super().__init__()
        self.n_species, self.n_radial, self.cutoff = n_species, n_radial, float(cutoff)
        dimensions = [n_species * n_radial + n_species, *hidden, n_output]
        layers: list[nn.Module] = []
        for input_size, output_size in zip(dimensions[:-2], dimensions[1:-1]):
            layers.extend((nn.Linear(input_size, output_size), nn.SiLU()))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, lattice: Tensor, fractional_positions: Tensor, species_index: Tensor) -> Tensor:
        descriptor = periodic_radial_descriptor(
            lattice, fractional_positions, species_index,
            n_species=self.n_species, n_radial=self.n_radial, cutoff=self.cutoff,
        )
        identity = torch.nn.functional.one_hot(species_index.long(), self.n_species).to(descriptor.dtype)
        return self.network(torch.cat((descriptor, identity), dim=-1))
