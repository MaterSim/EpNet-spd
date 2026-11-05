"""Rotationally invariant atom-centred s/p separable projectors."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor, nn

from epnet.hamiltonian.torch_pw import fold_to_first_zone
from epnet.physics import separable_nonlocal_operator


class SPProjectorCorrection(nn.Module):
    r"""Species-specific Kleinman--Bylander-like s/p correction.

    Radial projector shapes are fixed Gaussians. Each species learns one energy
    coefficient for every radial s channel and one coefficient shared by the
    three degenerate Cartesian p components. Sharing the p coefficient makes
    the assembled operator rotationally invariant. Coefficients start at zero,
    so attaching this module leaves the parent Hamiltonian unchanged.
    """

    def __init__(self, n_species: int, radial_scales: tuple[float, ...] = (1.5, 3.5)) -> None:
        super().__init__()
        if n_species < 1 or not radial_scales or any(value <= 0 for value in radial_scales):
            raise ValueError("n_species and radial scales must be positive")
        self.n_species = int(n_species)
        self.register_buffer("radial_scales", torch.tensor(radial_scales, dtype=torch.float32))
        # Last axis: l=0 (s), l=1 (shared across px, py, pz).
        self.coefficients = nn.Parameter(torch.zeros(self.n_species, len(radial_scales), 2))

    @property
    def n_radial(self) -> int:
        return int(self.radial_scales.numel())

    def projectors(self, packed: Any, k_frac: Tensor, g_indices: Tensor) -> Tensor:
        k_frac = fold_to_first_zone(k_frac, packed.lattice)
        reciprocal = 2.0 * torch.pi * torch.linalg.inv(packed.lattice).transpose(0, 1)
        g_cart = g_indices.to(reciprocal) @ reciprocal
        k_cart = k_frac.to(reciprocal) @ reciprocal
        momentum = k_cart[:, None, :] + g_cart[None, :, :]
        q = torch.linalg.vector_norm(momentum, dim=-1)
        direction = momentum / q.clamp_min(1e-8).unsqueeze(-1)
        scale = self.radial_scales.to(device=q.device, dtype=q.dtype)
        x = q.unsqueeze(-1) / scale
        radial_s = torch.exp(-0.5 * x.square())
        radial_p = x * radial_s
        phase = torch.exp(-1j * torch.einsum("kgc,nc->kgn", momentum, packed.positions.to(momentum)))
        norm = math.sqrt(float(packed.volume))
        s = phase.unsqueeze(-1) * radial_s.unsqueeze(-2)
        p = (
            phase[..., None, None]
            * radial_p[:, :, None, :, None]
            * direction[:, :, None, None, :]
            * math.sqrt(3.0)
        )
        # (k,g,atom,radial,1+3) -> (k,g,channel)
        return torch.cat((s.unsqueeze(-1), p), dim=-1).flatten(start_dim=2) / norm

    def channel_coefficients(
        self,
        species: Tensor,
        n_k: int,
        environment_factors: Tensor | None = None,
        environment_delta: Tensor | None = None,
    ) -> Tensor:
        values = self.coefficients[species.to(dtype=torch.long)]
        if environment_factors is not None:
            if tuple(environment_factors.shape) != tuple(values.shape):
                raise ValueError(
                    f"environment_factors shape {tuple(environment_factors.shape)} "
                    f"must match per-atom coefficients {tuple(values.shape)}"
                )
            values = values * environment_factors.to(values)
        if environment_delta is not None:
            if tuple(environment_delta.shape) != tuple(values.shape):
                raise ValueError(
                    f"environment_delta shape {tuple(environment_delta.shape)} "
                    f"must match per-atom coefficients {tuple(values.shape)}"
                )
            values = values + environment_delta.to(values)
        s = values[..., 0:1]
        p = values[..., 1:2].expand(*values.shape[:-1], 3)
        channels = torch.cat((s, p), dim=-1).flatten()
        return channels.unsqueeze(0).expand(int(n_k), -1)

    def forward(
        self,
        packed: Any,
        k_frac: Tensor,
        g_indices: Tensor,
        *,
        projectors: Tensor | None = None,
        environment_factors: Tensor | None = None,
        environment_delta: Tensor | None = None,
    ) -> Tensor:
        basis = self.projectors(packed, k_frac, g_indices) if projectors is None else projectors
        coefficients = self.channel_coefficients(
            packed.species,
            int(k_frac.shape[0]),
            environment_factors=environment_factors,
            environment_delta=environment_delta,
        )
        return separable_nonlocal_operator(basis, coefficients)


class SPDProjectorCorrection(nn.Module):
    r"""Species-specific rotationally invariant s/p/d separable correction.

    The five real d components form a normalized traceless-quadrupole basis.
    One coefficient is shared across each angular multiplet, preserving
    rotational invariance of the assembled operator.
    """

    def __init__(self, n_species: int, radial_scales: tuple[float, ...] = (1.5, 3.5)) -> None:
        super().__init__()
        if n_species < 1 or not radial_scales or any(value <= 0 for value in radial_scales):
            raise ValueError("n_species and radial scales must be positive")
        self.n_species = int(n_species)
        self.register_buffer("radial_scales", torch.tensor(radial_scales, dtype=torch.float32))
        # Last axis: l=0, l=1, l=2; each coefficient is shared over 2l+1 components.
        self.coefficients = nn.Parameter(torch.zeros(self.n_species, len(radial_scales), 3))

    @property
    def n_radial(self) -> int:
        return int(self.radial_scales.numel())

    def projectors(self, packed: Any, k_frac: Tensor, g_indices: Tensor) -> Tensor:
        k_frac = fold_to_first_zone(k_frac, packed.lattice)
        reciprocal = 2.0 * torch.pi * torch.linalg.inv(packed.lattice).transpose(0, 1)
        g_cart = g_indices.to(reciprocal) @ reciprocal
        k_cart = k_frac.to(reciprocal) @ reciprocal
        momentum = k_cart[:, None, :] + g_cart[None, :, :]
        q = torch.linalg.vector_norm(momentum, dim=-1)
        direction = momentum / q.clamp_min(1e-8).unsqueeze(-1)
        ux, uy, uz = direction.unbind(dim=-1)
        d_angular = torch.stack(
            (
                math.sqrt(3.0) * ux * uy,
                math.sqrt(3.0) * uy * uz,
                math.sqrt(3.0) * uz * ux,
                0.5 * math.sqrt(3.0) * (ux.square() - uy.square()),
                0.5 * (3.0 * uz.square() - 1.0),
            ),
            dim=-1,
        )
        scale = self.radial_scales.to(device=q.device, dtype=q.dtype)
        x = q.unsqueeze(-1) / scale
        radial_s = torch.exp(-0.5 * x.square())
        radial_p = x * radial_s
        radial_d = x.square() * radial_s
        phase = torch.exp(-1j * torch.einsum("kgc,nc->kgn", momentum, packed.positions.to(momentum)))
        norm = math.sqrt(float(packed.volume))
        s = phase.unsqueeze(-1) * radial_s.unsqueeze(-2)
        p = (
            phase[..., None, None]
            * radial_p[:, :, None, :, None]
            * direction[:, :, None, None, :]
            * math.sqrt(3.0)
        )
        d = (
            phase[..., None, None]
            * radial_d[:, :, None, :, None]
            * d_angular[:, :, None, None, :]
            * math.sqrt(5.0)
        )
        # (k,g,atom,radial,1+3+5) -> (k,g,channel)
        return torch.cat((s.unsqueeze(-1), p, d), dim=-1).flatten(start_dim=2) / norm

    def channel_coefficients(self, species: Tensor, n_k: int) -> Tensor:
        values = self.coefficients[species.to(dtype=torch.long)]
        s = values[..., 0:1]
        p = values[..., 1:2].expand(*values.shape[:-1], 3)
        d = values[..., 2:3].expand(*values.shape[:-1], 5)
        channels = torch.cat((s, p, d), dim=-1).flatten()
        return channels.unsqueeze(0).expand(int(n_k), -1)

    def forward(
        self,
        packed: Any,
        k_frac: Tensor,
        g_indices: Tensor,
        *,
        projectors: Tensor | None = None,
    ) -> Tensor:
        basis = self.projectors(packed, k_frac, g_indices) if projectors is None else projectors
        coefficients = self.channel_coefficients(packed.species, int(k_frac.shape[0]))
        return separable_nonlocal_operator(basis, coefficients)


class SPDFullRadialProjectorCorrection(SPDProjectorCorrection):
    r"""Rotationally invariant s/p/d projectors with full radial coupling.

    For each species and angular momentum, the learned coefficient is a real
    symmetric matrix over radial channels. The same matrix is shared over all
    2l+1 angular components, and contributions are summed independently over
    atoms. Only the lower triangle is stored, so the trainable parameter count
    equals the number of independent entries.
    """

    def __init__(
        self,
        n_species: int,
        radial_scales: tuple[float, ...] = (1.5, 3.5),
    ) -> None:
        super().__init__(n_species, radial_scales)
        del self.coefficients
        rows, columns = torch.tril_indices(self.n_radial, self.n_radial)
        self.register_buffer("_coupling_rows", rows)
        self.register_buffer("_coupling_columns", columns)
        n_triangular = self.n_radial * (self.n_radial + 1) // 2
        self.packed_couplings = nn.Parameter(
            torch.zeros(self.n_species, 3, n_triangular)
        )

    @property
    def independent_parameter_count(self) -> int:
        return int(self.packed_couplings.numel())

    def coupling_matrices(self, species: Tensor, *, active_l: int = 2) -> Tensor:
        if active_l not in (0, 1, 2):
            raise ValueError("active_l must be 0, 1, or 2")
        packed = self.packed_couplings[species.to(dtype=torch.long)]
        matrices = packed.new_zeros(
            *packed.shape[:-1], self.n_radial, self.n_radial
        )
        rows = self._coupling_rows
        columns = self._coupling_columns
        matrices[..., rows, columns] = packed
        matrices[..., columns, rows] = packed
        if active_l < 2:
            mask = torch.arange(3, device=matrices.device) <= active_l
            matrices = matrices * mask.view(1, 3, 1, 1)
        return matrices

    def initialize_from_diagonal(
        self,
        coefficients: Tensor,
        radial_scales: Tensor,
    ) -> tuple[int, ...]:
        """Embed a diagonal-projector checkpoint in the full radial model."""
        coefficients = torch.as_tensor(coefficients)
        radial_scales = torch.as_tensor(radial_scales)
        expected = (self.n_species, int(radial_scales.numel()), 3)
        if tuple(coefficients.shape) != expected:
            raise ValueError(
                f"expected diagonal coefficients with shape {expected}, "
                f"received {tuple(coefficients.shape)}"
            )
        target_indices = []
        for scale in radial_scales:
            matches = torch.isclose(
                self.radial_scales,
                scale.to(self.radial_scales),
                atol=1e-7,
                rtol=1e-6,
            ).nonzero(as_tuple=False).flatten()
            if int(matches.numel()) != 1:
                raise ValueError(
                    f"source radial scale {float(scale)} does not map uniquely "
                    "to the full-radial basis"
                )
            target_indices.append(int(matches.item()))
        with torch.no_grad():
            self.packed_couplings.zero_()
            for source_index, target_index in enumerate(target_indices):
                packed_index = (
                    (self._coupling_rows == target_index)
                    & (self._coupling_columns == target_index)
                ).nonzero(as_tuple=False)
                if int(packed_index.numel()) != 1:
                    raise RuntimeError("could not resolve packed diagonal index")
                self.packed_couplings[..., int(packed_index.item())].copy_(
                    coefficients[:, source_index, :].to(self.packed_couplings)
                )
        return tuple(target_indices)

    def forward(
        self,
        packed: Any,
        k_frac: Tensor,
        g_indices: Tensor,
        *,
        projectors: Tensor | None = None,
        active_l: int = 2,
        species: Tensor | None = None,
    ) -> Tensor:
        basis = (
            self.projectors(packed, k_frac, g_indices)
            if projectors is None
            else projectors
        )
        n_k, n_g = int(basis.shape[0]), int(basis.shape[1])
        n_atoms = int(packed.species.numel())
        expected_channels = n_atoms * self.n_radial * 9
        if int(basis.shape[-1]) != expected_channels:
            raise ValueError(
                f"expected {expected_channels} projector channels, "
                f"received {int(basis.shape[-1])}"
            )
        resolved = basis.reshape(n_k, n_g, n_atoms, self.n_radial, 9)
        atom_species = packed.species if species is None else species
        if int(atom_species.numel()) != n_atoms:
            raise ValueError("species must contain one index per atom")
        couplings = self.coupling_matrices(
            atom_species.to(device=basis.device), active_l=active_l
        ).to(dtype=basis.dtype)
        operator = basis.new_zeros(n_k, n_g, n_g)
        for angular_index, angular_slice in enumerate(
            (slice(0, 1), slice(1, 4), slice(4, 9))
        ):
            if angular_index > active_l:
                break
            angular_basis = resolved[..., angular_slice]
            operator = operator + torch.einsum(
                "kgarm,ars,khasm->kgh",
                angular_basis,
                couplings[:, angular_index],
                angular_basis.conj(),
            )
        return operator


__all__ = [
    "SPProjectorCorrection",
    "SPDProjectorCorrection",
    "SPDFullRadialProjectorCorrection",
]
