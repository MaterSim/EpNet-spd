"""Exact v1 E(3)-equivariant encoder based on periodic neighbor moments.

This module intentionally does not require :mod:`e3nn`.  Instead, neighbor
density moments are accumulated in Cartesian irreducible representations:
scalars (l=0), vectors (l=1), and symmetric traceless tensors (l=2).  The
construction is exactly equivariant (up to floating point roundoff), while an
invariant MLP supplies trainable channel gates.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn


def _as_bool3(pbc: bool | Tensor, device: torch.device) -> Tensor:
    value = torch.as_tensor(pbc, dtype=torch.bool, device=device)
    if value.ndim == 0:
        value = value.repeat(3)
    if value.shape != (3,):
        raise ValueError("pbc must be a bool or a length-3 tensor")
    return value


def periodic_displacements(
    positions: Tensor,
    edge_index: Tensor,
    cell: Optional[Tensor] = None,
    cell_shifts: Optional[Tensor] = None,
) -> Tensor:
    """Return edge vectors ``r_j + shift @ cell - r_i``.

    ``edge_index`` has shape ``(2, E)`` with center atoms in row 0 and
    neighbors in row 1. ``cell_shifts`` contains integer lattice coordinates
    and may be omitted for a non-periodic graph.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape (2, E)")
    i, j = edge_index.to(device=positions.device, dtype=torch.long)
    displacement = positions.index_select(0, j) - positions.index_select(0, i)
    if cell_shifts is not None:
        if cell is None:
            raise ValueError("cell is required when cell_shifts are supplied")
        shifts = cell_shifts.to(device=positions.device, dtype=positions.dtype)
        if shifts.shape != (edge_index.shape[1], 3):
            raise ValueError("cell_shifts must have shape (E, 3)")
        if cell.ndim != 2 or cell.shape != (3, 3):
            raise ValueError("periodic_displacements expects cell shape (3, 3)")
        displacement = displacement + shifts @ cell.to(displacement)
    return displacement


def build_periodic_neighbor_graph(
    positions: Tensor,
    cell: Optional[Tensor],
    cutoff: float,
    pbc: bool | Tensor = True,
    include_self: bool = False,
) -> Tuple[Tensor, Tensor]:
    """Build a deterministic all-images periodic radius graph.

    This O(N^2 * images) reference implementation is intended for small
    structures and tests.  It supports arbitrary (including triclinic) cells
    and includes every image that can lie inside ``cutoff``.  Returned lattice
    shifts obey the convention used by :func:`periodic_displacements`.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if cutoff <= 0:
        raise ValueError("cutoff must be positive")
    device, dtype = positions.device, positions.dtype
    periodic = _as_bool3(pbc, device)
    if bool(periodic.any()):
        if cell is None or cell.shape != (3, 3):
            raise ValueError("a (3, 3) cell is required for periodic axes")
        cell = cell.to(device=device, dtype=dtype)
        # For row-vector cells, reciprocal column norms bound the necessary
        # integer translations for any pair after reduction to a nearby image.
        inv_cell = torch.linalg.inv(cell)
        fractional = positions @ inv_cell
        span = fractional.amax(dim=0) - fractional.amin(dim=0)
        bounds = torch.ceil(
            span + cutoff * torch.linalg.vector_norm(inv_cell, dim=0)
        ).to(torch.long) + 1
    else:
        cell = torch.eye(3, device=device, dtype=dtype) if cell is None else cell.to(device=device, dtype=dtype)
        bounds = torch.zeros(3, device=device, dtype=torch.long)

    ranges = [
        range(-int(bounds[k]), int(bounds[k]) + 1) if bool(periodic[k]) else range(0, 1)
        for k in range(3)
    ]
    shifts = torch.tensor(list(itertools.product(*ranges)), device=device, dtype=torch.long)
    n_atoms = positions.shape[0]
    centers = torch.arange(n_atoms, device=device).repeat_interleave(n_atoms)
    neighbors = torch.arange(n_atoms, device=device).repeat(n_atoms)
    pair_i = centers.repeat_interleave(shifts.shape[0])
    pair_j = neighbors.repeat_interleave(shifts.shape[0])
    pair_shifts = shifts.repeat(n_atoms * n_atoms, 1)
    vectors = positions[pair_j] + pair_shifts.to(dtype) @ cell - positions[pair_i]
    distance2 = (vectors * vectors).sum(-1)
    mask = distance2 < float(cutoff) ** 2
    if not include_self:
        mask &= distance2 > 100.0 * torch.finfo(dtype).eps
    edge_index = torch.stack((pair_i[mask], pair_j[mask]))
    return edge_index, pair_shifts[mask]


INVARIANT_NORMS = ("sqrt", "smooth")


def norm_invariant(squared: Tensor, mode: str = "sqrt", scale: float = 0.05) -> Tensor:
    """Rotation invariant from a squared dipole/quadrupole norm.

    ``sqrt``: |x| (Si-v1d, G1-G3 v2). It has a cusp at x = 0: on sites where the moment vanishes by symmetry
    (T_d, D_3h, inversion centres) band energies respond to a displacement u linearly in |u|, which biases
    second-order electron-phonon quantities (zero-point renormalization). ``smooth``: s (sqrt(1 + x^2/s^2) - 1),
    analytic and even in x: quadratic for |x| << s, ~|x| - s for |x| >> s (adopted 2026-10-13, G3 v3).
    """
    if mode == "sqrt":
        return torch.sqrt(squared + 1e-12)
    if mode == "smooth":
        return scale * (torch.sqrt(1.0 + squared / (scale * scale)) - 1.0)
    raise ValueError(f"unknown invariant norm {mode!r}")


def packing_gate(cn10: Tensor, center: float = 142.0, alpha: float = 2.0) -> Tensor:
    """Turn a cage channel on for open packing and off for compact 4-fold crystals.

    ``g = σ(α (C0 − CN10))``. Cages (CN10 ~ 122–141) give g≈1; diamond / ST12
    (CN10 ~ 143–170) give g≈0. Used on env Δv and, when residual_pack_gate is
    set, to blend ``(1−g)·frozen + g·live`` residual. No family name is used.
    """
    if float(alpha) <= 0:
        raise ValueError("alpha must be positive")
    return torch.sigmoid(float(alpha) * (float(center) - cn10))


_GATE_TERM = re.compile(r"^\s*cn\s*([0-9.]+)\s*([<>])\s*([-+0-9.]+)\s*@\s*([0-9.]+)\s*$")


def parse_gate_window(spec: str) -> list[dict[str, Any]]:
    """Parse ``"cn10<142@2,cn5<17@5"`` into gate terms.

    Each comma-separated term is ``cn<R><op><center>@<alpha>``: a sigmoid that is
    ~1 when the plateau-cosine coordination at radius R Å is below (``<``) or
    above (``>``) ``center`` with steepness ``alpha``. Terms in one window are
    multiplied (logical AND).
    """
    terms: list[dict[str, Any]] = []
    for raw in str(spec).split(","):
        if not raw.strip():
            continue
        match = _GATE_TERM.match(raw)
        if match is None:
            raise ValueError(f"bad gate term {raw!r}; expected e.g. 'cn10<142@2'")
        radius, op, center, alpha = match.groups()
        if float(alpha) <= 0 or float(radius) <= 0:
            raise ValueError(f"gate term {raw!r}: radius and alpha must be positive")
        terms.append(
            {"radius": float(radius), "center": float(center), "alpha": float(alpha), "below": op == "<"}
        )
    if not terms:
        raise ValueError("empty gate window")
    return terms


def gate_window_radii(windows: list[list[dict[str, Any]]]) -> tuple[float, ...]:
    return tuple(sorted({float(term["radius"]) for window in windows for term in window}))


def format_gate_windows(windows: list[list[dict[str, Any]]] | None) -> str:
    if not windows:
        return "legacy"
    return " | ".join(
        ",".join(
            f"cn{term['radius']:g}{'<' if term.get('below', True) else '>'}{term['center']:g}@{term['alpha']:g}"
            for term in window
        )
        for window in windows
    )


def packing_gate_windows(coordination: dict[float, Tensor], windows: list[list[dict[str, Any]]]) -> Tensor:
    """Smooth OR over windows of smooth AND over terms.

    ``g = 1 − Π_w (1 − Π_t σ(±α_t (c_t − CN_{R_t})))``. With one window and one
    term ``cn10<C0@α`` this is exactly :func:`packing_gate`. Every factor is in
    (0, 1), so ``g`` is bounded and differentiable; only geometry enters.
    """
    if not windows:
        raise ValueError("windows must not be empty")
    off: Tensor | None = None
    for window in windows:
        g_w: Tensor | None = None
        for term in window:
            cn = coordination[float(term["radius"])]
            x = float(term["alpha"]) * (float(term["center"]) - cn)
            if not term.get("below", True):
                x = -x
            s = torch.sigmoid(x)
            g_w = s if g_w is None else g_w * s
        assert g_w is not None
        off = (1.0 - g_w) if off is None else off * (1.0 - g_w)
    assert off is not None
    return 1.0 - off


def smooth_coordination(
    positions: Tensor,
    cell: Tensor,
    cutoffs: tuple[float, ...] | list[float],
    edge_index: Tensor | None = None,
    cell_shifts: Tensor | None = None,
) -> Tensor:
    """Plateau-cosine neighbor counts at one or more real-space cutoffs.

    Each neighbor inside ``0.8 R`` contributes 1; the weight falls as a
    raised cosine from ``0.8 R`` to ``R``. Self-images are excluded.
    Shape ``(N, n_cutoffs)``. Diamond first-neighbor Si–Si (~2.35 Å) then
    counts as 4 at ``R = 3`` Å, while ``R = 10`` Å sees cage-scale packing.
    """
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    radii = [float(r) for r in cutoffs]
    if not radii or any(r <= 0 for r in radii):
        raise ValueError("cutoffs must be positive")
    rmax = max(radii)
    if edge_index is None:
        edge_index, cell_shifts = build_periodic_neighbor_graph(positions, cell, rmax)
    displacement = periodic_displacements(positions, edge_index, cell, cell_shifts)
    distance = torch.linalg.vector_norm(displacement, dim=-1)
    center = edge_index.to(device=positions.device, dtype=torch.long)[0]
    n_atoms = positions.shape[0]
    out = positions.new_zeros(n_atoms, len(radii))
    for k, radius in enumerate(radii):
        x = distance / radius
        inner = x <= 0.8
        t = ((x - 0.8) / 0.2).clamp(0.0, 1.0)
        weight = torch.where(inner, torch.ones_like(distance), 0.5 * (1.0 + torch.cos(math.pi * t)))
        weight = torch.where((distance > 0) & (distance < radius), weight, torch.zeros_like(weight))
        out[:, k].index_add_(0, center, weight)
    return out


class SmoothCompactRadialBasis(nn.Module):
    """Sine radial basis multiplied by a C2 compact polynomial envelope."""

    def __init__(self, num_basis: int, cutoff: float) -> None:
        super().__init__()
        if num_basis < 1 or cutoff <= 0:
            raise ValueError("num_basis and cutoff must be positive")
        self.num_basis = int(num_basis)
        self.cutoff = float(cutoff)
        frequencies = torch.arange(1, num_basis + 1, dtype=torch.get_default_dtype()) * math.pi
        self.register_buffer("frequencies", frequencies)

    def forward(self, distance: Tensor) -> Tensor:
        x = distance.unsqueeze(-1) / self.cutoff
        inside = x < 1
        # 1 - 10x^3 + 15x^4 - 6x^5 has zero first/second endpoint slopes.
        envelope = 1 - 10 * x.pow(3) + 15 * x.pow(4) - 6 * x.pow(5)
        basis = torch.sinc(self.frequencies.to(x) * x / math.pi)
        return torch.where(inside, envelope * basis, torch.zeros_like(basis))


class EquivariantEncoder(nn.Module):
    """Periodic l<=2 neighbor-density encoder (the v1 equivariant encoder).

    Parameters
    ----------
    num_species:
        Number of integer atom types. Types must be in ``[0, num_species)``.
    channels:
        Number of channels in each l sector.
    cutoff, num_radial:
        Radius and resolution of the compact radial expansion.

    Notes
    -----
    ``forward`` returns a dictionary with ``scalar`` (N,C), ``vector``
    (N,C,3), and ``tensor`` (N,C,3,3).  The last is exactly symmetric and
    traceless.  Rotations act only on the Cartesian axes, never on channels.
    """

    def __init__(
        self,
        num_species: int,
        channels: int = 32,
        cutoff: float = 5.0,
        num_radial: int = 8,
        l_max: int = 2,
        interactions: int = 1,
        invariant_norm: str = "sqrt",
        invariant_scale: float = 0.05,
    ) -> None:
        super().__init__()
        if invariant_norm not in INVARIANT_NORMS:
            raise ValueError(f"invariant_norm must be one of {INVARIANT_NORMS}")
        self.invariant_norm, self.invariant_scale = str(invariant_norm), float(invariant_scale)
        if num_species < 1 or channels < 1:
            raise ValueError("num_species and channels must be positive")
        if l_max not in (0, 1, 2):
            raise ValueError("v1 encoder supports l_max in {0, 1, 2}")
        self.num_species = int(num_species)
        self.channels = int(channels)
        self.cutoff = float(cutoff)
        self.l_max = int(l_max)
        self.species_embedding = nn.Embedding(num_species, channels)
        self.radial_basis = SmoothCompactRadialBasis(num_radial, cutoff)
        self.radial_projection = nn.Linear(num_radial, channels, bias=False)
        invariant_size = channels * (1 + int(l_max >= 1) + int(l_max >= 2))
        self.gating = nn.Sequential(
            nn.Linear(invariant_size, 2 * channels),
            nn.SiLU(),
            nn.Linear(2 * channels, channels * (1 + int(l_max >= 1) + int(l_max >= 2))),
        )
        if interactions not in (1, 2):
            raise ValueError("interactions must be 1 or 2")
        self.interactions = int(interactions)
        if self.interactions == 2:
            # Second interaction: neighbours enter through their own first-layer features (bond connectivity,
            # molecular ions, second shell), not only through their species embedding.
            self.radial_projection2 = nn.Linear(num_radial, channels, bias=False)
            self.neighbor_mix = nn.Linear(invariant_size, channels)
            nn.init.zeros_(self.neighbor_mix.weight)
            nn.init.zeros_(self.neighbor_mix.bias)
            self.gating2 = nn.Sequential(
                nn.Linear(invariant_size, 2 * channels),
                nn.SiLU(),
                nn.Linear(2 * channels, channels * (1 + int(l_max >= 1) + int(l_max >= 2))),
            )

    @staticmethod
    def _scatter_sum(values: Tensor, index: Tensor, size: int) -> Tensor:
        output = values.new_zeros((size,) + values.shape[1:])
        output.index_add_(0, index, values)
        return output

    @staticmethod
    def _select_valid_edges(
        edge_index: Tensor,
        displacement: Tensor,
        distance: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        keep = torch.nonzero(valid, as_tuple=False).reshape(-1)
        return (
            edge_index.index_select(1, keep),
            displacement.index_select(0, keep),
            distance.index_select(0, keep),
        )

    def forward(
        self,
        positions: Tensor,
        species: Tensor,
        cell: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        cell_shifts: Optional[Tensor] = None,
        pbc: bool | Tensor = True,
    ) -> Dict[str, Tensor]:
        """Encode one structure; optionally consume a precomputed graph."""
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("positions must have shape (N, 3)")
        species = species.to(device=positions.device, dtype=torch.long)
        if species.shape != (positions.shape[0],):
            raise ValueError("species must have shape (N,)")
        if edge_index is None:
            edge_index, cell_shifts = build_periodic_neighbor_graph(
                positions, cell, self.cutoff, pbc=pbc
            )
        edge_index = edge_index.to(device=positions.device, dtype=torch.long)
        displacement = periodic_displacements(positions, edge_index, cell, cell_shifts)
        distance = torch.linalg.vector_norm(displacement, dim=-1)
        valid = (distance > torch.finfo(distance.dtype).eps) & (distance < self.cutoff)
        edge_index, displacement, distance = self._select_valid_edges(
            edge_index, displacement, distance, valid
        )
        center, neighbor = edge_index

        unit = displacement / distance.clamp_min(torch.finfo(distance.dtype).eps).unsqueeze(-1)
        radial = self.radial_projection(self.radial_basis(distance))
        weights = radial * self.species_embedding(species.index_select(0, neighbor))
        n_atoms = positions.shape[0]
        scalar_moment = self._scatter_sum(weights, center, n_atoms)
        scalar = self.species_embedding(species) + scalar_moment
        features: Dict[str, Tensor] = {}

        vector = None
        if self.l_max >= 1:
            vector = self._scatter_sum(weights[..., None] * unit[:, None, :], center, n_atoms)
        tensor = None
        if self.l_max >= 2:
            eye = torch.eye(3, device=positions.device, dtype=positions.dtype)
            st = unit[:, :, None] * unit[:, None, :] - eye / 3
            tensor = self._scatter_sum(weights[..., None, None] * st[:, None], center, n_atoms)

        invariants = [scalar]
        if vector is not None:
            invariants.append(norm_invariant((vector * vector).sum(-1), self.invariant_norm, self.invariant_scale))
        if tensor is not None:
            invariants.append(norm_invariant((tensor * tensor).sum((-1, -2)), self.invariant_norm, self.invariant_scale))
        gates = self.gating(torch.cat(invariants, dim=-1))
        chunks = gates.split(self.channels, dim=-1)
        features["scalar"] = scalar + chunks[0]
        offset = 1
        if vector is not None:
            features["vector"] = vector * torch.sigmoid(chunks[offset])[..., None]
            offset += 1
        if tensor is not None:
            features["tensor"] = tensor * torch.sigmoid(chunks[offset])[..., None, None]
        if self.interactions == 2:
            features = self._second_interaction(features, center, neighbor, distance, unit, n_atoms)
        return features

    def _invariants(self, features: Dict[str, Tensor]) -> Tensor:
        parts = [features["scalar"]]
        if "vector" in features:
            parts.append(norm_invariant((features["vector"] ** 2).sum(-1), self.invariant_norm, self.invariant_scale))
        if "tensor" in features:
            parts.append(norm_invariant((features["tensor"] ** 2).sum((-1, -2)), self.invariant_norm, self.invariant_scale))
        return torch.cat(parts, dim=-1)

    def _second_interaction(self, features, center, neighbor, distance, unit, n_atoms):
        h = self.neighbor_mix(self._invariants(features))                       # (N, C) invariant, environment-aware
        w = self.radial_projection2(self.radial_basis(distance)) * h.index_select(0, neighbor)
        out = {"scalar": features["scalar"] + self._scatter_sum(w, center, n_atoms)}
        if "vector" in features:
            out["vector"] = features["vector"] + self._scatter_sum(w[..., None] * unit[:, None, :], center, n_atoms)
        if "tensor" in features:
            eye = torch.eye(3, device=unit.device, dtype=unit.dtype)
            st = unit[:, :, None] * unit[:, None, :] - eye / 3
            out["tensor"] = features["tensor"] + self._scatter_sum(w[..., None, None] * st[:, None], center, n_atoms)
        chunks = self.gating2(self._invariants(out)).split(self.channels, dim=-1)
        out["scalar"] = out["scalar"] + chunks[0]
        offset = 1
        if "vector" in out:
            out["vector"] = out["vector"] * torch.sigmoid(chunks[offset])[..., None]
            offset += 1
        if "tensor" in out:
            out["tensor"] = out["tensor"] * torch.sigmoid(chunks[offset])[..., None, None]
        return out


# Descriptive alias retained for callers that want to pin the implementation.
NeighborDensityEquivariantEncoder = EquivariantEncoder

