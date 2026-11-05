"""Compact cosine-collocation empirical pseudopotential baseline."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class CosineRadialEP(nn.Module):
    """Species-dependent smooth radial potential on ``[0, q_cutoff]``.

    Coefficients multiply a cosine basis and a raised-cosine envelope, making
    both the value and first derivative vanish at the reciprocal cutoff.
    """

    def __init__(self, n_species: int, n_basis: int = 12, q_cutoff: float = 12.0) -> None:
        super().__init__()
        if n_species < 1 or n_basis < 1 or q_cutoff <= 0:
            raise ValueError("n_species, n_basis, and q_cutoff must be positive")
        self.n_species = int(n_species)
        self.n_basis = int(n_basis)
        self.q_cutoff = float(q_cutoff)
        self.coefficients = nn.Parameter(0.02 * torch.randn(n_species, n_basis))

    def basis(self, q: Tensor) -> Tensor:
        x = (q / self.q_cutoff).clamp(0, 1)
        envelope = 0.5 * (1.0 + torch.cos(math.pi * x))
        orders = torch.arange(self.n_basis, device=q.device, dtype=q.dtype)
        return envelope.unsqueeze(-1) * torch.cos(math.pi * x.unsqueeze(-1) * orders)

    def forward(self, species_index: Tensor, q: Tensor) -> Tensor:
        basis = self.basis(q)
        coeff = self.coefficients[species_index.to(dtype=torch.long)]
        return (coeff * basis).sum(dim=-1)

    def per_atom(self, species_index: Tensor, q: Tensor) -> Tensor:
        """Return ``v_Z(q)`` with shape ``(n_atoms, n_q)``."""
        if species_index.ndim != 1:
            raise ValueError("species_index must have shape (n_atoms,)")
        basis = self.basis(q)
        coeff = self.coefficients[species_index.to(dtype=torch.long)]
        return torch.einsum("nb,qb->nq", coeff, basis)

    def fit_collocation(self, species_index: int, q: Tensor, values: Tensor, ridge: float = 1e-10) -> None:
        basis = self.basis(q)
        gram = basis.T @ basis + ridge * torch.eye(self.n_basis, device=q.device, dtype=q.dtype)
        rhs = basis.T @ values.to(basis)
        coeff = torch.linalg.solve(gram, rhs)
        with torch.no_grad():
            self.coefficients[species_index].copy_(coeff)

    def collocate_from(self, other: "CosineRadialEP", n_q: int = 81, ridge: float = 1e-10) -> None:
        """Match ``other``'s \(v_Z(q)\) on a uniform grid (wider/narrower basis)."""
        if int(other.n_species) != int(self.n_species):
            raise ValueError("collocate_from requires the same n_species")
        q_max = min(float(self.q_cutoff), float(other.q_cutoff))
        q = torch.linspace(0.0, q_max, int(n_q), device=other.coefficients.device, dtype=other.coefficients.dtype)
        for i in range(self.n_species):
            idx = torch.full(q.shape, i, dtype=torch.long, device=q.device)
            self.fit_collocation(i, q, other(idx, q), ridge=ridge)


class EnvironmentRadialDelta(nn.Module):
    """Map SO(3)-invariant local features to a cosine-basis isotropic Δv(q).

    The last linear layer is zero-initialized so a freshly attached module
    leaves the species CosineRadialEP unchanged.
    """

    def __init__(self, in_features: int, n_basis: int, hidden: int = 32) -> None:
        super().__init__()
        if in_features < 1 or n_basis < 1 or hidden < 1:
            raise ValueError("in_features, n_basis, and hidden must be positive")
        self.in_features = int(in_features)
        self.n_basis = int(n_basis)
        self.net = nn.Sequential(
            nn.Linear(self.in_features, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.n_basis),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, invariants: Tensor) -> Tensor:
        if invariants.ndim != 2 or invariants.shape[-1] != self.in_features:
            raise ValueError(f"invariants must have shape (N, {self.in_features})")
        return self.net(invariants)


class DensityConditionedSphereDelta(nn.Module):
    """Per-atom polynomial correction of the species cosine coefficients in local density.

    ``Δc_{a,n} = Σ_{k=1..order} C_{k,n} t_a^k`` with ``t_a = (CN10_a − center) / scale``.
    Zero-initialised, so attaching it reproduces the species sphere exactly; the
    zeroth order stays in ``CosineRadialEP``. This is the EPM volume-scaling idea
    made smooth and per atom: no step, no family name, no encoder features.
    """

    def __init__(self, n_basis: int, order: int = 2, center: float = 151.0, scale: float = 30.0) -> None:
        super().__init__()
        if n_basis < 1 or order < 1 or scale <= 0:
            raise ValueError("n_basis and order must be positive and scale > 0")
        self.n_basis = int(n_basis)
        self.order = int(order)
        self.center = float(center)
        self.scale = float(scale)
        self.coefficients = nn.Parameter(torch.zeros(self.order, self.n_basis))

    def reduced_density(self, cn10: Tensor) -> Tensor:
        return (cn10 - self.center) / self.scale

    def forward(self, cn10: Tensor) -> Tensor:
        if cn10.ndim != 1:
            raise ValueError("cn10 must have shape (N,)")
        t = self.reduced_density(cn10)
        powers = torch.stack([t.pow(k) for k in range(1, self.order + 1)], dim=-1)
        return powers @ self.coefficients.to(powers)


def silicon_cohen_form_factor(q: Tensor) -> Tensor:
    """Cohen–Bergstresser 1966 Si atomic form factor in eV (q in Å⁻¹).

    Discrete symmetric form factors are converted with ``v(q) = V_S Ω / 2``
    using the diamond primitive volume ``Ω = a³/4``, ``a = 5.431 Å``:

    ``|G| = √3, √8, √11 × 2π/a`` and ``V_S = −0.21, +0.04, +0.08 Ry``.
    """
    ry = 13.605693122994
    a_ang = 5.431
    omega = (a_ang**3) / 4.0
    two_pi_over_a = 2.0 * math.pi / a_ang
    knots = q.new_tensor(
        [0.0, two_pi_over_a * math.sqrt(3.0), two_pi_over_a * math.sqrt(8.0), two_pi_over_a * math.sqrt(11.0), 6.5]
    )
    values = q.new_tensor([-0.21, -0.21, 0.04, 0.08, 0.0]) * (ry * omega / 2.0)
    qabs = q.to(dtype=values.dtype).abs()
    idx = torch.searchsorted(knots, qabs, right=True).clamp(1, knots.numel() - 1)
    left = idx - 1
    span = (knots[idx] - knots[left]).clamp_min(1e-8)
    t = ((qabs - knots[left]) / span).clamp(0, 1)
    # Raised-cosine interpolation so the collocation target is C1 at the stars.
    w = 0.5 * (1.0 - torch.cos(math.pi * t))
    return torch.where(qabs >= 6.5, torch.zeros_like(qabs), (1.0 - w) * values[left] + w * values[idx])


def screened_empty_core_form_factor(q: Tensor, z_valence: float, r_core: float, k_tf: float = 2.0) -> Tensor:
    """Thomas-Fermi screened Ashcroft empty core in eV·Å³ (q in Å⁻¹).

    ``v(q) = -4π Z e² cos(q r_c) / (q² + k_TF²)``. Finite at q = 0 and scaled
    by the element's valence, so anions start strongly attractive and cations
    weak: a species-dependent start that already orders ionic bands correctly.
    ``k_TF = 2 Å⁻¹`` is the Thomas-Fermi wavevector of a 0.2 e/Å³ valence gas.
    """
    e2 = 27.211386245988 * 0.529177210903  # e²/(4πε0) in eV·Å
    qabs = q.abs()
    return -4.0 * math.pi * float(z_valence) * e2 * torch.cos(qabs * float(r_core)) / (qabs.square() + float(k_tf) ** 2)


def empty_core_form_factor(q: Tensor, v0: float = -20.0, r_core: float = 1.2) -> Tensor:
    """Small Ashcroft empty-core guess in the same eV·Å³ units as Cohen Si.

    ``v(q) = v0 cos(q R_c) exp(-(q/4)²)``. Used only as a multi-element
    collocation start; ``V(G=0)`` is still zeroed in the Hamiltonian gauge.
    """
    qabs = q.abs()
    return qabs.new_tensor(v0) * torch.cos(qabs * r_core) * torch.exp(-(qabs / 4.0) ** 2)
