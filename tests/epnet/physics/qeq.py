"""Charge equilibration (QEq) for periodic cells.

Charges minimise  E(q) = sum_a chi_a q_a + 1/2 sum_a J_a q_a^2 + 1/2 sum_{a,b} q_a q_b C_ab
subject to sum_a q_a = total, where C is the periodic Coulomb interaction between
Gaussian charge clouds of width sigma_a (each atom's interaction with its own periodic
images included; its self-energy is carried by the hardness J_a). The solution is one
linear solve and is differentiable in chi and J.

C is evaluated by Ewald summation: erf(r/s)/r = [erf(r/s) - erf(r/eta)]/r + erf(r/eta)/r
with s = sqrt(sigma_a^2 + sigma_b^2) < eta; the first part is short ranged (real space),
the second is summed in reciprocal space. The G = 0 term is dropped, which shifts every
C_ab by the same constant and leaves the neutral solution unchanged.

Screening: the Coulomb matrix enters scaled by kappa (about 1/eps_inf, 0 < kappa < 1). Unscreened
point-ion lattice sums outweigh the hardness in ionic crystals (NaCl: Madelung about -8.9 eV e^-2
against (J_Na + J_Cl)/2 = 7.2 eV), which makes E(q) unbounded and the stationary point a saddle
with the wrong charge signs; electronic screening of partial charges in a solid restores a minimum.

Starting electronegativity and hardness: Mulliken values chi = (IE + EA)/2 and
J = IE - EA from first ionisation energies and electron affinities (eV); for Li and F
these reproduce Rappe & Goddard (1991) to 0.01 eV and 0.9 eV.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

COULOMB_EV_ANGSTROM = 14.399645

# (first ionisation energy, electron affinity) in eV; unbound anions (Be, N, Mg) set to 0.
IE_EA = {
    "H": (13.598, 0.754), "Li": (5.392, 0.618), "Be": (9.323, 0.0), "B": (8.298, 0.280), "C": (11.260, 1.262),
    "N": (14.534, 0.0), "O": (13.618, 1.461), "F": (17.423, 3.401), "Na": (5.139, 0.548), "Mg": (7.646, 0.0),
    "Al": (5.986, 0.433), "Si": (8.152, 1.390), "P": (10.487, 0.747), "S": (10.360, 2.077), "Cl": (12.968, 3.613),
    "K": (4.341, 0.501), "Ca": (6.113, 0.025), "Ga": (5.999, 0.430), "Ge": (7.900, 1.233), "As": (9.789, 0.814),
    "Se": (9.752, 2.021), "Br": (11.814, 3.364), "Rb": (4.177, 0.486), "Sr": (5.695, 0.048), "In": (5.786, 0.384),
    "Sn": (7.344, 1.112), "Sb": (8.608, 1.047), "Te": (9.010, 1.971), "I": (10.451, 3.059), "Cs": (3.894, 0.472),
    "Ba": (5.212, 0.145), "Tl": (6.108, 0.377), "Pb": (7.417, 0.364), "Bi": (7.286, 0.942),
}


def mulliken_parameters(symbols) -> tuple[Tensor, Tensor]:
    """(chi, J) in eV for each symbol: chi = (IE + EA)/2, J = IE - EA."""
    missing = [s for s in symbols if s not in IE_EA]
    if missing:
        raise ValueError(f"no IE/EA for {missing}")
    ie = torch.tensor([IE_EA[s][0] for s in symbols], dtype=torch.float32)
    ea = torch.tensor([IE_EA[s][1] for s in symbols], dtype=torch.float32)
    return 0.5 * (ie + ea), ie - ea


def coulomb_matrix(positions: Tensor, lattice: Tensor, sigma: Tensor, *, eta: float = 2.0,
                   r_cut: float | None = None, g_cut: float | None = None) -> Tensor:
    """Periodic Coulomb matrix C (eV per e^2) between Gaussian charges, shape (N, N).

    ``positions`` (N, 3) Cartesian in Angstrom, ``lattice`` (3, 3) rows, ``sigma`` (N,) widths.
    Computed in float64 without gradients (it depends on geometry only).
    """
    with torch.no_grad():
        pos = positions.detach().to(torch.float64).cpu()
        lat = lattice.detach().to(torch.float64).cpu()
        sig = sigma.detach().to(torch.float64).cpu()
        if float(sig.max()) * math.sqrt(2.0) >= eta:
            raise ValueError("eta must exceed the largest pair width")
        r_cut = r_cut if r_cut is not None else 4.0 * eta
        g_cut = g_cut if g_cut is not None else 8.0 / eta
        n = pos.shape[0]
        s2 = sig[:, None] ** 2 + sig[None, :] ** 2  # (N, N)
        s = torch.sqrt(s2)
        # real space: images within r_cut of every pair
        heights = 1.0 / torch.linalg.norm(torch.linalg.inv(lat).T, dim=1)  # plane spacings
        nmax = [int(math.ceil(r_cut / float(h))) + 1 for h in heights]
        rng = [torch.arange(-m, m + 1, dtype=torch.float64) for m in nmax]
        shifts = torch.stack(torch.meshgrid(*rng, indexing="ij"), dim=-1).reshape(-1, 3) @ lat  # (M, 3)
        d = pos[None, :, None, :] - pos[:, None, None, :] + shifts[None, None, :, :]  # (N, N, M, 3)
        r = torch.linalg.norm(d, dim=-1)
        mask = (r > 1e-8) & (r < r_cut)
        r_safe = torch.where(mask, r, torch.ones_like(r))
        short = (torch.erf(r_safe / s[..., None]) - torch.erf(r_safe / eta)) / r_safe
        real = torch.where(mask, short, torch.zeros_like(short)).sum(-1)
        # reciprocal space for erf(r/eta)/r, G != 0
        recip_lat = 2.0 * math.pi * torch.linalg.inv(lat).T
        gmax = [int(math.ceil(g_cut / float(torch.linalg.norm(recip_lat[i])))) + 1 for i in range(3)]
        grng = [torch.arange(-m, m + 1, dtype=torch.float64) for m in gmax]
        g = torch.stack(torch.meshgrid(*grng, indexing="ij"), dim=-1).reshape(-1, 3) @ recip_lat
        g2 = (g * g).sum(-1)
        keep = (g2 > 1e-12) & (g2 < g_cut**2)
        g, g2 = g[keep], g2[keep]
        volume = float(torch.abs(torch.linalg.det(lat)))
        weight = 4.0 * math.pi / volume * torch.exp(-(eta**2) * g2 / 4.0) / g2  # (K,)
        phase = torch.cos((pos[:, None, :] - pos[None, :, :]) @ g.T)  # (N, N, K)
        recip = (phase * weight).sum(-1)
        recip = recip - torch.eye(n, dtype=torch.float64) * (2.0 / (math.sqrt(math.pi) * eta))  # drop r=0 self term
        return (COULOMB_EV_ANGSTROM * (real + recip)).to(torch.float32)


def qeq_charges(coulomb: Tensor, chi: Tensor, hardness: Tensor, total: float = 0.0, kappa: Tensor | float = 0.2) -> Tensor:
    """Solve the QEq linear system with Coulomb screening ``kappa``; differentiable in chi, hardness, kappa."""
    n = chi.shape[0]
    h = kappa * coulomb.to(chi) + torch.diag(hardness)
    top = torch.cat([h, torch.ones(n, 1, dtype=chi.dtype, device=chi.device)], dim=1)
    bottom = torch.cat([torch.ones(1, n, dtype=chi.dtype, device=chi.device), torch.zeros(1, 1, dtype=chi.dtype, device=chi.device)], dim=1)
    rhs = torch.cat([-chi, torch.tensor([float(total)], dtype=chi.dtype, device=chi.device)])
    return torch.linalg.solve(torch.cat([top, bottom], dim=0), rhs)[:n]


__all__ = ["COULOMB_EV_ANGSTROM", "IE_EA", "coulomb_matrix", "mulliken_parameters", "qeq_charges"]
