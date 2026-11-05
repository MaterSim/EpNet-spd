"""d character of predicted bands (EpNet-spd d-channel plan, docs/d_channel_plan.md, T1).

For an eigenvector psi_nk of a model's plane-wave Hamiltonian, the d character of band n at k is its weight on the
Bloch sums of the phi_am, with phi_am the pseudo-d orbital of d atom a (epnet/data/atomic_channels, step 3a; u = r R,
int u^2 dr = 1) times the real l = 2 harmonic m. Its Bloch sum has the plane-wave components
<k+G|phi_am> = sqrt(4 pi) F(|k+G|) A_m(k+G) e^{-i(k+G).tau_a} / sqrt(Omega), up to the global phase (-i)^2, with
F(q) = int r u(r) j_2(q r) dr and A_m = sqrt(4 pi) Y_2m the projectors' angular factors (sum_m A_m^2 = 5). An isolated
orbital has sum_G |<k+G|phi>|^2 = (2 / pi) int q^2 F^2 dq = 1, less the part beyond the cutoff. Orbitals of
neighbouring atoms overlap (Bloch sums are not orthonormal), so the character is the weight on the d subspace,
<psi|Phi O^-1 Phi^dagger|psi> with O(k) = Phi^dagger Phi in the plane-wave basis: in [0, 1], and the plain sum of
|<phi_am|psi>|^2 when the orbitals do not overlap. With ultrasoft atomic channels (H psi = e S psi) every inner
product takes the S metric (<a|S|b>; psi S-normalized), so an atomic d state psi = phi has character 1.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
from scipy.special import spherical_jn

from epnet.data.valence import D_ELEMENTS
from epnet.models.general_epm import d_harmonics

DATA = Path(__file__).resolve().parents[1] / "data" / "atomic_channels"


def d_form_factor(symbol: str, q: np.ndarray, n_table: int = 4001) -> np.ndarray:
    """F(q) = int r u_d(r) j_2(q r) dr of the element's pseudo-d orbital. F depends on |q| only: it is tabulated once on a
    uniform grid to max(q) and linearly interpolated (relative error ~1e-6 vs direct quadrature at every q)."""
    z = np.load(DATA / f"{symbol}.npz")
    r, u = z["r"], z["u2"]
    q = np.asarray(q, dtype=float)
    grid = np.linspace(0.0, float(q.max()) + 1e-6, n_table)
    table = np.trapz((r * u)[None, :] * spherical_jn(2, np.outer(grid, r)), r, axis=1)
    return np.interp(q.reshape(-1), grid, table).reshape(q.shape)


def band_d_character(model, structure, kpoints: np.ndarray, n_bands: int, k_chunk: int = 16) -> tuple[np.ndarray, np.ndarray]:
    """(energies (k, n_bands) in eV, d character (k, n_bands)) from the model's dense Hamiltonians, ``k_chunk``
    k-points at a time (the dense matrices of a long path do not fit in memory at once)."""
    kpoints = np.asarray(kpoints)
    if len(kpoints) > k_chunk:
        parts = [_band_d_character(model, structure, kpoints[i:i + k_chunk], n_bands) for i in range(0, len(kpoints), k_chunk)]
        return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
    return _band_d_character(model, structure, kpoints, n_bands)


def _band_d_character(model, structure, kpoints: np.ndarray, n_bands: int) -> tuple[np.ndarray, np.ndarray]:
    """One chunk of k-points."""
    if model.config.get("d_overlap"):
        raise NotImplementedError("d character with the d overlap term (eigenvectors of the transformed problem)")
    device = model.element_features.device                    # the model's device (GPU scoring, Stage 5a)
    packed = model.pack(structure)
    packed = SimpleNamespace(**{n: (v.to(device) if torch.is_tensor(v) else v) for n, v in vars(packed).items()})
    k = torch.as_tensor(np.asarray(kpoints), dtype=packed.lattice.dtype, device=device)
    with torch.no_grad():
        hams, cache = model.band_path(packed, k, n_bands=n_bands, return_hamiltonians=True)
    d_atoms = [i for i, s in enumerate(structure.species) if s in D_ELEMENTS]
    recip = 2.0 * torch.pi * torch.linalg.inv(packed.lattice).transpose(0, 1)
    n_k = len(k)
    g_idx = cache.g_idx_k if getattr(cache, "g_idx_k", None) is not None else cache.g_idx[None].expand(n_k, -1, -1)
    momentum = (cache.k_frac.to(recip) @ recip)[:, None, :] + g_idx.to(recip) @ recip        # (k, g, 3), as projectors()
    valid = cache.k_valid if getattr(cache, "k_valid", None) is not None else torch.ones(momentum.shape[:2], dtype=torch.bool, device=device)
    qn = torch.linalg.vector_norm(momentum, dim=-1)
    ang = d_harmonics(momentum / qn.clamp_min(1e-8)[..., None])                               # (k, g, 5)
    radial = {s: torch.as_tensor(d_form_factor(s, qn.cpu().numpy()), dtype=qn.dtype, device=device) * valid.to(qn.dtype)
              for s in {structure.species[a] for a in d_atoms}}
    energies, chars, start = [], [], 0
    for h, s_mat in hams:
        sl = slice(start, start + h.shape[0])
        start += h.shape[0]
        if s_mat is None:
            e, v = torch.linalg.eigh(h)
        else:                                                                                  # H v = e S v, v^H S v = 1
            chol = torch.linalg.cholesky(s_mat)
            y = torch.linalg.solve_triangular(chol, h, upper=False)
            e, w = torch.linalg.eigh(torch.linalg.solve_triangular(chol, y.mH, upper=False))
            v = torch.linalg.solve_triangular(chol.mH, w, upper=True)
        e, v = e[:, :n_bands], v[:, :, :n_bands]                                              # (kc, g, n)
        if not d_atoms:
            energies.append(e)
            chars.append(torch.zeros(e.shape, dtype=e.dtype, device=e.device))
            continue
        phis = []
        for a in d_atoms:
            phase = torch.exp(-1j * (momentum[sl] @ packed.positions[a]))
            phi = (math.sqrt(4.0 * math.pi) * radial[structure.species[a]][sl])[..., None] * ang[sl] * phase[..., None]
            phis.append(phi / math.sqrt(float(packed.volume)))                                # (kc, g, 5)
        phi = torch.cat(phis, dim=-1).to(v.dtype)                                             # (kc, g, 5 n_d)
        s_phi = phi if s_mat is None else s_mat @ phi
        overlap = phi.mH @ s_phi                                                              # O(k), S metric
        proj = s_phi.mH @ v                                                                   # <phi|S|psi>
        char = (proj.conj() * torch.linalg.solve(overlap, proj)).real.sum(1)                 # <psi|P_d|psi>
        energies.append(e)
        chars.append(char)
    return torch.cat(energies).cpu().numpy(), torch.cat(chars).cpu().numpy()


__all__ = ["band_d_character", "d_form_factor"]
