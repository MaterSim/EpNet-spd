"""Pseudo-atom diagnostic for a GeneralEPM species (EpNet-spd d-channel plan, docs/d_channel_plan.md, T2).

One species of a trained model, isolated: its local form factor v(q) and its separable s/p/d projectors define a
spherical pseudo-atom. Each angular-momentum channel l is solved in a spherical-Bessel basis j_l(k_n r) on a sphere,
truncated at the same kinetic cutoff as the crystal's plane waves, so the spectrum is the one the crystal basis sees.

Conventions (crystal -> isolated atom):
- local: V_GG' = v(G - G') S / Omega, so the atomic potential is V(r) = (1 / 2 pi^2) int q^2 v(q) j_0(q r) dq;
- nonlocal: V_GG' = sum D f(k+G) f(k+G')* / Omega with f(q) = rad_l(q R) sqrt(4 pi) Y_lm(q_hat) e^{-i q.tau}. In the
  isolated limit this is the radial kernel (D / pi^3) B_l(r) B_l(r') with B_l(r) = int q^2 rad_l(q R) j_l(q r) dq,
  rad_0 = e^{-x^2/2}, rad_1 = x e^{-x^2/2}, rad_2 = x^2 e^{-x^2/2}, x = q R (closed form below).
The crystal sets V(G = 0) = 0; isolated levels relate to large-box crystal levels by + v(0) / Omega.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import eigh as scipy_eigh
from scipy.optimize import brentq
from scipy.special import spherical_jn

KIN = 3.809982116151619          # hbar^2 / 2 m_e in eV A^2 (as epnet.hamiltonian.torch_pw.KINETIC_EV_A2)


def bessel_zeros(l: int, n: int) -> np.ndarray:
    """First n positive zeros of the spherical Bessel function j_l."""
    zeros, x, step = [], 1e-6 + (l if l else 0.5), 0.05
    f_prev = spherical_jn(l, x)
    while len(zeros) < n:
        x_new = x + step
        f_new = spherical_jn(l, x_new)
        if f_prev * f_new < 0:
            zeros.append(brentq(lambda t: spherical_jn(l, t), x, x_new, xtol=1e-14))
        x, f_prev = x_new, f_new
    return np.array(zeros)


def projector_radial(l: int, radius: float, r: np.ndarray) -> np.ndarray:
    """B_l(r) = int_0^inf q^2 (q R)^l exp(-q^2 R^2 / 2) j_l(q r) dq, in closed form:
    R^l sqrt(pi) / 2^(l+2) r^l a^-(l+3/2) exp(-r^2 / 4a) with a = R^2 / 2."""
    a = radius * radius / 2.0
    return radius**l * math.sqrt(math.pi) / 2.0 ** (l + 2) * r**l * a ** (-(l + 1.5)) * np.exp(-r * r / (4.0 * a))


def local_radial(v_q: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """V(r) = (1 / 2 pi^2) int q^2 v(q) j_0(q r) dq (trapezoid on the given q grid)."""
    integrand = (q * q * v_q)[None, :] * spherical_jn(0, np.outer(r, q))
    return np.trapz(integrand, q, axis=1) / (2.0 * math.pi**2)


@dataclass
class Level:
    energy_eV: float
    mean_r_A: float
    peak_r_A: float
    basis_spread_eV: float = 0.0    # |E(r_max) - E(r_max + 4 A)|: large for states at the edge of the cutoff basis


@dataclass
class PseudoAtom:
    """Spherical pseudo-atom: local v(q) on ``q``, and per l a list of (D, R) separable terms; ``overlaps`` adds
    separable S terms (ultrasoft), solved as H c = e S c."""

    q: np.ndarray
    v_q: np.ndarray
    channels: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    overlaps: dict[int, list[tuple[float, float]]] = field(default_factory=dict)
    matrices: dict[int, tuple] = field(default_factory=dict)   # l -> (shapes, D (n, n), Q (n, n)): coupled projectors
    r_max: float = 12.0
    n_r: int = 6000

    def __post_init__(self) -> None:
        self.r = np.linspace(self.r_max / self.n_r, self.r_max, self.n_r)
        self.v_r = local_radial(self.v_q, self.q, self.r)

    def basis(self, l: int, ecut_ry: float) -> tuple[np.ndarray, np.ndarray]:
        """(k_n, phi_n(r)) of the spherical-Bessel basis with hbar^2 k^2 / 2m <= ecut (phi normalized on the sphere)."""
        k_cut = math.sqrt(ecut_ry) / 0.529177210903
        n_guess = int(k_cut * self.r_max / math.pi) + l + 4
        z = bessel_zeros(l, n_guess)
        k = z / self.r_max
        z, k = z[k <= k_cut], k[k <= k_cut]
        norm = math.sqrt(2.0) / (self.r_max**1.5 * np.abs(spherical_jn(l + 1, z)))
        phi = norm[:, None] * spherical_jn(l, np.outer(k, self.r))
        return k, phi

    def solve_checked(self, l: int, ecut_ry: float, local_only: bool = False, n_levels: int = 6,
                      dr_max: float = 4.0) -> list[Level]:
        """``solve`` plus each level's sensitivity to the sphere radius (r_max vs r_max + dr_max): a state converged in
        the cutoff barely moves; a compact state living at the basis edge (a ghost signature) moves by eV."""
        levels = self.solve(l, ecut_ry, local_only, n_levels)
        other = PseudoAtom(self.q, self.v_q, self.channels, self.overlaps, self.matrices, r_max=self.r_max + dr_max,
                           n_r=int(self.n_r * (self.r_max + dr_max) / self.r_max))
        alt = other.solve(l, ecut_ry, local_only, n_levels)
        for a, b in zip(levels, alt):
            a.basis_spread_eV = abs(a.energy_eV - b.energy_eV)
        return levels

    def _projector(self, l: int, shape) -> np.ndarray:
        """B_l(r) on self.r: a float is a Gaussian-type radius R (closed form); a (r_grid, values) pair is a table."""
        if isinstance(shape, (int, float, np.floating)):
            return projector_radial(l, float(shape), self.r)
        grid, values = shape
        return np.interp(self.r, grid, values, right=0.0)

    def solve_states(self, l: int, ecut_ry: float, local_only: bool = False, n_levels: int = 6):
        """(levels, radial functions R_j(r) on self.r) of channel l."""
        k, phi = self.basis(l, ecut_ry)
        w = self.r * self.r
        h = np.diag(KIN * k * k) + np.einsum("nr,r,mr->nm", phi, w * self.v_r, phi) * (self.r[1] - self.r[0])
        s = np.eye(len(k))
        if not local_only:
            for strength, shape in self.channels.get(l, []):
                p = (phi * (w * self._projector(l, shape))[None, :]).sum(1) * (self.r[1] - self.r[0])
                h = h + strength / math.pi**3 * np.outer(p, p)
            for strength, shape in self.overlaps.get(l, []):
                p = (phi * (w * self._projector(l, shape))[None, :]).sum(1) * (self.r[1] - self.r[0])
                s = s + strength / math.pi**3 * np.outer(p, p)
            if l in self.matrices:                          # sum_ij p_i (D_ij / pi^3) p_j, same for S
                shapes, d_mat, q_mat = self.matrices[l]
                ps = np.stack([(phi * (w * self._projector(l, sh))[None, :]).sum(1) * (self.r[1] - self.r[0]) for sh in shapes])
                h = h + ps.T @ (np.asarray(d_mat) / math.pi**3) @ ps
                s = s + ps.T @ (np.asarray(q_mat) / math.pi**3) @ ps
        self.last_s_min = float(np.linalg.eigvalsh(s).min())
        e, c = scipy_eigh(0.5 * (h + h.T), 0.5 * (s + s.T))      # c^T S c = 1
        out, radials = [], []
        for j in range(min(n_levels, len(e))):
            radial = c[:, j] @ phi                      # R(r)
            dens = w * radial * radial
            out.append(Level(float(e[j]), float((self.r * dens).sum() / dens.sum()), float(self.r[np.argmax(dens)])))
            radials.append(radial)
        return out, radials

    def solve(self, l: int, ecut_ry: float, local_only: bool = False, n_levels: int = 6) -> list[Level]:
        return self.solve_states(l, ecut_ry, local_only, n_levels)[0]


# ---------------------------------------------------------------------------------------------------- from a model
def model_q_grid(model, q_max: float | None = None, n_q: int = 4001) -> np.ndarray:
    c = model.config
    q_max = q_max or max(c["sphere_q_cutoff"], c.get("sphere_tail_q_cutoff", 0.0) if c.get("sphere_tail_n_basis") else 0.0,
                         20.0, float(model.prior_q[-1]) if getattr(model, "prior_table", None) is not None else 0.0)
    return np.linspace(0.0, q_max, n_q)


def species_form_factor(model, index: int, q: np.ndarray) -> np.ndarray:
    """Species base local form factor v_Z(q) (eV A^3): cosine basis plus the high-q tail, as in local_form_factor."""
    import torch
    from epnet.models.general_epm import cosine_basis

    qt = torch.as_tensor(q, dtype=torch.float64)
    with torch.no_grad():
        coeff, _ = model.species_parameters()
        v = cosine_basis(qt, model.n_basis, model.config["sphere_q_cutoff"]) @ coeff[index].double()
        tail = model.species_tail()
        if tail is not None:
            v = v + cosine_basis(qt, model.n_tail, model.config["sphere_tail_q_cutoff"]) @ tail[index].double()
        if getattr(model, "prior_table", None) is not None:
            v = v + model.prior_at(qt)[index].double()
    return v.numpy()


def species_radii(model, index: int) -> np.ndarray:
    """Real-space radius R (A) of each (radial function r, l): x = q R in the projector radial functions."""
    import torch

    c = model.config
    if c.get("projector_q_width_max") is not None or c.get("projector_radii_firstrow") is not None:
        raise NotImplementedError("pseudo-atom radii for projector_q_width_max / projector_radii_firstrow")
    with torch.no_grad():
        if c.get("projector_radius_mode") == "orbital":
            radius = model.orbital_multipliers[:, None] * model.orbital_radii[index][None, :]
            if model.projector_log_mult is not None:
                radius = (radius * torch.exp(model.projector_log_mult)[:, None]
                          * torch.exp(0.3 * torch.tanh(model.projector_log_offset[index]))[None, :])
        else:
            radius = (model.r_cov[index] / model.width_constants)[:, None].expand(-1, model.n_l)
    return radius.double().numpy()


def species_strengths(model, index: int) -> np.ndarray:
    """(R, n_l) species projector strengths D (eV A^3), with first-row l = 2 removed as in projector_strengths."""
    import torch

    with torch.no_grad():
        _, d = model.species_parameters()
        d = d[index].double().clone()
        if model.config.get("firstrow_lmax1") and model.n_l > 2 and bool(model.first_row[index]):
            d[:, 2] = 0.0
    return d.numpy()


def in_crystal_terms(model, structure, atom: int, q: np.ndarray) -> dict:
    """Per-atom isotropic additions in a given crystal: the routed l = 0 readout form factor on ``q`` and the per-atom
    projector strengths (species + head, as projector_strengths). The charge-transfer term is not included."""
    import torch

    packed = model.pack(structure)
    with torch.no_grad():
        k = torch.zeros(1, 3, dtype=packed.lattice.dtype)
        cache = model.make_pw_cache(packed, k)
        feats = model.encoder(packed.positions, packed.species, packed.lattice, edge_index=cache.edge_index,
                              cell_shifts=cache.cell_shifts)
        weights = model.mixture_weights(cache)[atom]
        qt = torch.as_tensor(q, dtype=feats["scalar"].dtype)
        mono = torch.zeros_like(qt)
        for w, ro in zip(weights, model.readouts):
            amp = ro.channel_readout[0](feats["scalar"][atom]) + ro.radial_bias[0]
            m = ro.radial_basis(qt) @ amp
            if ro.scalar_cosine is not None:
                m = m + ro.cosine_basis(qt) @ ro.scalar_cosine(feats["scalar"][atom])
            mono = mono + w * m
        strengths = model.projector_strengths(packed, cache, feats, None)[atom]
    return {"mono_q": mono.double().numpy(), "strengths": strengths.double().numpy(),
            "species_index": int(packed.species[atom])}


def model_pseudo_atom(model, index: int, extra: dict | None = None, r_max: float = 12.0) -> PseudoAtom:
    """Pseudo-atom of species ``index``: species-only, or with ``extra`` from in_crystal_terms (that atom's routed
    readout monopole added to v(q), its per-atom strengths replacing the species strengths)."""
    q = model_q_grid(model)
    v = species_form_factor(model, index, q)
    strengths = species_strengths(model, index)
    if extra is not None:
        v = v + extra["mono_q"]
        strengths = extra["strengths"]
    radii = species_radii(model, index)
    channels = {l: [(float(strengths[r, l]), float(radii[r, l])) for r in range(strengths.shape[0])]
                for l in range(strengths.shape[1])}
    return PseudoAtom(q, v, channels, r_max=r_max)


__all__ = ["KIN", "Level", "PseudoAtom", "bessel_zeros", "in_crystal_terms", "local_radial", "model_pseudo_atom",
           "model_q_grid", "projector_radial", "species_form_factor", "species_radii", "species_strengths"]
