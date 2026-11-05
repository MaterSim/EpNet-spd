"""Atomic s/p/d channels for d elements (EpNet-spd d-channel plan, docs/d_channel_plan.md, Step 3a).

For an element, the scalar-relativistic (ZORA-LDA) neutral atom of ``epnet/physics/atomic_radii.py`` gives the valence
ns, np (when bound) and occupied (n-1)d orbitals and levels. Each orbital is pseudized inside r_c into a smooth,
nodeless phi_l(r) = r^(l+1) (a + b r^2 + c r^4) that matches the all-electron value, slope and curvature at r_c and
equals it outside (norm is not conserved: EpNet fits energies only). r_c is the smallest radius in [rc_min, rc_max]
beyond the last node for which 99% of phi's plane-wave weight lies below ``ecut_ry``.

``kb_channel`` then builds the Kleinman-Bylander / Vanderbilt projector of a channel against a given radial local
potential V(r): chi = (eps - T - V) phi and D = 1 / <phi|chi>, so that T + V + |chi> D <chi| has phi as an
eigenfunction at eps (Kleinman & Bylander, PRL 48, 1425 (1982); Vanderbilt, PRB 41, 7892 (1990); Hamann, PRB 88,
085117 (2013)). Units: Angstrom and eV; radial functions in u-form (u = r R).

Ultrasoft form (step 3c, ``normalize=False``): phi is not renormalized, so it equals the all-electron orbital beyond
r_c (all-electron norm 1) and holds 1 - q inside; the deficit q is restored by an overlap S = 1 + |beta> q <beta|
(Vanderbilt 1990). Renormalizing instead (step 3a) scales the all-electron tail up by 1 / sqrt(1 - q), up to x3.8 for
the Zn 3d, and with it the neighbour overlap and the d bandwidth (Stage 1, docs/d_channel_plan.md).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import spherical_jn

from epnet.physics.atomic_radii import BOHR_A, HARTREE_EV, SYMBOLS, solve_atom

KIN = 3.809982116151619          # hbar^2 / 2 m_e (eV A^2)


@dataclass
class PseudoOrbital:
    l: int
    n: int
    eps_eV: float                # all-electron level (vacuum zero)
    rc_bohr: float
    ecut99_ry: float             # 99% of the plane-wave weight below this cutoff
    r: np.ndarray                # Angstrom, uniform grid
    u: np.ndarray                # pseudo orbital, u-form, int u^2 dr = 1 (A^-1/2)
    u_ae: np.ndarray             # all-electron orbital on the same grid, same normalization convention
    norm_deficit: float = 0.0    # q = 1 - int u^2 dr (0 when u is renormalized)


def plane_wave_cutoff(r: np.ndarray, u: np.ndarray, l: int, frac: float = 0.99) -> float:
    """Kinetic cutoff (Ry) holding ``frac`` of the plane-wave weight of the radial orbital u = r R (r in Angstrom)."""
    q = np.linspace(1e-4, 30.0, 3000)                                  # 1/A
    f = np.trapz((r * u)[None, :] * spherical_jn(l, np.outer(q, r)), r, axis=1)   # int r^2 R j_l = int r u j_l
    w = q * q * f * f
    cum = np.cumsum(w) / w.sum()
    q_frac = q[min(np.searchsorted(cum, frac), len(q) - 1)]
    return float((q_frac * BOHR_A) ** 2)                               # E (Ry) = (q bohr)^2


def kinetic_tail(r: np.ndarray, u: np.ndarray, l: int, ecut_ry: float) -> float:
    """Kinetic energy (eV) carried by the plane-wave components above ``ecut_ry``: <T> restricted to q > q_cut.
    It bounds the variational level error of the orbital in a basis truncated at that cutoff."""
    q = np.linspace(1e-4, 40.0, 4000)                                  # 1/A
    f = np.trapz((r * u)[None, :] * spherical_jn(l, np.outer(q, r)), r, axis=1)
    w = q * q * f * f
    w = w / np.trapz(w, q)
    q_cut = math.sqrt(ecut_ry) / BOHR_A
    hi = q > q_cut
    return float(np.trapz(KIN * q[hi] ** 2 * w[hi], q[hi]))


def _pseudize(r_b: np.ndarray, u_b: np.ndarray, l: int, rc: float) -> np.ndarray:
    """phi = r^(l+1) (a + b r^2 + c r^4) inside rc (bohr), matching u, u', u'' at rc; all-electron u outside."""
    spl = CubicSpline(r_b, u_b)
    u0, u1, u2 = float(spl(rc)), float(spl(rc, 1)), float(spl(rc, 2))
    p = l + 1
    g0 = u0 * rc ** -p
    g1 = u1 * rc ** -p - p * u0 * rc ** -(p + 1)
    g2 = u2 * rc ** -p - 2 * p * u1 * rc ** -(p + 1) + p * (p + 1) * u0 * rc ** -(p + 2)
    a, b, c = np.linalg.solve(np.array([[1, rc**2, rc**4], [0, 2 * rc, 4 * rc**3], [0, 2, 12 * rc**2]]), [g0, g1, g2])
    out = u_b.copy()
    inside = r_b < rc
    out[inside] = r_b[inside] ** p * (a + b * r_b[inside] ** 2 + c * r_b[inside] ** 4)
    return out


def atomic_orbitals(symbol: str, ecut_ry: float = 20.0, rc_min: float = 1.4, rc_max: float = 2.6, rc_max_sp: float = 3.6,
                    r_max_A: float = 12.0, n_r: int = 6000, max_kinetic_tail_eV: float = 0.05,
                    normalize: bool = True) -> dict[int, PseudoOrbital]:
    """Pseudo-orbitals for the valence s, p (if bound) and occupied (n-1)d of a d element. d: r_c in [rc_min, rc_max]
    (the production range of d channels); s and p are more extended (outer nodes of 5p/6p beyond 2.6 bohr), so their
    r_c may go up to rc_max_sp. r_c is chosen on the renormalized shape either way; ``normalize=False`` keeps the
    all-electron tail (ultrasoft form) and records the norm deficit."""
    z = SYMBOLS.index(symbol) + 1
    r_b, orbs, conf = solve_atom(z, relativistic=True)
    occ_d = [n for n, l, o in conf if l == 2 and o > 0]
    if not occ_d:
        raise ValueError(f"{symbol}: no occupied d shell")
    n_d = max(occ_d)
    wanted = {0: (n_d + 1, 0), 1: (n_d + 1, 1), 2: (n_d, 2)}
    r = np.linspace(r_max_A / n_r, r_max_A, n_r)
    out: dict[int, PseudoOrbital] = {}
    for l, key in wanted.items():
        if key not in orbs or orbs[key][0] >= 0:
            continue                                                    # unbound (e.g. some np): no channel
        eps_ha, u_b = orbs[key]
        u_b = u_b / math.sqrt(np.trapz(u_b * u_b, r_b))                  # int u^2 dr = 1 on the bohr grid
        if u_b[np.argmax(np.abs(u_b))] < 0:
            u_b = -u_b
        nodes = r_b[1:][np.sign(u_b[1:]) * np.sign(u_b[:-1]) < 0]
        last_node = float(nodes[nodes < 8.0].max()) if np.any(nodes < 8.0) else 0.0
        best = None
        hi = rc_max if l == 2 else max(rc_max_sp, 1.05 * last_node + 0.6)
        for rc in np.arange(max(rc_min, 1.05 * last_node + 0.05), hi + 1e-9, 0.05):
            ps = _pseudize(r_b, u_b, l, float(rc))
            if np.any(ps[r_b < rc] <= 0):                               # must be nodeless (positive) inside rc
                continue
            r_A = r_b * BOHR_A
            u_A = ps / math.sqrt(BOHR_A)                                 # u(bohr^-1/2) -> A^-1/2
            spl = CubicSpline(r_A, u_A)
            ug = spl(r)
            ug = ug / math.sqrt(np.trapz(ug * ug, r))
            # 2026-10-20: chosen by the kinetic energy above the cutoff (< max_kinetic_tail_eV), not by a 99% weight:
            # at 99% the remaining 1% of a hard 3d orbital carried up to ~3 eV of kinetic energy (first 3a run)
            e99 = plane_wave_cutoff(r, ug, l)
            best = (float(rc), e99, ug)
            if kinetic_tail(r, ug, l, ecut_ry) <= max_kinetic_tail_eV:
                break
        if best is None:
            raise ValueError(f"{symbol} l={l}: no nodeless pseudization in [{rc_min}, {rc_max}] bohr")
        rc, e99, ug = best
        ae = CubicSpline(r_b * BOHR_A, u_b / math.sqrt(BOHR_A))(r)
        ae_norm = math.sqrt(np.trapz(ae * ae, r))
        ae = ae / ae_norm
        deficit = 0.0
        if not normalize:   # back to the all-electron scale: equal to ae beyond r_c
            ps = CubicSpline(r_b * BOHR_A, _pseudize(r_b, u_b, l, rc) / math.sqrt(BOHR_A))(r) / ae_norm
            ug, deficit = ps, float(1.0 - np.trapz(ps * ps, r))
        out[l] = PseudoOrbital(l, key[0], float(eps_ha * HARTREE_EV), rc, e99, r, ug, ae, deficit)
    return out


def radial_kinetic(r: np.ndarray, u: np.ndarray, l: int) -> np.ndarray:
    """T u = KIN (-u'' + l(l+1) u / r^2) on a uniform grid (spline second derivative)."""
    spl = CubicSpline(r, u)
    return KIN * (-spl(r, 2) + l * (l + 1) * u / (r * r))


@dataclass
class KBChannel:
    l: int
    eps_eV: float
    chi_R: np.ndarray            # projector radial part chi(r) / r on the orbital grid (the B(r) of pseudo_atom)
    d_kb: float                  # H coefficient of |chi><chi|: (a + eps q) / a^2 with a = <phi|chi> (1 / a if q = 0)
    s_kb: float = 0.0            # S coefficient of |chi><chi|: q / a^2


def kb_channel(orb: PseudoOrbital, v_loc: np.ndarray, eps_shift: float = 0.0) -> KBChannel:
    """chi = (eps - T - V) phi (u-form) against the radial local potential ``v_loc`` (eV) on ``orb.r``; with the norm
    deficit q of phi, H = T + V + |chi> d_kb <chi| and S = 1 + |chi> s_kb <chi| give H phi = eps S phi and
    <phi|S|phi> = 1. In the pseudo-atom solver's convention the separable terms are (pi^3 d / pi^3) B B, B = chi / r."""
    eps = orb.eps_eV + eps_shift
    chi_u = (eps - v_loc) * orb.u - radial_kinetic(orb.r, orb.u, orb.l)
    a = float(np.trapz(orb.u * chi_u, orb.r))
    q = orb.norm_deficit
    return KBChannel(orb.l, eps, chi_u / orb.r, (a + eps * q) / a**2, q / a**2)


def ae_local_potential(symbol: str, r_A: np.ndarray, r_loc: float) -> np.ndarray:
    """Local potential (eV) on ``r_A`` equal to the self-consistent ZORA-LDA neutral-atom potential beyond ``r_loc`` (A),
    and a + b r^2 + c r^4 inside, matching value, slope and curvature at r_loc (step 5; DFT practice: Hamann 2013)."""
    info: dict = {}
    r_b, _, _ = solve_atom(SYMBOLS.index(symbol) + 1, relativistic=True, info=info)
    spl = CubicSpline(r_b * BOHR_A, info["potential"] * HARTREE_EV)
    v = spl(r_A)
    v0, v1, v2 = float(spl(r_loc)), float(spl(r_loc, 1)), float(spl(r_loc, 2))
    a, b, c = np.linalg.solve(np.array([[1, r_loc**2, r_loc**4], [0, 2 * r_loc, 4 * r_loc**3], [0, 2, 12 * r_loc**2]]),
                              [v0, v1, v2])
    inside = r_A < r_loc
    v[inside] = a + b * r_A[inside] ** 2 + c * r_A[inside] ** 4
    return v


def radial_form_factor(v_r: np.ndarray, r: np.ndarray, q: np.ndarray) -> np.ndarray:
    """v(q) = 4 pi int r^2 V(r) j_0(q r) dr (eV A^3), the inverse of V(r) = (1 / 2 pi^2) int q^2 v(q) j_0(q r) dq."""
    return 4.0 * math.pi * np.trapz((r * r * v_r)[None, :] * spherical_jn(0, np.outer(q, r)), r, axis=1)



def second_reference(symbol: str, orb: PseudoOrbital, delta_eV: float = 4.0, r_max_A: float = 6.0):
    """Second reference of a channel (Step 6): the all-electron ZORA solution at eps + delta (regular at the origin),
    pseudized at the same r_c with the same polynomial, on ``orb.r``; scaled to the first reference's all-electron
    amplitude at r_c. Beyond ``r_max_A`` both are zero (only r <= max(r_c, r_loc) enters the projectors).
    Returns (eps_2, phi_2, ae_2)."""
    from epnet.diagnostics.scattering import ae_radial

    eps2 = orb.eps_eV + delta_eV
    r_b, u_b = ae_radial(symbol, orb.l, eps2, r_max_A=r_max_A)
    ps_b = _pseudize(r_b, u_b, orb.l, orb.rc_bohr)
    r = orb.r
    lim = r <= r_b[-1] * BOHR_A
    ae2, ph2 = np.zeros_like(r), np.zeros_like(r)
    ae2[lim] = CubicSpline(r_b * BOHR_A, u_b / math.sqrt(BOHR_A))(r[lim])
    ph2[lim] = CubicSpline(r_b * BOHR_A, ps_b / math.sqrt(BOHR_A))(r[lim])
    rc = orb.rc_bohr * BOHR_A
    scale = float(np.interp(rc, r, orb.u_ae) / np.interp(rc, r, ae2))
    return eps2, ph2 * scale, ae2 * scale


def two_reference_channel(phis, aes, epss, v_loc: np.ndarray, r: np.ndarray, l: int, r_cut: float):
    """Vanderbilt (1990) multi-reference channel in u-form: chi_i = (eps_i - T - V) phi_i (zero beyond r_cut, where
    V = V_AE), B_ij = <phi_i|chi_j>, beta = B^-T chi, Q_ij = <AE_i|AE_j> - <phi_i|phi_j>, D_ij = B_ij + eps_j Q_ij.
    Returns (beta (n, n_r), D, Q, M) with M_ij = <AE_i|AE_j> (the per-atom level-shift kernel, Step 6c)."""
    inside = r <= r_cut
    chis = [np.where(inside, (e - v_loc) * ph - radial_kinetic(r, ph, l), 0.0) for ph, e in zip(phis, epss)]
    n = len(phis)
    B = np.array([[np.trapz(phis[i] * chis[j] * inside, r) for j in range(n)] for i in range(n)])
    M = np.array([[np.trapz(aes[i] * aes[j] * inside, r) for j in range(n)] for i in range(n)])
    P = np.array([[np.trapz(phis[i] * phis[j] * inside, r) for j in range(n)] for i in range(n)])
    Q = M - P
    beta = np.linalg.inv(B).T @ np.stack(chis)
    D = B + Q * np.asarray(epss)[None, :]
    return beta, D, Q, M


__all__ = ["KBChannel", "PseudoOrbital", "ae_local_potential", "atomic_orbitals", "kb_channel", "kinetic_tail", "plane_wave_cutoff",
           "radial_form_factor", "radial_kinetic", "second_reference", "two_reference_channel"]
