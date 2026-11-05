"""Scattering of an atomic channel vs the all-electron atom (EpNet-spd d-channel plan, docs/d_channel_plan.md, step 4).

The l-channel logarithmic derivative L(E) = u'(R) / u(R) at a radius R, for
- the all-electron ZORA-LDA atom (``epnet.physics.atomic_radii``): its self-consistent potential, integrated outward on
  the solver's own x = ln r discretization (the recursion of the tridiagonal operator whose eigenvectors are the
  atom's orbitals, so the bound levels are reproduced exactly);
- a separable (KB / ultrasoft) channel: (T + V - E) u + chi (d - E s) <chi|u> = 0, solved without a basis as
  u = u_h + k u_p with (T + V - E) u_h = 0, (T + V - E) u_p = -chi (Numerov, outward) and
  k = (d - E s) <chi|u_h> / (1 - (d - E s) <chi|u_p>).
Both are compared through the phase theta(E) = arctan(R L(E)), independent of the sign of u and unwrapped with period
pi across the poles of L (u(R) = 0); it falls monotonically with E. The energy
error dE(E) = E' - E is defined by theta_model(E') = theta_AE(E) (Hamann, PRB 88, 085117 (2013), Sec. III: log
derivatives as the transferability test). Units: eV and Angstrom.
"""

from __future__ import annotations

import math

import numpy as np

from epnet.physics.atomic_radii import BOHR_A, HARTREE_EV, SYMBOLS, _radial_matrix, solve_atom, zora_k

KIN = 3.809982116151619          # hbar^2 / 2 m_e (eV A^2)


def _phase(u: np.ndarray, du: np.ndarray, radius: float) -> np.ndarray:
    """arctan(R u'/u) over an energy grid (u, du at R per energy), unwrapped with period pi."""
    theta = np.arctan(radius * du / u)
    return np.unwrap(2.0 * theta) / 2.0


def ae_phase(symbol: str, l: int, energies_eV: np.ndarray, radius_A: float) -> np.ndarray:
    """All-electron phase theta(E) of channel l at radius R (scalar-relativistic ZORA-LDA neutral atom)."""
    info: dict = {}
    r, _, _ = solve_atom(SYMBOLS.index(symbol) + 1, relativistic=True, info=info)
    v, h = info["potential"], info["h"]
    diag, off = _radial_matrix(v, r, h, l, zora_k(v))
    energies = np.asarray(energies_eV, dtype=float) / HARTREE_EV
    i_r = int(np.searchsorted(r, radius_A / BOHR_A))
    n = i_r + 2
    vv = np.zeros((n, energies.size))
    vv[0] = 1.0
    vv[1] = -(diag[0] - energies * r[0] ** 2) * vv[0] / off[0]          # Dirichlet: v_{-1} = 0
    for i in range(1, n - 1):
        vv[i + 1] = -(off[i - 1] * vv[i - 1] + (diag[i] - energies * r[i] ** 2) * vv[i]) / off[i]
        big = np.abs(vv[i + 1]) > 1e100
        if big.any():                                                   # rescale (the phase is scale-free)
            vv[:, big] *= 1e-100
    u = np.sqrt(r[:n])[:, None] * vv
    j = i_r
    rr = r[j] * BOHR_A                                                  # interpolate to R on the log grid
    t = (math.log(radius_A / BOHR_A) - math.log(r[j - 1])) / h
    u_r = (1 - t) * u[j - 1] + t * u[j]
    du_dx = (u[j] - u[j - 2]) / (2 * h) * (1 - t) + (u[j + 1] - u[j - 1]) / (2 * h) * t
    du_dr = du_dx / (radius_A / BOHR_A) / BOHR_A                        # per Angstrom
    del rr
    return _phase(u_r, du_dr, radius_A)


def channel_phase(r: np.ndarray, v_loc: np.ndarray, l: int, chi_u: np.ndarray, d: float, s: float,
                  energies_eV: np.ndarray, radius_A: float, r_int_A: float = 6.0) -> np.ndarray:
    """Phase theta(E) at radius R of the channel T + V_loc + |chi> d <chi| with overlap 1 + |chi> s <chi| (u-form,
    uniform grid ``r`` in Angstrom starting at r[0] = dr)."""
    energies = np.asarray(energies_eV, dtype=float)
    dr = r[1] - r[0]
    n = int(np.searchsorted(r, r_int_A))
    rr, vv, cc = r[:n], v_loc[:n], chi_u[:n]
    f = l * (l + 1) / rr[:, None] ** 2 + (vv[:, None] - energies[None, :]) / KIN          # u'' = f u + g
    g = np.repeat((cc / KIN)[:, None], energies.size, axis=1)
    w = dr * dr / 12.0

    def numerov(u0, u1, gg):
        u = np.zeros((n, energies.size))
        u[0], u[1] = u0, u1
        for i in range(1, n - 1):
            u[i + 1] = (2 * u[i] * (1 + 5 * w * f[i]) - u[i - 1] * (1 - w * f[i - 1])
                        + w * (gg[i + 1] + 10 * gg[i] + gg[i - 1])) / (1 - w * f[i + 1])
        return u

    zero = np.zeros(energies.size)
    u_h = numerov(rr[0] ** (l + 1) + zero, rr[1] ** (l + 1) + zero, np.zeros_like(g))
    u_p = numerov(zero, zero, g)
    ch = cc[:, None]
    a_h, a_p = np.trapz(ch * u_h, rr, axis=0), np.trapz(ch * u_p, rr, axis=0)
    lam = d - energies * s
    k = lam * a_h / (1.0 - lam * a_p)
    u = u_h + k[None, :] * u_p
    j = int(np.searchsorted(rr, radius_A))
    t = (radius_A - rr[j - 1]) / dr
    u_r = (1 - t) * u[j - 1] + t * u[j]
    du = (1 - t) * (u[j] - u[j - 2]) / (2 * dr) + t * (u[j + 1] - u[j - 1]) / (2 * dr)
    return _phase(u_r, du, radius_A)


def energy_error(energies_eV: np.ndarray, theta_ae: np.ndarray, theta_model: np.ndarray) -> np.ndarray:
    """dE(E) = E' - E with theta_model(E') = theta_AE(E); NaN where theta_AE(E) is outside the model's range or the
    model phase is not monotonic."""
    e = np.asarray(energies_eV, dtype=float)
    tm = np.asarray(theta_model)
    if not np.all(np.diff(tm) < 1e-12):
        return np.full_like(e, np.nan)
    e_prime = np.interp(-np.asarray(theta_ae), -tm, e, left=np.nan, right=np.nan)
    return e_prime - e





def ae_radial(symbol: str, l: int, energy_eV: float, r_max_A: float = 4.0) -> tuple[np.ndarray, np.ndarray]:
    """All-electron ZORA-LDA radial solution u(r) (regular at the origin, arbitrary scale) of channel l at a given
    energy, on the solver's log grid (r in bohr) up to ``r_max_A``: the same outward recursion as ``ae_phase``."""
    info: dict = {}
    r, _, _ = solve_atom(SYMBOLS.index(symbol) + 1, relativistic=True, info=info)
    v, h = info["potential"], info["h"]
    diag, off = _radial_matrix(v, r, h, l, zora_k(v))
    e = energy_eV / HARTREE_EV
    n = int(np.searchsorted(r, r_max_A / BOHR_A)) + 2
    vv = np.zeros(n)
    vv[0] = 1.0
    vv[1] = -(diag[0] - e * r[0] ** 2) * vv[0] / off[0]
    for i in range(1, n - 1):
        vv[i + 1] = -(off[i - 1] * vv[i - 1] + (diag[i] - e * r[i] ** 2) * vv[i]) / off[i]
        if abs(vv[i + 1]) > 1e100:
            vv[: i + 2] *= 1e-100
    return r[:n], np.sqrt(r[:n]) * vv


def multi_channel_phase(r: np.ndarray, v_loc: np.ndarray, l: int, betas: np.ndarray, d_mat: np.ndarray, q_mat: np.ndarray,
                        energies_eV: np.ndarray, radius_A: float, r_int_A: float = 6.0) -> np.ndarray:
    """Phase theta(E) at R of T + V_loc + sum_ij |beta_i> D_ij <beta_j| with overlap 1 + sum_ij |beta_i> Q_ij <beta_j|
    (u-form; ``betas`` (n_proj, n_r)). u = u_h + sum_i w_i u_p,i with (T + V - E) u_p,i = -beta_i, w = K c,
    c = (1 - P K)^-1 h, K = D - E Q, h_j = <beta_j|u_h>, P_ji = <beta_j|u_p,i>."""
    energies = np.asarray(energies_eV, dtype=float)
    dr = r[1] - r[0]
    n = int(np.searchsorted(r, r_int_A))
    rr, vv, bb = r[:n], v_loc[:n], np.atleast_2d(betas)[:, :n]
    n_p = bb.shape[0]
    f = l * (l + 1) / rr[:, None] ** 2 + (vv[:, None] - energies[None, :]) / KIN
    w = dr * dr / 12.0

    def numerov(u0, u1, gg):
        u = np.zeros((n, energies.size))
        u[0], u[1] = u0, u1
        for i in range(1, n - 1):
            u[i + 1] = (2 * u[i] * (1 + 5 * w * f[i]) - u[i - 1] * (1 - w * f[i - 1])
                        + w * (gg[i + 1] + 10 * gg[i] + gg[i - 1])) / (1 - w * f[i + 1])
        return u

    zero = np.zeros(energies.size)
    u_h = numerov(rr[0] ** (l + 1) + zero, rr[1] ** (l + 1) + zero, np.zeros((n, energies.size)))
    u_p = [numerov(zero, zero, np.repeat((bb[i] / KIN)[:, None], energies.size, axis=1)) for i in range(n_p)]
    h = np.stack([np.trapz(bb[j][:, None] * u_h, rr, axis=0) for j in range(n_p)])                   # (n_p, E)
    P = np.stack([[np.trapz(bb[j][:, None] * u_p[i], rr, axis=0) for i in range(n_p)] for j in range(n_p)])   # (j, i, E)
    u = u_h.copy()
    for k in range(energies.size):
        K = d_mat - energies[k] * q_mat
        c = np.linalg.solve(np.eye(n_p) - P[:, :, k] @ K, h[:, k])
        wk = K @ c
        for i in range(n_p):
            u[:, k] += wk[i] * u_p[i][:, k]
    j = int(np.searchsorted(rr, radius_A))
    t = (radius_A - rr[j - 1]) / dr
    u_r = (1 - t) * u[j - 1] + t * u[j]
    du = (1 - t) * (u[j] - u[j - 2]) / (2 * dr) + t * (u[j + 1] - u[j - 1]) / (2 * dr)
    return _phase(u_r, du, radius_A)
__all__ = ["ae_phase", "ae_radial", "channel_phase", "energy_error", "multi_channel_phase"]
