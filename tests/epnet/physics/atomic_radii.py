"""Valence orbital radii from a small all-electron radial Kohn-Sham solver (LDA; non-relativistic or scalar-relativistic ZORA).

Used to place the nonlocal projectors of each element where its own valence orbitals live, instead of
scaling one set of Si-tuned constants by the covalent radius. For every element the neutral atom is
solved self-consistently (spherical, spin-unpolarized, ground-state configuration), and the radius of
the maximum of r^2 |R_nl(r)|^2 is reported for the valence s and p orbitals (an unoccupied valence p,
e.g. Na 3p, is taken as the lowest bound p eigenstate of the neutral-atom potential) and, for the d block,
the occupied (n-1)d orbital (``valence_radii_spd``).

Method: grid in x = ln r, u(r) = r R(r) = r^{1/2} v(x), which turns the radial equation into the
symmetric generalized problem  -1/2 v'' + [r^2 (V + l(l+1)/(2 r^2)) + 1/8] v = eps r^2 v.  Atomic
units internally. Non-relativistic by default: vs ZORA the 6s radii of Tl-Bi are ~13% too large (5s of In-I
and Cs 6s ~4%, 6p ~3%); the d-block entries of the radii table use ZORA (2026-10-19).

Scalar-relativistic option (``relativistic=True``): zeroth-order regular approximation (ZORA), kinetic
energy (1/2) p.K(r) p with K = 1 / (1 - V/(2 c^2)) (V = full Kohn-Sham potential, energy independent), i.e.

    -1/2 (K u')' + [K l(l+1)/(2 r^2) + K'/(2 r) + V] u = eps u        (' = d/dr, u = r R)

(+K'/(2r): from the quadratic form (1/2) int K |grad psi|^2 = (1/2) int K [(u' - u/r)^2 + l(l+1) u^2/r^2] dr;
the same R'-coupling as Koelling-Harmon with M -> 1/K). In x = ln r with v = u / sqrt(r) the energy is

    int [ K/2 v_x^2 + (K/8 + K_x/4 + K l(l+1)/2 + r^2 V) v^2 ] dx,   norm int r^2 v^2 dx,

discretized as a symmetric tridiagonal generalized problem (K at midpoints for the stiffness, central
difference for K_x). With K = 1 it is exactly the non-relativistic matrix, so ``relativistic=False`` keeps
the original discretization (and radii). For hydrogenic ions the ZORA s levels are exactly
2 c^2 E_D / (2 c^2 + E_D) (E_D = Dirac level), which the tests check.

The lowest k eigenpairs of the tridiagonal problem are found by bisection + inverse iteration
(LAPACK stebz/stein on the M^{-1/2} A M^{-1/2} form) instead of ARPACK shift-invert: same discrete
problem (eigenvalues agree to ~1e-10 Ha), ~1000x faster for heavy atoms.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import scipy.sparse as sp
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import eigh_tridiagonal
from scipy.sparse.linalg import eigsh

BOHR_A = 0.529177210903
HARTREE_EV = 27.211386
C_LIGHT = 137.035999        # speed of light, atomic units (1 / fine-structure constant)
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb "
           "Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi").split()
# Madelung (aufbau) filling order; exceptions irrelevant for s/p main-group elements
ORDER = ["1s", "2s", "2p", "3s", "3p", "4s", "3d", "4p", "5s", "4d", "5p", "6s", "4f", "5d", "6p"]
CAP = {"s": 2, "p": 6, "d": 10, "f": 14}
L_OF = {"s": 0, "p": 1, "d": 2, "f": 3}
# Experimental neutral ground-state configurations that break the Madelung order (d block, Z <= 83 without
# the lanthanides): {symbol: {shell: occupation}}; a shell set to 0 is removed from the configuration.
CONFIG_EXCEPTIONS = {
    "Cr": {"3d": 5, "4s": 1}, "Cu": {"3d": 10, "4s": 1},
    "Nb": {"4d": 4, "5s": 1}, "Mo": {"4d": 5, "5s": 1}, "Ru": {"4d": 7, "5s": 1}, "Rh": {"4d": 8, "5s": 1},
    "Pd": {"4d": 10, "5s": 0}, "Ag": {"4d": 10, "5s": 1},
    "Pt": {"5d": 9, "6s": 1}, "Au": {"5d": 10, "6s": 1},
}
# d block covered by valence_radii_spd / epnet/data/valence_orbital_radii.json (2026-10-19)
D_ELEMENTS = ("Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg").split()


def configuration(z: int) -> list[tuple[int, int, float]]:
    """Neutral ground-state configuration [(n, l, occupation)]: Madelung order plus ``CONFIG_EXCEPTIONS``."""
    out, left = [], float(z)
    for orb in ORDER:
        if left <= 0:
            break
        occ = min(CAP[orb[1]], left)
        out.append((int(orb[0]), L_OF[orb[1]], occ))
        left -= occ
    exc = CONFIG_EXCEPTIONS.get(SYMBOLS[z - 1]) if 1 <= z <= len(SYMBOLS) else None
    if exc:
        occ = {(n, l): o for n, l, o in out}
        for shell, o in exc.items():
            occ[(int(shell[0]), L_OF[shell[1]])] = float(o)
        out = [(n, l, occ[(n, l)]) for n, l, _ in out if occ[(n, l)] > 0]
        if abs(sum(o for _, _, o in out) - z) > 1e-12:
            raise AssertionError(f"configuration exception for Z={z} does not conserve the electron count")
    return out


def _lda_xc(rho: np.ndarray) -> np.ndarray:
    """Perdew-Zunger LDA exchange-correlation potential (Hartree)."""
    rho = np.maximum(rho, 1e-30)
    rs = (3.0 / (4.0 * math.pi * rho)) ** (1.0 / 3.0)
    vx = -(3.0 / math.pi * rho) ** (1.0 / 3.0)
    g, b1, b2 = -0.1423, 1.0529, 0.3334
    a, b, c, d = 0.0311, -0.048, 0.0020, -0.0116
    sq = np.sqrt(rs)
    ec_hi = g / (1 + b1 * sq + b2 * rs)
    vc_hi = ec_hi * (1 + 7.0 / 6.0 * b1 * sq + 4.0 / 3.0 * b2 * rs) / (1 + b1 * sq + b2 * rs)
    lnrs = np.log(rs)
    vc_lo = a * lnrs + (b - a / 3.0) + 2.0 / 3.0 * c * rs * lnrs + (2 * d - c) / 3.0 * rs
    return vx + np.where(rs >= 1.0, vc_hi, vc_lo)


def zora_k(v: np.ndarray, c_light: float = C_LIGHT) -> np.ndarray:
    """ZORA kinetic factor K(r) = 1 / (1 - V(r) / (2 c^2)) (= 2c^2 / (2c^2 - V))."""
    return 1.0 / (1.0 - v / (2.0 * c_light ** 2))


def _radial_matrix(v: np.ndarray, r: np.ndarray, h: float, l: int, kfac: np.ndarray | None = None):
    """Tridiagonal (diag, offdiag) of A in A v = eps r^2 v (x = ln r grid, v = u / sqrt(r), Dirichlet ends).

    ``kfac`` = ZORA K on the grid (None: non-relativistic, K = 1). Stiffness uses K at midpoints
    (K_{i+1/2} = (K_i + K_{i+1})/2, ghost values K_{-1/2} = K_0, K_{n-1/2} = K_{n-1}); K_x/4 is the central
    difference (K_{i+1/2} - K_{i-1/2}) / (4h). K = 1 reproduces the non-relativistic matrix exactly."""
    n = len(r)
    if kfac is None:
        diag = 1.0 / h ** 2 + r ** 2 * v + (l + 0.5) ** 2 / 2.0
        off = -0.5 / h ** 2 * np.ones(n - 1)
        return diag, off
    kfac = np.asarray(kfac, dtype=float)
    kmid = np.empty(n + 1)
    kmid[1:-1] = 0.5 * (kfac[1:] + kfac[:-1])
    kmid[0], kmid[-1] = kfac[0], kfac[-1]
    diag = ((kmid[:-1] + kmid[1:]) / (2.0 * h ** 2) + (kmid[1:] - kmid[:-1]) / (4.0 * h)
            + kfac * (0.125 + 0.5 * l * (l + 1)) + r ** 2 * v)
    off = -kmid[1:-1] / (2.0 * h ** 2)
    return diag, off


def radial_states(v: np.ndarray, r: np.ndarray, h: float, l: int, k: int, z: float,
                  kfac: np.ndarray | None = None, eig: str = "tridiagonal"):
    """Lowest k radial eigenstates of potential v(r) for angular momentum l: [(eps, u normalized)].

    ``kfac``: ZORA K(r) on the grid (``zora_k(v)``) for the scalar-relativistic operator; None = non-relativistic.
    ``eig``: "tridiagonal" (bisection + inverse iteration, default) or "arpack" (the original shift-invert eigsh)."""
    diag, off = _radial_matrix(v, r, h, l, kfac)
    if eig == "arpack":
        a = sp.diags([diag, off, off], [0, 1, -1], format="csc")
        m = sp.diags(r ** 2, 0, format="csc")
        eps, vec = eigsh(a, k=k, M=m, sigma=-0.5 * z ** 2 - 1.0, which="LM")   # shift-invert below the lowest level
        order = np.argsort(eps); eps, vec = eps[order], vec[:, order]
    elif eig == "tridiagonal":
        # standard form T = M^{-1/2} A M^{-1/2} (M = diag r^2), eigenvector w = r v; the default stebz tolerance
        # (eps * |T|_1, |T| ~ 1/(h r_min)^2) would be far too loose, so ask for 1e-13 Ha explicitly
        eps, w = eigh_tridiagonal(diag / r ** 2, off / (r[:-1] * r[1:]), select="i", select_range=(0, k - 1),
                                  lapack_driver="stebz", tol=1e-13)
        vec = w / r[:, None]
    else:
        raise ValueError("eig is 'tridiagonal' or 'arpack'")
    out = []
    for j in range(k):
        u = np.sqrt(r) * vec[:, j]
        u = u / math.sqrt(np.trapz(u ** 2 * r, dx=h))                    # int u^2 dr = int u^2 r dx
        if u[np.argmax(np.abs(u))] < 0:
            u = -u
        out.append((float(eps[j]), u))
    return out


def solve_atom(z: int, n_grid: int = 3000, r_max: float = 60.0, iters: int = 300, mix: float = 0.25, bare: bool = False,
               relativistic: bool = False, c_light: float = C_LIGHT, tol: float = 1e-8,
               conf: list[tuple[int, int, float]] | None = None, info: dict | None = None):
    """Self-consistent neutral atom (or bare -Z/r if ``bare``). Returns (r [bohr], {(n,l): (eps [Ha], u)}, conf).

    ``relativistic``: scalar-relativistic ZORA with K built from the current Kohn-Sham potential (``c_light`` a.u.).
    SCF: linear density mixing ``mix``, stopped when max_x |rho_out - rho_in| 4 pi r^3 < ``tol`` or after ``iters``.
    ``conf`` overrides ``configuration(z)``; a dict passed as ``info`` receives iterations / converged / residual /
    h (x-grid step) / potential (the V on the grid that the returned orbitals belong to)."""
    x = np.linspace(math.log(1e-5 / z), math.log(r_max), n_grid)
    h = x[1] - x[0]
    r = np.exp(x)
    conf = configuration(z) if conf is None else conf
    lmax_states = {l: max([nn for nn, ll, _ in conf if ll == l], default=l) - l + 2 for l in sorted({0, 1, 2} | {ll for _, ll, _ in conf})}
    v = -(1.0 + (z - 1.0) * np.exp(-r * z ** (1.0 / 3.0))) / r           # screened start
    if bare:
        v = -z / r
    rho_old = None
    orbs = {}
    res, it, v_orbs = float("nan"), 0, v
    for it in range(1 if bare else iters):
        orbs, v_orbs = {}, v
        kfac = zora_k(v, c_light) if relativistic else None
        for l, k in lmax_states.items():
            for j, (e, u) in enumerate(radial_states(v, r, h, l, k, z, kfac=kfac)):
                orbs[(l + 1 + j, l)] = (e, u)
        if bare:
            break
        rho = np.zeros_like(r)
        for n, l, occ in conf:
            rho += occ * orbs[(n, l)][1] ** 2 / (4 * math.pi * r ** 2)
        if rho_old is not None:
            res = float(np.max(np.abs(rho - rho_old) * 4 * math.pi * r ** 3))
            if res < tol:
                break
            rho = mix * rho + (1 - mix) * rho_old
        q_in = cumulative_trapezoid(4 * math.pi * r ** 3 * rho, dx=h, initial=0.0)
        outer = cumulative_trapezoid((4 * math.pi * r ** 2 * rho)[::-1], dx=h, initial=0.0)[::-1]
        v = -z / r + q_in / r + outer + _lda_xc(rho)
        rho_old = rho
    if info is not None:   # potential: the V(r) [Ha] the returned orbitals are eigenstates of
        info.update(iterations=it + 1, residual=res, converged=bool(bare or res < tol), h=h, potential=v_orbs)
    return r, orbs, conf


def _peak(r: np.ndarray, u: np.ndarray) -> float:
    """Radius (Angstrom) of the grid maximum of u^2."""
    return float(r[np.argmax(u ** 2)] * BOHR_A)


@lru_cache(maxsize=None)
def valence_radii(symbol: str) -> dict[str, float]:
    """Peak radii (Angstrom) of the valence s and p orbitals of a neutral atom (r of max u(r)^2); non-relativistic."""
    z = SYMBOLS.index(symbol) + 1
    r, orbs, conf = solve_atom(z)
    n_s = max(n for n, l, _ in conf if l == 0)
    occ_p = [n for n, l, _ in conf if l == 1]
    n_p = n_s if (not occ_p or max(occ_p) < n_s) else max(occ_p)             # valence p shares n with valence s
    out = {}
    for lab, key in (("s", (n_s, 0)), ("p", (n_p, 1))):
        if key in orbs and orbs[key][0] < 0:
            u = orbs[key][1]
            out[lab] = float(r[np.argmax(u ** 2)] * BOHR_A)
            out[lab + "_eps_eV"] = orbs[key][0] * 27.211386
    out["n_s"], out["n_p"] = n_s, n_p
    return out


def _config_string(conf, shells) -> str:
    occ = {(n, l): o for n, l, o in conf}
    return " ".join(f"{n}{'spdf'[l]}{int(occ.get((n, l), 0))}" for n, l in shells)


@lru_cache(maxsize=None)
def valence_radii_spd(symbol: str, relativistic: bool = True, c_light: float = C_LIGHT, n_grid: int = 3000,
                      iters: int = 300, mix: float = 0.25, tol: float = 1e-8) -> dict:
    """Peak radii (Angstrom, r of max u^2) of the valence ns, np and occupied (n-1)d orbitals of a d-block atom.

    s: valence ns (n = n_d + 1; unoccupied for Pd 4d10 5s0, then the bound 5s of the neutral atom);
    p: np with the same n, only if bound (eps < 0); d: the occupied (n-1)d. Eigenvalues in eV (``*_eps_eV``).
    Scalar-relativistic ZORA by default (``relativistic=False``: the non-relativistic solver)."""
    z = SYMBOLS.index(symbol) + 1
    info: dict = {}
    r, orbs, conf = solve_atom(z, n_grid=n_grid, iters=iters, mix=mix, relativistic=relativistic, c_light=c_light,
                               tol=tol, info=info)
    occ_d = [n for n, l, o in conf if l == 2 and o > 0]
    if not occ_d:
        raise ValueError(f"{symbol} has no occupied d shell")
    n_d = max(occ_d)
    n_s = n_p = n_d + 1
    out: dict = {}
    for lab, key in (("s", (n_s, 0)), ("p", (n_p, 1)), ("d", (n_d, 2))):
        if key in orbs and orbs[key][0] < 0:
            out[lab] = _peak(r, orbs[key][1])
            out[lab + "_eps_eV"] = orbs[key][0] * HARTREE_EV
    out.update(n_s=n_s, n_p=n_p, n_d=n_d, config=_config_string(conf, [(n_d, 2), (n_s, 0)]),
               method="zora-lda" if relativistic else "nonrel-lda",
               scf_iterations=info["iterations"], scf_converged=info["converged"])
    return out


__all__ = ["valence_radii", "valence_radii_spd", "solve_atom", "configuration", "radial_states", "zora_k",
           "CONFIG_EXCEPTIONS", "D_ELEMENTS", "C_LIGHT"]
