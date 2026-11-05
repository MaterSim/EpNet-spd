#!/usr/bin/env python3
"""Step 6d (revised, docs/d_channel_plan.md): conditioning safeguard for the two-reference d channel, CPU.

The d level shift no longer enters the construction (it is applied as D += delta <AE_i|AE_j>), so only the extent
scale and the second reference energy can move B. Per d element, on the tabulated all-electron-tail local:
- eps_2 offset: of eps_d + {3, 4, 5, 6} eV, the one with the smallest basis-free augmentation norm lambda_max(Q G) at
  the atomic construction (G = <beta_i|beta_j>; Q the PSD part, second reference normalized);
- extent bound: the largest e (0.0125 steps up to 0.15) such that for every scale s with |s - 1| <= e both
  lambda_max(Q G) and lambda_max(|D| G) stay <= 3x their atomic values. e < 0.025 -> the element's d extent is fixed.
Readings (recorded beforehand): >= 20 of 29 elements get e >= 0.025 (a learnable extent); Au, Nb reported. Writes
eps2b_offset and d_extent_max into the npz and results/checks/two_reference_safeguard_v2.json; then
scripts/build_two_reference.py rebuilds the second reference at the chosen offsets (gate 3e).
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

D_ELEMENTS = "Sc Ti V Cr Mn Fe Co Ni Cu Zn Y Zr Nb Mo Tc Ru Rh Pd Ag Cd Hf Ta W Re Os Ir Pt Au Hg".split()
OFFSETS = (3.0, 4.0, 5.0, 6.0)
EXTENTS = np.arange(0.0125, 0.15001, 0.0125)
LIMIT = 3.0
DATA = ROOT / "epnet/data/atomic_channels"


def operator_norms(r, v, phis, aes, epss, r_cut, scale):
    """(lambda_max(Q G), lambda_max(|D| G)) of the stable two-reference channel at an extent scale (no level shift)."""
    from scipy.interpolate import CubicSpline
    from epnet.physics.atomic_channels import radial_kinetic
    inside = r <= r_cut
    sph, chis = [], []
    for ph, e in zip(phis, epss):
        p_s = CubicSpline(r, ph)(np.clip(r / scale, r[0], r[-1])) / math.sqrt(scale)
        sph.append(p_s)
        chis.append(np.where(inside, (e - v) * p_s - radial_kinetic(r, p_s, 2), 0.0))
    B = np.array([[np.trapz(sph[i] * chis[j] * inside, r) for j in range(2)] for i in range(2)])
    if abs(np.linalg.det(B)) < 1e-12:
        return np.inf, np.inf
    M = np.array([[np.trapz(aes[i] * aes[j] * inside, r) for j in range(2)] for i in range(2)])
    P = np.array([[np.trapz(phis[i] * phis[j] * inside, r) for j in range(2)] for i in range(2)])   # scale-invariant
    lam, vec = np.linalg.eigh(M - P)
    Q = (vec * np.clip(lam, 0.0, None)) @ vec.T
    beta = np.linalg.inv(B).T @ np.stack(chis)
    G = np.array([[np.trapz(beta[i] * beta[j], r) for j in range(2)] for i in range(2)])
    D = B + Q * np.asarray(epss)[None, :]
    return float(np.max(np.linalg.eigvals(Q @ G).real)), float(np.max(np.abs(np.linalg.eigvals(D @ G))))


def one(symbol: str) -> dict:
    from epnet.diagnostics.pseudo_atom import local_radial
    from epnet.models.general_epm import prior_q_max
    from epnet.physics.atomic_channels import atomic_orbitals, radial_form_factor, second_reference
    from epnet.physics.atomic_radii import BOHR_A

    z = dict(np.load(DATA / f"{symbol}.npz"))
    orb = atomic_orbitals(symbol, ecut_ry=20.0, normalize=False)[2]
    r = orb.r
    q = np.linspace(0.0, prior_q_max([20.0, 30.0]), 4001)
    v = local_radial(radial_form_factor(z["vloc"], r, q), q, r)
    r_cut = max(orb.rc_bohr * BOHR_A, float(z["r_loc"])) + 0.3
    inside = r <= r_cut
    refs, out = {}, {"element": symbol, "offsets": {}}
    for off in OFFSETS:
        eps2, ph2, ae2 = second_reference(symbol, orb, off)
        sc = 1.0 / math.sqrt(np.trapz(ae2**2 * inside, r))
        refs[off] = ([orb.u, ph2 * sc], [orb.u_ae, ae2 * sc], [orb.eps_eV, eps2])
        lq, ld = operator_norms(r, v, *refs[off], r_cut, 1.0)
        out["offsets"][str(off)] = {"lam_QG": lq, "lam_DG": ld}
    best = min(OFFSETS, key=lambda o: out["offsets"][str(o)]["lam_QG"])
    lq0, ld0 = out["offsets"][str(best)]["lam_QG"], out["offsets"][str(best)]["lam_DG"]
    e_max, trace = 0.0, []
    for e in EXTENTS:
        worst = 0.0
        for s in (1.0 - e, 1.0 + e):
            lq, ld = operator_norms(r, v, *refs[best], r_cut, s)
            worst = max(worst, lq / lq0, ld / ld0)
        trace.append([float(e), worst])
        if worst <= LIMIT:
            e_max = float(e)
        else:
            break
    e_max = e_max if e_max >= 0.025 else 0.0
    out.update(offset=best, lam_QG=lq0, lam_DG=ld0, d_extent_max=e_max, extent_trace=trace, extent_fixed=e_max == 0.0)
    z.update(eps2b_offset=best, d_extent_max=e_max)
    np.savez(DATA / f"{symbol}.npz", **z)
    return out


def main() -> None:
    with ProcessPoolExecutor(29) as ex:
        rows = list(ex.map(one, D_ELEMENTS))
    (ROOT / "results/checks/two_reference_safeguard_v2.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{'El':3s} " + " ".join(f"lamQG@+{o:.0f}" for o in OFFSETS) + "  chosen  extent bound")
    for x in rows:
        lams = " ".join(f"{x['offsets'][str(o)]['lam_QG']:9.2f}" for o in OFFSETS)
        print(f"{x['element']:3s} {lams}   +{x['offset']:.0f} eV  +-{100 * x['d_extent_max']:.2f}%" + ("  FIXED" if x["extent_fixed"] else ""))
    n = sum(not x["extent_fixed"] for x in rows)
    print(f"elements with a learnable extent: {n}/29 -> {'SUPPORTED' if n >= 20 else 'not supported'} (>= 20); "
          f"fixed: {[x['element'] for x in rows if x['extent_fixed']]}")


if __name__ == "__main__":
    main()
