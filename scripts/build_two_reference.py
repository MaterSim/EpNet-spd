#!/usr/bin/env python3
"""Step 6 (6a, 6b; gate 3e) of docs/d_channel_plan.md: two-reference d channel for the 29 d elements.

6a: Y and Zr get the all-electron-tail local, choosing the r_loc candidate of the 3d gate
(results/checks/ae_local_report.json) that passes every other check with the smallest p-scattering error. All elements:
the second d reference (eps_d + 4 eV, or the element's eps2b_offset from Step 6d) is stored in epnet/data/atomic_channels/<El>.npz (u2b, ua2b, eps2b).
Gate 3e, pseudo-atom solver at 20 Ry with the d channel as a 2x2 (D, Q) block on the model's tabulated local:
d level within 0.1 eV of eps_d, no ghost (no l = 2 state below eps_d - 0.1 eV), S positive definite, basis spread
< 0.1 eV, tail within 2% of the all-electron orbital at 1.5 r_c (30 Ry). Gate >= 27/29. Writes
results/checks/two_reference_report.json.
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
DELTA, ECUT, ECUT_TAIL = 4.0, 20.0, 30.0
DATA = ROOT / "epnet/data/atomic_channels"


def pick_r_loc(symbol: str) -> tuple[float, dict]:
    """Y / Zr: the 3d-gate candidate passing every check except p scattering, with the smallest p-scattering error."""
    rep = next(x for x in json.loads((ROOT / "results/checks/ae_local_report.json").read_text()) if x["element"] == symbol)
    ok = []
    for c in rep["tried"]:
        ch = c["channels"]
        others = c["repr_ok"] and ch["0"]["pass"] and ch["2"]["pass"] and all(
            abs(x["level_err_eV"]) < 0.1 and not x["ghost"] and x["basis_spread_eV"] < 0.1 and x["s_min"] > 0
            and abs(x["tail_ratio_30ry"] - 1) < 0.02 for x in [ch["1"]])
        if others:
            ok.append((ch["1"]["scatter_max_dE_eV"], c["r_loc"]))
    if not ok:
        raise ValueError(f"{symbol}: no candidate passes the other checks")
    p_err, r_loc = min(ok)
    return r_loc, {"p_scatter_max_dE_eV": p_err, "candidates": sorted(ok)}


def one(symbol: str) -> dict:
    try:
        return _one(symbol)
    except Exception as exc:
        return {"element": symbol, "pass": False, "error": f"{type(exc).__name__}: {exc}"}


def _one(symbol: str) -> dict:
    from epnet.diagnostics.pseudo_atom import PseudoAtom, local_radial
    from epnet.models.general_epm import prior_q_max
    from epnet.physics.atomic_channels import (ae_local_potential, atomic_orbitals, kb_channel, radial_form_factor,
                                               second_reference, two_reference_channel)
    from epnet.physics.atomic_radii import BOHR_A

    path = DATA / f"{symbol}.npz"
    z = dict(np.load(path))
    orbs = atomic_orbitals(symbol, ecut_ry=ECUT, normalize=False)
    r = orbs[2].r
    out = {"element": symbol}
    if "vloc" not in z:                                   # Y, Zr (6a)
        r_loc, info = pick_r_loc(symbol)
        z["vloc"], z["r_loc"] = ae_local_potential(symbol, r, r_loc), r_loc
        out["ae_local_added"] = {"r_loc": r_loc, **info}
    q = np.linspace(0.0, prior_q_max([ECUT, 30.0]), 4001)
    v_q = radial_form_factor(z["vloc"], r, q)
    v = local_radial(v_q, q, r)                           # the local as the model represents it
    orb = orbs[2]
    delta = float(z["eps2b_offset"]) if "eps2b_offset" in z else DELTA     # Step 6d (revised): per-element offset
    eps2, ph2, ae2 = second_reference(symbol, orb, delta)
    out["eps2_offset"] = delta
    rc = orb.rc_bohr * BOHR_A
    r_cut = max(rc, float(z["r_loc"])) + 0.3
    beta, D, Q, M = two_reference_channel([orb.u, ph2], [orb.u_ae, ae2], [orb.eps_eV, eps2], v, r, 2, r_cut)
    channels, overlaps = {}, {}
    for l in (0, 1):
        if l in orbs:
            kb = kb_channel(orbs[l], v)
            channels[l] = [(math.pi**3 * kb.d_kb, (r, kb.chi_R))]
            overlaps[l] = [(math.pi**3 * kb.s_kb, (r, kb.chi_R))]
    shapes = [(r, beta[0] / r), (r, beta[1] / r)]
    atom = PseudoAtom(q, v_q, channels, overlaps, {2: (shapes, math.pi**3 * D, math.pi**3 * Q)}, r_max=12.0, n_r=6000)
    levels, _ = atom.solve_states(2, ECUT, n_levels=4)
    s_min = atom.last_s_min
    spread = atom.solve_checked(2, ECUT, n_levels=1)[0].basis_spread_eV
    _, rad30 = atom.solve_states(2, ECUT_TAIL, n_levels=1)
    tail = abs(float(np.interp(1.5 * rc, atom.r, atom.r * rad30[0]) / np.interp(1.5 * rc, r, orb.u_ae)))
    err = levels[0].energy_eV - orb.eps_eV
    ghost = abs(err) >= 0.1 and any(abs(x.energy_eV - orb.eps_eV) < 0.1 for x in levels[1:])
    out.update(eps_d=orb.eps_eV, eps2=eps2, r_loc=float(z["r_loc"]), level_err_eV=err, levels=[x.energy_eV for x in levels],
               ghost=bool(ghost), s_min=s_min, basis_spread_eV=spread, tail_ratio_30ry=tail,
               D=D.tolist(), Q=Q.tolist(), M=M.tolist(), D_asym=float(abs(D[0, 1] - D[1, 0])))
    out["pass"] = bool(abs(err) < 0.1 and not ghost and s_min > 0 and spread < 0.1 and abs(tail - 1) < 0.02)
    z.update(u2b=ph2, ua2b=ae2, eps2b=eps2)
    np.savez(path, **z)
    return out


def main() -> None:
    with ProcessPoolExecutor(29) as ex:
        rows = list(ex.map(one, D_ELEMENTS))
    (ROOT / "results/checks/two_reference_report.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(f"{'El':3s} {'eps_d':>7s} {'err':>7s} {'ghost':5s} {'Smin':>5s} {'spread':>6s} {'tail30':>6s} {'Dasym':>6s} pass   levels")
    for x in rows:
        if "error" in x:
            print(x["element"], "ERROR", x["error"]); continue
        print(f"{x['element']:3s} {x['eps_d']:7.2f} {x['level_err_eV']:+7.3f} {str(x['ghost']):5s} {x['s_min']:5.2f} "
              f"{x['basis_spread_eV']:6.3f} {x['tail_ratio_30ry']:6.3f} {x['D_asym']:6.3f} {x['pass']}  {[round(v, 2) for v in x['levels']]}"
              + (f"  [ae local added: r_loc {x['ae_local_added']['r_loc']}, p scatter {x['ae_local_added']['p_scatter_max_dE_eV']:.2f}]" if "ae_local_added" in x else ""))
    n = sum(x.get("pass", False) for x in rows)
    print(f"gate 3e: {n}/29 -> {'PASSED' if n >= 27 else 'FAILED'} (>= 27); failing: {[x['element'] for x in rows if not x.get('pass')]}")


if __name__ == "__main__":
    main()
