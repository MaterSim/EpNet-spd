#!/usr/bin/env python3
"""Step 5 (3d gate) of docs/d_channel_plan.md: all-electron-tail local potential for the 29 d elements.

Per element, r_loc is the smallest of R_LOC for which every channel passes (checks recorded before the run):
- at 20 Ry: level within 0.1 eV of eps_l, no ghost (the lowest state of channel l is the physical one), basis spread
  < 0.1 eV, S positive definite;
- at 30 Ry: the S-normalized state within 2% of the all-electron orbital at 1.5 r_c;
- representation: |V_model - V_loc| < 0.05 eV on [r_loc + 0.2, 5] A (V_model: the tabulated prior as the model evaluates
  it, to prior_q_max and zero beyond; first run: a cosine + tail basis fit, which failed, see the plan's amendment);
- scattering: s and p max |dE| < 0.2 eV over E - eps in [-4, +6] eV at R = 2.4 A (d reported).
Everything is built on V_model. Gate: >= 27 of 29. Adds vloc / r_loc to epnet/data/atomic_channels/<El>.npz for
passing elements (and removes them for failing ones); writes results/checks/ae_local_report.json.
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
R_LOC = (1.2, 1.4, 1.6, 1.8, 2.0)
ECUT, ECUT_TAIL = 20.0, 30.0


def candidate(symbol: str, orbs: dict, r_loc: float) -> dict:
    from epnet.diagnostics.pseudo_atom import PseudoAtom, local_radial
    from epnet.diagnostics.scattering import ae_phase, channel_phase, energy_error
    from epnet.models.general_epm import prior_q_max
    from epnet.physics.atomic_channels import ae_local_potential, kb_channel, radial_form_factor
    from epnet.physics.atomic_radii import BOHR_A

    r = next(iter(orbs.values())).r
    v_target = ae_local_potential(symbol, r, r_loc)
    q = np.linspace(0.0, prior_q_max([ECUT, 30.0]), 4001)
    v_q = radial_form_factor(v_target, r, q)
    v_basis = local_radial(v_q, q, r)
    sel = (r >= r_loc + 0.2) & (r <= 5.0)
    repr_err = float(np.abs(v_basis[sel] - v_target[sel]).max())
    channels, overlaps, kbs = {}, {}, {}
    for l, orb in orbs.items():
        kb = kb_channel(orb, v_basis)
        kbs[l] = kb
        channels[l] = [(math.pi**3 * kb.d_kb, (r, kb.chi_R))]
        overlaps[l] = [(math.pi**3 * kb.s_kb, (r, kb.chi_R))]
    atom = PseudoAtom(q, v_q, channels, overlaps, r_max=12.0, n_r=6000)
    out = {"r_loc": r_loc, "repr_err_eV": repr_err, "repr_ok": repr_err < 0.05, "channels": {}}
    ok = out["repr_ok"]
    for l, orb in orbs.items():
        levels, _ = atom.solve_states(l, ECUT, n_levels=4)
        s_min = atom.last_s_min
        spread = atom.solve_checked(l, ECUT, n_levels=1)[0].basis_spread_eV
        lv30, rad30 = atom.solve_states(l, ECUT_TAIL, n_levels=1)
        r_t = 1.5 * orb.rc_bohr * BOHR_A
        tail_ratio = abs(float(np.interp(r_t, atom.r, atom.r * rad30[0]) / np.interp(r_t, orb.r, orb.u_ae)))
        err = levels[0].energy_eV - orb.eps_eV
        ghost = abs(err) >= 0.1 and any(abs(x.energy_eV - orb.eps_eV) < 0.1 for x in levels[1:])
        e = orb.eps_eV + np.arange(-4.0, 6.0001, 0.02)
        de = energy_error(e, ae_phase(symbol, l, e, 2.4), channel_phase(r, v_basis, l, kbs[l].chi_R * r, kbs[l].d_kb,
                                                                        kbs[l].s_kb, e, 2.4))
        max_de = float(np.nanmax(np.abs(de))) if np.isfinite(de).any() else float("inf")
        c = {"eps_eV": orb.eps_eV, "level_err_eV": err, "levels": [x.energy_eV for x in levels], "ghost": bool(ghost),
             "s_min": s_min, "basis_spread_eV": spread, "tail_ratio_30ry": tail_ratio, "scatter_max_dE_eV": max_de,
             "scatter_nan_frac": float(np.mean(~np.isfinite(de))), "d_kb": kbs[l].d_kb, "s_kb": kbs[l].s_kb}
        c["pass"] = bool(abs(err) < 0.1 and not ghost and spread < 0.1 and s_min > 0 and abs(tail_ratio - 1) < 0.02
                         and (l == 2 or max_de < 0.2))
        ok &= c["pass"]
        out["channels"][str(l)] = c
    out["pass"] = bool(ok)
    out["v_target"] = v_target
    return out


def one(symbol: str) -> dict:
    try:
        from epnet.physics.atomic_channels import atomic_orbitals
        orbs = atomic_orbitals(symbol, ecut_ry=ECUT, normalize=False)
        tried = []
        for r_loc in R_LOC:
            c = candidate(symbol, orbs, r_loc)
            tried.append(c)
            if c["pass"]:
                break
        best = next((c for c in tried if c["pass"]), None)
        path = ROOT / "epnet/data/atomic_channels" / f"{symbol}.npz"
        z = dict(np.load(path))
        z.pop("vloc", None), z.pop("r_loc", None)
        if best is not None:
            z["vloc"], z["r_loc"] = best["v_target"], best["r_loc"]
        np.savez(path, **z)
        for c in tried:
            c.pop("v_target")
        return {"element": symbol, "pass": best is not None, "r_loc": None if best is None else best["r_loc"], "tried": tried}
    except Exception as exc:   # record and continue
        return {"element": symbol, "pass": False, "r_loc": None, "tried": [], "error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    with ProcessPoolExecutor(29) as ex:
        reps = list(ex.map(one, D_ELEMENTS))
    (ROOT / "results/checks/ae_local_report.json").write_text(json.dumps(reps, indent=1) + "\n")
    print(f"{'El':3s} {'r_loc':>5s} {'repr':>5s} | per l: err / ghost / tail30 / scatter max dE / pass")
    for rep in reps:
        for c in rep["tried"]:
            mark = "*" if rep["r_loc"] == c["r_loc"] else " "
            chans = "  ".join(f"{l}:{x['level_err_eV']:+.2f}/{'G' if x['ghost'] else '-'}/{x['tail_ratio_30ry']:.3f}/"
                              f"{x['scatter_max_dE_eV']:.2f}/{'ok' if x['pass'] else 'X'}" for l, x in c["channels"].items())
            print(f"{rep['element']:3s} {c['r_loc']:5.1f}{mark}{c['repr_err_eV']:5.3f} | {chans}")
        if rep.get("error"):
            print(rep["element"], "ERROR", rep["error"])
    n = sum(r["pass"] for r in reps)
    print(f"elements passing: {n}/29 -> gate {'PASSED' if n >= 27 else 'FAILED'} (>= 27)")
    print("failing:", [r["element"] for r in reps if not r["pass"]])
    print("r_loc:", {r["element"]: r["r_loc"] for r in reps})


if __name__ == "__main__":
    main()
