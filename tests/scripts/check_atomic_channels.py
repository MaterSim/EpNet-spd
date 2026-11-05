#!/usr/bin/env python3
"""Steps 3a / 3c of docs/d_channel_plan.md: build and check the atomic s/p/d channels of the 29 d elements.

3c (default, ultrasoft): the pseudo-orbitals keep the all-electron tail and the norm deficit q is restored by an
overlap term; the pseudo-atom is solved as H c = e S c. Checks per channel (recorded before the run): level within
0.1 eV of eps_l, S-normalized eigenstate within 2% of the all-electron orbital at 1.5 r_c, no ghost, basis spread
< 0.1 eV, S positive definite. ``--renormalized`` reproduces 3a (u renormalized, no S; overlap with phi >= 0.99).

Per element: pseudo-orbitals from the ZORA-LDA atom (epnet/physics/atomic_channels.py), Kleinman-Bylander / Vanderbilt
projectors against the model's shallow local potential (GeneralEPM base prior with d_prior_valence="s"), and the
pseudo-atom solved at 20 Ry (epnet/diagnostics/pseudo_atom.py). Checks per channel (recorded before the run):
level within 0.1 eV of eps_l, overlap with phi_l >= 0.99, no level below eps_l (no ghost), basis spread < 0.1 eV.
Gate: >= 27 of 29 elements pass every check. Writes epnet/data/atomic_channels/<El>.npz and
results/checks/atomic_channels_report.json.
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
ECUT = 20.0


def local_potential_q(symbol: str):
    import torch
    from epnet.diagnostics.pseudo_atom import model_q_grid, species_form_factor
    from epnet.models.general_epm import GeneralEPM
    torch.set_num_threads(1)
    model = GeneralEPM((symbol,), channels=8, n_readouts=1, ecut_ry=ECUT, k_centred_basis=True, projector_radius_mode="orbital",
                       projector_d_radius="d", element_feature_set="spd-v1", base_prior=True, d_prior_valence="s").double()
    q = model_q_grid(model)
    return q, species_form_factor(model, 0, q)


def one(symbol: str) -> dict:
    try:
        return _one(symbol)
    except Exception as exc:   # record and continue: one element must not stop the table
        return {"element": symbol, "channels": {}, "pass": False, "error": f"{type(exc).__name__}: {exc}"}


ULTRASOFT = "--renormalized" not in sys.argv


def _one(symbol: str) -> dict:
    from epnet.diagnostics.pseudo_atom import PseudoAtom, local_radial
    from epnet.physics.atomic_channels import atomic_orbitals, kb_channel
    from epnet.physics.atomic_radii import BOHR_A

    orbs = atomic_orbitals(symbol, ecut_ry=ECUT, normalize=not ULTRASOFT)
    q, v_q = local_potential_q(symbol)
    r = next(iter(orbs.values())).r
    v_r = local_radial(v_q, q, r)
    channels, overlaps, kbs = {}, {}, {}
    for l, orb in orbs.items():
        kb = kb_channel(orb, v_r)
        kbs[l] = kb
        channels[l] = [(math.pi**3 * kb.d_kb, (r, kb.chi_R))]
        overlaps[l] = [(math.pi**3 * kb.s_kb, (r, kb.chi_R))] if ULTRASOFT else []
    atom = PseudoAtom(q, v_q, channels, overlaps, r_max=12.0, n_r=6000)
    rep = {"element": symbol, "channels": {}}
    ok_all = True
    for l, orb in orbs.items():
        levels, radials = atom.solve_states(l, ECUT, n_levels=3)
        s_min = atom.last_s_min
        checked = atom.solve_checked(l, ECUT, n_levels=1)[0]
        local = atom.solve(l, ECUT, local_only=True, n_levels=1)[0].energy_eV
        target = orb.eps_eV
        j = int(np.argmin([abs(x.energy_eV - target) for x in levels]))
        phi_on_atom = np.interp(atom.r, orb.r, orb.u) / atom.r          # R = u / r
        dr = atom.r[1] - atom.r[0]
        overlap = abs(float(np.sum(atom.r**2 * radials[j] * phi_on_atom) * dr)) / math.sqrt(
            float(np.sum(atom.r**2 * radials[j] ** 2) * dr) * float(np.sum(atom.r**2 * phi_on_atom**2) * dr))
        r_t = 1.5 * orb.rc_bohr * BOHR_A                                  # tail point: psi (S-normalized) vs AE
        tail_ratio = float(np.interp(r_t, atom.r, atom.r * radials[j]) / np.interp(r_t, orb.r, orb.u_ae))
        tail_ratio = abs(tail_ratio)
        ghost = any(x.energy_eV < target - 0.1 for x in levels[:j])
        c = {"n": orb.n, "eps_eV": target, "rc_bohr": orb.rc_bohr, "ecut99_ry": orb.ecut99_ry,
             "level_eV": levels[j].energy_eV, "level_err_eV": levels[j].energy_eV - target, "overlap": overlap,
             "ghost": bool(ghost), "levels": [x.energy_eV for x in levels], "basis_spread_eV": checked.basis_spread_eV,
             "local_only_lowest_eV": local, "d_kb": kbs[l].d_kb, "s_kb": kbs[l].s_kb, "norm_deficit": orb.norm_deficit,
             "tail_ratio": tail_ratio, "s_min": s_min, "mean_r_A": levels[j].mean_r_A}
        shape_ok = abs(tail_ratio - 1.0) < 0.02 and s_min > 0 if ULTRASOFT else overlap >= 0.99
        c["pass"] = bool(abs(c["level_err_eV"]) < 0.1 and shape_ok and not ghost and checked.basis_spread_eV < 0.1)
        ok_all &= c["pass"]
        rep["channels"][str(l)] = c
    rep["pass"] = bool(ok_all)
    out = ROOT / "epnet/data/atomic_channels"
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / f"{symbol}.npz", r=r, **{f"u{l}": o.u for l, o in orbs.items()}, **{f"ua{l}": o.u_ae for l, o in orbs.items()},
             **{f"q{l}": o.norm_deficit for l, o in orbs.items()},
             **{f"eps{l}": o.eps_eV for l, o in orbs.items()}, **{f"rc{l}": o.rc_bohr for l, o in orbs.items()},
             **{f"n{l}": o.n for l, o in orbs.items()},
             method=("ultrasoft: all-electron tail kept, norm deficit q" if ULTRASOFT else "renormalized")
             + "; zora-lda pseudized r^(l+1)(a+br^2+cr^4), kinetic tail above 20 Ry < 0.05 eV")
    return rep


def main() -> None:
    with ProcessPoolExecutor(29) as ex:
        reps = list(ex.map(one, D_ELEMENTS))
    name = "atomic_channels_report.json" if not ULTRASOFT else "atomic_channels_us_report.json"
    (ROOT / "results/checks" / name).write_text(json.dumps(reps, indent=1) + "\n")
    print(f"{'El':3s} {'l':>2s} {'eps':>7s} {'err':>6s} {'ovl':>5s} {'q':>5s} {'tail':>5s} {'rc':>4s} {'spread':>6s} {'loc-only':>8s} {'D':>8s} {'S':>8s} {'Smin':>5s} ghost pass")
    for rep in reps:
        for l, c in rep["channels"].items():
            print(f"{rep['element']:3s} {l:>2s} {c['eps_eV']:7.2f} {c['level_err_eV']:+6.3f} {c['overlap']:5.3f} {c['norm_deficit']:5.2f} "
                  f"{c['tail_ratio']:5.3f} {c['rc_bohr']:4.2f} {c['basis_spread_eV']:6.3f} {c['local_only_lowest_eV']:8.2f} "
                  f"{c['d_kb']:+8.4f} {c['s_kb']:+8.4f} {c['s_min']:5.2f} {str(c['ghost']):5s} {c['pass']}")
    n_pass = sum(r["pass"] for r in reps)
    print(f"elements passing every check: {n_pass}/29 -> gate {'PASSED' if n_pass >= 27 else 'FAILED'} (>= 27)")
    print("failing:", [(r["element"], r.get("error") or [l for l, c in r["channels"].items() if not c["pass"]]) for r in reps if not r["pass"]])


if __name__ == "__main__":
    main()
