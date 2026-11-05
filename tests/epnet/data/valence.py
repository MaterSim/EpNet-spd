"""Valence contract ``zm-v2``: which bands of a PAW label the EPM must reproduce.

MP line-mode bands start at the lowest state of the VASP PAW valence, which
includes semicore shells (Na_pv 2p, Ba_sv 5s5p, Ga_d 3d, Ti_pv 3p, Y_sv 4s4p, ...).
The EPM only models the chemical valence ``Z_m``; the lowest ``n_drop`` target
bands are discarded. Semicore shells are closed, so ``Z_PAW - Z_m`` is even for
every element and ``n_drop`` is an integer by construction.

``zm-v1`` (EpNet-sp): main group plus groups 11-12. ``Z_PAW`` was recovered from
3,930 gapped nonmagnetic cells (``results/checks/valence_contract`` in EpNet-sp).
``zm-v2`` (EpNet-spd, 2026-10-19) adds groups 3-10 with ``Z_m = ns + (n-1)d``.
Their ``Z_PAW`` is the MP default POTCAR set, not a fit: the EpNet-sp fit gave
Cr 11, Mn 11, Fe 13 from few cells, odd semicore counts for Cr and Fe. Every
gapped cell checks it again (``apply_contract``: ``paw_count_mismatch``). zm-v2
changes no zm-v1 value, so zm-v1 manifests stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike

CONTRACT_VERSION = "zm-v2"
# manifests built under these contracts load unchanged (zm-v2 extends zm-v1 without changing a value)
COMPATIBLE_VERSIONS = frozenset({"zm-v1", "zm-v2"})

# Chemical valence modelled by the EPM: ns+np for the main group (filled d of
# Ga..Bi dropped), ns+(n-1)d for the d block.
Z_MODEL = {
    "H": 1, "Li": 1, "Be": 2, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "Na": 1, "Mg": 2, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
    "K": 1, "Ca": 2, "Ga": 3, "Ge": 4, "As": 5, "Se": 6, "Br": 7,
    "Rb": 1, "Sr": 2, "In": 3, "Sn": 4, "Sb": 5, "Te": 6, "I": 7,
    "Cs": 1, "Ba": 2, "Tl": 3, "Pb": 4, "Bi": 5,
    "Cu": 11, "Zn": 12, "Ag": 11, "Cd": 12, "Au": 11, "Hg": 12,
    # zm-v2: groups 3-10
    "Sc": 3, "Ti": 4, "V": 5, "Cr": 6, "Mn": 7, "Fe": 8, "Co": 9, "Ni": 10,
    "Y": 3, "Zr": 4, "Nb": 5, "Mo": 6, "Tc": 7, "Ru": 8, "Rh": 9, "Pd": 10,
    "Hf": 4, "Ta": 5, "W": 6, "Re": 7, "Os": 8, "Ir": 9, "Pt": 10,
}

# MP VASP POTCAR valence (Li_sv, Be_sv, Na_pv, Mg_pv, K_sv, Ca_sv, Ga_d, Ge_d,
# Rb_sv, Sr_sv, In_d, Sn_d, Cs_sv, Ba_sv, Tl_d, Pb_d, Cu_pv; Bi without d).
# zm-v2: Sc_sv, Ti_pv, V_pv, Cr_pv, Mn_pv, Fe_pv, Co, Ni_pv, Y_sv, Zr_sv, Nb_pv,
# Mo_pv, Tc_pv, Ru_pv, Rh_pv, Pd, Hf_pv, Ta_pv, W_pv, Re_pv, Os_pv, Ir, Pt.
Z_PAW = {
    "H": 1, "Li": 3, "Be": 4, "B": 3, "C": 4, "N": 5, "O": 6, "F": 7,
    "Na": 7, "Mg": 8, "Al": 3, "Si": 4, "P": 5, "S": 6, "Cl": 7,
    "K": 9, "Ca": 10, "Ga": 13, "Ge": 14, "As": 5, "Se": 6, "Br": 7,
    "Rb": 9, "Sr": 10, "In": 13, "Sn": 14, "Sb": 5, "Te": 6, "I": 7,
    "Cs": 9, "Ba": 10, "Tl": 13, "Pb": 14, "Bi": 5,
    "Cu": 17, "Zn": 12, "Ag": 11, "Cd": 12, "Au": 11, "Hg": 12,
    "Sc": 11, "Ti": 10, "V": 11, "Cr": 12, "Mn": 13, "Fe": 14, "Co": 9, "Ni": 16,
    "Y": 11, "Zr": 12, "Nb": 11, "Mo": 12, "Tc": 13, "Ru": 14, "Rh": 15, "Pd": 10,
    "Hf": 10, "Ta": 11, "W": 12, "Re": 13, "Os": 14, "Ir": 9, "Pt": 10,
}

# Scope: main group + d block. Noble gases (22 MP cells) have no element features;
# La and the f block (empty 4f near E_F) are outside the contract.
IN_SCOPE = frozenset(Z_MODEL)
D10_ELEMENTS = frozenset({"Cu", "Zn", "Ag", "Cd", "Au", "Hg"})
# EpNet-sp scope (2026-09-29): main group only.
SP_ELEMENTS = frozenset({
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl",
    "K", "Ca", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "In", "Sn", "Sb", "Te", "I",
    "Cs", "Ba", "Tl", "Pb", "Bi",
})
D_ELEMENTS = IN_SCOPE - SP_ELEMENTS
# EpNet-spd tiers (docs/spd_plan.md): a cell belongs to the highest tier among its d elements.
D_TIERS = {
    1: D10_ELEMENTS,
    2: frozenset({"Sc", "Y", "Ti", "Zr", "Hf", "V", "Nb", "Ta"}),
    3: frozenset({"Mo", "Tc", "Ru", "Rh", "Pd", "W", "Re", "Os", "Ir", "Pt"}),
    4: frozenset({"Cr", "Mn", "Fe", "Co", "Ni"}),
}
# MP applies DFT+U to oxides and fluorides of these elements: those labels are not PBE.
MP_U_ELEMENTS = frozenset({"Co", "Cr", "Fe", "Mn", "Mo", "Ni", "V", "W"})
MP_U_ANIONS = frozenset({"O", "F"})


def d_tier(elements: ArrayLike) -> int:
    """0 for an s/p-only cell, otherwise the highest tier (1-4) among its d elements."""
    present = {str(e) for e in elements}
    return max((t for t, members in D_TIERS.items() if present & members), default=0)


def mp_plus_u(elements: ArrayLike) -> bool:
    """True when MP labels this chemistry with GGA+U (a U element together with O or F)."""
    present = {str(e) for e in elements}
    return bool(present & MP_U_ELEMENTS) and bool(present & MP_U_ANIONS)

# Loss window reaches VBM - 3 eV; a dropped band must sit below VBM - 4 eV.
DEFAULT_MIN_DEPTH_EV = 4.0
# Kept bands whose minimum lies within this of the dropped top are not a clean
# valence state: the count-based drop can cut a degenerate multiplet when a shallow
# cation semicore p (K 3p, Cs 5p, Ba 5p) meets the anion s (CsF, KCl, BaO, ...).
DEEP_AMBIGUITY_MARGIN_EV = 1.0


def model_valence(species: ArrayLike) -> int:
    return int(sum(Z_MODEL[str(s)] for s in species))


def paw_valence(species: ArrayLike) -> int:
    return int(sum(Z_PAW[str(s)] for s in species))


def n_drop(species: ArrayLike) -> int:
    extra = paw_valence(species) - model_valence(species)
    if extra % 2:
        raise ValueError(f"odd semicore electron count {extra} for {sorted(set(map(str, species)))}")
    return extra // 2


@dataclass(frozen=True)
class ContractResult:
    """Outcome of applying ``zm-v1`` to one labelled cell."""

    ok: bool
    reason: str | None
    gapped: bool
    n_occupied_paw: int | None
    n_drop: int
    n_occupied_model: int | None
    n_valence_model: int
    dropped_top_minus_edge_ev: float | None
    gap_ev: float | None
    drop_boundary_gap_ev: float | None = None
    n_deep_ambiguous: int = 0
    lowest_paw_band_minus_edge_ev: float | None = None

    def metadata(self) -> dict:
        return {
            "z_model_version": CONTRACT_VERSION,
            "n_occupied_paw": self.n_occupied_paw,
            "band_offset": self.n_drop,
            "n_occupied_model": self.n_occupied_model,
            "n_valence_model": self.n_valence_model,
            "dropped_top_minus_edge_eV": self.dropped_top_minus_edge_ev,
            "gapped": self.gapped,
            "gap_eV": self.gap_ev,
            "drop_boundary_gap_eV": self.drop_boundary_gap_ev,
            "n_deep_ambiguous": self.n_deep_ambiguous,
            "lowest_paw_band_minus_edge_eV": self.lowest_paw_band_minus_edge_ev,
        }


def apply_contract(
    species: ArrayLike,
    energies: ArrayLike,
    fermi_energy: float,
    *,
    is_metal: bool,
    min_depth_ev: float = DEFAULT_MIN_DEPTH_EV,
    min_gap_ev: float = 0.01,
    tol: float = 1e-4,
) -> ContractResult:
    """Check a PAW-labelled band array against ``zm-v1``.

    Gapped cells (``is_metal`` false and a path gap above ``min_gap_ev`` at the
    PAW band count) must place E_F exactly at ``Z_PAW/2`` occupied bands; this
    is the per-cell check that the POTCAR table is right. Metals use the band
    index of the highest state at or below E_F. In both cases the top of the
    dropped block must lie ``min_depth_ev`` below the VBM (or E_F).
    """
    symbols = tuple(str(s) for s in species)
    e = np.asarray(energies, dtype=float)
    if e.ndim != 2:
        raise ValueError("energies must have shape (n_k, n_bands)")

    def result(ok, reason, gapped=False, n_paw=None, drop=0, n_model=None, depth=None, gap=None):
        return ContractResult(ok, reason, gapped, n_paw, drop, n_model, nv, depth, gap)

    outside = sorted(set(symbols) - IN_SCOPE)
    nv = sum(Z_MODEL.get(s, 0) for s in symbols)
    if outside:
        return result(False, f"out_of_scope:{'-'.join(outside)}")
    drop = n_drop(symbols)
    occupied = np.flatnonzero(np.any(e <= float(fermi_energy) + tol, axis=0))
    if not occupied.size:
        return result(False, "no_occupied_band", drop=drop)
    n_fermi = int(occupied.max()) + 1
    n_expected = paw_valence(symbols) // 2 if paw_valence(symbols) % 2 == 0 else None

    gapped = False
    gap = None
    if not is_metal and n_expected is not None and 0 < n_expected < e.shape[1]:
        gap = float(e[:, n_expected].min() - e[:, n_expected - 1].max())
        gapped = gap > min_gap_ev
    if gapped:
        if n_fermi != n_expected:
            return result(False, "paw_count_mismatch", gapped, n_fermi, drop, gap=gap)
        n_paw = n_expected
        edge = float(e[:, n_paw - 1].max())
    else:
        n_paw = n_fermi
        edge = float(fermi_energy)
    n_model = n_paw - drop
    if n_model < 1:
        return result(False, "no_valence_band", gapped, n_paw, drop, gap=gap)
    if n_paw >= e.shape[1]:
        return result(False, "no_conduction_band", gapped, n_paw, drop, gap=gap)
    depth = None
    if drop:
        depth = float(e[:, drop - 1].max() - edge)
        if depth > -float(min_depth_ev):
            return result(False, "semicore_too_shallow", gapped, n_paw, drop, n_model, depth, gap)
    boundary, ambiguous = drop_boundary(e, drop, n_model)
    return replace(result(True, None, gapped, n_paw, drop, n_model, depth, gap),
                   drop_boundary_gap_ev=boundary, n_deep_ambiguous=ambiguous,
                   lowest_paw_band_minus_edge_ev=float(e.min() - edge))


def drop_boundary(energies: ArrayLike, band_offset: int, n_occupied_model: int,
                  margin_ev: float = DEEP_AMBIGUITY_MARGIN_EV) -> tuple[float | None, int]:
    """Gap between the dropped block and the kept bands, and how many kept bands are ambiguous.

    The gap is ``min E[:, offset] - max E[:, offset-1]``; zero or negative means the
    drop cut through a multiplet. Ambiguous bands are the lowest kept occupied bands,
    counted from the bottom, whose minimum lies within ``margin_ev`` of the dropped
    top: they belong to the semicore cluster, not to a clean valence state. They stay
    in the target (the window never reaches them) but deep-band losses skip them.
    """
    e = np.asarray(energies, dtype=float)
    offset = int(band_offset)
    if offset == 0:
        return None, 0
    top = float(e[:, offset - 1].max())
    ambiguous = 0
    while ambiguous < n_occupied_model and float(e[:, offset + ambiguous].min()) <= top + margin_ev:
        ambiguous += 1
    return float(e[:, offset].min() - top), ambiguous


def model_target(energies: ArrayLike, band_offset: int) -> np.ndarray:
    """Target bands for the EPM: the PAW array with the dropped block removed."""
    e = np.asarray(energies)
    if not 0 <= int(band_offset) < e.shape[1]:
        raise ValueError(f"band_offset {band_offset} outside 0..{e.shape[1] - 1}")
    return e[:, int(band_offset):]


__all__ = [
    "COMPATIBLE_VERSIONS",
    "CONTRACT_VERSION",
    "ContractResult",
    "D10_ELEMENTS",
    "D_ELEMENTS",
    "D_TIERS",
    "MP_U_ANIONS",
    "MP_U_ELEMENTS",
    "DEEP_AMBIGUITY_MARGIN_EV",
    "DEFAULT_MIN_DEPTH_EV",
    "IN_SCOPE",
    "SP_ELEMENTS",
    "Z_MODEL",
    "Z_PAW",
    "apply_contract",
    "d_tier",
    "drop_boundary",
    "model_target",
    "model_valence",
    "mp_plus_u",
    "n_drop",
    "paw_valence",
]
