"""Versioned periodic-table features for the Phase 15 element encoder."""

from __future__ import annotations

import hashlib
import json

import torch
from torch import Tensor


FEATURE_VERSION = "periodic-table-v1"
FEATURE_NAMES = (
    "constant",
    "atomic_number_div_100",
    "period_div_7",
    "group_div_18",
    "valence_s_div_2",
    "valence_p_div_6",
    "valence_d_div_10",
    "block_s",
    "block_p",
    "block_d",
    "pauling_electronegativity_div_4",
    "covalent_radius_angstrom_div_2p5",
    "atomic_number_squared_div_10000",
)

# Z, period, group, valence s/p/d occupancy, block, Pauling electronegativity,
# and Cordero-style single-bond covalent radius in angstrom. The table covers
# the frozen 45-species Phase 15 domain and intentionally has no learned or
# endpoint-derived values.
_ELEMENT_DATA = {
    "H": (1, 1, 1, 1, 0, 0, "s", 2.20, 0.31),
    "Li": (3, 2, 1, 1, 0, 0, "s", 0.98, 1.28),
    "Be": (4, 2, 2, 2, 0, 0, "s", 1.57, 0.96),
    "B": (5, 2, 13, 2, 1, 0, "p", 2.04, 0.84),
    "C": (6, 2, 14, 2, 2, 0, "p", 2.55, 0.76),
    "N": (7, 2, 15, 2, 3, 0, "p", 3.04, 0.71),
    "O": (8, 2, 16, 2, 4, 0, "p", 3.44, 0.66),
    "F": (9, 2, 17, 2, 5, 0, "p", 3.98, 0.57),
    "Na": (11, 3, 1, 1, 0, 0, "s", 0.93, 1.66),
    "Mg": (12, 3, 2, 2, 0, 0, "s", 1.31, 1.41),
    "Al": (13, 3, 13, 2, 1, 0, "p", 1.61, 1.21),
    "Si": (14, 3, 14, 2, 2, 0, "p", 1.90, 1.11),
    "P": (15, 3, 15, 2, 3, 0, "p", 2.19, 1.07),
    "S": (16, 3, 16, 2, 4, 0, "p", 2.58, 1.05),
    "Cl": (17, 3, 17, 2, 5, 0, "p", 3.16, 1.02),
    "K": (19, 4, 1, 1, 0, 0, "s", 0.82, 2.03),
    "Ca": (20, 4, 2, 2, 0, 0, "s", 1.00, 1.76),
    "Cu": (29, 4, 11, 1, 0, 10, "d", 1.90, 1.32),
    "Zn": (30, 4, 12, 2, 0, 10, "d", 1.65, 1.22),
    "Ga": (31, 4, 13, 2, 1, 10, "p", 1.81, 1.22),
    "Ge": (32, 4, 14, 2, 2, 10, "p", 2.01, 1.20),
    "As": (33, 4, 15, 2, 3, 10, "p", 2.18, 1.19),
    "Se": (34, 4, 16, 2, 4, 10, "p", 2.55, 1.20),
    "Br": (35, 4, 17, 2, 5, 10, "p", 2.96, 1.20),
    "Rb": (37, 5, 1, 1, 0, 0, "s", 0.82, 2.20),
    "Sr": (38, 5, 2, 2, 0, 0, "s", 0.95, 1.95),
    "Y": (39, 5, 3, 2, 0, 1, "d", 1.22, 1.90),
    "Zr": (40, 5, 4, 2, 0, 2, "d", 1.33, 1.75),
    "Nb": (41, 5, 5, 1, 0, 4, "d", 1.60, 1.64),
    "Pd": (46, 5, 10, 0, 0, 10, "d", 2.20, 1.39),
    "Ag": (47, 5, 11, 1, 0, 10, "d", 1.93, 1.45),
    "Cd": (48, 5, 12, 2, 0, 10, "d", 1.69, 1.44),
    "In": (49, 5, 13, 2, 1, 10, "p", 1.78, 1.42),
    "Sn": (50, 5, 14, 2, 2, 10, "p", 1.96, 1.39),
    "Sb": (51, 5, 15, 2, 3, 10, "p", 2.05, 1.39),
    "Te": (52, 5, 16, 2, 4, 10, "p", 2.10, 1.38),
    "I": (53, 5, 17, 2, 5, 10, "p", 2.66, 1.39),
    "Cs": (55, 6, 1, 1, 0, 0, "s", 0.79, 2.44),
    "Ba": (56, 6, 2, 2, 0, 0, "s", 0.89, 2.15),
    "Pt": (78, 6, 10, 1, 0, 9, "d", 2.28, 1.36),
    # EpNet-spd (2026-10-19): the rest of groups 3-10. Neutral ground-state occupancies
    # (Cr 3d5 4s1, Mo 4d5 5s1, Ru 4d7 5s1, Rh 4d8 5s1); Cordero 2008 radii, low-spin for
    # Mn/Fe/Co (the model's cells are non-magnetic).
    "Sc": (21, 4, 3, 2, 0, 1, "d", 1.36, 1.70),
    "Ti": (22, 4, 4, 2, 0, 2, "d", 1.54, 1.60),
    "V": (23, 4, 5, 2, 0, 3, "d", 1.63, 1.53),
    "Cr": (24, 4, 6, 1, 0, 5, "d", 1.66, 1.39),
    "Mn": (25, 4, 7, 2, 0, 5, "d", 1.55, 1.39),
    "Fe": (26, 4, 8, 2, 0, 6, "d", 1.83, 1.32),
    "Co": (27, 4, 9, 2, 0, 7, "d", 1.88, 1.26),
    "Ni": (28, 4, 10, 2, 0, 8, "d", 1.91, 1.24),
    "Mo": (42, 5, 6, 1, 0, 5, "d", 2.16, 1.54),
    "Tc": (43, 5, 7, 2, 0, 5, "d", 1.90, 1.47),
    "Ru": (44, 5, 8, 1, 0, 7, "d", 2.20, 1.46),
    "Rh": (45, 5, 9, 1, 0, 8, "d", 2.28, 1.42),
    "Hf": (72, 6, 4, 2, 0, 2, "d", 1.30, 1.75),
    "Ta": (73, 6, 5, 2, 0, 3, "d", 1.50, 1.70),
    "W": (74, 6, 6, 2, 0, 4, "d", 2.36, 1.62),
    "Re": (75, 6, 7, 2, 0, 5, "d", 1.90, 1.51),
    "Os": (76, 6, 8, 2, 0, 6, "d", 2.20, 1.44),
    "Ir": (77, 6, 9, 2, 0, 7, "d", 2.20, 1.41),
    "Au": (79, 6, 11, 1, 0, 10, "d", 2.54, 1.36),
    "Hg": (80, 6, 12, 2, 0, 10, "d", 2.00, 1.32),
    "Tl": (81, 6, 13, 2, 1, 10, "p", 1.62, 1.45),
    "Pb": (82, 6, 14, 2, 2, 10, "p", 2.33, 1.46),
    "Bi": (83, 6, 15, 2, 3, 10, "p", 2.02, 1.48),
}


def _feature_row(symbol: str) -> list[float]:
    try:
        z, period, group, s_occ, p_occ, d_occ, block, electronegativity, radius = (
            _ELEMENT_DATA[symbol]
        )
    except KeyError as exc:
        raise KeyError(f"no {FEATURE_VERSION} features for element {symbol}") from exc
    return [
        1.0,
        z / 100.0,
        period / 7.0,
        group / 18.0,
        s_occ / 2.0,
        p_occ / 6.0,
        d_occ / 10.0,
        float(block == "s"),
        float(block == "p"),
        float(block == "d"),
        electronegativity / 4.0,
        radius / 2.5,
        (z * z) / 10000.0,
    ]


def periodic_table_features(
    symbols: list[str] | tuple[str, ...],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> Tensor:
    """Return the fixed feature matrix in the supplied species-table order."""
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError("symbols must be a nonempty unique sequence")
    values = torch.tensor([_feature_row(symbol) for symbol in symbols], dtype=dtype)
    if not torch.isfinite(values).all():
        raise ValueError("periodic-table features must be finite")
    return values.to(device=device)


def feature_fingerprint(symbols: list[str] | tuple[str, ...]) -> str:
    values = periodic_table_features(symbols, dtype=torch.float64).tolist()
    payload = {
        "version": FEATURE_VERSION,
        "feature_names": list(FEATURE_NAMES),
        "symbols": list(symbols),
        "values": values,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


__all__ = [
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "feature_fingerprint",
    "periodic_table_features",
]
