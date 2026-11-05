"""Frozen element index table and MP first-slice chemistry filters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

PLUS_U_METALS = frozenset({"V", "Cr", "Mn", "Fe", "Co", "Ni"})
LANTHANIDES = frozenset(
    "La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()
)
ACTINIDES = frozenset("Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr".split())
F_BLOCK = LANTHANIDES | ACTINIDES
EXCLUDED_ELEMENTS = PLUS_U_METALS | F_BLOCK

III_V_CATIONS = frozenset({"B", "Al", "Ga", "In"})
III_V_ANIONS = frozenset({"N", "P", "As", "Sb"})

# Common oxidation states for the MP first-slice probe. Not Bader charges.
# Noble gases are 0. Ambiguous metals use the most common ionic value.
COMMON_OXIDATION = {
    "H": 1,
    "He": 0,
    "Li": 1,
    "Be": 2,
    "B": 3,
    "C": 4,
    "N": -3,
    "O": -2,
    "F": -1,
    "Ne": 0,
    "Na": 1,
    "Mg": 2,
    "Al": 3,
    "Si": 4,
    "P": -3,
    "S": -2,
    "Cl": -1,
    "Ar": 0,
    "K": 1,
    "Ca": 2,
    "Sc": 3,
    "Ti": 4,
    "Zn": 2,
    "Ga": 3,
    "Ge": 4,
    "As": -3,
    "Se": -2,
    "Br": -1,
    "Kr": 0,
    "Rb": 1,
    "Sr": 2,
    "Y": 3,
    "Zr": 4,
    "Nb": 5,
    "Mo": 4,
    "Tc": 4,
    "Ru": 4,
    "Rh": 3,
    "Pd": 2,
    "Ag": 1,
    "Cd": 2,
    "In": 3,
    "Sn": 4,
    "Sb": -3,
    "Te": -2,
    "I": -1,
    "Xe": 0,
    "Cs": 1,
    "Ba": 2,
    "Hf": 4,
    "Ta": 5,
    "W": 4,
    "Re": 4,
    "Os": 4,
    "Ir": 3,
    "Pt": 2,
    "Au": 1,
    "Hg": 2,
    "Tl": 3,
    "Pb": 2,
    "Bi": 3,
    "Cu": 2,
}


def oxidation_charges(symbols: tuple[str, ...] | list[str]) -> list[float]:
    """Return common oxidation states for ``symbols``. Raises on unknown elements."""
    missing = [str(s) for s in symbols if str(s) not in COMMON_OXIDATION]
    if missing:
        raise KeyError(f"no common oxidation state for {missing}")
    return [float(COMMON_OXIDATION[str(s)]) for s in symbols]


def is_excluded_element(symbol: str) -> bool:
    return str(symbol) in EXCLUDED_ELEMENTS


def is_iii_v_chemsys(chemsys: str) -> bool:
    parts = [p for p in str(chemsys).split("-") if p]
    if len(parts) != 2:
        return False
    a, b = parts
    return (a in III_V_CATIONS and b in III_V_ANIONS) or (b in III_V_CATIONS and a in III_V_ANIONS)


def is_elemental_si(chemsys: str) -> bool:
    return str(chemsys) == "Si"


def infer_n_occupied(energies: NDArray, fermi: float, tol: float = 1e-4) -> int | None:
    """Last band index (1-based count) occupied at any k, using ``E <= E_F + tol``."""
    occupied = np.any(np.asarray(energies) <= float(fermi) + float(tol), axis=0)
    if not occupied.any():
        return None
    return int(np.max(np.flatnonzero(occupied)) + 1)


@dataclass(frozen=True)
class SpeciesTable:
    """Map element symbols to a dense index used by CosineRadialEP / the encoder."""

    symbols: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.symbols or len(set(self.symbols)) != len(self.symbols):
            raise ValueError("species table must have unique non-empty symbols")
        object.__setattr__(self, "index", {s: i for i, s in enumerate(self.symbols)})

    def __len__(self) -> int:
        return len(self.symbols)

    def __contains__(self, symbol: str) -> bool:
        return symbol in self.index

    def encode_one(self, symbol: str) -> int:
        try:
            return self.index[symbol]
        except KeyError as exc:
            raise KeyError(f"element {symbol!r} is not in the species table") from exc

    def encode(self, species: tuple[str, ...] | list[str]) -> list[int]:
        return [self.encode_one(s) for s in species]

    def to_json(self) -> dict:
        return {"symbols": list(self.symbols)}

    @classmethod
    def from_symbols(cls, symbols: list[str] | tuple[str, ...]) -> SpeciesTable:
        return cls(tuple(str(s) for s in symbols))

    @classmethod
    def from_json(cls, payload: dict) -> SpeciesTable:
        return cls.from_symbols(payload["symbols"])
