"""Validated, lightweight data objects used throughout EpNet."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _array(value: ArrayLike, shape: tuple[int | None, ...], dtype: Any) -> NDArray:
    out = np.asarray(value, dtype=dtype)
    if out.ndim != len(shape) or any(s is not None and out.shape[i] != s for i, s in enumerate(shape)):
        raise ValueError(f"expected shape {shape}, got {out.shape}")
    if not np.all(np.isfinite(out)):
        raise ValueError("arrays must contain only finite values")
    return out.copy()


@dataclass(frozen=True)
class Structure:
    """Periodic structure with row lattice vectors and fractional positions."""

    lattice: ArrayLike
    species: tuple[str, ...] | list[str]
    fractional_positions: ArrayLike
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lattice = _array(self.lattice, (3, 3), float)
        positions = _array(self.fractional_positions, (None, 3), float)
        species = tuple(str(s) for s in self.species)
        if len(species) != len(positions) or not species:
            raise ValueError("species and positions must have the same non-zero length")
        if abs(np.linalg.det(lattice)) < 1e-12:
            raise ValueError("lattice must be nonsingular")
        object.__setattr__(self, "lattice", lattice)
        object.__setattr__(self, "fractional_positions", positions % 1.0)
        object.__setattr__(self, "species", species)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def n_atoms(self) -> int:
        return len(self.species)

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.lattice)))

    @property
    def cartesian_positions(self) -> NDArray[np.float64]:
        return self.fractional_positions @ self.lattice

    @property
    def reciprocal_lattice(self) -> NDArray[np.float64]:
        return 2.0 * np.pi * np.linalg.inv(self.lattice).T


@dataclass(frozen=True)
class ReciprocalPotential:
    """Complex local potential sampled at integer reciprocal indices."""

    g_indices: ArrayLike
    values: ArrayLike
    gauge_zero: bool = True

    def __post_init__(self) -> None:
        g = _array(self.g_indices, (None, 3), int)
        values = np.asarray(self.values, dtype=np.complex128)
        if values.shape != (len(g),) or not np.all(np.isfinite(values)):
            raise ValueError("values must be a finite vector matching g_indices")
        if len(np.unique(g, axis=0)) != len(g):
            raise ValueError("g_indices must be unique")
        if self.gauge_zero:
            zero = np.flatnonzero(np.all(g == 0, axis=1))
            if len(zero) and abs(values[zero[0]]) > 1e-10:
                raise ValueError("the G=0 gauge value must be zero")
        object.__setattr__(self, "g_indices", g)
        object.__setattr__(self, "values", values.copy())


@dataclass(frozen=True)
class BandData:
    kpoints: ArrayLike
    energies: ArrayLike
    weights: ArrayLike | None = None
    fermi_energy: float | None = None

    def __post_init__(self) -> None:
        kpoints = _array(self.kpoints, (None, 3), float)
        energies = _array(self.energies, (len(kpoints), None), float)
        weights = (
            np.full(len(kpoints), 1.0 / len(kpoints))
            if self.weights is None
            else _array(self.weights, (len(kpoints),), float)
        )
        if weights.sum() <= 0:
            raise ValueError("k-point weights must have positive sum")
        object.__setattr__(self, "kpoints", kpoints)
        object.__setattr__(self, "energies", energies)
        object.__setattr__(self, "weights", weights / weights.sum())


@dataclass(frozen=True)
class Sample:
    structure: Structure
    potential: ReciprocalPotential | None = None
    bands: BandData | None = None
    sample_id: str | None = None
