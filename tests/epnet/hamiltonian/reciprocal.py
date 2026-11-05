"""Reciprocal-space grids, phases, and local-potential assembly."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from epnet.data import Structure


def reciprocal_lattice(lattice: ArrayLike) -> NDArray[np.float64]:
    """Return row reciprocal vectors satisfying ``A @ B.T = 2 pi I``."""
    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3) or abs(np.linalg.det(lattice)) < 1e-12:
        raise ValueError("lattice must be a nonsingular 3x3 matrix")
    return 2.0 * np.pi * np.linalg.inv(lattice).T


def reciprocal_vectors(lattice: ArrayLike, cutoff: float, *, include_zero: bool = True) -> tuple[NDArray, NDArray]:
    """Enumerate integer indices and Cartesian G vectors within a norm cutoff."""
    if cutoff < 0:
        raise ValueError("cutoff must be non-negative")
    b = reciprocal_lattice(lattice)
    # ||n B|| >= sigma_min(B) ||n|| bounds a complete integer search cube.
    nmax = int(np.ceil(cutoff / np.linalg.svd(b, compute_uv=False).min())) + 1
    axis = np.arange(-nmax, nmax + 1)
    indices = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1).reshape(-1, 3)
    vectors = indices @ b
    mask = np.linalg.norm(vectors, axis=1) <= cutoff + 1e-12
    if not include_zero:
        mask &= np.any(indices != 0, axis=1)
    indices, vectors = indices[mask], vectors[mask]
    order = np.lexsort((indices[:, 2], indices[:, 1], indices[:, 0], np.linalg.norm(vectors, axis=1)))
    return indices[order], vectors[order]


def structure_factor(structure: Structure, g_indices: ArrayLike, *, per_species: bool = False):
    """Compute ``sum_i exp(-2 pi i h.r_i)`` using fractional coordinates."""
    g = np.asarray(g_indices, dtype=int)
    if g.ndim != 2 or g.shape[1] != 3:
        raise ValueError("g_indices must have shape (n, 3)")
    phases = np.exp(-2j * np.pi * (g @ structure.fractional_positions.T))
    if not per_species:
        return phases.sum(axis=1)
    return {s: phases[:, np.array(structure.species) == s].sum(axis=1) for s in dict.fromkeys(structure.species)}


def assemble_local_potential(
    structure: Structure,
    g_indices: ArrayLike,
    form_factors: Mapping[str, Callable[[NDArray], ArrayLike]],
    *,
    zero_gauge: bool = True,
) -> NDArray[np.complex128]:
    """Assemble atom-centered radial form factors into a crystal potential."""
    g = np.asarray(g_indices, dtype=int)
    cart = g @ structure.reciprocal_lattice
    norms = np.linalg.norm(cart, axis=1)
    factors = structure_factor(structure, g, per_species=True)
    missing = set(factors).difference(form_factors)
    if missing:
        raise KeyError(f"missing form factors for {sorted(missing)}")
    values = sum(np.asarray(form_factors[s](norms), dtype=np.complex128) * sf for s, sf in factors.items())
    values = np.asarray(values) / structure.volume
    if zero_gauge:
        values[np.all(g == 0, axis=1)] = 0.0
    return values


def enforce_conjugate_symmetry(g_indices: ArrayLike, values: ArrayLike) -> NDArray[np.complex128]:
    """Average available +/-G pairs so that V(-G)=V(G)*."""
    g = np.asarray(g_indices, dtype=int)
    out = np.asarray(values, dtype=np.complex128).copy()
    lookup = {tuple(row): i for i, row in enumerate(g)}
    for i, row in enumerate(g):
        j = lookup.get(tuple(-row))
        if j is not None:
            average = 0.5 * (out[i] + out[j].conjugate())
            out[i], out[j] = average, average.conjugate()
    return out
