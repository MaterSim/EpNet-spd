"""Dense local plane-wave Hamiltonian reference implementation."""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray


def assemble_hamiltonian(
    basis_indices: ArrayLike,
    reciprocal_lattice: ArrayLike,
    kpoint: ArrayLike,
    potential: Mapping[tuple[int, int, int], complex] | tuple[ArrayLike, ArrayLike],
    *,
    kinetic_prefactor: float = 0.5,
    strict: bool = False,
) -> NDArray[np.complex128]:
    """Build H_GG'=T(k+G)delta_GG'+V(G-G') in atomic units."""
    basis = np.asarray(basis_indices, dtype=int)
    b = np.asarray(reciprocal_lattice, dtype=float)
    k = np.asarray(kpoint, dtype=float)
    if basis.ndim != 2 or basis.shape[1] != 3 or b.shape != (3, 3) or k.shape != (3,):
        raise ValueError("invalid basis, reciprocal lattice, or k-point shape")
    if isinstance(potential, tuple):
        pg, pv = potential
        table = {tuple(row): complex(v) for row, v in zip(np.asarray(pg, int), np.asarray(pv, complex))}
    else:
        table = {tuple(key): complex(value) for key, value in potential.items()}
    differences = basis[:, None, :] - basis[None, :, :]
    matrix = np.empty((len(basis), len(basis)), dtype=np.complex128)
    for i in range(len(basis)):
        for j in range(len(basis)):
            key = tuple(differences[i, j])
            if strict and key not in table:
                raise KeyError(f"potential does not contain reciprocal difference {key}")
            matrix[i, j] = table.get(key, 0.0)
    cartesian_k = k @ b  # k is in fractional reciprocal coordinates.
    momenta = basis @ b + cartesian_k
    matrix[np.diag_indices_from(matrix)] += kinetic_prefactor * np.einsum("ij,ij->i", momenta, momenta)
    residual = np.max(np.abs(matrix - matrix.conjugate().T), initial=0.0)
    if residual > 1e-10:
        raise ValueError(f"potential produces a non-Hermitian Hamiltonian (residual {residual:.3g})")
    return 0.5 * (matrix + matrix.conjugate().T)


def solve_bands(
    basis_indices: ArrayLike,
    reciprocal_lattice: ArrayLike,
    kpoints: ArrayLike,
    potential: Mapping[tuple[int, int, int], complex] | tuple[ArrayLike, ArrayLike],
    *,
    n_bands: int | None = None,
    return_eigenvectors: bool = False,
):
    """Diagonalize the dense Hamiltonian independently at each k-point."""
    ks = np.asarray(kpoints, dtype=float)
    if ks.ndim != 2 or ks.shape[1] != 3:
        raise ValueError("kpoints must have shape (n_k, 3)")
    size = len(np.asarray(basis_indices))
    n = size if n_bands is None else n_bands
    if not 1 <= n <= size:
        raise ValueError("n_bands must lie between one and the basis size")
    energies, vectors = [], []
    for k in ks:
        h = assemble_hamiltonian(basis_indices, reciprocal_lattice, k, potential)
        eigenvalues, eigenvectors = np.linalg.eigh(h)
        energies.append(eigenvalues[:n])
        vectors.append(eigenvectors[:, :n])
    e = np.stack(energies)
    return (e, np.stack(vectors)) if return_eigenvectors else e
