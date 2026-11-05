"""Fast band evaluation for large screens (2026-11-01): the same model and Hamiltonian, solved along a path with CheFSI
warm-started from the neighbouring k batch (torch_pw.chain_band_energies, fft_solver.fft_chain_band_energies) instead
of a dense eigh (or a cold FFT start) at every k. Opt-in; band_path is unchanged without ``cache.chain``."""

from __future__ import annotations

import math

import numpy as np
import torch


def interleaved(n: int, batch: int) -> np.ndarray:
    """k order for chained batches: the path is cut into ``batch`` consecutive sub-paths of length L and batch t holds
    point t of each, so member i of a batch follows member i of the previous one. Short sub-paths repeat the last k."""
    length = math.ceil(n / batch)
    return np.array([min(j * length + t, n - 1) for t in range(length) for j in range(batch)], dtype=np.int64)


@torch.no_grad()
def chain_bands(model, structure, kfrac: np.ndarray, n_bands: int, batch: int = 8, tol: float = 1e-3,
                max_pass: int = 12) -> tuple[np.ndarray, dict]:
    """Lowest ``n_bands`` energies at ``kfrac`` (in path order), in one band_path call with the warm chain; returns the
    energies (n_k, n_bands) and the solver counts (eigh / chefsi / fallback / fft_warm / fft_cold / fft_unconverged).
    ``tol``: CheFSI residual tolerance in eV (both paths); ``max_pass``: dense-path passes before the eigh fallback."""
    packed = model.pack(structure)
    dev = model.element_features.device
    for name, value in vars(packed).items():
        if isinstance(value, torch.Tensor):
            setattr(packed, name, value.to(dev))
    order = interleaved(len(kfrac), batch)
    k = torch.as_tensor(np.asarray(kfrac)[order], dtype=packed.lattice.dtype, device=dev)
    cache = model.make_pw_cache(packed, k)
    cache.chain = {"tol": tol, "max_pass": max_pass}
    kb0 = model.config["k_batch"]
    model.config["k_batch"] = batch
    try:
        e = model.band_path(packed, k, n_bands=n_bands, cache=cache)
    finally:
        model.config["k_batch"] = kb0
    out = np.empty((len(kfrac), n_bands))
    out[order] = e.detach().cpu().numpy().astype(float)
    return out, dict(cache.chain.get("stats", {}))
