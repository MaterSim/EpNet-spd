"""Differentiable local plane-wave Hamiltonian with energies in eV."""

from __future__ import annotations

import math

import torch
from torch import Tensor

# T[eV] = KINETIC_EV_A2 * |k[Å^-1]|^2  (ħ²/2m_e).
KINETIC_EV_A2 = 3.809982116151619
BOHR_ANGSTROM = 0.529177210903
HARTREE_EV = 27.211386245988
# 4π e²/(4πε0) in eV·Å, so V[eV] = COULOMB_EV_A * 4π S(G) / (Ω[Å³] |G[Å⁻¹]|²).
COULOMB_EV_A = HARTREE_EV * BOHR_ANGSTROM


def g_cutoff_angstrom(ecut_ry: float) -> float:
    """Return the |G| cutoff in Å⁻¹ for a kinetic cutoff in Rydberg."""
    if ecut_ry <= 0:
        raise ValueError("ecut_ry must be positive")
    return math.sqrt(ecut_ry) / BOHR_ANGSTROM


def pairwise_g_differences(g_indices: Tensor) -> Tensor:
    if g_indices.ndim != 2 or g_indices.shape[-1] != 3:
        raise ValueError("g_indices must have shape (n_g, 3)")
    return g_indices[:, None, :] - g_indices[None, :, :]


def pairwise_g_cartesian(g_indices: Tensor, reciprocal_lattice: Tensor) -> Tensor:
    diffs = pairwise_g_differences(g_indices).to(reciprocal_lattice)
    return diffs @ reciprocal_lattice


def pairwise_momentum_transfer(g_indices: Tensor, reciprocal_lattice: Tensor) -> Tensor:
    return pairwise_g_cartesian(g_indices, reciprocal_lattice)


def unique_g_transfers(g_indices: Tensor, reciprocal_lattice: Tensor) -> tuple[Tensor, Tensor]:
    """Return unique Cartesian ΔG and the (n_g, n_g) inverse index map.

    Uniqueness is on integer Miller differences, then mapped with the reciprocal lattice.
    """
    diffs = pairwise_g_differences(g_indices)
    unique_idx, inverse = torch.unique(diffs.reshape(-1, 3), dim=0, return_inverse=True)
    unique_cart = unique_idx.to(reciprocal_lattice) @ reciprocal_lattice
    return unique_cart, inverse.reshape(g_indices.shape[0], g_indices.shape[0])


def unique_g_transfers_encoded(g_indices: Tensor, reciprocal_lattice: Tensor) -> tuple[Tensor, Tensor]:
    """Same result as ``unique_g_transfers`` (up to the order of the unique transfers): the integer differences are
    packed into one int64 key, so the unique is a 1-D sort. Used to rebuild the (n_g, n_g) map on the device
    instead of storing it (it grows as n_g^2 and dominates large-cell caches)."""
    g = g_indices.to(torch.int64)
    gmax = int(g.abs().max().item()) if g.numel() else 0
    span, off = 4 * gmax + 1, 2 * gmax                            # differences lie in [-2 gmax, 2 gmax]
    d = g[:, None, :] - g[None, :, :] + off                      # (n, n, 3), each component in [0, span)
    key = (d[..., 0] * span + d[..., 1]) * span + d[..., 2]
    uniq, inverse = torch.unique(key.reshape(-1), return_inverse=True)
    m = torch.stack((uniq // (span * span), (uniq // span) % span, uniq % span), dim=-1) - off
    return m.to(reciprocal_lattice) @ reciprocal_lattice, inverse.reshape(g.shape[0], g.shape[0])


def structure_phases(g_indices: Tensor, fractional_positions: Tensor) -> Tensor:
    """Return Σ_a exp(-2πi (G_i-G_j)·r_a) with r in crystal coordinates."""
    diffs = pairwise_g_differences(g_indices).to(fractional_positions)
    n_g, n_atoms = diffs.shape[0], fractional_positions.shape[0]
    # Large external cells (e.g. 82-atom mS164) need n_g^2 * n_atoms complex entries at once; sum over atom
    # chunks instead. Cells below the limit keep the single-shot path unchanged.
    atoms_per_chunk = max(1, int(1.5e9 // max(n_g * n_g, 1)))
    if n_atoms <= atoms_per_chunk:
        dots = torch.einsum("ijc,ac->ija", diffs, fractional_positions)
        return torch.exp(-2j * math.pi * dots).sum(dim=-1)
    total = None
    for a0 in range(0, n_atoms, atoms_per_chunk):
        dots = torch.einsum("ijc,ac->ija", diffs, fractional_positions[a0 : a0 + atoms_per_chunk])
        part = torch.exp(-2j * math.pi * dots).sum(dim=-1)
        total = part if total is None else total + part
    return total


def fold_to_first_zone(k_frac: Tensor, lattice: Tensor, reach: int = 2, tie_tol: float = 1e-5) -> Tensor:
    """Each fractional k mapped to its shortest image k - G (first Brillouin zone).

    The plane-wave basis is a fixed |G| < gcut sphere centred at G = 0, so it is only
    k-invariant for k near the origin; MP band paths can leave the first zone (e.g. X_1 or
    T in centred lattices), where the truncated basis gives energies several eV too high.
    """
    k = k_frac - torch.round(k_frac)
    reciprocal = 2.0 * torch.pi * torch.linalg.inv(lattice.detach().to(torch.float64)).transpose(0, 1)
    r = torch.arange(-reach, reach + 1, dtype=torch.float64, device=k.device)
    shifts = torch.stack(torch.meshgrid(r, r, r, indexing="ij"), dim=-1).reshape(-1, 3)
    shifts = shifts[torch.argsort(shifts.abs().sum(-1), stable=True)]  # unshifted first
    cand = k.to(torch.float64)[:, None, :] - shifts[None]
    norm = torch.linalg.vector_norm(cand @ reciprocal.to(k.device), dim=-1)
    # zone-boundary ties: take the first image in the fixed shift order, not the float argmin,
    # so the choice depends only on fractional k (rotation invariant)
    best = (norm <= norm.min(dim=1, keepdim=True).values + tie_tol).to(norm.dtype).argmax(dim=1)
    return cand[torch.arange(k.shape[0], device=k.device), best].to(k_frac.dtype)


def kinetic_diagonal(g_indices: Tensor, reciprocal_lattice: Tensor, k_frac: Tensor) -> Tensor:
    """Return T(k+G) in eV. ``k_frac`` is crystal coordinates, shape (3,) or (n_k, 3)."""
    g_cart = g_indices.to(reciprocal_lattice) @ reciprocal_lattice
    k = k_frac.to(reciprocal_lattice)
    if k.ndim == 1:
        k = k[None, :]
    k_cart = k @ reciprocal_lattice
    momentum = k_cart[:, None, :] + g_cart[None, :, :]
    return KINETIC_EV_A2 * (momentum * momentum).sum(dim=-1)


def local_potential_matrix(
    phases: Tensor,
    form_factor: Tensor,
    volume: Tensor | float,
    *,
    q: Tensor | None = None,
    zero_gauge: bool = True,
) -> Tensor:
    """Assemble V_{GG'} = S(G-G') v(|G-G'|) / Ω, then Hermitian-symmetrize."""
    values = form_factor.to(phases)
    if values.ndim == 1:
        matrix = phases * values
    else:
        matrix = phases * values
    matrix = matrix / float(volume)
    if zero_gauge:
        n = matrix.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=matrix.device)
        matrix = matrix.masked_fill(eye, 0)
    return 0.5 * (matrix + matrix.conj().transpose(-1, -2))


def add_multipole_potential(
    spherical: Tensor,
    multipole_values: Tensor,
    inverse: Tensor,
    volume: Tensor | float,
    q: Tensor,
    *,
    zero_gauge: bool = True,
) -> Tensor:
    """Add a unique-ΔG multipole form factor onto a spherical V_{GG'}."""
    extra = multipole_values.to(spherical)[inverse.to(dtype=torch.int64)] / float(volume)
    if zero_gauge:
        n = extra.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=extra.device)
        extra = extra.masked_fill(eye, 0)
    out = spherical + extra
    return 0.5 * (out + out.conj().transpose(-1, -2))


# Batched MKL eigvalsh beats scipy subset until n_G is well above this catalog.
# Tests pass subset_ng_min=0 to exercise the lowest-n autograd path.
SUBSET_NG_MIN = 4096


def _subset_eigh(ham: Tensor, n_bands: int) -> tuple[Tensor, Tensor]:
    """Lowest ``n_bands`` eigenpairs of a batched Hermitian matrix (CPU)."""
    import numpy as np
    import scipy.linalg as sla

    mats = ham.detach().cpu().contiguous().numpy().copy()
    n_k, n_g, _ = mats.shape
    evals = np.empty((n_k, n_bands), dtype=np.float64)
    evecs = np.empty((n_k, n_g, n_bands), dtype=mats.dtype)
    for ik in range(n_k):
        values, vectors = sla.eigh(
            mats[ik],
            subset_by_index=(0, n_bands - 1),
            driver="evr",
            check_finite=False,
            overwrite_a=True,
        )
        evals[ik] = values
        evecs[ik] = vectors
    return (
        torch.as_tensor(evals, device=ham.device, dtype=ham.real.dtype),
        torch.as_tensor(evecs, device=ham.device, dtype=ham.dtype),
    )


class _LowestEigvalsh(torch.autograd.Function):
    """Eigenvalues of the lowest ``n_bands`` Hermitian subspace, with projector backward."""

    @staticmethod
    def forward(ctx, ham: Tensor, n_bands: int) -> Tensor:
        evals, evecs = _subset_eigh(ham, int(n_bands))
        ctx.save_for_backward(evecs)
        return evals

    @staticmethod
    def backward(ctx, grad_evals: Tensor):
        (evecs,) = ctx.saved_tensors
        scaled = evecs * grad_evals.to(dtype=evecs.real.dtype).unsqueeze(-2).to(evecs)
        grad_ham = scaled @ evecs.transpose(-1, -2).conj()
        grad_ham = 0.5 * (grad_ham + grad_ham.transpose(-1, -2).conj())
        return grad_ham, None


class _TruncatedEigvalsh(torch.autograd.Function):
    """Full dense eigh, but backward only through the lowest ``n_bands`` evecs.

    The catalog loss never sees the high spectrum, so this is the exact
    ``eigvalsh`` gradient with a much cheaper ``U_s diag(g) U_s†``.
    """

    @staticmethod
    def forward(ctx, ham: Tensor, n_bands: int) -> Tensor:
        evals, evecs = torch.linalg.eigh(ham)
        n_keep = int(n_bands)
        ctx.save_for_backward(evecs[..., :n_keep].contiguous())
        return evals[..., :n_keep]

    @staticmethod
    def backward(ctx, grad_evals: Tensor):
        (evecs,) = ctx.saved_tensors
        scaled = evecs * grad_evals.to(dtype=evecs.real.dtype).unsqueeze(-2).to(evecs)
        grad_ham = scaled @ evecs.transpose(-1, -2).conj()
        grad_ham = 0.5 * (grad_ham + grad_ham.transpose(-1, -2).conj())
        return grad_ham, None


class _ScatterHermitianPotential(torch.autograd.Function):
    """``V = herm(v_unique[inverse] / Ω)`` without graphing the n_G×n_G gather."""

    @staticmethod
    def forward(ctx, v_unique: Tensor, inverse: Tensor, volume: float) -> Tensor:
        # Ampere (A40/A100) CUDA gather asserts on int32 complex index; Ada L40S does not.
        inverse = inverse.to(dtype=torch.int64)
        ctx.save_for_backward(inverse)
        ctx.n_unique = int(v_unique.shape[0])
        ctx.volume = float(volume)
        matrix = v_unique[inverse] / ctx.volume
        eye = torch.eye(matrix.shape[0], dtype=torch.bool, device=matrix.device)
        matrix = matrix.masked_fill(eye, 0)
        return 0.5 * (matrix + matrix.conj().transpose(-1, -2))

    @staticmethod
    def backward(ctx, grad_matrix: Tensor):
        (inverse,) = ctx.saved_tensors
        g = 0.5 * (grad_matrix + grad_matrix.conj().transpose(-1, -2)) / ctx.volume
        eye = torch.eye(g.shape[0], dtype=torch.bool, device=g.device)
        g = g.masked_fill(eye, 0)
        dv = torch.zeros(ctx.n_unique, dtype=g.dtype, device=g.device)
        dv.scatter_add_(0, inverse.reshape(-1), g.reshape(-1))
        return dv, None, None


class _GatherRowsCols(torch.autograd.Function):
    """``out[k] = matrix[idx[k][:, None], idx[k][None, :]]`` for a k-centred basis. Plain advanced indexing backs
    propagates with a sort-based accumulating index_put (radix sort, ~27% of warm-step GPU time, 2026-10-17 profile);
    here the backward is one atomic scatter_add over the flattened matrix (same sum, summation order aside)."""

    @staticmethod
    def forward(ctx, matrix: Tensor, idx: Tensor) -> Tensor:
        n = int(matrix.shape[-1])
        idx = idx.to(torch.int64)
        lin = idx[:, :, None] * n + idx[:, None, :]
        ctx.save_for_backward(lin)
        ctx.n = n
        return matrix.reshape(-1)[lin]

    @staticmethod
    def backward(ctx, grad: Tensor):
        (lin,) = ctx.saved_tensors
        out = torch.zeros(ctx.n * ctx.n, dtype=grad.dtype, device=grad.device)
        out.scatter_add_(0, lin.reshape(-1), grad.reshape(-1))
        return out.view(ctx.n, ctx.n), None


def gather_rows_cols(matrix: Tensor, idx: Tensor) -> Tensor:
    return _GatherRowsCols.apply(matrix, idx)


def scatter_hermitian_potential(v_unique: Tensor, inverse: Tensor, volume: float) -> Tensor:
    return _ScatterHermitianPotential.apply(v_unique, inverse, float(volume))


def band_energies(
    kinetic: Tensor,
    potential: Tensor,
    *,
    n_bands: int | None = None,
    subset_ng_min: int = SUBSET_NG_MIN,
) -> Tensor:
    if kinetic.ndim != 2:
        raise ValueError("kinetic must have shape (n_k, n_g)")
    ham = potential + torch.diag_embed(kinetic.to(dtype=potential.dtype))
    n_g = int(ham.shape[-1])
    use_subset = (
        n_bands is not None
        and 0 < int(n_bands) < n_g
        and n_g >= int(subset_ng_min)
        and ham.device.type == "cpu"
    )
    if use_subset:
        return _LowestEigvalsh.apply(ham, int(n_bands))
    if n_bands is not None and 0 < int(n_bands) < n_g and ham.requires_grad:
        return _TruncatedEigvalsh.apply(ham, int(n_bands))
    evals = torch.linalg.eigvalsh(ham)
    if n_bands is not None:
        evals = evals[:, :n_bands]
    return evals


# ---------------------------------------------------------------------------------------------------------------
# Warm-started Chebyshev-filtered subspace iteration (CheFSI) for the lowest bands (training path, 2026-10-12).
# Dense eigh spends ~all its time reducing H to tridiagonal form (~9 n^3), whatever the number of bands wanted.
# CheFSI needs ~10 n^2 m flops per pass (GEMMs), m = bands + buffer, and 2-5 passes from the previous visit's
# eigenvectors. Benchmark on real G3 Hamiltonians: 3.8-4.1x faster than eigh and 0.04-0.20 meV from fp64
# (fp32 eigh itself errs up to ~1.9 meV). Padded k-centred rows (kinetic 1e4 eV, decoupled) are clamped to the
# top of the physical spectrum so they do not stretch the filter interval; they never enter the lowest bands.

PAD_THRESHOLD_EV = 5.0e3


def _spectral_upper_bound(ham: Tensor, steps: int = 8) -> Tensor:
    """Power-iteration estimate of the largest eigenvalue per batch member (x1.05 + 1 eV safety)."""
    v = torch.randn(ham.shape[0], ham.shape[-1], 1, dtype=ham.dtype, device=ham.device)
    for _ in range(steps):
        v = ham @ v
        v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True)
    lam = (v.mH @ ham @ v).real[:, 0, 0]
    return lam * 1.05 + 1.0


def _lanczos_upper_bound(ham: Tensor, steps: int = 16) -> Tensor:
    """k-step Lanczos bound per batch member: largest Ritz value + |beta_k| + 1 eV (as fft_solver.PWOperator.upper_bound).
    Used when the power-iteration bound falls below the current Ritz block (2026-10-19: G4 crash, a negative damping
    ratio); unlike power iteration it cannot be pulled toward a large-|lambda| eigenvalue at the bottom."""
    b, n = ham.shape[0], ham.shape[-1]
    steps = min(steps, n)
    v = torch.randn(b, n, 1, dtype=ham.dtype, device=ham.device)
    v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True)
    alpha = torch.zeros(b, steps, dtype=v.real.dtype, device=v.device)
    beta = torch.zeros(b, steps, dtype=v.real.dtype, device=v.device)
    v_prev = torch.zeros_like(v)
    for j in range(steps):
        w = ham @ v
        alpha[:, j] = (v.mH @ w).real[:, 0, 0]
        w = w - alpha[:, j, None, None].to(w.dtype) * v - (beta[:, j - 1, None, None].to(w.dtype) * v_prev if j else 0)
        beta[:, j] = torch.linalg.vector_norm(w, dim=1)[:, 0]
        v_prev, v = v, w / beta[:, j, None, None].clamp_min(1e-30).to(w.dtype)
    tri = torch.diag_embed(alpha) + torch.diag_embed(beta[:, :-1], 1) + torch.diag_embed(beta[:, :-1], -1)
    return torch.linalg.eigvalsh(tri)[:, -1] + beta[:, -1] + 1.0


def _clamped(ham: Tensor, kinetic: Tensor) -> tuple[Tensor, Tensor]:
    """H with padded diagonal entries moved to the top of the physical spectrum, and that bound."""
    pad = kinetic > PAD_THRESHOLD_EV
    if not bool(pad.any()):
        return ham, _spectral_upper_bound(ham)
    diag = torch.diagonal(ham, dim1=-2, dim2=-1).real
    phys = ham - torch.diag_embed(torch.where(pad, diag, torch.zeros_like(diag)).to(ham.dtype))
    bound = _spectral_upper_bound(phys)
    shift = torch.where(pad, bound[:, None].expand_as(diag), torch.zeros_like(diag))
    return phys + torch.diag_embed(shift.to(ham.dtype)), bound


def _rayleigh_ritz(ham: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    q, _ = torch.linalg.qr(y)
    hs = q.mH @ (ham @ q)
    theta, v = torch.linalg.eigh(0.5 * (hs + hs.mH))
    return theta, q @ v


def _cheb_filter(ham: Tensor, x: Tensor, degree: int, a: Tensor, b: Tensor, a0: Tensor) -> Tensor:
    """Scaled Chebyshev filter (Zhou & Saad): damps [a, b], amplifies the spectrum below a; a0 ~ lowest eigenvalue."""
    e = ((b - a) / 2)[:, None, None].to(x.real.dtype)
    c = ((b + a) / 2)[:, None, None].to(x.real.dtype)
    sigma = e / (a0[:, None, None].to(x.real.dtype) - c)
    tau = 2 / sigma
    y = (ham @ x - c * x) * (sigma / e)
    x_prev = x
    for _ in range(2, degree + 1):
        sigma_new = 1 / (tau - sigma)
        y_new = 2 * sigma_new / e * (ham @ y - c * y) - sigma * sigma_new * x_prev
        x_prev, y, sigma = y, y_new, sigma_new
    return y


def _adaptive_degree(theta: Tensor, bound: Tensor, base: int) -> int:
    """Chebyshev degree for the damping ratio (b - a) / (a - a0) (as epnet.hamiltonian.fft_solver), 2026-10-18."""
    ratio = float(((bound - theta[:, -1]).clamp_min(0.0) / (theta[:, -1] - theta[:, 0]).clamp_min(1e-3)).max())
    if not math.isfinite(ratio):   # Stage 5a-g: a NaN here crashed a run; as a linear-algebra error --skip-nonfinite skips it
        raise torch.linalg.LinAlgError("non-finite Ritz values in CheFSI")
    return int(min(40, max(base, round(2.5 * ratio ** 0.5))))


def chefsi_lowest(ham: Tensor, kinetic: Tensor, x0: Tensor, n_bands: int, *, degree: int = 10, tol: float = 1e-3,
                  max_pass: int = 12, stats: dict | None = None) -> tuple[Tensor, Tensor, bool]:
    """Lowest ``n_bands`` eigenpairs of a batch of Hermitian ``ham`` from the start block ``x0`` (b, n, m >= n_bands).
    Returns (eigenvalues (b, n_bands), Ritz block (b, n, m), converged). Converged = every wanted residual < tol (eV)."""
    with torch.no_grad():
        h, bound = _clamped(ham, kinetic)
        # float32 floor (2026-10-18): residuals level off near eps |H| (20 Ry first-row cells: |H| ~ several hundred eV,
        # ~1e-3 eV), so a fixed 1e-3 eV tolerance sent 30% of warm visits in training to the dense fallback
        tol_eff = max(tol, 50.0 * torch.finfo(x0.real.dtype).eps * float(bound.abs().max()))
        theta, x = _rayleigh_ritz(h, x0)
        low = bound <= theta[:, -1]            # power-iteration bound below the Ritz block: the filter would be wrong
        if bool(low.any()):
            bound = torch.where(low, torch.maximum(_lanczos_upper_bound(h), theta[:, -1] + 1.0), bound)
            if stats is not None:
                stats["chefsi_bound_fix"] = stats.get("chefsi_bound_fix", 0) + 1
        history = []
        for _ in range(max_pass):
            x = _cheb_filter(h, x, _adaptive_degree(theta, bound, degree), theta[:, -1], bound, theta[:, 0])
            theta, x = _rayleigh_ritz(h, x)
            r = h @ x[..., :n_bands] - x[..., :n_bands] * theta[:, None, :n_bands].to(x.dtype)
            res = float(torch.linalg.vector_norm(r, dim=1).max())
            history.append(res)
            stalled = len(history) >= 3 and res > 0.9 * history[-3] and res < 20 * tol_eff
            if res < tol_eff or stalled:
                if stats is not None:
                    stats["chefsi_passes"] = stats.get("chefsi_passes", 0) + _ + 1
                return theta[:, :n_bands], x, True
        if stats is not None:
            stats["chefsi_passes"] = stats.get("chefsi_passes", 0) + max_pass
        return theta[:, :n_bands], x, False



def warm_pack(x: Tensor, store: dict) -> Tensor:
    """A warm-start block for the store: host memory if ``store["_host"]``, half precision if ``store["_half"]``
    (Stage 5c: the <= 24-site warm store is ~400 GB in single precision; it only seeds the next filter)."""
    x = x.detach()
    if store.get("_half"):
        x = x.to(torch.complex32 if x.is_complex() else torch.float16)
    return x.to("cpu") if store.get("_host", False) else x.contiguous()

class _StoredEigvalsh(torch.autograd.Function):
    """Lowest eigenvalues with the low-eigenvector backward of _TruncatedEigvalsh. Forward: CheFSI from the stored
    block when available (falls back to dense eigh if it does not converge), else dense eigh; refreshes the store."""

    @staticmethod
    def forward(ctx, ham: Tensor, kinetic: Tensor, n_bands: int, store: dict, key, n_block: int) -> Tensor:
        if not bool(torch.isfinite(ham).all()):
            raise torch.linalg.LinAlgError("non-finite Hamiltonian")
        x0 = store.get(key)
        if x0 is not None and not bool(torch.isfinite(x0).all()):   # a corrupted warm block: start this cell cold
            x0 = None
            store["warm_nonfinite"] = store.get("warm_nonfinite", 0) + 1
        host = bool(store.get("_host", False))  # warm store kept in host memory (it outgrows a 44 GB GPU on G3 v3)
        if host and x0 is not None:
            x0 = x0.to(ham.device, non_blocking=True)
        if x0 is not None:
            x0 = x0.to(ham.dtype)                       # a half-precision store (warm_pack) back to the solver's dtype
        evals = vecs = None
        tried = False
        if x0 is not None and x0.shape == (ham.shape[0], ham.shape[-1], n_block) and x0.device == ham.device:
            tried = True
            evals, block, ok = chefsi_lowest(ham, kinetic, x0, n_bands, stats=store)
            if ok:
                vecs = block
                store["chefsi"] = store.get("chefsi", 0) + 1
        if vecs is None:
            ev, vv = torch.linalg.eigh(ham)
            evals, vecs = ev[..., :n_bands], vv[..., :n_block]
            store["eigh"] = store.get("eigh", 0) + 1
            if tried:                                       # warm block present but CheFSI did not converge
                store["chefsi_fallback"] = store.get("chefsi_fallback", 0) + 1
        store[key] = warm_pack(vecs, store)
        ctx.save_for_backward(vecs[..., :n_bands].contiguous())
        return evals

    @staticmethod
    def backward(ctx, grad_evals: Tensor):
        (evecs,) = ctx.saved_tensors
        scaled = evecs * grad_evals.to(dtype=evecs.real.dtype).unsqueeze(-2).to(evecs)
        grad_ham = scaled @ evecs.transpose(-1, -2).conj()
        grad_ham = 0.5 * (grad_ham + grad_ham.transpose(-1, -2).conj())
        return grad_ham, None, None, None, None, None


def map_block(x_prev: Tensor | None, idx_prev: Tensor | None, idx_cur: Tensor | None, valid_cur: Tensor | None) -> Tensor | None:
    """Eigenvector block of the previous k batch (b, n, m) re-indexed to the current batch's k-centred bases: rows follow
    the plane-wave (union) index, so a G present in both spheres keeps its coefficient and the rest start at zero."""
    if x_prev is None:
        return None
    if idx_cur is None:                                   # one basis for all k
        return x_prev
    if idx_prev is None or idx_prev.shape != idx_cur.shape or x_prev.shape[:2] != idx_cur.shape:
        return None
    b, n = idx_prev.shape
    n_u = int(torch.maximum(idx_prev.max(), idx_cur.max())) + 1
    inv = torch.full((b, n_u), -1, dtype=torch.long, device=idx_prev.device)
    inv.scatter_(1, idx_prev, torch.arange(n, device=idx_prev.device).expand(b, n))
    slots = torch.gather(inv, 1, idx_cur)
    keep = (slots >= 0) if valid_cur is None else (slots >= 0) & valid_cur
    x0 = torch.gather(x_prev, 1, slots.clamp_min(0)[..., None].expand(-1, -1, x_prev.shape[-1]))
    return x0 * keep[..., None].to(x0.real.dtype)


def chain_band_energies(kinetic: Tensor, potential: Tensor, *, n_bands: int, chain: dict, k_index: Tensor | None = None,
                        valid: Tensor | None = None) -> Tensor:
    """Evaluation (no gradients): lowest bands of a k batch by CheFSI warm-started from the previous batch along the path
    (``chain``: its Ritz block and k_index), dense eigh for the first batch and whenever CheFSI does not converge. The
    caller orders k so that member i of each batch follows member i of the previous one (screen.interleaved)."""
    ham = potential + torch.diag_embed(kinetic.to(dtype=potential.dtype))
    n_block = min(int(ham.shape[-1]), n_bands + max(8, n_bands // 4))
    stats = chain.setdefault("stats", {})
    x0 = map_block(chain.get("x"), chain.get("k_index"), k_index, valid)
    if x0 is not None and x0.shape[-1] == n_block:
        evals, block, ok = chefsi_lowest(ham, kinetic, x0.to(ham.dtype), n_bands, tol=float(chain.get("tol", 1e-3)),
                                         max_pass=int(chain.get("max_pass", 12)), stats=stats)
        if ok and bool(torch.isfinite(evals).all()):
            chain.update(x=block, k_index=k_index)
            stats["chefsi"] = stats.get("chefsi", 0) + 1
            return evals
        stats["fallback"] = stats.get("fallback", 0) + 1
    ev, vv = torch.linalg.eigh(ham)
    chain.update(x=vv[..., :n_block], k_index=k_index)
    stats["eigh"] = stats.get("eigh", 0) + 1
    return ev[..., :n_bands]


def stored_band_energies(kinetic: Tensor, potential: Tensor, *, n_bands: int, store: dict, key) -> Tensor:
    """Training-path band energies with a persistent warm-start ``store`` (one dict per cell, keyed by k-chunk)."""
    ham = potential + torch.diag_embed(kinetic.to(dtype=potential.dtype))
    n_block = min(int(ham.shape[-1]), n_bands + max(8, n_bands // 4))
    return _StoredEigvalsh.apply(ham, kinetic, int(n_bands), store, key, n_block)


def batched_band_energies(
    kinetic: Tensor,
    potential: Tensor,
    *,
    n_bands: int | None = None,
    k_batch: int | None = None,
    subset_ng_min: int = SUBSET_NG_MIN,
) -> Tensor:
    n_g = int(kinetic.shape[1])
    if k_batch is None:
        if kinetic.device.type == "cuda":
            # One cuSOLVER batch beats many small launches. 33×799 H is ~0.3 GB.
            k_batch = max(int(kinetic.shape[0]), 1)
        else:
            k_batch = 16 if n_g < 400 else 4
    chunks = []
    for start in range(0, kinetic.shape[0], k_batch):
        sl = slice(start, start + k_batch)
        chunk_potential = potential[sl] if potential.ndim == 3 else potential
        chunks.append(
            band_energies(
                kinetic[sl],
                chunk_potential,
                n_bands=n_bands,
                subset_ng_min=subset_ng_min,
            )
        )
    return torch.cat(chunks, dim=0)


def vbm_aligned_window_loss(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    *,
    half_width: float = 2.0,
    edge_weight: float = 5.0,
    anchor: str = "mid",
) -> Tensor:
    """MAE after aligning VBMs, restricted to ±half_width eV of the target midgap (``anchor="mid"``), or to
    [VBM - half_width, CBM + half_width] (``anchor="edges"``): a midgap window holds only the two edge bands
    once the gap exceeds 2*half_width, leaving the valence/conduction manifolds of wide-gap cells untrained."""
    pred = prediction
    tgt = target
    pred_vbm = pred[:, n_occupied - 1].amax()
    tgt_vbm = tgt[:, n_occupied - 1].amax()
    aligned = pred - pred_vbm + tgt_vbm
    tgt_cbm = tgt[:, n_occupied].amin()
    if anchor == "edges":
        mask = (tgt >= tgt_vbm - half_width) & (tgt <= tgt_cbm + half_width)
    elif anchor == "mid":
        mid = 0.5 * (tgt_vbm + tgt_cbm)
        mask = (tgt - mid).abs() <= half_width
    else:
        raise ValueError(f"unknown window anchor {anchor!r}")
    # A gap wider than 2*half_width would otherwise select no states and make
    # the window loss identically zero.  Band edges are always part of the
    # objective, irrespective of gap size.
    mask[:, n_occupied - 1] = True
    mask[:, n_occupied] = True
    weight = torch.ones_like(tgt)
    weight[:, n_occupied - 1] = edge_weight
    weight[:, n_occupied] = edge_weight
    weight = weight * mask.to(weight)
    denom = weight.sum().clamp_min(1.0)
    return ((aligned - tgt).abs() * weight).sum() / denom


def fermi_aligned_window_loss(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    *,
    half_width: float = 2.0,
) -> Tensor:
    """MAE after aligning Fermi proxies, restricted to ±half_width eV of target E_F.

    Used for metallic cells. The Fermi proxy is the mid-point of the highest
    occupied and lowest unoccupied eigenvalues; gap-edge band weights are not
    applied.
    """
    pred = prediction
    tgt = target
    pred_ef = 0.5 * (pred[:, n_occupied - 1].amax() + pred[:, n_occupied].amin())
    tgt_ef = 0.5 * (tgt[:, n_occupied - 1].amax() + tgt[:, n_occupied].amin())
    aligned = pred - pred_ef + tgt_ef
    mask = (tgt - tgt_ef).abs() <= half_width
    denom = mask.to(tgt).sum().clamp_min(1.0)
    return ((aligned - tgt).abs() * mask.to(tgt)).sum() / denom


def fermi_level_window_loss(
    prediction: Tensor,
    target: Tensor,
    fermi_energy: float,
    *,
    half_width: float = 2.0,
) -> tuple[Tensor, Tensor]:
    """Metal window loss centred on the true E_F (``fermi_level_window_mae``): MAE over target states within
    ±half_width eV of E_F after the best rigid shift (median of target - prediction there, detached). Returns the
    loss and the shift; the deep-band term reuses the shift."""
    diff = target - prediction
    mask = (target - float(fermi_energy)).abs() <= half_width
    if not bool(mask.any()):
        mask = torch.ones_like(mask)
    shift = diff[mask].median().detach()
    return (diff - shift).abs()[mask].mean(), shift


def deep_band_loss(
    prediction: Tensor,
    target: Tensor,
    n_occupied: int,
    gapped: bool,
    *,
    n_skip: int = 0,
    floor_rel: float | None = None,
    floor_margin: float = 2.0,
) -> Tensor:
    """MAE over the occupied bands the window never reaches, plus a ghost-state floor.

    Bands are aligned at the VBM (gapped) or at the Fermi proxy (metals), as in the
    window losses. Occupied bands ``n_skip:`` are matched to the target; the lowest
    ``n_skip`` are left free because the contract drop made them ambiguous (manifest
    ``n_deep_ambiguous``). ``floor_rel`` is the lowest PAW band relative to the target
    edge (manifest ``lowest_paw_band_minus_edge_eV``): any occupied predicted band more
    than ``floor_margin`` eV below it is penalized linearly, so a skipped band cannot
    become a ghost state.
    """
    def ref(x: Tensor) -> Tensor:
        top = x[:, n_occupied - 1].amax()
        return top if gapped else 0.5 * (top + x[:, n_occupied].amin())

    pred = prediction[:, :n_occupied] - ref(prediction)
    loss = prediction.new_zeros(())
    if n_skip < n_occupied:
        tgt = target[:, :n_occupied] - ref(target)
        loss = loss + (pred[:, n_skip:] - tgt[:, n_skip:]).abs().mean()
    if floor_rel is not None:
        loss = loss + torch.relu((float(floor_rel) - floor_margin) - pred).mean()
    return loss
