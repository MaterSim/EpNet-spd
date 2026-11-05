"""Operator-form (FFT) CheFSI for the training path (2026-10-18).

The dense training path builds H (k, m, m) per k-chunk and runs CheFSI (warm) or eigh (cold). Here H is never built:

    H X = kinetic * X + local(X) + B (D (B^H X)),     local(X)(G) = sum_G' V(G - G') X(G')  via the FFT grid
                                                                                          (epnet.hamiltonian.fft_grid)

on each k's own sphere (padded entries of a k-centred basis masked out: the search space stays in the valid plane
waves, so no spectral clamping is needed). Cold visits start from the lowest-kinetic plane waves instead of a dense
eigh. Gradients: Hellmann-Feynman via a surrogate E_n = <psi_n|H(theta)|psi_n> with psi detached, evaluated with the
same differentiable pieces (kinetic is constant; local through V(r); nonlocal through D and B), returned straight-through
on top of the solver eigenvalues. This equals the dense path's low-eigenvector backward (sum_n g_n |psi_n><psi_n|).

Ultrasoft atomic channels (EpNet-spd Stage 5b, 2026-10-23): H x = E S x with the low-rank S = 1 + B_a q B_a^H over the
augmented projector columns (q real symmetric, PSD in the safe two-reference construction). CheFSI runs on the Löwdin
operator A = S^-1/2 H S^-1/2, with S^-1/2 = 1 + U ((1 + M)^-1/2 - 1) U^H from the thin QR B_a = U R and M = R q R^H
(a c x c eigendecomposition per k, c = augmented columns): exact for this S, and A stays Hermitian, so the solver is
unchanged. Eigenvectors y of A give x = S^-1/2 y with x^H S x = 1; the surrogate becomes
E_n = <x_n|H(theta)|x_n> - E_n <x_n|S(theta)|x_n> (+ const) with x_n and E_n detached, the generalized eigenvalue
derivative x^H (dH - E dS) x that the dense path's Cholesky backward also gives.
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor

from epnet.hamiltonian.torch_pw import map_block, warm_pack


class PWOperator:
    """H for a chunk of k-points. kinetic (k, m) real; valid (k, m) bool or None; pos (k, m) flat FFT-grid positions;
    v_r (N_r,) real local potential on the grid; basis (k, m, c) projectors; chan (c,) real strengths.
    pair (c0, c1, d): symmetric couplings d (n,) between projector columns c0 and c1 (two-reference d channel).
    overlap (cols, q): S = 1 + B_cols q B_cols^H, q (n, n) real symmetric (ultrasoft atomic channels); ``apply`` is then
    the Löwdin operator S^-1/2 H S^-1/2 and ``apply_h`` H itself."""

    def __init__(self, kinetic: Tensor, valid: Tensor | None, pos: Tensor, shape: Sequence[int], v_r: Tensor, basis: Tensor,
                 chan: Tensor, pair: tuple[Tensor, Tensor, Tensor] | None = None,
                 overlap: tuple[Tensor, Tensor] | None = None) -> None:
        self.kinetic, self.pos, self.shape = kinetic, pos, tuple(int(s) for s in shape)
        self.valid = valid if valid is not None else torch.ones_like(kinetic, dtype=torch.bool)
        self.v_r, self.basis, self.chan = v_r, basis, chan
        self.n_r = self.shape[0] * self.shape[1] * self.shape[2]
        self.pair, self.overlap = pair, overlap
        self.lowdin = None if overlap is None else self._lowdin(*overlap)

    def mask(self, x: Tensor) -> Tensor:
        return x * self.valid[..., None].to(x.real.dtype)

    def _lowdin(self, cols: Tensor, q: Tensor) -> tuple[Tensor, Tensor]:
        """(U, W) with S^-1/2 = 1 + U W U^H: thin QR B_cols = U R (valid rows), W = (1 + M)^-1/2 - 1, M = R q R^H; the
        small eigendecomposition in float64. Householder QR keeps U orthonormal if B_cols is rank deficient (R's null
        rows give M null rows, so W ignores those columns). ``s_range``: extreme eigenvalues of S on the projector span."""
        with torch.no_grad():
            u, r = torch.linalg.qr(self.mask(self.basis[..., cols].detach()))
            wide = torch.complex128 if r.is_complex() else torch.float64
            r = r.to(wide)
            m = r @ q.detach().to(wide) @ r.mH
            lam, vec = torch.linalg.eigh(0.5 * (m + m.mH))
            self.s_range = tuple(torch.stack((1.0 + lam.min(), 1.0 + lam.max())).tolist())
            if self.s_range[0] <= 0.0:
                raise ValueError("S = 1 + B q B^H is not positive definite")
            w = (vec * ((1.0 + lam).rsqrt() - 1.0)[..., None, :].to(wide)) @ vec.mH
            return u, w.to(u.dtype)

    def inv_sqrt_s(self, x: Tensor) -> Tensor:
        """S^-1/2 x (identity without an overlap)."""
        if self.lowdin is None:
            return x
        u, w = self.lowdin
        return self.mask(x + (u @ (w @ (u.mH @ x.to(u.dtype)))).to(x.dtype))

    def real_space(self, x: Tensor) -> Tensor:
        """(k, m, nb) -> (k, nb, N_r) u(r) = sum_G x(G) e^{iGr} (padded entries carry zero)."""
        k, m, nb = x.shape
        grid = x.new_zeros(k, nb, self.n_r)
        grid.scatter_add_(2, self.pos[:, None, :].expand(k, nb, m), self.mask(x).transpose(1, 2))
        return torch.fft.ifftn(grid.reshape(k, nb, *self.shape), dim=(-3, -2, -1), norm="forward").reshape(k, nb, self.n_r)

    GRID_BYTES = 2 << 30   # Stage 5c: largest (k, bands, N_r) grid per FFT; more bands are done in blocks (an eval of a
                           # big cell asked for 31.6 GB at once on a 44 GB GPU)

    def band_block(self, k: int, nb: int, itemsize: int = 8) -> int:
        return max(1, min(nb, int(self.GRID_BYTES // max(1, k * self.n_r * itemsize))))

    def local(self, x: Tensor, v_r: Tensor | None = None) -> Tensor:
        k, m, nb = x.shape
        step = self.band_block(k, nb, x.element_size())
        if step < nb:                                    # the same per-band result, in blocks of bands
            return torch.cat([self._local(x[..., i:i + step], v_r) for i in range(0, nb, step)], dim=-1)
        return self._local(x, v_r)

    def _local(self, x: Tensor, v_r: Tensor | None = None) -> Tensor:
        k, m, nb = x.shape
        v = self.v_r if v_r is None else v_r
        w = self.real_space(x) * v.to(x.real.dtype)
        g = torch.fft.fftn(w.reshape(k, nb, *self.shape), dim=(-3, -2, -1), norm="forward").reshape(k, nb, self.n_r)
        return self.mask(torch.gather(g, 2, self.pos[:, None, :].expand(k, nb, m)).transpose(1, 2))

    def apply_h(self, x: Tensor) -> Tensor:
        b = self.basis.mH @ x.to(self.basis.dtype)
        coef = self.chan.to(b.real.dtype)[None, :, None] * b
        if self.pair is not None:
            c0, c1, d = self.pair
            d = d.to(b.real.dtype)[None, :, None]
            coef = coef.index_add(1, c0, d * b[:, c1]).index_add(1, c1, d * b[:, c0])
        nonlocal_x = self.basis @ coef
        return self.mask(self.kinetic[..., None].to(x.real.dtype) * x + self.local(x) + nonlocal_x.to(x.dtype))

    def apply(self, x: Tensor) -> Tensor:
        """The operator CheFSI diagonalizes: H, or S^-1/2 H S^-1/2 with an overlap."""
        if self.lowdin is None:
            return self.apply_h(x)
        return self.inv_sqrt_s(self.apply_h(self.inv_sqrt_s(x)))

    def upper_bound(self, steps: int = 16) -> Tensor:
        """k-step Lanczos bound per k: largest Ritz value + |beta_k| (Zhou & Saad's CheFSI estimate). The power
        iteration of the dense path (x1.05 + 1 eV) can sit below the top of a wide 20 Ry spectrum (SiO2, n_G 8282:
        unconverged visits, 2026-10-18), and an underestimated bound makes the filter amplify the unwanted top."""
        v = self.mask(torch.randn(*self.kinetic.shape, 1, dtype=self.basis.dtype, device=self.kinetic.device))
        v = v / torch.linalg.vector_norm(v, dim=1, keepdim=True).clamp_min(1e-30)
        k = v.shape[0]
        alpha = torch.zeros(k, steps, dtype=v.real.dtype, device=v.device)
        beta = torch.zeros(k, steps, dtype=v.real.dtype, device=v.device)
        v_prev = torch.zeros_like(v)
        for j in range(steps):
            w = self.apply(v)
            alpha[:, j] = (v.mH @ w).real[:, 0, 0]
            w = w - alpha[:, j, None, None].to(w.dtype) * v - (beta[:, j - 1, None, None].to(w.dtype) * v_prev if j else 0)
            beta[:, j] = torch.linalg.vector_norm(w, dim=1)[:, 0]
            v_prev, v = v, w / beta[:, j, None, None].clamp_min(1e-30).to(w.dtype)
        tri = torch.diag_embed(alpha) + torch.diag_embed(beta[:, :-1], 1) + torch.diag_embed(beta[:, :-1], -1)
        return torch.linalg.eigvalsh(tri)[:, -1] + beta[:, -1] + 1.0


def _rayleigh_ritz(op: PWOperator, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    q, _ = torch.linalg.qr(op.mask(y))
    hq = op.apply(q)
    hs = q.mH @ hq
    theta, v = torch.linalg.eigh(0.5 * (hs + hs.mH))
    return theta, q @ v, hq @ v


def _cheb_filter(op: PWOperator, x: Tensor, degree: int, a: Tensor, b: Tensor, a0: Tensor) -> Tensor:
    e = ((b - a) / 2)[:, None, None].to(x.real.dtype)
    c = ((b + a) / 2)[:, None, None].to(x.real.dtype)
    sigma = e / (a0[:, None, None].to(x.real.dtype) - c)
    tau = 2 / sigma
    y = (op.apply(x) - c * x) * (sigma / e)
    x_prev = x
    for _ in range(2, degree + 1):
        sigma_new = 1 / (tau - sigma)
        y_new = 2 * sigma_new / e * (op.apply(y) - c * y) - sigma * sigma_new * x_prev
        x_prev, y, sigma = y, y_new, sigma_new
    return y


def _degree(theta: Tensor, bound: Tensor, base: int) -> int:
    """Chebyshev degree for the damping ratio (b - a) / (a - a0): large cells (wide spectrum at 20 Ry, n_G ~ 8000)
    need more than the dense path's 10 (SiO2 n_G 8282 did not converge at 10, 2026-10-18)."""
    ratio = float(((bound - theta[:, -1]) / (theta[:, -1] - theta[:, 0]).clamp_min(1e-3)).max())
    if ratio != ratio or ratio in (float("inf"), float("-inf")):   # non-finite: as a linear-algebra error (skippable)
        raise torch.linalg.LinAlgError("non-finite Ritz values in CheFSI")
    return int(min(60, max(base, round(2.5 * ratio ** 0.5))))


def chefsi_operator(op: PWOperator, x0: Tensor, n_bands: int, *, degree: int = 10, tol: float = 1e-3,
                    max_pass: int = 6) -> tuple[Tensor, Tensor, bool, int]:
    """Lowest ``n_bands`` eigenpairs of the operator from the start block x0 (k, m, n_block)."""
    with torch.no_grad():
        bound = op.upper_bound()
        # floating-point floor: residuals of float32 FFT products on large grids level off near eps |H| (SiO2, n_G 8282,
        # |H| ~ several hundred eV: ~1e-3 eV, so a fixed 1e-3 eV tolerance was never met; 2026-10-18)
        eps = torch.finfo(x0.real.dtype).eps
        tol_eff = max(tol, 50.0 * eps * float(bound.abs().max()))
        theta, x, hx = _rayleigh_ritz(op, x0)
        history = []
        for p in range(max_pass):
            x = _cheb_filter(op, x, _degree(theta, bound, degree), theta[:, -1], bound, theta[:, 0])
            theta, x, hx = _rayleigh_ritz(op, x)
            r = hx[..., :n_bands] - x[..., :n_bands] * theta[:, None, :n_bands].to(x.dtype)
            res = float(torch.linalg.vector_norm(r, dim=1).max())
            if res < tol_eff:
                return theta[:, :n_bands], x, True, p + 1
            history.append(res)
            if len(history) >= 3 and history[-1] > 0.9 * history[-3] and res < 20 * tol_eff:
                return theta[:, :n_bands], x, True, p + 1          # stagnated at the precision floor
        return theta[:, :n_bands], x, False, max_pass


def cold_start(op: PWOperator, n_block: int, noise: float = 0.0) -> Tensor:
    """Unit vectors on the n_block lowest-kinetic valid plane waves of each k (padding has huge kinetic). ``noise``: a
    seeded random admixture (norm ~noise per column). At a high-symmetry k the lowest plane waves can span only some
    irreducible representations and the filter never creates the others: AlSi3W2 at (1/2, 1/2, 1/2) lost two bands
    0.9 eV above E_F without it (Stage 5b, 2026-10-23)."""
    k, m = op.kinetic.shape
    idx = torch.argsort(torch.where(op.valid, op.kinetic, torch.full_like(op.kinetic, float("inf"))), dim=1)[:, :n_block]
    x = torch.zeros(k, m, n_block, dtype=op.basis.dtype, device=op.kinetic.device)
    kk = torch.arange(k, device=x.device)[:, None].expand(k, n_block)
    jj = torch.arange(n_block, device=x.device)[None, :].expand(k, n_block)
    x[kk, idx, jj] = 1.0
    if noise:
        r = torch.randn(k, m, n_block, generator=torch.Generator().manual_seed(0), dtype=x.real.dtype).to(x.device)
        x = op.mask(x + (noise / m ** 0.5) * r)
    return x


def hellmann_feynman(op: PWOperator, psi: Tensor, evals: Tensor | None = None) -> Tensor:
    """Differentiable E_n = <psi_n|H(theta)|psi_n> (k, nb) with psi detached: d/dtheta gives psi^H dH psi. With an
    overlap, psi are eigenvectors of S^-1/2 H S^-1/2, x = S^-1/2 psi (x^H S x = 1) and, with ``evals`` E_n (k, nb),
    E_n = <x_n|H|x_n> - E_n <x_n|S - 1|x_n>: d/dtheta gives x^H (dH - E_n dS) x."""
    psi = op.inv_sqrt_s(psi.detach())
    kin = (op.kinetic[..., None].to(psi.real.dtype) * psi.abs().square()).sum(1)
    step = op.band_block(psi.shape[0], psi.shape[2], psi.element_size())
    loc = torch.cat([(op.real_space(psi[..., i:i + step]).abs().square() * op.v_r.to(psi.real.dtype)[None, None, :]).sum(-1)
                     for i in range(0, psi.shape[2], step)], dim=1) / op.n_r       # sum_r |u_n(r)|^2 V(r), blocks of bands
    b = op.basis.mH @ psi.to(op.basis.dtype)                        # (k, c, nb)
    nl = (op.chan.to(b.real.dtype)[None, :, None] * b.abs().square()).sum(1)
    if op.pair is not None:
        c0, c1, d = op.pair
        nl = nl + 2.0 * (d.to(b.real.dtype)[None, :, None] * (b[:, c0].conj() * b[:, c1]).real).sum(1)
    out = kin + loc + nl
    if op.overlap is not None:
        cols, q = op.overlap
        ba = b[:, cols]
        aug = (ba.conj() * (q.to(ba.dtype) @ ba)).real.sum(1)      # <x|S - 1|x>
        out = out - evals.detach().to(out.dtype) * aug
    return out


def fft_stored_band_energies(op: PWOperator, *, n_bands: int, store: dict, key) -> Tensor:
    """Training-path band energies with the persistent warm-start ``store`` (same format as the dense path). With an
    overlap the stored block holds the eigenvectors of S^-1/2 H S^-1/2. ``store["_tol"]``: residual tolerance (eV,
    default 1e-3; tighter for float64 checks)."""
    m = int(op.kinetic.shape[1])
    n_block = min(int(op.valid.sum(1).min()), m, n_bands + max(8, n_bands // 4))
    x0 = store.get(key)
    if x0 is not None and not bool(torch.isfinite(x0.to(torch.complex64) if x0.dtype == torch.complex32 else x0).all()):
        x0 = None                                        # a corrupted warm block: start this cell cold
        store["warm_nonfinite"] = store.get("warm_nonfinite", 0) + 1
    host = bool(store.get("_host", False))
    tol = float(store.get("_tol", 1e-3))
    if x0 is not None:
        x0 = x0.to(op.kinetic.device, non_blocking=True)
    warm = x0 is not None and x0.shape == (op.kinetic.shape[0], m, n_block)
    if warm:
        evals, x, ok, _ = chefsi_operator(op, x0.to(op.basis.dtype), n_bands, tol=tol)
        if not ok:                                       # a warm block that drifted too far: keep filtering from it
            evals, x, ok, _ = chefsi_operator(op, x, n_bands, tol=tol, max_pass=30)
        store["fft_warm"] = store.get("fft_warm", 0) + 1
    else:
        # the symmetry-breaking start for the ultrasoft operator only: s/p cells keep their start (and bands) bitwise
        x0 = cold_start(op, n_block, noise=0.1 if op.overlap is not None else 0.0)
        evals, x, ok, _ = chefsi_operator(op, x0, n_bands, tol=tol, max_pass=80)
        store["fft_cold"] = store.get("fft_cold", 0) + 1
    if not ok:
        store["fft_unconverged"] = store.get("fft_unconverged", 0) + 1
    store[key] = warm_pack(x, store)
    sur = hellmann_feynman(op, x[..., :n_bands], evals)
    return evals.detach().to(sur.dtype) + (sur - sur.detach())


def fft_chain_band_energies(op: PWOperator, *, n_bands: int, chain: dict, k_index: Tensor | None, valid: Tensor | None,
                            tol: float = 1e-3) -> Tensor:
    """Evaluation (no gradients): as torch_pw.chain_band_energies on the FFT path. The block of the previous k batch
    along the path, re-indexed to this batch's spheres, seeds the filter; the first batch starts cold as in training."""
    m = int(op.kinetic.shape[1])
    n_block = min(int(op.valid.sum(1).min()), m, n_bands + max(8, n_bands // 4))
    stats = chain.setdefault("stats", {})
    tol = float(chain.get("tol", tol))
    x0 = map_block(chain.get("x"), chain.get("k_index"), k_index, valid)
    if x0 is not None and x0.shape[-1] == n_block:
        evals, x, ok, passes = chefsi_operator(op, x0.to(op.basis.dtype), n_bands, tol=tol)
        if not ok:
            evals, x, ok, more = chefsi_operator(op, x, n_bands, tol=tol, max_pass=30)
            passes += more
        stats["fft_warm"] = stats.get("fft_warm", 0) + 1
        stats["fft_warm_passes"] = stats.get("fft_warm_passes", 0) + passes
    else:
        evals, x, ok, passes = chefsi_operator(op, cold_start(op, n_block, noise=0.1 if op.overlap is not None else 0.0),
                                               n_bands, tol=tol, max_pass=80)
        stats["fft_cold"] = stats.get("fft_cold", 0) + 1
        stats["fft_cold_passes"] = stats.get("fft_cold_passes", 0) + passes
    if not ok:
        stats["fft_unconverged"] = stats.get("fft_unconverged", 0) + 1
    chain.update(x=x, k_index=k_index)
    return evals.detach()


__all__ = ["PWOperator", "chefsi_operator", "cold_start", "fft_chain_band_energies", "fft_stored_band_energies",
           "hellmann_feynman"]
