#!/usr/bin/env python3
"""Train one shared GeneralEPM on s/p cells of general_sp_v1 with the G1-sp recipe.

Recipe (G1-sp gate, 28/28 at 0.1 eV): window MAE + signed gap term switched on after
``--gap-off-fraction`` of the steps + plain deep-band term; AdamW with a cosine decay;
the species base (per-element residual and hypernetwork) at ``--base-lr``.
Minibatches are drawn with inverse-sqrt chemical-system weights. Cells come only from
``--train-split``/``--val-split`` via ``select_entries``, which refuses the sealed split.
Outputs under ``--out``: ``metrics.json`` (config, history, per-cell scores), the
best-validation checkpoint ``best.pt`` and the final ``last.pt``.

Data parallel: launched with several ranks (``srun`` or ``torchrun``; see
``train_general_sp_ddp.sbatch``), every rank builds all cell caches and draws the same
minibatch from the same RNG; each rank backpropagates its share of the batch, the
gradients are summed, and the identical update runs everywhere. The recipe (batch,
learning rates, sampling) is unchanged. Scoring is split across ranks; rank 0 writes files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import datetime
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from epnet.data.general_manifest import load_general_manifest, select_entries  # noqa: E402
from epnet.models.general_epm import GeneralEPM  # noqa: E402
from epnet.train_cells import gather_lists, reduce_float, reduce_grads  # noqa: E402
from epnet.train_general import (  # noqa: E402
    cell_loss, convergence_loss, move_row, ordered_subset, prepare_cell, prepare_cell_cached, router_normalization, row_nbytes,
    sampling_weights, score_cell, summarize, thin_row, METAL_WINDOWS,
)

G2_ELEMENTS = ("B", "C", "N", "O", "F", "Al", "Si", "P", "S", "Cl", "Ga", "Ge", "As", "Se", "Br", "In", "Sn", "Sb",
               "Te", "I", "Li", "Na", "K", "Mg", "Ca")


def pick(raw: dict, split: str, a, n: int | None, max_sites: int | None = None, exempt: frozenset = frozenset()) -> list[dict]:
    """Cells of one split within the element set and size caps; ids in ``exempt`` skip the size caps."""
    allowed, required = set(a.elements), set(a.require_any or ())
    max_sites = a.max_sites if max_sites is None else max_sites
    entries = [e for e in select_entries(raw, [split])
               if set(e["metadata"]["elements"]) <= allowed
               and (e["id"] in exempt or (e["metadata"]["nsites"] <= max_sites and e["metadata"][f"n_g_ecut8"] <= a.max_ng8))
               and (not required or set(e["metadata"]["elements"]) & required)]
    return ordered_subset(entries, n, a.seed)


def cheapest_per_chemsys(entries: list[dict], k: int) -> list[dict]:
    """Keep the k cheapest cells (n_k * n_G^3 at 8 Ry) of each chemical system: chemical
    diversity at a fraction of the cost of every polymorph (BN alone has 16 small cells)."""
    by: dict[str, list[dict]] = {}
    for e in entries:
        by.setdefault(e["metadata"]["chemsys"], []).append(e)
    cost = lambda e: (e["metadata"]["n_g_ecut8"] ** 3 * e["metadata"]["n_k"], e["id"])  # noqa: E731
    keep = {e["id"] for v in by.values() for e in sorted(v, key=cost)[:k]}
    return [e for e in entries if e["id"] in keep]


def inner_holdout(entries: list[dict], frac: float, salt: str, by: str = "chemsys") -> tuple[list[dict], list[dict]]:
    """Split train cells by chemical-system hash into (train, inner holdout). The inner holdout
    is carved from the train split, so ablations never touch the official holdouts."""
    def u(chemsys: str) -> float:
        return int(hashlib.sha256(f"{salt}:{chemsys}".encode()).hexdigest()[:15], 16) / 16**15
    key = (lambda e: e["metadata"]["chemsys"]) if by == "chemsys" else (lambda e: e["id"])  # "cell": random cells
    hold = [e for e in entries if u(key(e)) < frac]
    keep = [e for e in entries if u(key(e)) >= frac]
    return keep, hold


def balanced_share(batch, costs, rank: int, world: int) -> list[int]:
    """This rank's cells of the minibatch: largest estimated cost first, each to the least
    loaded rank (LPT). Deterministic, so every rank computes the same split; the summed
    gradient is unchanged, only the per-step wall time (set by the slowest rank) improves."""
    if world <= 1:
        return list(batch)
    loads, owner = [0.0] * world, {}
    for i in sorted(batch, key=lambda j: (-costs[j], j)):
        r = min(range(world), key=lambda q: (loads[q], q))
        loads[r] += costs[i]
        owner[i] = r
    return [i for i in batch if owner[i] == rank]


def init_distributed(requested: str) -> tuple[int, int, torch.device]:
    """(rank, world, device). Reads torchrun variables, or Slurm's when launched by ``srun``."""
    env = os.environ
    world = int(env.get("WORLD_SIZE", env.get("SLURM_NTASKS", "1")))
    if world <= 1:
        return 0, 1, torch.device(requested)
    rank = int(env.get("RANK", env.get("SLURM_PROCID", "0")))
    local = int(env.get("LOCAL_RANK", env.get("SLURM_LOCALID", "0")))
    env.setdefault("RANK", str(rank)); env.setdefault("WORLD_SIZE", str(world))
    if not str(requested).startswith("cuda"):   # CPU ranks over gloo (tests of the multi-rank path)
        torch.distributed.init_process_group(backend="gloo", rank=rank, world_size=world)
        return rank, world, torch.device("cpu")
    torch.cuda.set_device(local)
    # 4 h, not NCCL's 10 min: in the final full-path scoring at <= 24 sites one rank's shard can take more than 10 min
    # longer than the others, and the waiting ranks aborted in the all-gather (5c-L s1 links 4 and 5, 2026-10-30)
    torch.distributed.init_process_group(backend="nccl", rank=rank, world_size=world, timeout=datetime.timedelta(hours=4))
    return rank, world, torch.device(f"cuda:{local}")


def cell_costs(rows, cost_model: str, fft_threshold) -> tuple[list[float], list[float]]:
    """(warm cost, cold multiplier) per training row for --balance-ranks (see --cost-model)."""
    if cost_model == "legacy":
        return [r["n_k"] * float(r["n_g"]) ** 3 for r in rows], [4.0] * len(rows)   # dense eigensolve dominates the step
    costs, mult = [], []
    for r in rows:
        n_k, n_g = float(r["n_k"]), float(r["n_g"])
        fft = fft_threshold is not None and n_g >= float(fft_threshold)
        costs.append(1.045e-5 * n_k * n_g ** 1.02 if fft else 4.98e-7 * n_k * n_g ** 1.42)
        mult.append(5.6 if fft else 1.9)
    return costs, mult


def evaluate(model, rows, device, rank: int = 0, world: int = 1, presharded: bool = False, metal_window: str = "proxy",
             warm: bool = False) -> list[dict]:
    """Score rows split across ranks; every rank gets all scores back (in rank-major order).
    ``presharded``: ``rows`` already holds only this rank's cells (``--shard``). ``warm``: each row keeps a warm store."""
    model.eval()
    mine_ = rows if presharded else rows[rank::world]
    scores = [score_cell(model, r, device, metal_window=metal_window,
                         warm_store=r.setdefault("eval_warm", {"_host": True, "_half": True}) if warm else None) for r in mine_]
    model.train()
    return gather_lists(scores)


def cell_ecut(entry: dict, a) -> float:
    table = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in (a.ecut_by_element or [])}
    return max([table.get(el, a.ecut_ry) for el in entry["metadata"]["elements"]] + [a.ecut_ry if not table else 0.0])


_PREP_MODEL = None


def _prep_job(job):
    entry, root, cache_dir, kw = job
    torch.set_num_threads(1)
    prepare_cell_cached(_PREP_MODEL, entry, root, cache_dir, **kw)
    return entry["id"]


def build_caches(model, jobs, root, cache_dir, workers: int, say) -> None:
    """Fill the disk cache in parallel on CPU (fork: the workers share the model)."""
    import multiprocessing as mp
    global _PREP_MODEL
    _PREP_MODEL = model
    t = time.time()
    todo = [(e, root, cache_dir, kw) for e, kw in jobs]
    with mp.get_context("fork").Pool(workers) as pool:
        for n, _ in enumerate(pool.imap_unordered(_prep_job, todo, chunksize=1), 1):
            if n % 200 == 0 or n == len(todo):
                say(f"  cached {n}/{len(todo)} cells, {time.time() - t:.0f}s", flush=True)


def distributed_router_normalization(rows, world: int):
    """Mean/std of router inputs over all ranks' (sharded) training rows."""
    x = torch.cat([r["cache"].router_inputs for r in rows], dim=0).double()
    stats = torch.cat([x.sum(0), x.square().sum(0), torch.tensor([float(x.shape[0])], dtype=x.dtype)])
    if world > 1:
        stats = stats.cuda() if torch.distributed.get_backend() == "nccl" else stats   # gloo: CPU ranks (tests)
        torch.distributed.all_reduce(stats)
        stats = stats.cpu()
    d = x.shape[1]
    n = stats[-1]
    mean = stats[:d] / n
    var = (stats[d:2 * d] / n - mean.square()) * n / (n - 1)
    return mean.float(), var.clamp_min(0).sqrt().float()


def build_parser() -> argparse.ArgumentParser:
    """The trainer's command line (also used by scripts/train_stage.py to replay a recorded stage config)."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=ROOT / "data/manifests/general_sp_v1.json")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--elements", nargs="+", default=list(G2_ELEMENTS))
    p.add_argument("--max-sites", type=int, default=8)
    p.add_argument("--eval-max-sites", type=int, default=None,
                   help="site cap for val, selection and inner-holdout cells (default --max-sites); keeps them comparable when training on larger cells")
    p.add_argument("--per-chemsys", type=int, default=None, help="train on the k cheapest cells of each chemical system")
    p.add_argument("--max-ng8", type=int, default=1500, help="skip cells with more plane waves at 8 Ry")
    p.add_argument("--train-split", default="train")
    p.add_argument("--val-split", default="val")
    p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--require-any", nargs="+", default=None, help="keep only cells containing at least one of these elements")
    p.add_argument("--inner-holdout-frac", type=float, default=0.0,
                   help="fraction of train chemical systems held out internally, scored once at the end with best.pt")
    p.add_argument("--inner-salt", default="inner-v1")
    p.add_argument("--inner-by", choices=("chemsys", "cell"), default="chemsys",
                   help="inner holdout by chemical system (default) or by individual cell (interpolation check)")
    p.add_argument("--select-frac", type=float, default=0.0,
                   help="fraction of train chemical systems scored with val at every eval for model selection (protocol of 2026-10-03)")
    p.add_argument("--select-salt", default="select-v1")
    p.add_argument("--smooth", type=int, default=3, help="evals averaged (centred) for the smoothed selection statistic")
    p.add_argument("--max-val", type=int, default=None)
    p.add_argument("--max-select", type=int, default=None,
                   help="score only this many selection cells (a fixed subset, seed 0); the rest stay out of training")
    p.add_argument("--max-inner", type=int, default=None,
                   help="score only this many inner-holdout cells at the end (a fixed subset, seed 0); all stay out of training")
    p.add_argument("--monitor-train", type=int, default=64, help="train cells scored on the full path at each eval")
    p.add_argument("--ecut-ry", type=float, default=8.0)
    p.add_argument("--channels", type=int, default=32)
    p.add_argument("--readouts", type=int, default=4)
    p.add_argument("--projector-widths", type=float, nargs="+", default=None,
                   help="radial projector width constants (default model: 1.665 3.885); a third adds a radial function")
    p.add_argument("--sphere-basis", type=int, default=None, help="cosine terms in each element's base potential (default 12)")
    p.add_argument("--q-cutoff", type=float, default=None,
                   help="q range (1/A) of the base form factor and the readouts (default 8; 12 Ry needs |G-G'| up to 13.1)")
    p.add_argument("--readout-basis", type=int, default=None, help="cosine terms of the readout form factors (default 12)")
    p.add_argument("--conv-weight", type=float, default=0.0,
                   help="weight of the basis-convergence penalty (bands at --conv-ecut vs the model cutoff), one cell per rank per step")
    p.add_argument("--conv-ecut", type=float, default=16.0)
    p.add_argument("--env-weight-decay", type=float, default=0.0,
                   help="AdamW weight decay on the environment readouts and projector head only")
    p.add_argument("--metal-window", choices=METAL_WINDOWS, default="proxy",
                   help="metal window and alignment in the loss and the scores: the band mid-point proxy (default, G1-G3 v2) "
                        "or the stored true E_F with the best rigid shift ('ef', adopted for the next version)")
    p.add_argument("--window-anchor", choices=("mid", "edges"), default="mid",
                   help="gapped-cell training window: +-2 eV of midgap (default) or [VBM-2, CBM+2] eV")
    p.add_argument("--projector-radius-mode", choices=("rcov", "orbital"), default="rcov",
                   help="projector radii from r_cov with Si constants (default) or from each element's valence orbital radii")
    p.add_argument("--projector-orbital-multipliers", type=float, nargs="+", default=None,
                   help="orbital mode: radius of each radial projector as a multiple of the orbital radius (default 1.0 0.5)")
    p.add_argument("--learn-projector-radii", action="store_true", help="orbital mode: learn global multipliers + bounded per-element offsets")
    p.add_argument("--firstrow-lmax1", action="store_true", help="no l=2 projector channel on B, C, N, O, F atoms")
    p.add_argument("--firstrow-projector-radii", type=float, nargs="+", default=None,
                   help="real-space peak radii (A) of the radial projectors for B, C, N, O, F (one per radial function)")
    p.add_argument("--k-centred", action="store_true", help="standard k-centred plane-wave basis {|k+G| < gcut}")
    p.add_argument("--projector-q-max", type=float, default=None,
                   help="cap on projector Gaussian q-widths (1/A); e.g. 3.8 keeps first-row projectors inside the 12 Ry basis")
    p.add_argument("--base-q-cutoff", type=float, default=None,
                   help="q range of the species base form factor only (readouts keep theirs); overrides --q-cutoff for the base")
    p.add_argument("--charge-coupling", action="store_true",
                   help="charge-coupled species base: v_Z(q) + q_a w_Z(q), D_Z + q_a E_Z (GeneralEPM charge_coupling)")
    p.add_argument("--qeq", action="store_true", help="QEq charges feed the charge coupling (implies --charge-coupling)")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--base-lr", type=float, default=0.3)
    p.add_argument("--gap-weight", type=float, default=1.0)
    p.add_argument("--gap-off-fraction", type=float, default=0.75)
    p.add_argument("--deep-weight", type=float, default=0.3)
    p.add_argument("--k-stride", type=int, default=2)
    p.add_argument("--ecut-by-element", nargs="+", default=None, metavar="EL=RY",
                   help="per-element cutoffs; a cell uses the highest of its elements (others --ecut-ry), e.g. N=20 O=20 F=20 B=16 C=16")
    p.add_argument("--conv-all-bands", action="store_true",
                   help="convergence penalty on all loss-window bands (default: occupied only)")
    p.add_argument("--base-prior", action="store_true",
                   help="physical species start as a fixed feature-derived buffer; residual table starts at zero")
    p.add_argument("--residual-weight", type=float, default=0.0, help="L2 weight on the per-element base residual table")
    p.add_argument("--embed-residual-weight", type=float, default=0.0, help="L2 weight on the per-species embedding residual")
    p.add_argument("--solver", choices=("eigh", "chefsi", "chefsi_fft"), default="eigh",
                   help="training-path eigensolver: dense eigh, or warm-started Chebyshev-filtered subspace iteration")
    p.add_argument("--fft-ng-threshold", type=int, default=None,
                   help="cells with n_G >= this use the FFT operator solver in every path (training, convergence penalty, "
                        "evaluation); dense below (crossover ~4,500 on A40, dense OOM above ~8,000 at 48 GB)")
    p.add_argument("--projector-d-radius", choices=("p", "d"), default="p",
                   help="orbital mode: l = 2 channel at the p radius (default) or the tabulated (n-1)d radius")
    p.add_argument("--encoder-layers", type=int, default=1, choices=(1, 2), help="2: neighbours enter with their own environment")
    p.add_argument("--invariant-norm", choices=("sqrt", "smooth"), default="sqrt",
                   help="dipole/quadrupole norm invariants: |x| (cusp at symmetric sites, G1-G3 v2) or the analytic s(sqrt(1+x^2/s^2)-1)")
    p.add_argument("--invariant-scale", type=float, default=0.05, help="s of the smooth invariant (raw-moment units)")
    p.add_argument("--train-frac", type=float, default=1.0,
                   help="learning curve: keep this fraction of training cells (nested per seed); held-out cells unchanged")
    p.add_argument("--extra-species", nargs="+", default=[],
                   help="species with no training cells (held-out elements: feature prior and hypernetwork only)")
    p.add_argument("--cache-dir", type=Path, default=None, help="persistent per-cell cache (shared by seeds and ablations)")
    p.add_argument("--prep-only", action="store_true", help="fill --cache-dir and exit (run on CPU with --prep-workers)")
    p.add_argument("--prep-workers", type=int, default=1, help="CPU processes for --prep-only")
    p.add_argument("--shard", action="store_true",
                   help="each rank loads and samples only its share of training and evaluation cells (memory, setup / world)")
    p.add_argument("--extra-train", type=Path, default=None,
                   help="JSON with a 'picked' list of {id}: train-split cells that skip --max-sites/--max-ng8/--max-train-ng "
                        "(e.g. novelty-selected larger cells); they must already be in the manifest's train split")
    p.add_argument("--k-select", choices=("stride", "capped"), default="stride",
                   help="training-path k selection with --max-train-k: 'stride' (old; vertices, duplicated junctions and band "
                        "edges are added on top, 40-55 k at 32) or 'capped' (endpoints, vertices, VBM/CBM always kept, duplicated "
                        "junction copies dropped, stride points thinned to fit; soft cap)")
    p.add_argument("--cost-model", choices=("legacy", "measured"), default="legacy",
                   help="per-cell cost for --balance-ranks: legacy n_k n_G^3 (x4 on a cold visit); measured: the fit to warm "
                        "forward + backward times on an L40S (5c-N speed review, 2026-10-30; results/post5cL/profile_train_cells.py): "
                        "dense path 4.98e-7 n_k n_G^1.42 s (x1.9 cold), FFT path 1.045e-5 n_k n_G^1.02 s (x5.6 cold)")
    p.add_argument("--log-timing", type=int, default=0,
                   help="every N steps, log the per-rank step time split: computing (forward + backward) vs waiting at the "
                        "gradient all-reduce, and the cells drawn per step (0: off)")
    p.add_argument("--eval-warm", action="store_true",
                   help="keep a warm-start store per evaluation cell (FFT-path cells start from their eigenvectors of the "
                        "previous evaluation; same results within the solver tolerance)")
    p.add_argument("--loss-per-drawn", action="store_true",
                   help="with --shard --balance-ranks, divide the summed cell losses by the cells drawn on all ranks (the "
                        "step's true mean) instead of --batch; before 2026-10-29 about 26 cells were drawn per step at --batch 16")
    p.add_argument("--balance-ranks", action="store_true",
                   help="sharded runs: each rank draws cells (same probabilities) until a cost budget = (batch/world) x its "
                        "expected cost per draw (n_k n_G^3, x4 for a first, cold visit), instead of a fixed count; the expected "
                        "gradient is unchanged and ranks wait less for each other at the all-reduce")
    p.add_argument("--profile-steps", type=int, default=0,
                   help="timing mode: run this many training steps with per-phase CUDA-synced timers, no evals, then print and exit")
    p.add_argument("--boost-lr-params", nargs="+", default=None,
                   help="parameter-name prefixes trained at --boost-lr (e.g. a zero-initialized layer added to a warm start)")
    p.add_argument("--boost-lr", type=float, default=None)
    p.add_argument("--oversample", type=Path, default=None,
                   help="JSON with a 'picked' list of {id}: training cells whose sampling weight is multiplied by --oversample-factor")
    p.add_argument("--oversample-factor", type=float, default=1.0)
    p.add_argument("--max-train-ng", type=float, default=None,
                   help="train only on cells with estimated n_G <= this at their own cutoff (larger cells are still scored)")
    p.add_argument("--eval-max-k", type=int, default=None,
                   help="intermediate evals on paths thinned to about this many k (final scoring on the full path)")
    p.add_argument("--k-budget-sites", type=int, default=None,
                   help="cells above this many sites train on max_train_k * N / nsites k-points (min 8): band count grows with\n"
                        "the cell, so (bands x k) per cell stays about constant; cells up to N sites are unchanged")
    p.add_argument("--max-train-k", type=int, default=None,
                   help="raise the k stride of long training paths so about this many k remain (eigh is ~90%% of a step)")
    p.add_argument("--max-train-k-large", type=int, default=None,
                   help="Stage 5c: the --max-train-k of training cells with more than --large-sites sites")
    p.add_argument("--large-sites", type=int, default=8)
    p.add_argument("--k-batch", type=int, default=None, help="k-points per Hamiltonian chunk (needed at 12 Ry on 44 GB GPUs)")
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-from", type=Path, default=None, help="warm start (use a low --lr/--base-lr)")
    p.add_argument("--expand-species", action="store_true",
                   help="EpNet-spd: warm start that adds the training elements missing from the --init-from checkpoint "
                        "(epnet.models.species_expansion; old elements unchanged, new ones from the shared hypernetwork)")
    p.add_argument("--element-feature-set", choices=("v1", "spd-v1"), default="v1",
                   help="element features: v1 (G4) or spd-v1 (+ model d count; EpNet-spd)")
    p.add_argument("--d-overlap", action="store_true", help="EpNet-spd M4: d overlap operator S (generalized eigenproblem)")
    p.add_argument("--d-crystal-field", action="store_true", help="EpNet-spd M5: sigma angular-overlap crystal field on the d block")
    # EpNet-spd d channel (docs/d_channel_plan.md): atomic-orbital channels, ultrasoft, all-electron-tail local, guard B
    p.add_argument("--atomic-channels", type=int, nargs="+", default=None, help="l values of the atomic channels of d elements")
    p.add_argument("--d-prior-valence", choices=("full", "s", "ae"), default=None, help="local start of d elements")
    p.add_argument("--no-free-d-slot", action="store_true", help="guard B: no free (Gaussian) l = 2 slot on d atoms")
    p.add_argument("--d-species-learn-scale", type=float, default=None, help="guard B: d species learned deviation x this")
    p.add_argument("--atomic-learn-l", type=int, nargs="+", default=None, help="atomic channels with learned shift/extent")
    p.add_argument("--atomic-two-ref-d", action="store_true", help="Step 6: two-reference d channel (second l = 2 slot)")
    p.add_argument("--freeze-shared", action="store_true",
                   help="Stage 3d: freeze every parameter except the d species' rows (base residual, embedding residual, "
                        "projector offsets), the atomic-channel shifts/extents and the per-atom d-level head; s/p cells keep "
                        "the warm start exactly")
    p.add_argument("--atomic-atom-shift", action="store_true", help="Step 6c: per-atom d level from the atom's invariants")
    p.add_argument("--atomic-two-ref-safe", action="store_true",
                   help="Step 6d (revised): d level shift as D += shift <AE_i|AE_j> (B unchanged), per-element extent bounds")
    p.add_argument("--atomic-two-ref-stable", action="store_true",
                   help="Stage 3d fix: normalized second d reference and the PSD part of Q (S >= 1 by construction)")
    p.add_argument("--d-atomic-start", action="store_true",
                   help="after --expand-species: the added d species start at atomic parameters (base residual cancels the "
                        "hypernetwork); d species already in the checkpoint are not reset")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", action="store_true", help="continue from <out>/state.pt if present")
    p.add_argument("--save-every", type=int, default=None, help="also save the resumable state every N steps (chain links resume)")
    p.add_argument("--keep-eval-ckpts", action="store_true",
                   help="save <out>/ckpt/step<N>.pt at every eval even without --select-frac (always on with a selection set)")
    p.add_argument("--conv-max-ng", type=float, default=None,
                   help="convergence-penalty cell: n_G at --conv-ecut must not exceed this (its dense solve + backward set peak memory)")
    p.add_argument("--skip-oom", action="store_true", help="skip (and log) a cell whose loss or backward runs out of GPU memory")
    p.add_argument("--skip-nonfinite", action="store_true",
                   help="Stage 3d: skip (and log) a cell whose solve raises a linear-algebra error, and a whole step whose "
                        "gradients are not finite (a finite loss can still give NaN gradients and corrupt the parameters)")
    p.add_argument("--warm-store", choices=("device", "host"), default="device",
                   help="where CheFSI keeps each cell's last eigenvectors: GPU (default) or host memory (large datasets on <=48 GB GPUs)")
    p.add_argument("--cb-manifold", action="store_true",
                   help="Stage 5a-w: gapped training cells with non-d10 d atoms keep the lowest 5 N conduction bands in the labels "
                        "and the loss window (the empty d manifold)")
    p.add_argument("--legacy-k-rows", action="store_true",
                   help="training rows as in Stages 3-5c-L: built with the stride k selection whatever --k-select (the k-cap bug, "
                        "fixed 2026-10-29), under their old cache keys; reproduces those runs")
    p.add_argument("--warm-half", action="store_true",
                   help="keep the warm-start blocks in half precision (Stage 5c: the <= 24-site store is ~400 GB in single)")
    p.add_argument("--max-hours", type=float, default=None,
                   help="save state and exit cleanly after this wall time (chain jobs with --resume)")
    p.add_argument("--final-scoring", choices=("full", "none"), default="full",
                   help="after the last step: score val, monitor and the inner holdout on full paths (full), or only select the "
                        "checkpoint and write metrics.json (none; 5c-L, user 2026-10-30: the readings score these cells thinned)")
    p.add_argument("--stop-check-every", type=int, default=100,
                   help="with --max-hours, also check the time every N steps between evaluations (save state and exit there; "
                        "0: only at evaluations, as before 2026-10-29)")
    p.add_argument("--stop-at-step", type=int, default=None,
                   help="save state and exit at this step (checked at evaluations and every --stop-check-every steps; testing, chaining)")
    p.add_argument("--cache-device", choices=("auto", "host", "device"), default="auto",
                   help="keep per-cell caches on the GPU (auto: if they use < 50%% of its memory)")
    return p


def main() -> None:
    p = build_parser()
    a = p.parse_args()
    rank, world, device = (0, 1, torch.device("cpu")) if a.prep_only else init_distributed(a.device)  # prep is single-process CPU
    main_rank = rank == 0
    say = print if main_rank else (lambda *x, **k: None)
    a.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)

    raw = load_general_manifest(a.manifest)
    root = a.manifest.resolve().parent
    eval_sites = a.max_sites if a.eval_max_sites is None else a.eval_max_sites
    small = lambda es: [e for e in es if e["metadata"]["nsites"] <= eval_sites]  # noqa: E731
    extra = frozenset(x["id"] for x in json.loads(a.extra_train.read_text())["picked"]) if a.extra_train else frozenset()
    train_entries = pick(raw, a.train_split, a, a.max_train, exempt=extra)
    if extra:
        found = extra & {e["id"] for e in train_entries}
        say(f"--extra-train: {len(found)}/{len(extra)} cells found in the {a.train_split} split", flush=True)
        if found != extra:
            raise SystemExit(f"--extra-train cells missing from the train split: {sorted(extra - found)[:10]}")
    train_entries, select_entries_ = inner_holdout(train_entries, a.select_frac, a.select_salt)
    train_entries, inner_entries = inner_holdout(train_entries, a.inner_holdout_frac, a.inner_salt, by=a.inner_by)
    select_entries_, inner_entries = small(select_entries_), small(inner_entries)
    if a.max_select:   # Stage 5a: fewer scored cells per eval at <= 8 sites; carved out of training either way
        select_entries_ = ordered_subset(select_entries_, a.max_select, 0)
    if a.max_inner:
        inner_entries = ordered_subset(inner_entries, a.max_inner, 0)
    if a.per_chemsys:
        train_entries = cheapest_per_chemsys(train_entries, a.per_chemsys)
    train_elements = sorted({el for e in train_entries for el in e["metadata"]["elements"]} | set(a.extra_species))
    if a.train_frac < 1.0:  # learning curve: subset after the inner split, species list and held-out set unchanged
        train_entries = ordered_subset(train_entries, max(1, round(a.train_frac * len(train_entries))), a.seed)
    if a.max_train_ng:  # n_G scales as ecut^1.5 from the manifest's 8 Ry count
        n_before = len(train_entries)
        train_entries = [e for e in train_entries
                         if e["id"] in extra or e["metadata"]["n_g_ecut8"] * (cell_ecut(e, a) / 8.0) ** 1.5 <= a.max_train_ng]
        say(f"--max-train-ng {a.max_train_ng:.0f}: {len(train_entries)}/{n_before} training cells kept", flush=True)
    val_entries = [e for e in pick(raw, a.val_split, a, a.max_val, max_sites=eval_sites) if set(e["metadata"]["elements"]) <= set(train_elements)]
    spd_extra = {}
    if a.element_feature_set != "v1":
        spd_extra["element_feature_set"] = a.element_feature_set
    if a.d_overlap:
        spd_extra["d_overlap"] = True
    if a.d_crystal_field:
        spd_extra["d_crystal_field"] = True
    if a.atomic_channels:
        spd_extra["atomic_channels"] = tuple(a.atomic_channels)
    if a.d_prior_valence:
        spd_extra["d_prior_valence"] = a.d_prior_valence
    if a.no_free_d_slot:
        spd_extra["atomic_free_d_slot"] = False
    if a.d_species_learn_scale is not None:
        spd_extra["d_species_learn_scale"] = a.d_species_learn_scale
    if a.atomic_learn_l is not None:
        spd_extra["atomic_learn_l"] = tuple(a.atomic_learn_l)
    if a.atomic_two_ref_d:
        spd_extra["atomic_two_ref_d"] = True
    if a.atomic_atom_shift:
        spd_extra["atomic_atom_shift"] = True
    if a.atomic_two_ref_stable:
        spd_extra["atomic_two_ref_stable"] = True
    if a.atomic_two_ref_safe:
        spd_extra["atomic_two_ref_safe"] = True
    if a.init_from and a.expand_species:
        from epnet.models.species_expansion import expand_species
        payload = torch.load(a.init_from, map_location="cpu", weights_only=False)
        override = {"ecut_ry": a.ecut_ry, "k_batch": a.k_batch, "solver": a.solver, **spd_extra}
        if a.fft_ng_threshold is not None:
            override["fft_ng_threshold"] = a.fft_ng_threshold
        if a.ecut_by_element:
            override["ecut_by_element"] = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.ecut_by_element}
        added = [el for el in train_elements if el not in payload["species_symbols"]]
        model = expand_species(payload, added, **override)
        say(f"--expand-species: {len(payload['species_symbols'])} -> {len(model.species_symbols)} elements (+{added})", flush=True)
        if a.d_atomic_start:   # added d species at atomic parameters (d species already in the checkpoint keep theirs)
            from epnet.models.species_expansion import atomic_start
            say(f"--d-atomic-start: {atomic_start(model, added)}", flush=True)
    elif a.init_from:
        payload = torch.load(a.init_from, map_location="cpu", weights_only=False)
        # the state dict does not depend on the cutoff: rebuild at --ecut-ry and load the same weights
        override = {"ecut_ry": a.ecut_ry, "k_batch": a.k_batch, "solver": a.solver}
        if a.fft_ng_threshold is not None:
            override["fft_ng_threshold"] = a.fft_ng_threshold
        if a.projector_d_radius != payload["config"].get("projector_d_radius", "p"):
            override["projector_d_radius"] = a.projector_d_radius  # d-channel fix on a warm start
        model = GeneralEPM(payload["species_symbols"], **{**payload["config"], **override})
        fresh_radii = model.orbital_radii.clone() if hasattr(model, "orbital_radii") else None
        model.load_state_dict(payload["state_dict"])
        if "projector_d_radius" in override and fresh_radii is not None:
            # the checkpoint stores the old radii buffer: keep the new l = 2 radii, and reset the learned per-element
            # l = 2 scale of the elements whose radius moved (it was fitted to the old placement)
            moved = (fresh_radii[:, 2] - model.orbital_radii[:, 2]).abs() > 1e-6
            with torch.no_grad():
                model.orbital_radii.copy_(fresh_radii)
                if model.projector_log_offset is not None:
                    model.projector_log_offset[moved, 2] = 0.0
            say(f"d-channel radius moved for {[model.species_symbols[i] for i in torch.nonzero(moved).flatten().tolist()]}", flush=True)
        if set(train_elements) - set(model.species_symbols):
            raise ValueError("warm-start checkpoint lacks elements of this training set")
    else:
        extra = {}
        if a.projector_widths:
            extra["projector_width_constants"] = tuple(a.projector_widths)
        if a.sphere_basis:
            extra["sphere_n_basis"] = a.sphere_basis
        if a.q_cutoff:
            extra["sphere_q_cutoff"] = extra["scalar_cosine_cutoff"] = a.q_cutoff
        if a.readout_basis:
            extra["scalar_cosine_basis"] = a.readout_basis
        if a.k_centred:
            extra["k_centred_basis"] = True
        if a.base_prior:
            extra["base_prior"] = True
        if a.solver != "eigh":
            extra["solver"] = a.solver
        if a.fft_ng_threshold is not None:
            extra["fft_ng_threshold"] = a.fft_ng_threshold
        if a.projector_d_radius != "p":
            extra["projector_d_radius"] = a.projector_d_radius
        if a.encoder_layers != 1:
            extra["encoder_layers"] = a.encoder_layers
        if a.invariant_norm != "sqrt":
            extra["invariant_norm"], extra["invariant_scale"] = a.invariant_norm, a.invariant_scale
        if a.ecut_by_element:
            extra["ecut_by_element"] = {kv.split("=")[0]: float(kv.split("=")[1]) for kv in a.ecut_by_element}
        if a.firstrow_lmax1:
            extra["firstrow_lmax1"] = True
        if a.projector_radius_mode != "rcov":
            extra["projector_radius_mode"] = a.projector_radius_mode
        if a.projector_orbital_multipliers:
            extra["projector_orbital_multipliers"] = tuple(a.projector_orbital_multipliers)
        if a.learn_projector_radii:
            extra["learn_projector_radii"] = True
        if a.firstrow_projector_radii:
            extra["projector_radii_firstrow"] = tuple(a.firstrow_projector_radii)
        if a.projector_q_max:
            extra["projector_q_width_max"] = a.projector_q_max
        if a.base_q_cutoff:
            extra["sphere_q_cutoff"] = a.base_q_cutoff
        if a.charge_coupling or a.qeq:
            extra["charge_coupling"] = True
        if a.qeq:
            extra["charge_model"] = "qeq"
        model = GeneralEPM(train_elements, channels=a.channels, cutoff=6.0, n_readouts=a.readouts, ecut_ry=a.ecut_ry,
                           projector_l_max=2, charge_transfer=True, subset_ng_min=10**9, k_batch=a.k_batch, **extra, **spd_extra)
    t0 = time.time()
    monitor_ids = {e["id"] for e in train_entries[: a.monitor_train]}
    monitor_entries = [e for e in train_entries if e["id"] in monitor_ids]
    inner_entries = [e for e in inner_entries if set(e["metadata"]["elements"]) <= set(train_elements)]
    select_entries_ = [e for e in select_entries_ if set(e["metadata"]["elements"]) <= set(train_elements)]
    train_kw = {"k_stride": a.k_stride, "full_path": False, "max_k": a.max_train_k, "k_select": a.k_select}
    if a.cb_manifold:
        train_kw["cb_manifold"] = True
    if a.legacy_k_rows:
        train_kw["legacy_k_rows"] = True

    def kw_for(e: dict) -> dict:  # per-cell k budget (--k-budget-sites; --max-train-k-large above --large-sites)
        n, ref = int(e["metadata"]["nsites"]), a.k_budget_sites
        if a.max_train_k_large and n > a.large_sites:
            return {**train_kw, "max_k": a.max_train_k_large}
        if not (ref and a.max_train_k and n > ref):
            return train_kw
        return {**train_kw, "max_k": max(8, round(a.max_train_k * ref / n))}
    full_kw = {"k_stride": 1, "full_path": True}
    if a.prep_only:
        if not a.cache_dir:
            raise ValueError("--prep-only needs --cache-dir")
        jobs = [(e, kw_for(e)) for e in train_entries] + [(e, full_kw) for e in monitor_entries + val_entries + select_entries_ + inner_entries]
        say(f"prep-only: {len(jobs)} cell jobs, {a.prep_workers} workers -> {a.cache_dir}", flush=True)
        build_caches(model, jobs, root, a.cache_dir, a.prep_workers, say)
        return
    sharded = a.shard and world > 1
    mine = (lambda es: es[rank::world]) if sharded else (lambda es: es)  # noqa: E731
    global_chemsys = Counter(e["metadata"]["chemsys"] for e in train_entries)
    train_entries_local = mine(train_entries)
    prep = lambda es, kw: [prepare_cell_cached(model, e, root, a.cache_dir, **kw) for e in es]  # noqa: E731
    train_rows = [prepare_cell_cached(model, e, root, a.cache_dir, **kw_for(e)) for e in train_entries_local]
    if a.warm_store == "host":
        for r in train_rows:
            r.setdefault("warm", {})["_host"] = True
    if a.warm_half:
        for r in train_rows:
            r.setdefault("warm", {})["_half"] = True
    monitor_rows = prep(mine(monitor_entries), full_kw)
    val_rows = prep(mine(val_entries), full_kw)
    select_rows = prep(mine(select_entries_), full_kw)
    inner_rows = prep(mine(inner_entries), full_kw)
    if not a.init_from:
        model.set_router_normalization(*(distributed_router_normalization(train_rows, world) if sharded else router_normalization(train_rows)))
    model.to(device).train()
    if world > 1:  # same seed gives the same init already; broadcast to be certain
        for value in model.state_dict().values():
            torch.distributed.broadcast(value, 0)
    rows_all = train_rows + monitor_rows + val_rows
    cache_gb = sum(row_nbytes(r) for r in rows_all) / 1e9
    resident = a.cache_device == "device" or (
        a.cache_device == "auto" and device.type == "cuda" and cache_gb < 0.5 * torch.cuda.get_device_properties(device).total_memory / 1e9)
    if resident:
        train_rows, monitor_rows, val_rows, inner_rows, select_rows = (
            [move_row(r, device) for r in rows] for rows in (train_rows, monitor_rows, val_rows, inner_rows, select_rows))
    say(f"ranks {world}{' (sharded)' if sharded else ''}; caches {cache_gb:.2f} GB{' on rank 0' if sharded else ''}, resident on device: {resident}", flush=True)
    if a.eval_max_k:  # cheap views for intermediate evals; final scoring uses the full rows
        mon_eval, val_eval, sel_eval = ([thin_row(r, a.eval_max_k) for r in rows] for rows in (monitor_rows, val_rows, select_rows))
    else:
        mon_eval, val_eval, sel_eval = monitor_rows, val_rows, select_rows
    say(f"cells: train {len(train_entries)} (monitor {len(monitor_entries)}), val {len(val_entries)}, inner holdout {len(inner_entries)}; elements {len(train_elements)}; "
          f"prep {time.time() - t0:.0f}s; params {sum(p.numel() for p in model.parameters() if p.requires_grad)}", flush=True)

    if a.freeze_shared:   # Stage 3d: only d-species rows and d-only modules learn (all groups have zero weight decay)
        from epnet.data.valence import D_ELEMENTS
        d_rows = torch.tensor([sym in D_ELEMENTS for sym in model.species_symbols], dtype=torch.bool)
        row_params = {"base_residual", "encoder.species_embedding.species_residual.weight", "projector_log_offset"}
        free_params = {"ac_eps_shift", "ac_extent"}
        n_train = 0
        for name, prm in model.named_parameters():
            if name in row_params:
                mask = d_rows.to(prm.device, prm.dtype).reshape(-1, *([1] * (prm.dim() - 1)))
                prm.register_hook(lambda g, m=mask: g * m.to(g))
                n_train += int(d_rows.sum()) * (prm[0].numel() if prm.dim() > 1 else 1)
            elif name in free_params or name.startswith("ac_shift_head."):
                n_train += prm.numel()
            else:
                prm.requires_grad_(False)
        say(f"--freeze-shared: {n_train} trainable values (d rows and d-only modules)", flush=True)
    base = [q for q in [model.base_residual, *model.base_hyper.parameters()] if q.requires_grad]
    if getattr(model, "d_overlap_raw", None) is not None:   # species-level d overlap: trained with the species base
        base.append(model.d_overlap_raw)
    if getattr(model, "projector_offdiag", None) is not None:   # per-species projector coupling: a species-base quantity
        base.append(model.projector_offdiag)
    for name in ("charge_hyper", "charge_residual", "qeq_hyper", "qeq_residual", "qeq_gate", "qeq_kappa_raw"):   # per-species charge terms
        mod = getattr(model, name, None)
        if mod is not None:
            base.extend(mod.parameters() if isinstance(mod, torch.nn.Module) else [mod])
    base_ids = {id(q) for q in base}
    groups = [{"params": base, "lr": a.base_lr, "weight_decay": 0.0}]
    if a.env_weight_decay > 0:  # decoupled decay on the environment readouts and the per-atom projector head only
        env = [*model.readouts.parameters(), *model.projector_head.parameters()]
        env_ids = {id(q) for q in env}
        groups.append({"params": [q for q in env if q.requires_grad], "lr": a.lr, "weight_decay": a.env_weight_decay})
        base_ids |= env_ids
    if a.boost_lr_params:   # new layers on a warm start: their own (from-scratch) learning rate
        boosted = [q for n_, q in model.named_parameters() if q.requires_grad and id(q) not in base_ids
                   and any(n_.startswith(pfx) for pfx in a.boost_lr_params)]
        groups.append({"params": boosted, "lr": a.boost_lr or a.lr, "weight_decay": 0.0})
        base_ids |= {id(q) for q in boosted}
        say(f"--boost-lr {a.boost_lr}: {sum(q.numel() for q in boosted)} parameters in {a.boost_lr_params}", flush=True)
    rest = [q for q in model.parameters() if q.requires_grad and id(q) not in base_ids]
    groups.append({"params": rest, "lr": a.lr, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps, eta_min=a.lr * 0.01)
    if sharded:  # inverse-sqrt chemsys weights from the global counts, restricted to this rank's cells
        weights = np.array([1.0 / np.sqrt(global_chemsys[r["chemsys"]]) for r in train_rows])
        weights = weights / weights.sum()
    else:
        weights = sampling_weights(train_rows)
    if a.oversample and a.oversample_factor != 1.0:   # short targeted runs: few cells must be visited often enough
        over = {x["id"] for x in json.loads(a.oversample.read_text())["picked"]}
        weights = weights * np.array([a.oversample_factor if r["id"] in over else 1.0 for r in train_rows])
        weights = weights / weights.sum()
        say(f"--oversample x{a.oversample_factor:g}: {sum(r['id'] in over for r in train_rows)} of this rank's cells", flush=True)
    costs, cold_mult = cell_costs(train_rows, a.cost_model, model.config.get("fft_ng_threshold"))
    history, best, t1, start = [], None, time.time(), 0
    state_path = a.out / "state.pt"
    if a.resume and state_path.exists():
        st = torch.load(state_path, map_location=device, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]; torch.set_rng_state(st["torch_rng"].cpu())  # map_location moved it to the GPU
        history, best, start = st["history"], st["best"], st["step"] + 1
        say(f"resumed from step {st['step']} (best {best})", flush=True)
    if start > a.steps:
        if (a.out / "metrics.json").exists():
            say("run already complete", flush=True)
            return
        # the previous link was stopped (wall limit) during the final scoring: no training, score and write metrics.json
        say("training complete; final scoring only (no metrics.json from the previous link)", flush=True)

    def save_state(step: int) -> None:
        if not main_rank:
            return
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                    "history": history, "best": best, "step": step}, state_path)

    PT = {}   # --profile-steps: phase -> seconds (CUDA-synced)
    TIMING = {"compute": 0.0, "wait": 0.0, "drawn": 0, "steps": 0}   # --log-timing

    def tick(name, t_prev):
        if not a.profile_steps:
            return t_prev
        if device.type == "cuda":
            torch.cuda.synchronize()
        now = time.perf_counter(); PT[name] = PT.get(name, 0.0) + now - t_prev
        return now

    for step in range(start, a.steps + 1):
        if a.profile_steps and step - start >= a.profile_steps:
            tot = PT.pop("_wall"); PT["unaccounted"] = tot - sum(PT.values())
            say(f"--profile-steps {a.profile_steps} (rank {rank} of {world}): {tot / a.profile_steps:.2f} s/step", flush=True)
            for k, v in sorted(PT.items(), key=lambda kv: -kv[1]):
                say(f"  {k:28s} {v / a.profile_steps:7.3f} s/step  {v / tot:6.1%}", flush=True)
            agg = {}
            for r_ in train_rows:
                for k_, v_ in r_.get("warm", {}).items():
                    if isinstance(v_, (int, float)) and not isinstance(v_, bool):
                        agg[k_] = agg.get(k_, 0) + v_
            n_chefsi = agg.get("chefsi", 0) + agg.get("chefsi_fallback", 0)
            say(f"  solver counts (k-chunks, rank {rank}): {agg}; warm CheFSI attempts {n_chefsi}, fallbacks to eigh "
                f"{agg.get('chefsi_fallback', 0)} ({agg.get('chefsi_fallback', 0) / max(n_chefsi, 1):.0%}), mean passes per attempt "
                f"{agg.get('chefsi_passes', 0) / max(n_chefsi, 1):.2f}", flush=True)
            return
        if (step % a.eval_every == 0 or step == a.steps) and not (step == start and start > 0) and not a.profile_steps:
            mon = summarize(evaluate(model, mon_eval, device, rank, world, sharded, a.metal_window, warm=a.eval_warm))
            val_scores_now = evaluate(model, val_eval, device, rank, world, sharded, a.metal_window, warm=a.eval_warm)
            val = summarize(val_scores_now)
            entry = {"step": step, "monitor": mon, "val": val, "elapsed_s": time.time() - t1}
            if select_entries_:  # selection statistic (every rank takes part, even with an empty shard): median over val + the selection set; checkpoint kept per eval
                entry["select"] = summarize(val_scores_now + evaluate(model, sel_eval, device, rank, world, sharded, a.metal_window,
                                                                      warm=a.eval_warm))
            if (select_entries_ or a.keep_eval_ckpts) and main_rank:
                (a.out / "ckpt").mkdir(exist_ok=True)
                torch.save(model.checkpoint_payload(), a.out / "ckpt" / f"step{step}.pt")
            history.append(entry)
            say(f"step {step:5d}  monitor median {mon.get('median_window_mae_eV', float('nan')):.3f} pass0.1 "
                  f"{mon.get('pass_0.1')}/{mon['n']}  val median {val.get('median_window_mae_eV', float('nan')):.3f} "
                  f"pass0.1 {val.get('pass_0.1')}/{val['n']}  {time.time() - t1:.0f}s", flush=True)
            if val["n"] and (best is None or val["median_window_mae_eV"] < best[1]):
                best = (step, val["median_window_mae_eV"])
                if main_rank:
                    torch.save(model.checkpoint_payload(), a.out / "best.pt")
            if step < a.steps:
                stop = (a.max_hours and time.time() - t0 > a.max_hours * 3600) or (a.stop_at_step is not None and step >= a.stop_at_step)
                if reduce_float(float(bool(stop)), device, op="max") > 0:  # every rank stops together
                    save_state(step - 1)  # the step-th update has not run yet
                    say(f"time limit: saved state at step {step}; resume with --resume", flush=True)
                    return
        elif ((a.max_hours or a.stop_at_step is not None) and a.stop_check_every and start < step < a.steps
              and step % a.stop_check_every == 0 and not a.profile_steps):   # between evaluations too (a link stops near its budget)
            stop = (a.max_hours and time.time() - t0 > a.max_hours * 3600) or (a.stop_at_step is not None and step >= a.stop_at_step)
            if reduce_float(float(bool(stop)), device, op="max") > 0:   # every rank together
                save_state(step - 1)  # the step-th update has not run yet
                say(f"time limit: saved state at step {step}; resume with --resume", flush=True)
                return
        if step == a.steps:
            break
        gap_on = step >= a.gap_off_fraction * a.steps
        if a.profile_steps and device.type == "cuda":
            torch.cuda.synchronize()
        tp = step_t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        if sharded:  # each rank draws its share of the batch from its own cells; stateless RNG, so resume is exact
            n_local = a.batch // world + int(rank < a.batch % world)
            local_rng = np.random.default_rng([a.seed, rank, step])
            if a.balance_ranks:   # draw until this rank's expected-cost budget (Wald: expected count stays n_local)
                eff = np.array([costs[i] * (cold_mult[i] if len(train_rows[i].get("warm", {})) <= 1 else 1.0) for i in range(len(train_rows))])
                budget = n_local * float(np.dot(weights, eff))
                stream = local_rng.choice(len(train_rows), size=min(4 * n_local + 4, len(train_rows)), replace=False, p=weights)
                batch, spent = [], 0.0
                for i in stream:
                    batch.append(int(i)); spent += eff[i]
                    if spent >= budget:
                        break
            else:
                batch = list(local_rng.choice(len(train_rows), size=min(n_local, len(train_rows)), replace=False, p=weights))
            batch_size = a.batch
            if a.balance_ranks and a.loss_per_drawn:
                batch_size = max(1, int(round(reduce_float(float(len(batch)), device))))   # cells drawn on all ranks
                if step == start:
                    say(f"--loss-per-drawn: {batch_size} cells drawn at step {step} (--batch {a.batch})", flush=True)
        else:
            batch = rng.choice(len(train_rows), size=min(a.batch, len(train_rows)), replace=False, p=weights)
            batch_size = len(batch)
        total, conv_total = 0.0, 0.0
        my_cells = batch if sharded else balanced_share(batch, costs, rank, world)
        tp = tick("sample batch", tp)
        for i in my_cells:  # this rank's share; per-cell backward keeps peak memory at one cell
            try:
                cold = len(train_rows[i].get("warm", {})) <= 1
                loss, _ = cell_loss(model, train_rows[i], device, gap_on=gap_on, gap_weight=a.gap_weight, deep_weight=a.deep_weight,
                                    window_anchor=a.window_anchor, metal_window=a.metal_window)
                tp = tick("cell loss fwd (cold visit)" if cold else "cell loss fwd (warm)", tp)
                if not torch.isfinite(loss):
                    continue
                (loss / batch_size).backward()
                tp = tick("cell loss backward", tp)
                total += float(loss) / batch_size
            except torch.cuda.OutOfMemoryError:
                if not a.skip_oom:
                    raise
                loss = None
                torch.cuda.empty_cache()
                print(f"[rank{rank}] step {step}: OOM on cell {train_rows[i]['id']} (n_G {train_rows[i]['n_g']}, n_k {train_rows[i]['n_k']}); skipped", flush=True)
            except torch.linalg.LinAlgError as exc:
                if not a.skip_nonfinite:
                    raise
                loss = None
                print(f"[rank{rank}] step {step}: linear-algebra error on cell {train_rows[i]['id']} ({str(exc)[:80]}); skipped", flush=True)
        share = my_cells
        # only cells solved below --conv-ecut can drift (element-aware cutoffs: first-row cells are already there)
        conv_share = [i for i in share if train_rows[i].get("ecut", a.ecut_ry) < a.conv_ecut] if share else []
        if a.conv_max_ng:  # n_G grows as (conv_ecut / ecut)^1.5 at the penalty cutoff
            conv_share = [i for i in conv_share
                          if train_rows[i]["n_g"] * (a.conv_ecut / train_rows[i].get("ecut", a.ecut_ry)) ** 1.5 <= a.conv_max_ng]
        if a.conv_weight > 0 and conv_share:
            j = min(conv_share, key=lambda i: costs[i])  # cheapest cell of this rank's share keeps the overhead small
            try:
                conv = convergence_loss(model, train_rows[j], device, ecut_hi=a.conv_ecut, all_bands=a.conv_all_bands)
                if torch.isfinite(conv):
                    (a.conv_weight * conv / world).backward()
                    conv_total += float(conv) / world
            except torch.cuda.OutOfMemoryError:
                if not a.skip_oom:
                    raise
                conv = None
                torch.cuda.empty_cache()
                print(f"[rank{rank}] step {step}: OOM in the convergence penalty on {train_rows[j]['id']}; skipped", flush=True)
        tp = tick("convergence penalty", tp)
        if main_rank and (a.residual_weight > 0 or a.embed_residual_weight > 0):  # once per step (grads are summed over ranks)
            pen = a.residual_weight * model.base_residual_penalty() + a.embed_residual_weight * model.embedding_residual_penalty()
            pen.backward()
        tp = tick("residual penalties", tp)
        if a.log_timing:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_comp = time.perf_counter()
        reduce_grads(model)  # sum of the per-rank shares = the single-GPU batch gradient
        total = reduce_float(total, device)
        if a.log_timing:
            if device.type == "cuda":
                torch.cuda.synchronize()
            TIMING["compute"] += t_comp - step_t0
            TIMING["wait"] += time.perf_counter() - t_comp
            TIMING["drawn"] += len(my_cells)
            TIMING["steps"] += 1
            if TIMING["steps"] == a.log_timing:
                n = TIMING["steps"]
                c_max = reduce_float(TIMING["compute"] / n, device, op="max")
                c_mean = reduce_float(TIMING["compute"] / n, device) / world
                c_min = -reduce_float(-TIMING["compute"] / n, device, op="max")
                w_mean = reduce_float(TIMING["wait"] / n, device) / world
                drawn = reduce_float(TIMING["drawn"] / n, device)
                say(f"  timing steps {step - n + 1}-{step}: compute per rank mean {c_mean:.2f} (min {c_min:.2f}, max {c_max:.2f}) s/step, "
                    f"all-reduce wait mean {w_mean:.2f} s/step, {drawn:.1f} cells drawn/step", flush=True)
                TIMING.update(compute=0.0, wait=0.0, drawn=0, steps=0)
        tp = tick("gradient all-reduce", tp)
        if a.skip_nonfinite and any(q.grad is not None and not bool(torch.isfinite(q.grad).all()) for q in model.parameters()):
            print(f"[rank{rank}] step {step}: non-finite gradient; step skipped", flush=True)
            opt.zero_grad(set_to_none=True)
        else:
            opt.step()
        sched.step()
        tp = tick("optimizer step", tp)
        if a.profile_steps:
            PT["_wall"] = PT.get("_wall", 0.0) + tp - step_t0
        if ((step + 1) % a.eval_every == 0 or (a.save_every and (step + 1) % a.save_every == 0)) and not a.profile_steps:
            save_state(step)
        if step % 25 == 0:
            history.append({"step": step, "loss": total, "elapsed_s": time.time() - t1})
            if step % 100 == 0:
                conv_msg = f" conv {reduce_float(conv_total, device):.3f}" if a.conv_weight > 0 else ""
                say(f"  step {step:5d} loss {total:.4f}{conv_msg} gap_on {gap_on}  {time.time() - t1:.0f}s", flush=True)
    if main_rank:
        torch.save(model.checkpoint_payload(), a.out / "last.pt")
    save_state(a.steps)  # marks the run complete: a later --resume job exits instead of retraining the tail
    full = a.final_scoring == "full"
    val_scores = evaluate(model, val_rows, device, rank, world, sharded, a.metal_window) if full else []
    mon_scores = evaluate(model, monitor_rows, device, rank, world, sharded, a.metal_window) if full else []
    best_scores, selection = {}, None
    sel_hist = [h for h in history if "select" in h and "median_window_mae_eV" in h["select"]]
    if sel_hist:  # smoothed selection: centred mean of the selection median over --smooth evals
        meds = [h["select"]["median_window_mae_eV"] for h in sel_hist]
        half = a.smooth // 2
        smoothed = [float(np.mean(meds[max(0, i - half): i + half + 1])) for i in range(len(meds))]
        i = int(np.argmin(smoothed))
        selection = {"step": sel_hist[i]["step"], "smoothed_median_eV": smoothed[i], "raw_median_eV": meds[i],
                     "window": a.smooth, "n_select_cells": len(select_entries_) + len(val_entries),
                     "checkpoint": str(a.out / "ckpt" / f"step{sel_hist[i]['step']}.pt")}
        if main_rank:
            import shutil
            shutil.copyfile(selection["checkpoint"], a.out / "best_smoothed.pt")
    if world > 1:
        torch.distributed.barrier()
    chosen = a.out / ("best_smoothed.pt" if selection else "best.pt")
    if full and chosen.exists():  # the selected checkpoint, scored once on val and the inner holdout
        model.load_state_dict(torch.load(chosen, map_location=device, weights_only=False)["state_dict"])
        best_scores = {"val": evaluate(model, val_rows, device, rank, world, sharded, a.metal_window),
                       "inner_holdout": evaluate(model, inner_rows, device, rank, world, sharded, a.metal_window)}
    out = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "elements": train_elements, "n_train": len(train_entries), "n_val": len(val_entries),
        "train_ids": [e["id"] for e in train_entries], "best_step": best[0] if best else None,
        "final": {"val": summarize(val_scores), "monitor": summarize(mon_scores)},
        "best": {k: summarize(v) for k, v in best_scores.items()}, "best_scores": best_scores,
        "inner_ids": [e["id"] for e in inner_entries], "select_ids": [e["id"] for e in select_entries_],
        "selection": selection,
        "val_scores": val_scores, "monitor_scores": mon_scores, "history": history,
        "train_seconds": time.time() - t1,
    }
    out["ranks"] = world
    out["final_scoring"] = a.final_scoring
    if main_rank:
        (a.out / "metrics.json").write_text(json.dumps(out, indent=1) + "\n")
    say("final", json.dumps(out["final"]), flush=True)
    if world > 1:
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
