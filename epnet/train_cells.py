"""Cell-parallel gradient accumulation with the same weighted catalog loss."""

from __future__ import annotations

import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Callable

import torch
from torch import Tensor, nn

from epnet.hamiltonian.torch_pw import fermi_aligned_window_loss, vbm_aligned_window_loss


_TRAIN_ROW_KEYS = ("k", "target")
_TRAIN_CACHE_KEYS = (
    "g_idx",
    "unique_cart",
    "inverse",
    "kinetic",
    "species",
    "positions",
    "unique_q",
    "inv_q",
    "phases",
    "spherical",
    "edge_index",
    "cell_shifts",
    "sp_projectors",
)


def _maybe_pin(value: Any) -> Any:
    if (
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and torch.cuda.is_available()
        and not value.is_pinned()
    ):
        return value.pin_memory()
    return value


def _tensor_to(value: Any, device: torch.device, *, pin: bool = False) -> Any:
    if not torch.is_tensor(value):
        return value
    if pin:
        value = _maybe_pin(value)
    return value.to(device, non_blocking=device.type == "cuda")


def move_pw_cache(
    cache: Any,
    device: torch.device,
    *,
    pin: bool = False,
    train_only: bool = False,
) -> Any:
    """Move cached PW tensors to ``device`` without cloning host storage."""
    if cache is None:
        return None
    if train_only:
        payload = {key: _tensor_to(getattr(cache, key, None), device, pin=pin) for key in _TRAIN_CACHE_KEYS}
        payload["volume"] = cache.volume
        payload["n_g"] = cache.n_g
        return SimpleNamespace(**payload)
    payload = {key: _tensor_to(val, device, pin=pin) for key, val in vars(cache).items()}
    return SimpleNamespace(**payload)


def move_packed(packed: Any, device: torch.device, *, pin: bool = False) -> Any:
    payload = {key: _tensor_to(val, device, pin=pin) for key, val in vars(packed).items()}
    return SimpleNamespace(**payload)


def move_cell_row(
    row: dict[str, Any],
    device: torch.device,
    *,
    pin: bool = False,
    train_only: bool = False,
) -> dict[str, Any]:
    """Return a shallow row copy with packed/cache/k/target on ``device``."""
    out = dict(row)
    out["packed"] = move_packed(row["packed"], device, pin=pin)
    out["cache"] = move_pw_cache(row["cache"], device, pin=pin, train_only=train_only)
    keys = _TRAIN_ROW_KEYS if train_only else ("k", "k_full", "target", "e_full")
    for key in keys:
        if key in out:
            out[key] = _tensor_to(out[key], device, pin=pin)
    return out


def _record_row_stream(row: dict[str, Any], stream: torch.cuda.Stream) -> None:
    """Keep side-stream copies alive until ``stream`` is done with them."""

    def rec(value: Any) -> None:
        if torch.is_tensor(value) and value.is_cuda:
            value.record_stream(stream)

    packed = row.get("packed")
    if packed is not None:
        for val in vars(packed).values():
            rec(val)
    cache = row.get("cache")
    if cache is not None:
        for val in vars(cache).values():
            rec(val)
    for key in ("k", "target", "k_full", "e_full"):
        if key in row:
            rec(row[key])


def pin_cell_row(row: dict[str, Any]) -> dict[str, Any]:
    """Pin host tensors so later CUDA copies can be non-blocking."""
    packed = row["packed"]
    for key, val in vars(packed).items():
        pinned = _maybe_pin(val)
        if pinned is not val:
            setattr(packed, key, pinned)
    cache = row.get("cache")
    if cache is not None:
        for key, val in vars(cache).items():
            pinned = _maybe_pin(val)
            if pinned is not val:
                setattr(cache, key, pinned)
    for key in ("k", "target"):
        if key in row:
            row[key] = _maybe_pin(row[key])
    return row


def _tensor_nbytes(value: Any) -> int:
    if torch.is_tensor(value):
        return int(value.numel() * value.element_size())
    return 0


def row_host_nbytes(row: dict[str, Any]) -> int:
    """Host-resident tensor bytes for one cached catalog cell."""
    n = 0
    for key in ("k", "k_full", "target", "e_full"):
        n += _tensor_nbytes(row.get(key))
    packed = row.get("packed")
    if packed is not None:
        n += sum(_tensor_nbytes(val) for val in vars(packed).values())
    cache = row.get("cache")
    if cache is not None:
        n += sum(_tensor_nbytes(val) for val in vars(cache).values())
    return n


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device


def resolve_thread_plan(cell_workers: int, threads_per_worker: int = 0) -> tuple[int, int]:
    """Return ``(workers, threads)`` and pin BLAS/PyTorch to ``threads`` per process."""
    workers = max(1, int(cell_workers))
    if threads_per_worker and int(threads_per_worker) > 0:
        threads = int(threads_per_worker)
    else:
        slurm = os.environ.get("SLURM_CPUS_PER_TASK")
        cpus = int(slurm) if slurm else (os.cpu_count() or 8)
        threads = max(1, int(cpus) // workers)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = str(threads)
    return workers, threads


def partition_by_work(work: list[int], world_size: int) -> list[int]:
    """Assign each index to a rank, packing the largest remaining item onto the lightest rank."""
    world = max(1, int(world_size))
    if world == 1:
        return [0] * len(work)
    loads = [0] * world
    assign = [0] * len(work)
    for i in sorted(range(len(work)), key=lambda idx: work[idx], reverse=True):
        rank = min(range(world), key=lambda r: loads[r])
        assign[i] = rank
        loads[rank] += int(work[i])
    return assign


def reduce_grads(model: nn.Module) -> None:
    """Sum parameter grads across ranks. No-op when not distributed."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return
    if torch.distributed.get_world_size() <= 1:
        return
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            param.grad = torch.zeros_like(param)
        torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)


def reduce_float(value: float, device: torch.device, *, op: str = "sum") -> float:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return float(value)
    if torch.distributed.get_world_size() <= 1:
        return float(value)
    tensor = torch.tensor([float(value)], device=device, dtype=torch.float64)
    red = torch.distributed.ReduceOp.SUM if op == "sum" else torch.distributed.ReduceOp.MAX
    torch.distributed.all_reduce(tensor, op=red)
    return float(tensor.item())


def gather_lists(items: list[Any]) -> list[Any]:
    """Concatenate per-rank Python lists onto every rank."""
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return list(items)
    world = torch.distributed.get_world_size()
    if world <= 1:
        return list(items)
    gathered: list[Any] = [None] * world
    torch.distributed.all_gather_object(gathered, list(items))
    out: list[Any] = []
    for part in gathered:
        out.extend(part)
    return out


def weighted_cell_loss(
    model: nn.Module,
    row: dict[str, Any],
    *,
    gap_weight: float,
    env_l2: float,
    refresh_sphere: bool,
    n_families: int,
    family_count: dict[str, int],
    edge_weight: float = 5.0,
    residual: bool = True,
    gap_mode: str = "clamp",
) -> tuple[Tensor, float]:
    """Return ``(weighted_loss, raw_loss)`` for one catalog cell.

    ``raw_loss`` is a detached tensor so CUDA callers can accumulate without
    a per-cell host sync.
    """
    device = _model_device(model)
    if row["k"].device != device:
        row = move_cell_row(row, device, train_only=True)
    pred = model.band_path(
        row["packed"],
        row["k"],
        n_bands=row["n_bands"],
        cache=row["cache"],
        residual=residual,
        refresh_sphere=refresh_sphere,
    )
    if row["use_gap"]:
        loss = vbm_aligned_window_loss(
            pred, row["target"], row["n_occ"], edge_weight=float(edge_weight)
        )
        # signed keeps the band-overlap depth, so a closed predicted gap still has a gradient.
        pred_gap = pred[:, row["n_occ"]].amin() - pred[:, row["n_occ"] - 1].amax()
        if gap_mode == "clamp":
            pred_gap = torch.clamp(pred_gap, min=0.0)
        elif gap_mode != "signed":
            raise ValueError(f"gap_mode must be clamp or signed, got {gap_mode!r}")
        tgt_gap = torch.clamp(
            row["target"][:, row["n_occ"]].amin() - row["target"][:, row["n_occ"] - 1].amax(), min=0.0
        )
        loss = loss + float(gap_weight) * (pred_gap - tgt_gap).abs()
    else:
        loss = fermi_aligned_window_loss(pred, row["target"], row["n_occ"])
    env_delta = getattr(model, "_last_env_delta", None)
    if env_l2 and env_delta is not None:
        loss = loss + float(env_l2) * env_delta.pow(2).mean()
    weight = 1.0 / (n_families * family_count[row["family"]])
    return loss * weight, loss.detach()


def _add_grads(dst: nn.Module, src: nn.Module) -> None:
    for p_dst, p_src in zip(dst.parameters(), src.parameters()):
        if p_src.grad is None:
            continue
        if p_dst.grad is None:
            p_dst.grad = p_src.grad.detach().clone()
        else:
            p_dst.grad.add_(p_src.grad)


def accumulate_cell_grads(
    model: nn.Module,
    rows: list[dict[str, Any]],
    *,
    cell_workers: int,
    gap_weight: float,
    env_l2: float,
    refresh_sphere: bool,
    n_families: int,
    family_count: dict[str, int],
    edge_weight: float = 5.0,
    epoch_id: int = 0,
    residual: bool = True,
    gap_mode: str = "clamp",
) -> float:
    """Backward the catalog loss onto ``model``. Returns the unweighted mean numerator (sum of raw losses)."""
    if not rows:
        return 0.0
    if _model_device(model).type == "cuda":
        workers = 1
    else:
        workers = max(1, int(cell_workers))
    loss_kwargs = dict(
        gap_weight=gap_weight,
        env_l2=env_l2,
        refresh_sphere=refresh_sphere,
        n_families=n_families,
        family_count=family_count,
        edge_weight=edge_weight,
        residual=residual,
        gap_mode=gap_mode,
    )
    if _model_device(model).type == "cuda":
        return _accumulate_cuda(model, rows, loss_kwargs)
    if workers <= 1 or len(rows) == 1:
        running = None
        for row in rows:
            weighted, raw = weighted_cell_loss(model, row, **loss_kwargs)
            weighted.backward()
            running = raw if running is None else running + raw
        return float(running)

    ordered = sorted(rows, key=lambda r: int(getattr(r["cache"], "n_g", 0)), reverse=True)
    local = threading.local()
    lock = threading.Lock()
    running = 0.0

    def replica() -> nn.Module:
        if not hasattr(local, "model"):
            local.model = copy.deepcopy(model)
            local.epoch_id = None
        return local.model

    def run(row: dict[str, Any], epoch_id: int) -> float:
        worker = replica()
        if local.epoch_id != epoch_id:
            worker.load_state_dict(model.state_dict())
            worker.train(model.training)
            local.epoch_id = epoch_id
        worker.zero_grad(set_to_none=True)
        weighted, raw = weighted_cell_loss(worker, row, **loss_kwargs)
        weighted.backward()
        with lock:
            _add_grads(model, worker)
        return float(raw)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        raws = list(pool.map(lambda row: run(row, int(epoch_id)), ordered))
    return float(sum(raws))


def _accumulate_cuda(model: nn.Module, rows: list[dict[str, Any]], loss_kwargs: dict[str, Any]) -> float:
    """Serial GPU cells with one-cell lookahead copy on a side stream."""
    device = _model_device(model)
    ordered = sorted(rows, key=lambda r: int(getattr(r["cache"], "n_g", 0)), reverse=True)
    compute = torch.cuda.current_stream()
    copy_stream = torch.cuda.Stream()

    def upload(row: dict[str, Any]) -> dict[str, Any]:
        with torch.cuda.stream(copy_stream):
            # Pin once on the host catalog. Per-cell pin_memory() serializes CUDA.
            moved = move_cell_row(row, device, pin=False, train_only=True)
        _record_row_stream(moved, compute)
        return moved

    nxt = upload(ordered[0])
    running = None
    for i, _row in enumerate(ordered):
        compute.wait_stream(copy_stream)
        cur = nxt
        nxt = upload(ordered[i + 1]) if i + 1 < len(ordered) else None
        weighted, raw = weighted_cell_loss(model, cur, **loss_kwargs)
        weighted.backward()
        running = raw if running is None else running + raw
        del cur, weighted
    return float(running)


def prepare_rows_parallel(
    prepare: Callable[[Any], dict[str, Any]],
    samples: list[Any],
    cell_workers: int,
) -> list[dict[str, Any]]:
    workers = max(1, int(cell_workers))
    if workers <= 1 or len(samples) <= 1:
        return [prepare(sample) for sample in samples]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(prepare, samples))
