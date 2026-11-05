#!/usr/bin/env python3
"""Replay one training stage of the paper. configs/stages/<stage>_s<seed>.json holds the exact arguments of that run
(recorded by the trainer in its metrics.json), with the paths of this repository: "starts_from" is the checkpoint it was
initialized from, "writes" the checkpoint of it that is published, "ranks" the number of GPUs (data-parallel ranks) it
ran on and "train_hours" its training time. This script turns the arguments back into the scripts/train_general_sp.py
command line, checks that it parses to the recorded arguments, and prints or runs it.

    python scripts/train_stage.py configs/stages/5cN_s0.json --dry-run          # print the command
    python scripts/train_stage.py configs/stages/5cN_s0.json                    # run it with the recorded ranks
    python scripts/train_stage.py configs/stages/5cN_s0.json --args             # the trainer arguments only (srun, ...)

Runs with one rank are launched as one process; runs with more (5aL, 5cL, 5cN: 4) with torchrun. With several ranks the
training cells are sharded and each rank draws its share of every batch of 16 from its own cells, so the sampling, and
with it the trajectory, depends on the number of ranks: an exact replay needs the recorded number (--ranks changes it).
Extra trainer arguments after "--" are appended (e.g. -- --device cpu)."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def trainer_parser() -> argparse.ArgumentParser:
    sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
    import train_general_sp
    return train_general_sp.build_parser()


def argv_from_config(parser: argparse.ArgumentParser, cfg: dict) -> list[str]:
    argv = []
    for act in parser._actions:
        if not act.option_strings or act.dest == "help" or act.dest not in cfg:
            continue
        v, opt = cfg[act.dest], max(act.option_strings, key=len)
        if isinstance(act, argparse._StoreTrueAction):
            argv += [opt] if v else []
        elif isinstance(act, argparse._StoreFalseAction):
            argv += [] if v else [opt]
        elif v is None:
            continue
        elif isinstance(v, list):
            argv += [opt] + [str(x) for x in v] if (v or act.nargs == "*") else []
        else:
            argv += [opt, str(v)]
    return argv


def check(parser: argparse.ArgumentParser, argv: list[str], cfg: dict) -> tuple[dict, list[str]]:
    """recorded arguments that the command line does not reproduce, and recorded keys the trainer no longer has"""
    ns = vars(parser.parse_args(argv))

    def norm(v):
        return [norm(x) for x in v] if isinstance(v, (list, tuple)) else (str(v) if isinstance(v, Path) else v)
    diff = {k: (cfg[k], ns[k]) for k in cfg if k in ns and norm(ns[k]) != norm(cfg[k])}
    return diff, sorted(set(cfg) - set(ns))


def main() -> None:
    raw = sys.argv[1:]
    extra = raw[raw.index("--") + 1:] if "--" in raw else []
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", type=Path)
    ap.add_argument("--dry-run", action="store_true", help="print the command, do not run it")
    ap.add_argument("--args", action="store_true", help="print only the trainer arguments (for srun or another launcher)")
    ap.add_argument("--ranks", type=int, default=None, help="number of GPUs / ranks (default: the paper's, from the config)")
    a = ap.parse_args(raw[:raw.index("--")] if "--" in raw else raw)
    rec = json.loads(a.config.read_text())
    parser = trainer_parser()
    argv = argv_from_config(parser, rec["args"])
    diff, unknown = check(parser, argv, rec["args"])
    if diff:
        sys.exit(f"the command line does not reproduce the recorded arguments: {diff}")
    if unknown:
        print(f"note: recorded keys not in the trainer (ignored): {', '.join(unknown)}", file=sys.stderr)
    argv += extra
    if a.args:
        print(shlex.join(argv))
        return
    ranks = a.ranks or int(rec.get("ranks") or 1)
    launch = ["torchrun", "--nproc_per_node", str(ranks)] if ranks > 1 else [sys.executable]
    cmd = launch + ["scripts/train_general_sp.py"] + argv
    print(f"# stage {rec['stage']} seed {rec['seed']}: from {rec['starts_from']}, published as {rec['writes']}; "
          f"paper: {rec.get('ranks', 1)} rank(s), {rec.get('train_hours')} h", file=sys.stderr)
    if ranks != int(rec.get("ranks") or 1):
        print(f"# note: {ranks} ranks instead of the paper's {rec.get('ranks')}: the cell sampling differs", file=sys.stderr)
    print(shlex.join(cmd))
    if not a.dry_run:
        sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
