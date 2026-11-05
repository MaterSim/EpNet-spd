#!/usr/bin/env python3
"""Band structure and density of states of a crystal structure with EpNet, from the command line (CPU or GPU).

    python scripts/predict.py STRUCTURE [--out DIR] [--ckpt CHECKPOINT] [--threads N] [--workers P]

STRUCTURE: CIF, POSCAR/CONTCAR/.vasp, or (by extension) extxyz/xyz, Quantum ESPRESSO input, FHI-aims geometry.in,
.res, .gen, CASTEP .cell, .xsf, .cfg. The cell is standardized to its primitive cell; bands follow the
Setyawan-Curtarolo path; the DOS uses a Gamma-centred mesh reduced by symmetry. Writes <formula>_epnet_bands.csv,
_dos.csv, _bands.png, _summary.json and _structure.vasp into --out and prints the summary.
--workers P > 1 splits the k points over P processes of threads/P threads each (faster on many-core CPUs)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("structure", type=Path)
    ap.add_argument("--out", type=Path, default=Path("."), help="output directory (default: current directory)")
    ap.add_argument("--ckpt", default=None, help="model checkpoint (default: the final model, seed 0)")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads (default: all cores)")
    ap.add_argument("--workers", type=int, default=1, help="k-point-parallel processes (default 1)")
    ap.add_argument("--kpoints-per-invA", type=float, default=25.0, help="k-point density along the path (per 1/A)")
    ap.add_argument("--empty-bands", type=int, default=8, help="minimum empty bands (more are added for d atoms)")
    ap.add_argument("--emin", type=float, default=-6.0)
    ap.add_argument("--emax", type=float, default=6.0)
    a = ap.parse_args()
    threads = a.threads or os.cpu_count() or 1
    os.environ["EPNET_THREADS"] = str(threads)
    from apps.epnet_bands import core
    if a.ckpt:
        os.environ["EPNET_CKPT"] = str(Path(a.ckpt).resolve())
    if a.workers > 1:
        core.start_pool(a.workers, max(1, threads // a.workers), a.ckpt)
    p = core.predict(a.structure.read_bytes(), a.structure.name, line_density=a.kpoints_per_invA, n_empty=a.empty_bands,
                     ckpt=a.ckpt)
    a.out.mkdir(parents=True, exist_ok=True)
    stem = a.out / f"{p.formula}_epnet"
    Path(f"{stem}_bands.csv").write_text(core.to_csv(p))
    Path(f"{stem}_dos.csv").write_text(core.to_dos_csv(p))
    Path(f"{stem}_bands.png").write_bytes(core.figure_png(p, a.emin, a.emax))
    Path(f"{stem}_summary.json").write_text(json.dumps(core.summary(p), indent=1))
    Path(f"{stem}_structure.vasp").write_text(p.poscar)
    print(json.dumps(core.summary(p), indent=1))
    print(f"wrote {stem}_{{bands.csv,dos.csv,bands.png,summary.json,structure.vasp}}", file=sys.stderr)


if __name__ == "__main__":
    main()
