"""Run one EpNet prediction in a Slurm step (the web UI on the login node calls this through srun).
  python -m apps.epnet_bands.worker JOB_DIR      (JOB_DIR has input.bin and params.json; writes result.pkl or error.txt)"""
import json
import pickle
import sys
from pathlib import Path

from apps.epnet_bands.core import predict

job = Path(sys.argv[1]); prm = json.loads((job / "params.json").read_text())
try:
    p = predict((job / "input.bin").read_bytes(), prm["filename"], line_density=prm["line_density"], n_empty=prm["n_empty"],
                dos_sigma=prm.get("dos_sigma", 0.1))
    (job / "result.pkl").write_bytes(pickle.dumps(p))
except ValueError as ex:
    (job / "error.txt").write_text(str(ex))
