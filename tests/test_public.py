"""Checks of the public repository: every checkpoint loads, every recorded training stage replays to its exact
arguments, the split table is complete, and predictions reproduce the reference gaps of the example structures."""
import csv
import gzip
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

# gaps (eV) of the examples with the final model, seed 0 (None: metal)
REFERENCE = {"Al_mp-134.cif": None, "Cu_mp-30.cif": None, "GaAs_mp-2534.cif": 0.2424, "Si_mp-149.cif": 0.5747,
             "ZnO_mp-2133.cif": 0.7186, "TiO2_mp-2657.cif": 1.7886}


# elements per checkpoint: s/p (34), then the d groups as the curriculum adds them (d10 +6, early d +8, 4d/5d +10,
# open-shell 3d +5); 63 from 4c-S on
ELEMENTS = {"epnet_sp_g4_s0.pt": 34, "J-A2_s1.pt": 40, "4a-A_s0.pt": 48, "4b-S_s1.pt": 58}


def test_checkpoints_load():
    from epnet.models.general_epm import GeneralEPM
    files = sorted((ROOT / "models").rglob("*.pt"))
    assert len(files) == 13          # EpNet-sp start, 10 stage checkpoints, the final model (2 seeds)
    for f in files:
        model = GeneralEPM.from_checkpoint(torch.load(f, map_location="cpu", weights_only=False))
        n = len(model.species_symbols)
        assert n == ELEMENTS.get(f.name, 63), (f.name, n)


def test_stage_configs_replay():
    from train_stage import argv_from_config, check, trainer_parser
    parser = trainer_parser()
    configs = sorted((ROOT / "configs/stages").glob("*.json"))
    assert len(configs) == 12
    for f in configs:
        rec = json.loads(f.read_text())
        diff, _ = check(parser, argv_from_config(parser, rec["args"]), rec["args"])
        assert not diff, (f.name, diff)
        assert (ROOT / rec["starts_from"]).exists() and (ROOT / rec["writes"]).exists(), f.name
        for k, v in rec["args"].items():                  # the cell lists the run read
            if isinstance(v, str) and v.startswith("configs/"):
                assert (ROOT / v).exists(), (f.name, k, v)


def test_splits_table():
    with gzip.open(ROOT / "data/splits/general_spd_v1_splits.csv.gz", "rt") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 22553
    assert {r["split"] for r in rows} >= {"train", "val", "holdout_chemsys", "holdout_prototype", "sealed"}


@pytest.mark.parametrize("name,ref", sorted(REFERENCE.items()))
def test_example_gaps(name, ref):
    from apps.epnet_bands.core import predict
    f = ROOT / "apps/epnet_bands/examples" / name
    p = predict(f.read_bytes(), f.name)
    if ref is None:
        assert p.metal
    else:
        assert not p.metal and abs(p.gap - ref) < 0.01, (name, p.gap, ref)


def test_training_chain():
    """each stage starts from the checkpoint its predecessor published; every config records its ranks"""
    recs = {f.stem: json.loads(f.read_text()) for f in (ROOT / "configs/stages").glob("*.json")}
    written = {r["writes"] for r in recs.values()} | {"models/stages/epnet_sp_g4_s0.pt"}
    for name, r in recs.items():
        assert r["starts_from"] in written, (name, r["starts_from"])
        assert isinstance(r["ranks"], int) and r["ranks"] >= 1, name
        assert r["args"]["init_from"] == r["starts_from"], name
    assert recs["5cN_s0"]["writes"] == "models/epnet_spd_5cN_s0.pt" and recs["5cN_s1"]["writes"] == "models/epnet_spd_5cN_s1.pt"


def test_manifests():
    """the paper's manifest agrees with the split table; every evaluated cell is in a published manifest"""
    man = json.loads((ROOT / "data/manifests/general_spd_v1.json").read_text())["samples"]
    assert len(man) == 22553 and all(e["path"].startswith("../mp_bands/npz/") for e in man)
    with gzip.open(ROOT / "data/splits/general_spd_v1_splits.csv.gz", "rt") as fh:
        table = {r["id"]: r["split"] for r in csv.DictReader(fh)}
    assert table == {e["id"]: e["metadata"]["split"] for e in man}
    known = {e["id"] for e in man}
    for name in ("bigcell_sp_v1.json", "bigcell_spd_v1.json"):
        known |= {e["id"] for e in json.loads((ROOT / "data/manifests" / name).read_text())["samples"]}
    for cells in ("results/stage5/eval_cells_5cL.json", "results/stage5/closeout/cells.json"):
        missing = [c["id"] for c in json.loads((ROOT / cells).read_text()) if c["id"] not in known]
        assert not missing, (cells, len(missing), missing[:5])


def test_documented_paths_exist():
    """every repository path the READMEs mention exists (generated outputs excepted)"""
    import re
    generated = ("data/mp_bands", "data/alexandria", "runs/", "results/stage5/5cN/", "results/alexandria/5cN_s0", "data/manifests/<")
    pat = re.compile(r"((?:apps|configs|data|epnet|models|results|scripts|tests)/[A-Za-z0-9_./{},<>-]+)")
    for doc in ("README.md", "models/README.md", "apps/epnet_bands/README.md"):
        for m in pat.finditer((ROOT / doc).read_text()):
            p = m.group(1).rstrip(".,)")
            paths = [p.replace("{0,1}", s) for s in "01"] if "{0,1}" in p else [p]
            for q in paths:
                if not q.startswith(generated) and "<" not in q:
                    assert (ROOT / q).exists(), (doc, q)
