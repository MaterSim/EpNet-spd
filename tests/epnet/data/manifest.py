"""Immutable JSON manifests and NPZ sample loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import numpy as np

from .structures import BandData, ReciprocalPotential, Sample, Structure


class NPZDataset:
    """Lazy dataset described by a JSON manifest.

    The manifest is ``{"version": 1, "samples": [{"id": ..., "path": ...}]}``.
    Paths are resolved relative to the manifest and may not escape its directory
    unless ``allow_external`` is explicitly enabled.
    """

    def __init__(self, manifest: str | Path, *, allow_external: bool = False):
        self.manifest_path = Path(manifest).resolve()
        raw = json.loads(self.manifest_path.read_text())
        if raw.get("version") != 1 or not isinstance(raw.get("samples"), list):
            raise ValueError("manifest must have version 1 and a samples list")
        self.entries = tuple(raw["samples"])
        self.root = self.manifest_path.parent
        self.allow_external = allow_external
        ids = [entry.get("id") for entry in self.entries]
        if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
            raise ValueError("sample ids must be non-empty and unique")

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[Sample]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> Sample:
        entry = self.entries[index]
        path = (self.root / entry["path"]).resolve()
        if not self.allow_external and self.root not in path.parents:
            raise ValueError(f"sample path escapes manifest directory: {path}")
        with np.load(path, allow_pickle=False) as data:
            required = {"lattice", "species", "fractional_positions"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(f"{path} is missing arrays: {sorted(missing)}")
            structure = Structure(
                data["lattice"],
                tuple(data["species"].astype(str)),
                data["fractional_positions"],
                metadata=entry.get("metadata", {}),
            )
            potential = None
            if {"g_indices", "potential_real"}.issubset(data.files):
                imag = data["potential_imag"] if "potential_imag" in data.files else 0.0
                potential = ReciprocalPotential(data["g_indices"], data["potential_real"] + 1j * imag)
            bands = None
            if {"kpoints", "energies"}.issubset(data.files):
                weights = data["kpoint_weights"] if "kpoint_weights" in data.files else None
                fermi = float(data["fermi_energy"]) if "fermi_energy" in data.files else None
                bands = BandData(data["kpoints"], data["energies"], weights, fermi)
        return Sample(structure, potential, bands, entry["id"])


def write_manifest(path: str | Path, samples: list[dict]) -> None:
    """Write a stable, human-readable v1 manifest."""
    Path(path).write_text(json.dumps({"version": 1, "samples": samples}, indent=2, sort_keys=True) + "\n")
