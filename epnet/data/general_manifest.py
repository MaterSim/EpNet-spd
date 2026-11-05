"""General (multi-species) manifest: records, preregistered splits, and loading.

Entries follow the v1 manifest layout (``{"id", "path", "metadata"}``). Every
entry carries the ``zm-v1`` valence contract; ``band_offset`` target bands are
dropped at load time, and ``n_occupied_model`` is the occupied-band count the
EPM sees. There is deliberately no ``n_occupied`` key, so legacy loaders cannot
silently reuse the PAW count.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from math import gcd
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from epnet.hamiltonian import reciprocal_vectors
from epnet.hamiltonian.torch_pw import g_cutoff_angstrom

from .structures import BandData, Sample, Structure
from .valence import COMPATIBLE_VERSIONS, CONTRACT_VERSION, apply_contract, d_tier, model_target, mp_plus_u

MANIFEST_KIND = "general-v1"
SPLITS = ("train", "val", "holdout_chemsys", "holdout_element", "holdout_prototype", "sealed")


def anonymized_formula(species: Iterable[str]) -> str:
    """Reduced, element-blind stoichiometry, e.g. ``NaCl -> AB``, ``Al2O3 -> A2B3``."""
    counts = sorted(Counter(str(s) for s in species).values())
    divisor = 0
    for c in counts:
        divisor = gcd(divisor, c)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "".join(f"{letters[i]}{c // divisor if c // divisor > 1 else ''}" for i, c in enumerate(counts))


def prototype_key(species: Iterable[str], spacegroup_number: int) -> str:
    return f"{anonymized_formula(species)}|{int(spacegroup_number)}"


def _unit_hash(salt: str, kind: str, key: str) -> float:
    digest = hashlib.sha256(f"{salt}:{kind}:{key}".encode()).hexdigest()
    return int(digest[:15], 16) / float(16**15)


def assign_split(
    record: dict[str, Any],
    *,
    salt: str,
    holdout_elements: frozenset[str],
    sealed_frac: float,
    chemsys_frac: float,
    prototype_frac: float,
    val_frac: float,
) -> tuple[str, str | None]:
    """Deterministic split for one record; the order makes every holdout disjoint.

    1. any holdout element -> ``holdout_element``
    2. chemical system hashed into the sealed band -> ``sealed``
    3. chemical system hashed into the next band -> ``holdout_chemsys``
    4. prototype (anonymized formula + space group) -> ``holdout_prototype``
    5. id hash -> ``val``, otherwise ``train``
    """
    elements = set(record["elements"])
    hit = sorted(elements & holdout_elements)
    if hit:
        return "holdout_element", "-".join(hit)
    u = _unit_hash(salt, "chemsys", record["chemsys"])
    if u < sealed_frac:
        return "sealed", None
    if u < sealed_frac + chemsys_frac:
        return "holdout_chemsys", None
    if _unit_hash(salt, "prototype", record["prototype"]) < prototype_frac:
        return "holdout_prototype", record["prototype"]
    if _unit_hash(salt, "id", record["id"]) < val_frac:
        return "val", None
    return "train", None


def build_entry(material: dict[str, Any], npz_path: Path, rel_path: str, *, ecut_ry: float = 8.0) -> dict[str, Any]:
    """Apply the contract to one MP NPZ; returns an entry or ``{"id", "rejected"}``."""
    mid = material["material_id"]
    with np.load(npz_path, allow_pickle=True) as data:
        species = tuple(str(s) for s in data["species"])
        if mp_plus_u(species):  # MP labels this chemistry with GGA+U, not PBE
            return {"id": mid, "rejected": "plus_u"}
        if "energies" not in data.files or {"energies_up", "energies_down"} & set(data.files):
            return {"id": mid, "rejected": "spin_polarized"}
        lattice = np.asarray(data["lattice"], dtype=float)
        energies = np.asarray(data["energies"], dtype=float)
        fermi = float(data["fermi_energy"])
        n_k = int(energies.shape[0])
    result = apply_contract(species, energies, fermi, is_metal=bool(material["is_metal"]))
    if not result.ok:
        return {"id": mid, "rejected": result.reason}
    g_idx, _ = reciprocal_vectors(lattice, g_cutoff_angstrom(ecut_ry))
    n_g = int(len(g_idx))
    n_bands_model = int(energies.shape[1] - result.n_drop)
    if result.n_occupied_model >= n_g:
        return {"id": mid, "rejected": "basis_smaller_than_valence"}
    meta = {
        "material_id": mid,
        "formula": material["formula"],
        "chemsys": material["chemsys"],
        "elements": sorted(set(species)),
        "nelements": int(material["nelements"]),
        "nsites": int(material["nsites"]),
        "spacegroup_number": int(material["spacegroup_number"]),
        "prototype": prototype_key(species, material["spacegroup_number"]),
        "is_stable": bool(material["is_stable"]),
        "energy_above_hull": float(material["energy_above_hull"]),
        "mp_band_gap": float(material["band_gap"]),
        "fermi_energy": fermi,
        "label_source": "mp-gga",
        "spin": 1,
        "plus_u": False,
        "n_k": n_k,
        "n_bands_paw": int(energies.shape[1]),
        "n_bands_model": n_bands_model,
        f"n_g_ecut{int(ecut_ry)}": n_g,
        "use_gap_loss": bool(result.gapped),
        "d_tier": d_tier(species),
        **result.metadata(),
    }
    return {"id": mid, "path": rel_path, "metadata": meta}


def write_general_manifest(path: str | Path, entries: list[dict], provenance: dict[str, Any]) -> None:
    payload = {
        "version": 1,
        "kind": MANIFEST_KIND,
        "z_model_version": CONTRACT_VERSION,
        "provenance": provenance,
        "samples": entries,
    }
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")


def load_general_manifest(path: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    if raw.get("kind") != MANIFEST_KIND or raw.get("z_model_version") not in COMPATIBLE_VERSIONS:
        raise ValueError(f"{path} is not a {MANIFEST_KIND} manifest under {sorted(COMPATIBLE_VERSIONS)}")
    return raw


def select_entries(raw: dict[str, Any], splits: Iterable[str], *, allow_sealed: bool = False) -> list[dict[str, Any]]:
    """Entries of the requested splits. ``sealed`` is refused unless ``allow_sealed``
    (reserved for the final gate; see ``data/manifests/general_sp_v1.seal.json``)."""
    wanted = set(splits)
    unknown = wanted - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits: {sorted(unknown)}")
    if "sealed" in wanted and not allow_sealed:
        raise PermissionError("the sealed split is reserved for the final gate; pass allow_sealed=True only there")
    return [e for e in raw["samples"] if e["metadata"]["split"] in wanted]


def load_general_sample(entry: dict[str, Any], root: str | Path) -> Sample:
    """Load one entry with the dropped semicore block already removed from the bands."""
    meta = entry["metadata"]
    offset = int(meta["band_offset"])
    with np.load(Path(root) / entry["path"], allow_pickle=True) as data:
        structure = Structure(
            data["lattice"],
            tuple(str(s) for s in data["species"]),
            data["fractional_positions"],
            metadata=meta,
        )
        bands = BandData(
            data["kpoints"],
            model_target(data["energies"], offset),
            data["kpoint_weights"] if "kpoint_weights" in data.files else None,
            float(data["fermi_energy"]),
        )
    if bands.energies.shape[1] != int(meta["n_bands_model"]):
        raise ValueError(f"{entry['id']}: band count changed since the manifest was built")
    return Sample(structure, None, bands, entry["id"])


__all__ = [
    "MANIFEST_KIND",
    "SPLITS",
    "anonymized_formula",
    "assign_split",
    "build_entry",
    "load_general_manifest",
    "load_general_sample",
    "prototype_key",
    "select_entries",
    "write_general_manifest",
]
