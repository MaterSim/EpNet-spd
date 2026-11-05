"""Si family names and occupied-band count. Catalog factories live elsewhere."""

from __future__ import annotations

from typing import Any

TRAIN_FAMILIES = (
    "diamond",
    "bct",
    "bc8",
    "beta_tin",
    "r8",
    "simple_hex",
    "tI16",
    "oC12",
    "clathrate_i",
)
HOLDOUT_FAMILIES = ("lonsdaleite", "st12", "clathrate_ii")
ALL_FAMILIES = TRAIN_FAMILIES + HOLDOUT_FAMILIES


def silicon_n_occupied(structure: Any) -> int:
    """Spinless valence count: 2 electrons per Si atom."""
    if isinstance(structure, int):
        return 2 * int(structure)
    if hasattr(structure, "n_atoms"):
        return 2 * int(structure.n_atoms)
    metadata = getattr(structure, "metadata", None) or {}
    if "n_occupied" in metadata:
        return int(metadata["n_occupied"])
    species = getattr(structure, "species", ())
    return 2 * len(tuple(species))
