"""Grow a trained GeneralEPM to more elements without changing it on the old ones (EpNet-spd D0).

``expand_species(payload, add)`` builds a GeneralEPM whose species table is the checkpoint's followed by
``add`` and copies every trained parameter. Old elements keep their rows exactly, so a cell made only of
old elements gets the same bands as before. New elements start from what the shared parts predict for
them: the species hypernetwork and the feature-based embedding read their element features, the
per-element residual tables start at zero, and the fixed buffers (features, physical prior, orbital radii,
QEq parameters) are computed for them as for any element.

With ``element_feature_set="spd-v1"`` the feature table gains a column (the model d count); every layer
that reads element features gets a zero input weight for it. The column is 0 for every s/p element, so
this does not change the old elements either.
"""

from __future__ import annotations

from typing import Any, Sequence

import torch

from epnet.models.general_epm import GeneralEPM

# per-species rows: copied for old elements; new elements keep the fresh model's values (zero for residual
# tables, computed for buffers)
SPECIES_ROW_KEYS = (
    "base_residual", "base_prior", "z_model", "r_cov", "chi", "orbital_radii",
    "projector_log_offset", "charge_residual", "qeq_residual", "qeq_chi0", "qeq_j0", "projector_offdiag",
    "encoder.species_embedding.species_residual.weight", "d_overlap_raw", "ac_eps_shift", "ac_extent",
)
# element-feature tables: recomputed (they may gain columns); the old block must match the checkpoint
FEATURE_TABLE_KEYS = ("element_features", "encoder.species_embedding.element_features")
# first layers that read the element-feature vector: old input columns copied, new columns zero
FEATURE_INPUT_KEYS = (
    "base_hyper.0.weight", "charge_hyper.0.weight", "qeq_hyper.0.weight",
    "encoder.species_embedding.feature_projection.weight", "encoder.species_embedding.feature_residual.0.weight",
)


def expand_species(payload: dict[str, Any], add: Sequence[str], **config_overrides: Any) -> GeneralEPM:
    """A GeneralEPM over the checkpoint's species plus ``add``, equal to the checkpoint on old elements."""
    old_symbols = list(payload["species_symbols"])
    add = [str(s) for s in add if str(s) not in old_symbols]
    config = {**payload["config"], **config_overrides}
    model = GeneralEPM(old_symbols + add, **config)
    old_state, new_state = payload["state_dict"], model.state_dict()
    unknown = set(old_state) - set(new_state)
    if unknown:
        raise ValueError(f"checkpoint keys missing from the expanded model: {sorted(unknown)}")
    n_old = len(old_symbols)
    merged = {}
    for key, fresh in new_state.items():
        if key not in old_state:          # a module only the new config has: keep its initialization
            merged[key] = fresh
            continue
        old = old_state[key].to(fresh.dtype)
        if key in FEATURE_TABLE_KEYS:
            if not torch.equal(fresh[:n_old, : old.shape[1]], old):
                raise ValueError(f"{key}: recomputed element features differ from the checkpoint's")
            merged[key] = fresh
        elif key in SPECIES_ROW_KEYS:
            value = fresh.clone()
            value[:n_old] = old            # old species occupy the first rows, in the same order
            merged[key] = value
        elif key in FEATURE_INPUT_KEYS:
            value = torch.zeros_like(fresh)
            value[:, : old.shape[1]] = old
            merged[key] = value
        elif old.shape == fresh.shape:
            merged[key] = old.clone()
        else:
            raise ValueError(f"{key}: checkpoint shape {tuple(old.shape)} vs expanded {tuple(fresh.shape)}")
    model.load_state_dict(merged)
    return model


def atomic_start(model: GeneralEPM, symbols: Sequence[str]) -> list[str]:
    """Put the given d species at atomic parameters: their learned species deviation (hypernetwork + residual) is
    cancelled, so their species base is the physical prior. Only these rows change; returns the species reset.
    Used for the species added by an expansion (a warm start from a model that already trained other d species
    must not reset those)."""
    from epnet.data.valence import D_ELEMENTS

    rows = [i for i, s in enumerate(model.species_symbols) if s in set(symbols) and s in D_ELEMENTS]
    with torch.no_grad():
        model.base_residual[rows] = -model.base_hyper(model.element_features)[rows]
    return [model.species_symbols[i] for i in rows]


__all__ = ["FEATURE_INPUT_KEYS", "FEATURE_TABLE_KEYS", "SPECIES_ROW_KEYS", "atomic_start", "expand_species"]
