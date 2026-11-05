"""Shared gated Si EPM: species radial sphere, optional env Δv(q), l<=2 residual."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from epnet.data.elements import SpeciesTable
from epnet.data.si_prototypes import silicon_n_occupied
from epnet.data.structures import Structure
from epnet.hamiltonian import reciprocal_vectors
from epnet.hamiltonian.torch_pw import (
    COULOMB_EV_A,
    add_multipole_potential,
    batched_band_energies,
    fold_to_first_zone,
    g_cutoff_angstrom,
    kinetic_diagonal,
    local_potential_matrix,
    scatter_hermitian_potential,
    structure_phases,
    unique_g_transfers,
)
from epnet.models.baselines.radial import (
    CosineRadialEP,
    DensityConditionedSphereDelta,
    EnvironmentRadialDelta,
    empty_core_form_factor,
    silicon_cohen_form_factor,
)
from epnet.models.environment_ionic import EnvironmentIonicCorrection
from epnet.models.equivariant_encoder import (
    EquivariantEncoder,
    build_periodic_neighbor_graph,
    gate_window_radii,
    packing_gate,
    packing_gate_windows,
    parse_gate_window,
    smooth_coordination,
)
from epnet.models.multipole_form_factor import MultipoleFormFactor
from epnet.models.sp_projector import SPProjectorCorrection
from epnet.physics.long_range import neutralize_effective_charges, reciprocal_coulomb_potential

_ENV_PREFIX = "env_radial."
_CHARGE_PREFIX = "effective_charge"
_SP_PREFIX = "sp_projector."
_IONIC_PREFIX = "ionic_correction."
_FROZEN_PREFIXES = ("encoder_frozen.", "readout_frozen.")
_SPHERE_DENSITY_PREFIX = "sphere_density."
_QUAD_SCALE_PREFIX = "quad_scale."
_SP_HEAD_PREFIX = "sp_head."
_COORD_PREFIX = "coord_proj."
_EXPERT_PREFIXES = ("readout_extra.", "expert_router.")
COORD_RADII = (3.0, 4.0, 5.0, 6.0, 8.0, 10.0)
_OPTIONAL_PREFIXES = (
    _ENV_PREFIX,
    _SP_HEAD_PREFIX,
    _COORD_PREFIX,
    _CHARGE_PREFIX,
    _SP_PREFIX,
    _IONIC_PREFIX,
    _SPHERE_DENSITY_PREFIX,
    _QUAD_SCALE_PREFIX,
) + _FROZEN_PREFIXES + _EXPERT_PREFIXES


class QuadrupoleEnvScale(nn.Module):
    """Per-atom multiplier on the l=2 residual term as a smooth function of local density.

    ``s_a = exp(θ₁ t_a + θ₂ t_a²)`` with ``t_a = (CN10_a − center) / scale``. Zero-initialised
    (``s_a ≡ 1``) and exactly 1 at the diamond density whatever θ is, so the diamond
    cells never see it. Motivation (Phase 3 channel ablation, 15 Sep 2026): one
    species-level quadrupole profile is right for bct (CN10 ≈ 147) and too strong
    for ST12 (≈ 159); a density-dependent scale can hold the former while reducing
    the latter without a family name.
    """

    def __init__(self, order: int = 2, center: float = 151.0, scale: float = 30.0) -> None:
        super().__init__()
        if order < 1 or scale <= 0:
            raise ValueError("order must be positive and scale > 0")
        self.order = int(order)
        self.center = float(center)
        self.scale = float(scale)
        self.coefficients = nn.Parameter(torch.zeros(self.order))

    def forward(self, cn10: Tensor) -> Tensor:
        if cn10.ndim != 1:
            raise ValueError("cn10 must have shape (N,)")
        t = (cn10 - self.center) / self.scale
        powers = torch.stack([t.pow(k) for k in range(1, self.order + 1)], dim=-1)
        # Overflow guard only: the dense metals sit at t ≈ 2-2.5 (CN10 213-225), where a
        # quadratic term could blow up. ±4 is far outside any useful scale (0.02-55×).
        exponent = (powers @ self.coefficients.to(powers)).clamp(-4.0, 4.0)
        return torch.exp(exponent)


def normalize_gate_windows(windows: Any) -> list[list[dict[str, Any]]] | None:
    """Accept None, a list of spec strings, or a list of term lists."""
    if windows is None:
        return None
    out: list[list[dict[str, Any]]] = []
    for window in windows:
        if isinstance(window, str):
            out.append(parse_gate_window(window))
        else:
            terms = []
            for term in window:
                terms.append(
                    {
                        "radius": float(term["radius"]),
                        "center": float(term["center"]),
                        "alpha": float(term["alpha"]),
                        "below": bool(term.get("below", True)),
                    }
                )
            if not terms:
                raise ValueError("gate window without terms")
            out.append(terms)
    return out or None


def pack_structure(
    structure: Structure,
    dtype: torch.dtype = torch.float32,
    species_table: SpeciesTable | None = None,
) -> SimpleNamespace:
    lattice = torch.as_tensor(np.asarray(structure.lattice), dtype=dtype)
    frac = torch.as_tensor(np.asarray(structure.fractional_positions), dtype=dtype)
    cart = frac @ lattice
    if species_table is None:
        species = torch.zeros(structure.n_atoms, dtype=torch.long)
    else:
        species = torch.as_tensor(species_table.encode(structure.species), dtype=torch.long)
    if "band_offset" in structure.metadata:
        # general-v1 cells drop semicore bands; the Si/MP loaders would pair the
        # model with the PAW band count. Use epnet.data.general_manifest instead.
        raise ValueError("general-v1 manifest entry: load with epnet.data.general_manifest and GeneralEPM")
    n_occ = structure.metadata.get("n_occupied")
    n_occupied = int(n_occ) if n_occ is not None else int(silicon_n_occupied(structure))
    return SimpleNamespace(
        lattice=lattice,
        fractional_positions=frac,
        positions=cart,
        species=species,
        volume=torch.as_tensor(structure.volume, dtype=dtype),
        n_atoms=structure.n_atoms,
        n_occupied=n_occupied,
    )


class GatedSiEPM(nn.Module):
    """Cohen sphere plus an optional environment-dependent isotropic Δv(q) and l<=2 residual."""

    def __init__(
        self,
        channels: int = 8,
        cutoff: float = 5.0,
        ecut_ry: float = 12.0,
        monopole_mode: str = "gate",
        num_radial: int = 8,
        residual_l_max: int = 2,
        residual_readout_bias: bool = True,
        env_isotropic: bool = False,
        env_density: bool = False,
        density_cutoffs: tuple[float, ...] = (3.0, 10.0),
        env_pack_gate: bool = False,
        residual_pack_gate: bool = False,
        pack_gate_center: float = 142.0,
        pack_gate_alpha: float = 2.0,
        pack_gate_windows: list[list[dict[str, Any]]] | list[str] | None = None,
        sphere_density_order: int = 0,
        sphere_density_center: float = 151.0,
        sphere_density_scale: float = 30.0,
        residual_anchor: bool = False,
        long_range: bool = False,
        environment_ionic: bool = False,
        ionic_cutoffs: tuple[float, ...] = (3.0, 5.0, 8.0),
        ionic_gaussian_width: float = 0.5,
        sp_nonlocal: bool = False,
        sp_environment: bool = False,
        sp_gated_head: bool = False,
        coordination_features: bool = False,
        sp_projector_scales: tuple[float, ...] = (1.5, 3.5),
        quad_scale_env: bool = False,
        quad_scale_center: float = 151.0,
        quad_scale_scale: float = 30.0,
        sphere_n_basis: int = 12,
        sphere_q_cutoff: float = 8.0,
        n_species: int = 1,
        species_symbols: tuple[str, ...] | list[str] | None = None,
        subset_ng_min: int | None = None,
        k_batch: int | None = None,
        residual_scalar_cosine_basis: int = 0,
        residual_scalar_cosine_cutoff: float = 8.0,
        residual_experts: int = 1,
    ) -> None:
        super().__init__()
        if monopole_mode not in {"gate", "off", "free"}:
            raise ValueError("monopole_mode must be gate, off, or free")
        self.channels = int(channels)
        self.ecut_ry = float(ecut_ry)
        self.monopole_mode = monopole_mode
        self.residual_l_max = int(residual_l_max)
        self.residual_num_radial = int(num_radial)
        self.residual_readout_bias = bool(residual_readout_bias)
        self.residual_scalar_cosine_basis = int(residual_scalar_cosine_basis)
        self.residual_scalar_cosine_cutoff = float(residual_scalar_cosine_cutoff)
        if self.residual_l_max not in (0, 1, 2):
            raise ValueError("residual_l_max must be 0, 1, or 2")
        if self.residual_num_radial < 1:
            raise ValueError("num_radial must be positive")
        self.env_isotropic = bool(env_isotropic)
        self.env_density = bool(env_density and self.env_isotropic)
        self.density_cutoffs = tuple(float(r) for r in density_cutoffs)
        self.env_pack_gate = bool(env_pack_gate)
        self.residual_pack_gate = bool(residual_pack_gate)
        self.pack_gate_center = float(pack_gate_center)
        self.pack_gate_alpha = float(pack_gate_alpha)
        self.pack_gate_windows = normalize_gate_windows(pack_gate_windows)
        self.sphere_density_order = int(sphere_density_order)
        self.sphere_density_center = float(sphere_density_center)
        self.sphere_density_scale = float(sphere_density_scale)
        self.residual_anchor = bool(residual_anchor)
        self.long_range = bool(long_range)
        self.environment_ionic = bool(environment_ionic)
        self.sp_nonlocal = bool(sp_nonlocal)
        self.sp_environment = bool(sp_environment)
        if self.sp_environment and not self.sp_nonlocal:
            raise ValueError("sp_environment requires sp_nonlocal")
        self.sp_gated_head = bool(sp_gated_head)
        self.coordination_features = bool(coordination_features)
        if self.sp_gated_head and not self.sp_nonlocal:
            raise ValueError("sp_gated_head requires sp_nonlocal")
        self.quad_scale_env = bool(quad_scale_env)
        self.quad_scale_center = float(quad_scale_center)
        self.quad_scale_scale = float(quad_scale_scale)
        if self.quad_scale_env and self.residual_l_max < 2:
            raise ValueError("quad_scale_env requires residual_l_max >= 2")
        self.sphere_n_basis = int(sphere_n_basis)
        self.sphere_q_cutoff = float(sphere_q_cutoff)
        if self.sphere_n_basis < 1 or self.sphere_q_cutoff <= 0:
            raise ValueError("sphere_n_basis and sphere_q_cutoff must be positive")
        self.n_species = int(n_species)
        if self.n_species < 1:
            raise ValueError("n_species must be positive")
        if species_symbols is None:
            if self.n_species != 1:
                raise ValueError("species_symbols is required when n_species > 1")
            species_symbols = ("Si",)
        self.species_symbols = tuple(str(s) for s in species_symbols)
        if len(self.species_symbols) != self.n_species:
            raise ValueError("species_symbols length must match n_species")
        self.subset_ng_min = 4096 if subset_ng_min is None else int(subset_ng_min)
        self.k_batch = k_batch
        if self.env_density and (not self.density_cutoffs or any(r <= 0 for r in self.density_cutoffs)):
            raise ValueError("density_cutoffs must be positive")
        if self.env_pack_gate or self.residual_pack_gate:
            if self.pack_gate_windows is None and not self.env_density:
                raise ValueError(
                    "packing gates require env_density so CN(10 Å) exists, or explicit pack_gate_windows"
                )
            if self.pack_gate_alpha <= 0:
                raise ValueError("pack_gate_alpha must be positive")
        if self.residual_anchor and not self.residual_pack_gate:
            raise ValueError("residual_anchor requires residual_pack_gate")
        if self.sphere_density_order < 0 or self.sphere_density_scale <= 0:
            raise ValueError("sphere_density_order must be >= 0 and sphere_density_scale positive")
        self._last_pack_gate: Tensor | None = None
        self._anchor_reference: dict[str, Tensor] = {}
        self.sphere = CosineRadialEP(
            self.n_species, n_basis=self.sphere_n_basis, q_cutoff=self.sphere_q_cutoff
        )
        q_fit = torch.linspace(0.0, self.sphere_q_cutoff, 81)
        for i, symbol in enumerate(self.species_symbols):
            target = silicon_cohen_form_factor(q_fit) if symbol == "Si" else empty_core_form_factor(q_fit)
            self.sphere.fit_collocation(i, q_fit, target)
        self.sphere.coefficients.requires_grad_(False)
        self._frozen_sphere_idx: tuple[int, ...] = ()
        self._frozen_sphere_rows: Tensor | None = None
        self.sphere_density: DensityConditionedSphereDelta | None
        self._last_sphere_density_delta: Tensor | None = None
        if self.sphere_density_order > 0:
            self.sphere_density = DensityConditionedSphereDelta(
                self.sphere.n_basis,
                order=self.sphere_density_order,
                center=self.sphere_density_center,
                scale=self.sphere_density_scale,
            )
        else:
            self.sphere_density = None
        self.encoder = EquivariantEncoder(num_species=self.n_species, channels=channels, cutoff=cutoff, l_max=2)
        self.readout = MultipoleFormFactor(
            channels=channels,
            num_radial=num_radial,
            l_max=self.residual_l_max,
            include_monopole=monopole_mode != "off",
            trainable_scalar_bias=self.residual_readout_bias,
            scalar_cosine_basis=self.residual_scalar_cosine_basis,
            scalar_cosine_cutoff=self.residual_scalar_cosine_cutoff,
        )
        self.residual_experts = int(residual_experts)
        if self.residual_experts < 1:
            raise ValueError("residual_experts must be >= 1")
        if self.residual_experts > 1 and self.residual_pack_gate:
            raise ValueError("residual_experts replaces the packing gate; do not combine them")
        self.readout_extra: nn.ModuleList | None
        self.expert_router: nn.Sequential | None
        self._last_expert_weights: Tensor | None = None
        if self.residual_experts > 1:
            # Soft mixture of residual readouts on one shared encoder. Per-atom weights are a softmax of a
            # small MLP of standardized log local densities (COORD_RADII): continuous, no thresholds.
            self.readout_extra = nn.ModuleList(
                [
                    MultipoleFormFactor(
                        channels=channels,
                        num_radial=num_radial,
                        l_max=self.residual_l_max,
                        include_monopole=monopole_mode != "off",
                        trainable_scalar_bias=self.residual_readout_bias,
                        scalar_cosine_basis=self.residual_scalar_cosine_basis,
                        scalar_cosine_cutoff=self.residual_scalar_cosine_cutoff,
                    )
                    for _ in range(self.residual_experts - 1)
                ]
            )
            self.expert_router = nn.Sequential(
                nn.Linear(len(COORD_RADII), 16), nn.SiLU(), nn.Linear(16, self.residual_experts)
            )
            self.register_buffer("router_mean", torch.zeros(len(COORD_RADII)))
            self.register_buffer("router_std", torch.ones(len(COORD_RADII)))
        else:
            self.readout_extra = None
            self.expert_router = None
        self.encoder_frozen: EquivariantEncoder | None
        self.readout_frozen: MultipoleFormFactor | None
        if self.residual_pack_gate:
            self.encoder_frozen = EquivariantEncoder(
                num_species=self.n_species, channels=channels, cutoff=cutoff, l_max=2
            )
            self.readout_frozen = MultipoleFormFactor(
                channels=channels,
                num_radial=num_radial,
                l_max=self.residual_l_max,
                include_monopole=monopole_mode != "off",
                trainable_scalar_bias=self.residual_readout_bias,
                scalar_cosine_basis=self.residual_scalar_cosine_basis,
                scalar_cosine_cutoff=self.residual_scalar_cosine_cutoff,
            )
            self.snapshot_frozen_residual()
        else:
            self.encoder_frozen = None
            self.readout_frozen = None
        self.env_radial: EnvironmentRadialDelta | None
        self._last_env_delta: Tensor | None = None
        if self.env_isotropic:
            n_in = channels * (1 + int(self.encoder.l_max >= 1) + int(self.encoder.l_max >= 2))
            if self.env_density:
                n_in += len(self.density_cutoffs)
            self.env_radial = EnvironmentRadialDelta(n_in, self.sphere.n_basis)
        else:
            self.env_radial = None
        self.effective_charge: nn.Parameter | None
        if self.long_range:
            # Zero-init: H is the fitted v_Z until Z_i move. Neutralization
            # makes elemental cells (one species) stay exactly H-neutral.
            self.effective_charge = nn.Parameter(torch.zeros(self.n_species))
        else:
            self.effective_charge = None
        self.ionic_correction: EnvironmentIonicCorrection | None
        if self.environment_ionic:
            self.ionic_correction = EnvironmentIonicCorrection(
                self.n_species,
                coordination_cutoffs=ionic_cutoffs,
                gaussian_width=ionic_gaussian_width,
            )
        else:
            self.ionic_correction = None
        self.sp_projector: SPProjectorCorrection | None
        if self.sp_nonlocal:
            self.sp_projector = SPProjectorCorrection(self.n_species, sp_projector_scales)
            if self.sp_environment and self.sp_projector.n_radial != 2:
                raise ValueError("sp_environment requires exactly two projector radial scales")
        else:
            self.sp_projector = None
        self.sp_head: nn.Sequential | None
        if self.sp_gated_head:
            # Porous-branch invariants (h, |d|, |Q|) + CN3/4, CN10/150 -> additive per-atom s/p strengths.
            # Zero-initialised output: attaching the head leaves the parent Hamiltonian unchanged.
            n_in = channels * (1 + int(self.encoder.l_max >= 1) + int(self.encoder.l_max >= 2)) + 2
            self.sp_head = nn.Sequential(nn.Linear(n_in, 16), nn.SiLU(), nn.Linear(16, self.sp_projector.n_radial * 2))
            nn.init.zeros_(self.sp_head[-1].weight)
            nn.init.zeros_(self.sp_head[-1].bias)
        else:
            self.sp_head = None
        self._last_sp_head_delta: Tensor | None = None
        self.coord_proj: nn.Linear | None
        if self.coordination_features:
            # Smooth local number densities CN(r)/(4/3 pi r^3) at several radii, element-agnostic, added to the
            # encoder's scalar channels. Zero-initialised: attaching it leaves the parent model unchanged.
            self.coord_proj = nn.Linear(len(COORD_RADII), channels, bias=False)
            nn.init.zeros_(self.coord_proj.weight)
        else:
            self.coord_proj = None
        self.quad_scale: QuadrupoleEnvScale | None
        self._last_quad_scale: Tensor | None = None
        if self.quad_scale_env:
            self.quad_scale = QuadrupoleEnvScale(
                order=2, center=self.quad_scale_center, scale=self.quad_scale_scale
            )
        else:
            self.quad_scale = None

    def load_sphere(self, path, *, freeze: bool = True) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        state = payload["sphere"] if isinstance(payload, dict) and "sphere" in payload else payload
        self.sphere.load_state_dict(state)
        self.sphere.coefficients.requires_grad_(not freeze)

    def set_sphere_trainable(self, trainable: bool) -> None:
        self.sphere.coefficients.requires_grad_(trainable)
        if not trainable:
            self._frozen_sphere_idx = ()
            self._frozen_sphere_rows = None

    def freeze_sphere_symbols(self, symbols: tuple[str, ...] | list[str]) -> None:
        """Keep listed species rows fixed through optimizer steps (AdamW / weight decay)."""
        wanted = tuple(str(s) for s in symbols)
        missing = [s for s in wanted if s not in self.species_symbols]
        if missing:
            raise ValueError(f"freeze_sphere_symbols unknown species: {missing}")
        idx = tuple(self.species_symbols.index(s) for s in wanted)
        self.sphere.coefficients.requires_grad_(True)
        self._frozen_sphere_idx = idx
        if idx:
            rows = self.sphere.coefficients.detach()[list(idx)].clone()
            self._frozen_sphere_rows = rows
        else:
            self._frozen_sphere_rows = None

    def restore_frozen_sphere_rows(self) -> None:
        if not self._frozen_sphere_idx or self._frozen_sphere_rows is None:
            return
        with torch.no_grad():
            self.sphere.coefficients[list(self._frozen_sphere_idx)] = self._frozen_sphere_rows.to(
                device=self.sphere.coefficients.device,
                dtype=self.sphere.coefficients.dtype,
            )

    def snapshot_frozen_residual(self) -> None:
        """Copy live encoder/readout into the compact-site residual snapshot."""
        if self.encoder_frozen is None or self.readout_frozen is None:
            raise RuntimeError("snapshot_frozen_residual requires residual_pack_gate")
        self.encoder_frozen.load_state_dict(self.encoder.state_dict())
        self.readout_frozen.load_state_dict(self.readout.state_dict())
        self.encoder_frozen.eval()
        self.readout_frozen.eval()
        for param in (*self.encoder_frozen.parameters(), *self.readout_frozen.parameters()):
            param.requires_grad_(False)
        if self.residual_anchor:
            self.set_anchor_reference()

    def compact_residual_parameters(self) -> list[tuple[str, nn.Parameter]]:
        """Named parameters of the compact-site residual copy (frozen or anchored)."""
        out: list[tuple[str, nn.Parameter]] = []
        for name, param in self.named_parameters():
            if name.startswith(_FROZEN_PREFIXES):
                out.append((name, param))
        return out

    def set_anchor_reference(self) -> None:
        """Make the compact-site residual live, anchored (by L2) to its current values.

        Phase 3 step 4: instead of a hard freeze on compact 4-fold sites, the
        compact copy trains with a parameter-space pull toward the Phase-1
        snapshot. The reference is taken at load; ``anchor_penalty`` measures
        the drift. Bias entries the readout keeps frozen (l>0) stay frozen.
        """
        if self.encoder_frozen is None or self.readout_frozen is None:
            raise RuntimeError("set_anchor_reference requires residual_pack_gate")
        self.residual_anchor = True
        self._anchor_reference = {}
        for name, param in self.compact_residual_parameters():
            live_name = name.replace("encoder_frozen.", "encoder.", 1).replace("readout_frozen.", "readout.", 1)
            live = dict(self.named_parameters()).get(live_name)
            trainable = True if live is None else bool(live.requires_grad)
            if name.startswith("readout_frozen.radial_bias.") and not name.endswith(".0"):
                trainable = False
            param.requires_grad_(trainable)
            self._anchor_reference[name] = param.detach().clone()

    def anchor_penalty(self) -> Tensor:
        """Sum of squared drift of the compact residual copy from its reference."""
        params = dict(self.named_parameters())
        total = None
        for name, ref in self._anchor_reference.items():
            param = params.get(name)
            if param is None or not param.requires_grad:
                continue
            diff = (param - ref.to(param)).pow(2).sum()
            total = diff if total is None else total + diff
        if total is None:
            return torch.zeros((), dtype=self.sphere.coefficients.dtype, device=self.sphere.coefficients.device)
        return total

    def zero_residual_readout(self) -> None:
        """Make the live l<=2 form factor identically zero (H-neutral attach)."""
        with torch.no_grad():
            for layer in self.readout.channel_readout:
                layer.weight.zero_()
                if layer.bias is not None:
                    layer.bias.zero_()
            for bias in self.readout.radial_bias:
                bias.zero_()
            if self.readout.scalar_cosine is not None:
                self.readout.scalar_cosine.weight.zero_()
                if self.readout.scalar_cosine.bias is not None:
                    self.readout.scalar_cosine.bias.zero_()

    def _gate_features(self, features: dict[str, Tensor]) -> dict[str, Tensor]:
        out = dict(features)
        if self.monopole_mode != "gate" or "scalar" not in out:
            return out
        d = out.get("vector")
        q = out.get("tensor")
        aniso = out["scalar"].new_zeros(out["scalar"].shape[0])
        if d is not None:
            aniso = aniso + (d * d).sum(dim=(-1, -2))
        if q is not None:
            aniso = aniso + (q * q).sum(dim=(-1, -2, -3))
        gate = 1.0 - torch.exp(-aniso)
        out["scalar"] = out["scalar"] * gate[:, None]
        return out

    def encode(
        self,
        packed: SimpleNamespace,
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
        frozen: bool = False,
    ) -> dict[str, Tensor]:
        encoder = self.encoder_frozen if frozen else self.encoder
        if encoder is None:
            raise RuntimeError("frozen encoder requires residual_pack_gate")
        features = encoder(
            packed.positions,
            packed.species,
            packed.lattice,
            edge_index=edge_index,
            cell_shifts=cell_shifts,
        )
        if self.coord_proj is not None and not frozen:
            features = dict(features)
            features["scalar"] = features["scalar"] + self.coord_proj(self.coordination_descriptor(packed).to(features["scalar"]))
        return self._gate_features(features)

    def coordination_descriptor(self, packed: SimpleNamespace) -> Tensor:
        """(n_atoms, len(COORD_RADII)) local number densities in units of 0.05 per cubic Angstrom (~1 for Si)."""
        cn = self._coordination(packed, COORD_RADII)
        return torch.stack([cn[r] / (4.0 / 3.0 * torch.pi * r**3) / 0.05 for r in COORD_RADII], dim=-1)

    def router_inputs(self, packed: SimpleNamespace) -> Tensor:
        """Unstandardized router inputs: log local number densities, (n_atoms, len(COORD_RADII))."""
        return torch.log(self.coordination_descriptor(packed).clamp_min(1e-3))

    def set_router_normalization(self, mean: Tensor, std: Tensor) -> None:
        if self.expert_router is None:
            raise RuntimeError("set_router_normalization requires residual_experts > 1")
        with torch.no_grad():
            self.router_mean.copy_(mean.to(self.router_mean))
            self.router_std.copy_(std.clamp_min(1e-3).to(self.router_std))

    def expert_weights(self, packed: SimpleNamespace) -> Tensor:
        """Per-atom mixture weights over the residual readouts, (n_atoms, residual_experts)."""
        if self.expert_router is None:
            raise RuntimeError("expert_weights requires residual_experts > 1")
        x = self.router_inputs(packed).to(self.router_mean)
        weights = torch.softmax(self.expert_router((x - self.router_mean) / self.router_std), dim=-1)
        self._last_expert_weights = weights
        return weights

    def _environment_invariants(
        self,
        features: dict[str, Tensor],
        packed: SimpleNamespace | None = None,
    ) -> Tensor:
        parts = [features["scalar"]]
        vector = features.get("vector")
        tensor = features.get("tensor")
        if vector is not None:
            parts.append(torch.sqrt((vector * vector).sum(-1) + 1e-12))
        if tensor is not None:
            parts.append(torch.sqrt((tensor * tensor).sum((-1, -2)) + 1e-12))
        if self.env_density:
            if packed is None:
                raise ValueError("packed structure is required when env_density=True")
            parts.append(smooth_coordination(packed.positions, packed.lattice, self.density_cutoffs))
        return torch.cat(parts, dim=-1)

    def _cn10(self, packed: SimpleNamespace) -> Tensor:
        cn = smooth_coordination(packed.positions, packed.lattice, self.density_cutoffs)
        idx = min(range(len(self.density_cutoffs)), key=lambda i: abs(self.density_cutoffs[i] - 10.0))
        return cn[:, idx]

    @property
    def gate_radii(self) -> tuple[float, ...]:
        if self.pack_gate_windows is None:
            idx = min(range(len(self.density_cutoffs)), key=lambda i: abs(self.density_cutoffs[i] - 10.0))
            return (float(self.density_cutoffs[idx]),)
        return gate_window_radii(self.pack_gate_windows)

    def _coordination(self, packed: SimpleNamespace, radii: tuple[float, ...]) -> dict[float, Tensor]:
        """Plateau-cosine coordination at ``radii``, memoised on the packed cell (geometry is fixed)."""
        key = tuple(float(r) for r in radii)
        store = getattr(packed, "_coordination_cache", None)
        if store is None:
            store = {}
            try:
                packed._coordination_cache = store
            except AttributeError:
                pass
        cached = store.get(key)
        if cached is None or cached[0].device != packed.positions.device:
            cn = smooth_coordination(packed.positions, packed.lattice, key)
            cached = tuple(cn[:, i] for i in range(len(key)))
            store[key] = cached
        return dict(zip(key, cached))

    def _pack_gate_values(self, packed: SimpleNamespace) -> Tensor:
        if self.pack_gate_windows is None:
            gate = packing_gate(self._cn10(packed), self.pack_gate_center, self.pack_gate_alpha)
        else:
            gate = packing_gate_windows(self._coordination(packed, self.gate_radii), self.pack_gate_windows)
        self._last_pack_gate = gate
        return gate

    def quadrupole_scale_values(self, packed: SimpleNamespace) -> Tensor | None:
        """Per-atom l=2 multiplier from CN(10 Å), or ``None`` when the channel is off."""
        if self.quad_scale is None:
            return None
        cn10 = self._coordination(packed, (10.0,))[10.0]
        scale = self.quad_scale(cn10)
        self._last_quad_scale = scale
        return scale

    def sphere_density_form_factor(self, packed: SimpleNamespace, unique_cart: Tensor) -> Tensor:
        """Per-atom isotropic Δv(q) from the density-conditioned sphere coefficients."""
        if self.sphere_density is None:
            raise RuntimeError("sphere_density_form_factor requires sphere_density_order > 0")
        cn10 = self._coordination(packed, (10.0,))[10.0]
        delta = self.sphere_density(cn10)
        self._last_sphere_density_delta = delta
        q = torch.linalg.vector_norm(unique_cart, dim=-1)
        basis = self.sphere.basis(q)
        values = torch.einsum("nb,gb->ng", delta.to(basis), basis)
        phase = torch.exp(-1j * torch.einsum("ga,na->ng", unique_cart.to(packed.positions), packed.positions))
        total = (values.to(phase) * phase).sum(dim=0)
        return total.masked_fill(q < 1e-8, 0)

    def environment_form_factor(
        self,
        packed: SimpleNamespace,
        unique_cart: Tensor,
        features: dict[str, Tensor],
    ) -> Tensor:
        if self.env_radial is None:
            raise RuntimeError("environment_form_factor requires env_isotropic=True")
        delta = self.env_radial(self._environment_invariants(features, packed))
        if self.env_pack_gate:
            delta = delta * self._pack_gate_values(packed).unsqueeze(-1)
        elif not self.residual_pack_gate:
            self._last_pack_gate = None
        self._last_env_delta = delta
        q = torch.linalg.vector_norm(unique_cart, dim=-1)
        basis = self.sphere.basis(q)
        values = torch.einsum("nb,gb->ng", delta, basis)
        phase = torch.exp(-1j * torch.einsum("ga,na->ng", unique_cart.to(packed.positions), packed.positions))
        total = (values.to(phase) * phase).sum(dim=0)
        return total.masked_fill(q < 1e-8, 0)

    def long_range_form_factor(self, packed: SimpleNamespace, g_cart: Tensor) -> Tensor:
        r"""Neutralized analytic \(Z_i/q^2\) on unique ΔG, before the 1/Ω scatter."""
        if not self.long_range or self.effective_charge is None:
            raise RuntimeError("long_range_form_factor requires long_range=True")
        raw = self.effective_charge[packed.species.to(dtype=torch.long)]
        charges = neutralize_effective_charges(raw)
        # volume=1: add_multipole_potential applies 1/Ω. Scale Å/Hartree → eV.
        values = reciprocal_coulomb_potential(
            packed.positions, charges, g_cart, volume=1.0
        )
        return values * values.new_tensor(COULOMB_EV_A)

    def ionic_form_factor(
        self,
        packed: SimpleNamespace,
        g_cart: Tensor,
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        if self.ionic_correction is None:
            raise RuntimeError("ionic_form_factor requires environment_ionic=True")
        return self.ionic_correction(
            packed,
            g_cart,
            edge_index=edge_index,
            cell_shifts=cell_shifts,
        )

    def projector_environment_factors(
        self,
        packed: SimpleNamespace,
        features: dict[str, Tensor],
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
    ) -> Tensor:
        """Four fixed invariant descriptors for the two radial s/p channels."""
        scalar = torch.tanh(features["scalar"].mean(dim=-1))
        vector = torch.tanh(torch.sqrt(features["vector"].square().sum(dim=-1) + 1e-12).mean(dim=-1))
        tensor = torch.tanh(
            torch.sqrt(features["tensor"].square().sum(dim=(-1, -2)) + 1e-12).mean(dim=-1)
        )
        coordination = smooth_coordination(
            packed.positions,
            packed.lattice,
            (float(self.encoder.cutoff),),
            edge_index=edge_index,
            cell_shifts=cell_shifts,
        )[:, 0]
        coordination = torch.tanh(torch.log1p(coordination) / 2.0)
        return torch.stack((scalar, vector, tensor, coordination), dim=-1).reshape(-1, 2, 2)

    def projector_gated_delta(self, packed: SimpleNamespace, features: dict[str, Tensor]) -> Tensor:
        """Per-atom additive s/p strengths, (n_atoms, n_radial, 2), switched on by the packing gate g_a."""
        if self.sp_head is None or self.sp_projector is None:
            raise RuntimeError("projector_gated_delta requires sp_gated_head=True")
        cn = self._coordination(packed, (3.0, 10.0))
        inputs = torch.cat(
            (self._environment_invariants(features), (cn[3.0] / 4.0)[:, None], (cn[10.0] / 150.0)[:, None]), dim=-1
        )
        delta = self.sp_head(inputs).reshape(-1, self.sp_projector.n_radial, 2)
        if self.residual_pack_gate or self.env_pack_gate:  # gated (Si Phase 4) variant; ungated = general form
            delta = delta * self._pack_gate_values(packed).to(delta)[:, None, None]
        self._last_sp_head_delta = delta
        return delta

    def residual_form_factor(
        self,
        packed: SimpleNamespace,
        g_cart: Tensor,
        *,
        edge_index: Tensor | None = None,
        cell_shifts: Tensor | None = None,
        features: dict[str, Tensor] | None = None,
    ) -> Tensor:
        if features is None:
            features = self.encode(packed, edge_index=edge_index, cell_shifts=cell_shifts)
        qscale = self.quadrupole_scale_values(packed)
        if not self.residual_pack_gate:
            if self.readout_extra is None:
                return self.readout(packed.positions, g_cart, features, quadrupole_scale=qscale)
            weights = self.expert_weights(packed)
            total = None
            for k, readout in enumerate((self.readout, *self.readout_extra)):
                _, per_atom = readout(packed.positions, g_cart, features, return_per_atom=True, quadrupole_scale=qscale)
                part = (per_atom * weights[:, k].to(per_atom.real.dtype).unsqueeze(-1)).sum(dim=0)
                total = part if total is None else total + part
            return total
        if self.readout_frozen is None:
            raise RuntimeError("residual_pack_gate requires a frozen residual snapshot")
        _, live_per_atom = self.readout(
            packed.positions, g_cart, features, return_per_atom=True, quadrupole_scale=qscale
        )
        frozen_features = self.encode(
            packed, edge_index=edge_index, cell_shifts=cell_shifts, frozen=True
        )
        _, frozen_per_atom = self.readout_frozen(
            packed.positions, g_cart, frozen_features, return_per_atom=True, quadrupole_scale=qscale
        )
        gate = self._pack_gate_values(packed).to(device=live_per_atom.device, dtype=live_per_atom.real.dtype)
        blended = frozen_per_atom + (live_per_atom - frozen_per_atom) * gate.unsqueeze(-1)
        total = blended.sum(dim=0)
        q = torch.linalg.vector_norm(g_cart, dim=-1)
        return total.masked_fill(q < 1e-8, 0)

    def make_pw_cache(
        self,
        packed: SimpleNamespace,
        k_frac: Tensor,
        *,
        store_spherical: bool = True,
        store_graph: bool | None = None,
    ) -> SimpleNamespace:
        """Precompute PW geometry. Species spherical V is cached; env Δv(q) is live.

        Multi-species caches omit the unused n_G×n_G pair maps (q_pair, phases,
        inv_q). Those are only needed for the single-species Cohen path.
        The neighbor graph is cached for env Δv, or when ``store_graph`` is set
        so a live residual does not rebuild it on every cell.
        """
        k_frac = fold_to_first_zone(k_frac, packed.lattice)  # E(k) = E(k+G); the G sphere is centred at 0
        cutoff = g_cutoff_angstrom(self.ecut_ry)
        g_idx_np, _ = reciprocal_vectors(packed.lattice.detach().cpu().numpy(), cutoff)
        g_idx = torch.as_tensor(g_idx_np, dtype=torch.long, device=k_frac.device)
        recip = 2.0 * torch.pi * torch.linalg.inv(packed.lattice.detach()).transpose(0, 1)
        unique_cart, inverse = unique_g_transfers(g_idx, recip)
        if int(self.sphere.n_species) == 1:
            q_of_unique = torch.linalg.vector_norm(unique_cart, dim=-1)
            unique_q, inv_small = torch.unique(q_of_unique, return_inverse=True)
            inv_q = inv_small[inverse.reshape(-1)].detach()
            unique_q = unique_q.detach()
            phases = structure_phases(g_idx, packed.fractional_positions.detach()).detach()
        else:
            unique_q = None
            inv_q = None
            phases = None
        if inverse.numel() and int(inverse.max()) >= int(unique_cart.shape[0]):
            raise RuntimeError(
                f"ΔG inverse max {int(inverse.max())} >= n_unique {int(unique_cart.shape[0])}"
            )
        inverse = inverse.to(dtype=torch.int32)
        kinetic = kinetic_diagonal(g_idx, recip, k_frac)
        need_graph = bool(
            self.env_isotropic or self.environment_ionic or self.sp_environment or self.sp_gated_head
        ) if store_graph is None else bool(store_graph)
        if need_graph:
            graph_cutoff = float(self.encoder.cutoff)
            if self.ionic_correction is not None:
                graph_cutoff = max(
                    graph_cutoff, max(self.ionic_correction.coordination_cutoffs)
                )
            edge_index, cell_shifts = build_periodic_neighbor_graph(
                packed.positions.detach(), packed.lattice.detach(), graph_cutoff
            )
            edge_index = edge_index.detach()
            cell_shifts = cell_shifts.detach()
        else:
            edge_index = None
            cell_shifts = None
        cache = SimpleNamespace(
            g_idx=g_idx.detach(),
            unique_q=unique_q,
            inv_q=inv_q,
            q_pair=None,
            phases=phases,
            unique_cart=unique_cart.detach(),
            inverse=inverse.detach(),
            kinetic=kinetic.detach(),
            volume=float(packed.volume.detach()),
            n_g=int(g_idx.shape[0]),
            edge_index=edge_index,
            cell_shifts=cell_shifts,
            species=packed.species.detach(),
            positions=packed.positions.detach(),
            spherical=None,
            sp_projectors=None,
        )
        if store_spherical:
            cache.spherical = self.spherical_potential(cache).detach()
        if self.sp_projector is not None:
            cache.sp_projectors = self.sp_projector.projectors(packed, k_frac, g_idx).detach()
        return cache

    def spherical_potential(self, cache: SimpleNamespace) -> Tensor:
        if int(self.sphere.n_species) == 1:
            zeros = torch.zeros(cache.unique_q.numel(), dtype=torch.long, device=cache.unique_q.device)
            v_unique = self.sphere(zeros, cache.unique_q)
            v_pair = v_unique[cache.inv_q].reshape(cache.n_g, cache.n_g)
            return local_potential_matrix(cache.phases, v_pair, cache.volume)
        q_cart = torch.linalg.vector_norm(cache.unique_cart, dim=-1)
        v_atom = self.sphere.per_atom(cache.species, q_cart)
        phase = torch.exp(-1j * torch.einsum("gc,nc->gn", cache.unique_cart.to(cache.positions), cache.positions))
        total = (v_atom.to(phase.dtype).transpose(0, 1) * phase).sum(dim=1)
        total = total.masked_fill(q_cart < 1e-8, 0)
        return scatter_hermitian_potential(total, cache.inverse, cache.volume)

    def band_path(
        self,
        packed: SimpleNamespace,
        k_frac: Tensor,
        n_bands: int,
        cache: SimpleNamespace | None = None,
        *,
        residual: bool = True,
        refresh_sphere: bool = False,
    ) -> Tensor:
        if cache is None:
            cache = self.make_pw_cache(packed, k_frac)
        spherical = self.spherical_potential(cache) if refresh_sphere else cache.spherical
        need_features = residual or self.env_isotropic or self.sp_environment or self.sp_gated_head
        features = None
        if need_features:
            features = self.encode(
                packed, edge_index=cache.edge_index, cell_shifts=cache.cell_shifts
            )
        if self.env_isotropic:
            env_ff = self.environment_form_factor(packed, cache.unique_cart, features)
            spherical = add_multipole_potential(
                spherical, env_ff, cache.inverse, cache.volume, cache.unique_cart
            )
        if self.sphere_density is not None:
            dens_ff = self.sphere_density_form_factor(packed, cache.unique_cart)
            spherical = add_multipole_potential(
                spherical, dens_ff, cache.inverse, cache.volume, cache.unique_cart
            )
        if residual:
            extra = self.residual_form_factor(
                packed,
                cache.unique_cart,
                edge_index=cache.edge_index,
                cell_shifts=cache.cell_shifts,
                features=features,
            )
            spherical = add_multipole_potential(
                spherical, extra, cache.inverse, cache.volume, cache.unique_cart
            )
        if self.long_range:
            lr_ff = self.long_range_form_factor(packed, cache.unique_cart)
            spherical = add_multipole_potential(
                spherical, lr_ff, cache.inverse, cache.volume, cache.unique_cart
            )
        if self.environment_ionic:
            ionic_ff = self.ionic_form_factor(
                packed,
                cache.unique_cart,
                edge_index=cache.edge_index,
                cell_shifts=cache.cell_shifts,
            )
            spherical = add_multipole_potential(
                spherical, ionic_ff, cache.inverse, cache.volume, cache.unique_cart
            )
        potential = spherical
        if self.sp_projector is not None:
            environment_factors = None
            if self.sp_environment:
                environment_factors = self.projector_environment_factors(
                    packed,
                    features,
                    edge_index=cache.edge_index,
                    cell_shifts=cache.cell_shifts,
                )
            potential = potential + self.sp_projector(
                packed,
                k_frac,
                cache.g_idx,
                projectors=cache.sp_projectors,
                environment_factors=environment_factors,
                environment_delta=self.projector_gated_delta(packed, features) if self.sp_gated_head else None,
            ).to(potential)
        return batched_band_energies(
            cache.kinetic,
            potential,
            n_bands=n_bands,
            k_batch=self.k_batch,
            subset_ng_min=self.subset_ng_min,
        )


def build_gated_si_epm(
    channels: int = 8,
    ecut_ry: float = 12.0,
    monopole_mode: str = "gate",
    residual_l_max: int = 2,
    residual_num_radial: int = 8,
    residual_readout_bias: bool = True,
    cutoff: float = 5.0,
    env_isotropic: bool = False,
    env_density: bool = False,
    density_cutoffs: tuple[float, ...] = (3.0, 10.0),
    env_pack_gate: bool = False,
    residual_pack_gate: bool = False,
    pack_gate_center: float = 142.0,
    pack_gate_alpha: float = 2.0,
    pack_gate_windows: list[list[dict[str, Any]]] | list[str] | None = None,
    sphere_density_order: int = 0,
    sphere_density_center: float = 151.0,
    sphere_density_scale: float = 30.0,
    residual_anchor: bool = False,
    long_range: bool = False,
    environment_ionic: bool = False,
    ionic_cutoffs: tuple[float, ...] = (3.0, 5.0, 8.0),
    ionic_gaussian_width: float = 0.5,
    sp_nonlocal: bool = False,
    sp_environment: bool = False,
    sp_gated_head: bool = False,
    coordination_features: bool = False,
    sp_projector_scales: tuple[float, ...] = (1.5, 3.5),
    quad_scale_env: bool = False,
    quad_scale_center: float = 151.0,
    quad_scale_scale: float = 30.0,
    sphere_n_basis: int = 12,
    sphere_q_cutoff: float = 8.0,
    n_species: int = 1,
    species_symbols: tuple[str, ...] | list[str] | None = None,
    subset_ng_min: int | None = None,
    k_batch: int | None = None,
    residual_scalar_cosine_basis: int = 0,
    residual_scalar_cosine_cutoff: float = 8.0,
    residual_experts: int = 1,
) -> GatedSiEPM:
    return GatedSiEPM(
        channels=channels,
        cutoff=cutoff,
        ecut_ry=ecut_ry,
        monopole_mode=monopole_mode,
        num_radial=residual_num_radial,
        residual_scalar_cosine_basis=residual_scalar_cosine_basis,
        residual_scalar_cosine_cutoff=residual_scalar_cosine_cutoff,
        residual_l_max=residual_l_max,
        residual_readout_bias=residual_readout_bias,
        env_isotropic=env_isotropic,
        env_density=env_density,
        density_cutoffs=density_cutoffs,
        env_pack_gate=env_pack_gate,
        residual_pack_gate=residual_pack_gate,
        pack_gate_center=pack_gate_center,
        pack_gate_alpha=pack_gate_alpha,
        pack_gate_windows=pack_gate_windows,
        sphere_density_order=sphere_density_order,
        sphere_density_center=sphere_density_center,
        sphere_density_scale=sphere_density_scale,
        residual_anchor=residual_anchor,
        long_range=long_range,
        environment_ionic=environment_ionic,
        ionic_cutoffs=ionic_cutoffs,
        ionic_gaussian_width=ionic_gaussian_width,
        quad_scale_env=quad_scale_env,
        quad_scale_center=quad_scale_center,
        quad_scale_scale=quad_scale_scale,
        sp_nonlocal=sp_nonlocal,
        sp_environment=sp_environment,
        sp_gated_head=sp_gated_head,
        coordination_features=coordination_features,
        sp_projector_scales=sp_projector_scales,
        sphere_n_basis=sphere_n_basis,
        sphere_q_cutoff=sphere_q_cutoff,
        n_species=n_species,
        species_symbols=species_symbols,
        subset_ng_min=subset_ng_min,
        k_batch=k_batch,
        residual_experts=residual_experts,
    )


def checkpoint_env_isotropic(payload: dict[str, Any] | None, state: dict[str, Tensor] | None = None) -> bool:
    if payload and payload.get("env_isotropic"):
        return True
    keys = state if state is not None else (payload or {}).get("state_dict") or {}
    return any(str(k).startswith(_ENV_PREFIX) for k in keys)


def checkpoint_env_density(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("env_density"))


def checkpoint_density_cutoffs(payload: dict[str, Any] | None) -> tuple[float, ...]:
    raw = (payload or {}).get("density_cutoffs", (3.0, 10.0))
    return tuple(float(r) for r in raw)


def checkpoint_env_pack_gate(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("env_pack_gate"))


def checkpoint_residual_pack_gate(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("residual_pack_gate"))


def checkpoint_has_frozen_residual(state: dict[str, Tensor] | None) -> bool:
    keys = state or {}
    return any(str(k).startswith("encoder_frozen.") for k in keys) and any(
        str(k).startswith("readout_frozen.") for k in keys
    )


def checkpoint_pack_gate_center(payload: dict[str, Any] | None) -> float:
    return float((payload or {}).get("pack_gate_center", 142.0))


def checkpoint_pack_gate_alpha(payload: dict[str, Any] | None) -> float:
    return float((payload or {}).get("pack_gate_alpha", 2.0))


def checkpoint_pack_gate_windows(payload: dict[str, Any] | None) -> list[list[dict[str, Any]]] | None:
    return normalize_gate_windows((payload or {}).get("pack_gate_windows"))


def checkpoint_sphere_density(payload: dict[str, Any] | None) -> tuple[int, float, float]:
    data = payload or {}
    return (
        int(data.get("sphere_density_order", 0)),
        float(data.get("sphere_density_center", 151.0)),
        float(data.get("sphere_density_scale", 30.0)),
    )


def checkpoint_residual_anchor(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("residual_anchor"))


def checkpoint_quad_scale(
    payload: dict[str, Any] | None, state: dict[str, Tensor] | None = None
) -> tuple[bool, float, float]:
    """(enabled, center, scale) of the environment-dependent quadrupole scale."""
    data = payload or {}
    enabled = bool(data.get("quad_scale_env")) or bool(
        state and any(str(k).startswith(_QUAD_SCALE_PREFIX) for k in state)
    )
    return enabled, float(data.get("quad_scale_center", 151.0)), float(data.get("quad_scale_scale", 30.0))


def model_payload_flags(model: GatedSiEPM) -> dict[str, Any]:
    """Architecture flags a checkpoint needs so ``load_gated_si_epm`` rebuilds the same model."""
    return {
        "ecut_ry": float(model.ecut_ry),
        "channels": int(model.channels),
        "encoder_cutoff": float(model.encoder.cutoff),
        "monopole_mode": str(model.monopole_mode),
        "residual_l_max": int(model.residual_l_max),
        "residual_num_radial": int(model.residual_num_radial),
        "residual_readout_bias": bool(model.residual_readout_bias),
        "residual_scalar_cosine_basis": int(model.residual_scalar_cosine_basis),
        "residual_scalar_cosine_cutoff": float(model.residual_scalar_cosine_cutoff),
        "env_isotropic": bool(model.env_isotropic),
        "env_density": bool(model.env_density),
        "density_cutoffs": list(model.density_cutoffs),
        "env_pack_gate": bool(model.env_pack_gate),
        "residual_pack_gate": bool(model.residual_pack_gate),
        "pack_gate_center": float(model.pack_gate_center),
        "pack_gate_alpha": float(model.pack_gate_alpha),
        "pack_gate_windows": model.pack_gate_windows,
        "sphere_density_order": int(model.sphere_density_order),
        "sphere_density_center": float(model.sphere_density_center),
        "sphere_density_scale": float(model.sphere_density_scale),
        "residual_anchor": bool(model.residual_anchor),
        "sp_nonlocal": bool(model.sp_nonlocal),
        "sp_environment": bool(model.sp_environment),
        "sp_gated_head": bool(model.sp_gated_head),
        "coordination_features": bool(model.coordination_features),
        "residual_experts": int(model.residual_experts),
        "sp_projector_scales": (
            [float(x) for x in model.sp_projector.radial_scales.tolist()] if model.sp_projector is not None else [1.5, 3.5]
        ),
        "quad_scale_env": bool(model.quad_scale_env),
        "quad_scale_center": float(model.quad_scale_center),
        "quad_scale_scale": float(model.quad_scale_scale),
        "long_range": bool(model.long_range),
        "environment_ionic": bool(model.environment_ionic),
        "sphere_n_basis": int(model.sphere_n_basis),
        "sphere_q_cutoff": float(model.sphere_q_cutoff),
        "n_species": int(model.n_species),
        "species_symbols": list(model.species_symbols),
    }


def checkpoint_long_range(payload: dict[str, Any] | None, state: dict[str, Tensor] | None = None) -> bool:
    if payload and payload.get("long_range"):
        return True
    keys = state if state is not None else (payload or {}).get("state_dict") or {}
    return any(str(k) == _CHARGE_PREFIX or str(k).startswith(_CHARGE_PREFIX + ".") for k in keys)


def checkpoint_sp_nonlocal(payload: dict[str, Any] | None, state: dict[str, Tensor] | None = None) -> bool:
    if payload and payload.get("sp_nonlocal"):
        return True
    keys = state if state is not None else (payload or {}).get("state_dict") or {}
    return any(str(k).startswith(_SP_PREFIX) for k in keys)


def checkpoint_sp_environment(payload: dict[str, Any] | None) -> bool:
    return bool(payload and payload.get("sp_environment"))


def checkpoint_environment_ionic(
    payload: dict[str, Any] | None, state: dict[str, Tensor] | None = None
) -> bool:
    if payload and payload.get("environment_ionic"):
        return True
    keys = state if state is not None else (payload or {}).get("state_dict") or {}
    return any(str(k).startswith(_IONIC_PREFIX) for k in keys)


def load_gated_si_epm_state(
    model: GatedSiEPM,
    state: dict[str, Tensor],
    *,
    ignore_prefixes: tuple[str, ...] = (),
) -> None:
    payload = {
        key: value
        for key, value in state.items()
        if not any(str(key).startswith(prefix) for prefix in ignore_prefixes)
    }
    src = payload.get("sphere.coefficients")
    collocated = False
    if src is not None and tuple(src.shape) != tuple(model.sphere.coefficients.shape):
        parent = CosineRadialEP(
            n_species=int(src.shape[0]),
            n_basis=int(src.shape[1]),
            q_cutoff=model.sphere.q_cutoff,
        )
        with torch.no_grad():
            parent.coefficients.copy_(src)
        model.sphere.collocate_from(parent)
        payload = {k: v for k, v in payload.items() if not str(k).startswith("sphere.")}
        collocated = True
    result = model.load_state_dict(payload, strict=False)
    missing = [
        k
        for k in result.missing_keys
        if not any(str(k).startswith(p) for p in _OPTIONAL_PREFIXES)
        and not any(str(k).startswith(p) for p in ignore_prefixes)
        and not (collocated and str(k).startswith("sphere."))
    ]
    unexpected = [
        k for k in result.unexpected_keys if not any(str(k).startswith(p) for p in _OPTIONAL_PREFIXES)
    ]
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch missing={missing} unexpected={unexpected}")
    if model.residual_pack_gate and not checkpoint_has_frozen_residual(state):
        model.snapshot_frozen_residual()
    elif model.residual_anchor:
        model.set_anchor_reference()


def load_gated_si_epm(path: str | Path, *, env_isotropic: bool | None = None) -> tuple[GatedSiEPM, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"checkpoint {path} is not a dict payload")
    state = payload["state_dict"] if "state_dict" in payload else payload
    ecut = float(payload.get("ecut_ry", 12.0))
    channels = int(payload.get("channels", 8))
    monopole = str(payload.get("monopole_mode", "free"))
    residual_l_max = int(payload.get("residual_l_max", 2))
    residual_num_radial = int(payload.get("residual_num_radial", 8))
    residual_readout_bias = bool(payload.get("residual_readout_bias", True))
    env_iso = bool(env_isotropic) if env_isotropic is not None else checkpoint_env_isotropic(payload, state)
    env_density = checkpoint_env_density(payload)
    density_cutoffs = checkpoint_density_cutoffs(payload)
    env_pack_gate = checkpoint_env_pack_gate(payload)
    residual_pack_gate = checkpoint_residual_pack_gate(payload)
    long_range = checkpoint_long_range(payload, state)
    environment_ionic = checkpoint_environment_ionic(payload, state)
    sp_nonlocal = checkpoint_sp_nonlocal(payload, state)
    sp_environment = checkpoint_sp_environment(payload)
    symbols = payload.get("species_symbols")
    n_species = int(payload.get("n_species", len(symbols) if symbols else 1))
    sphere_density_order, sphere_density_center, sphere_density_scale = checkpoint_sphere_density(payload)
    quad_scale_env, quad_scale_center, quad_scale_scale = checkpoint_quad_scale(payload, state)
    model = build_gated_si_epm(
        channels=channels,
        cutoff=float(payload.get("encoder_cutoff", 5.0)),
        ecut_ry=ecut,
        monopole_mode=monopole,
        residual_l_max=residual_l_max,
        residual_num_radial=residual_num_radial,
        residual_scalar_cosine_basis=int(payload.get("residual_scalar_cosine_basis", 0)),
        residual_scalar_cosine_cutoff=float(payload.get("residual_scalar_cosine_cutoff", 8.0)),
        residual_readout_bias=residual_readout_bias,
        env_isotropic=env_iso,
        env_density=env_density,
        density_cutoffs=density_cutoffs,
        env_pack_gate=env_pack_gate,
        residual_pack_gate=residual_pack_gate,
        pack_gate_center=checkpoint_pack_gate_center(payload),
        pack_gate_alpha=checkpoint_pack_gate_alpha(payload),
        pack_gate_windows=checkpoint_pack_gate_windows(payload),
        sphere_density_order=sphere_density_order,
        sphere_density_center=sphere_density_center,
        sphere_density_scale=sphere_density_scale,
        residual_anchor=checkpoint_residual_anchor(payload),
        long_range=long_range,
        environment_ionic=environment_ionic,
        ionic_cutoffs=tuple(payload.get("ionic_cutoffs", (3.0, 5.0, 8.0))),
        ionic_gaussian_width=float(payload.get("ionic_gaussian_width", 0.5)),
        sp_nonlocal=sp_nonlocal,
        sp_environment=sp_environment,
        sp_gated_head=bool(payload.get("sp_gated_head")),
        coordination_features=bool(payload.get("coordination_features")),
        sp_projector_scales=tuple(payload.get("sp_projector_scales", (1.5, 3.5))),
        quad_scale_env=quad_scale_env,
        quad_scale_center=quad_scale_center,
        quad_scale_scale=quad_scale_scale,
        sphere_n_basis=int(payload.get("sphere_n_basis", payload.get("n_basis", 12))),
        sphere_q_cutoff=float(payload.get("sphere_q_cutoff", 8.0)),
        n_species=n_species,
        species_symbols=tuple(symbols) if symbols else None,
        subset_ng_min=payload.get("subset_ng_min"),
        k_batch=payload.get("k_batch"),
        residual_experts=int(payload.get("residual_experts", 1)),
    )
    load_gated_si_epm_state(model, state)
    return model, payload


def predict_bands(
    model: GatedSiEPM,
    structure: Structure,
    kpoints: np.ndarray,
    n_bands: int,
    species_table: SpeciesTable | None = None,
) -> Tensor:
    packed = pack_structure(structure, dtype=next(model.parameters()).dtype, species_table=species_table)
    k = torch.as_tensor(np.asarray(kpoints), dtype=packed.lattice.dtype)
    return model.band_path(packed, k, n_bands=n_bands)
