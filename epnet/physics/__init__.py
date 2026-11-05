"""Physics extensions for reciprocal-space EpNet Hamiltonians."""

from importlib import import_module

from .long_range import (
    neutralize_effective_charges,
    neutralized_effective_charges,
    reciprocal_coulomb_potential,
    reciprocal_ewald_potential,
)

# ``nonlocal`` is a Python keyword, so importing this module requires the
# dynamic spelling while still presenting ordinary package-level APIs.
_nonlocal = import_module(".nonlocal", __name__)

apply_separable_nonlocal = _nonlocal.apply_separable_nonlocal
separable_nonlocal_matrix = _nonlocal.separable_nonlocal_matrix
separable_nonlocal_operator = _nonlocal.separable_nonlocal_operator
separable_spin_nonlocal_matrix = _nonlocal.separable_spin_nonlocal_matrix
separable_spin_nonlocal_operator = _nonlocal.separable_spin_nonlocal_operator
spin_orbit_channel_coefficients = _nonlocal.spin_orbit_channel_coefficients

__all__ = [
    "apply_separable_nonlocal",
    "neutralize_effective_charges",
    "neutralized_effective_charges",
    "reciprocal_coulomb_potential",
    "reciprocal_ewald_potential",
    "separable_nonlocal_matrix",
    "separable_nonlocal_operator",
    "separable_spin_nonlocal_matrix",
    "separable_spin_nonlocal_operator",
    "spin_orbit_channel_coefficients",
]
