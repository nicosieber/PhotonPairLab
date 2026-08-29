"""Optional JAX-accelerated fast path for phase-matching angle search / group-index sweeps.

This subpackage is only imported when a caller explicitly asks for it (e.g.
``APMPhaseMatching.find_phase_matching_angles_jax``); it is never imported at package load time, so
the ``jax`` dependency stays optional (``pip install photonpairlab[jax]``). It does not change the
behavior of the existing numpy/scipy phase-matching code — it's an additive batched alternative for
sweeping many wavelengths/temperatures at once.
"""

from __future__ import annotations

try:
    import jax

    jax.config.update("jax_enable_x64", True)  # Sellmeier math needs float64 precision.
except ImportError as e:  # pragma: no cover - exercised only when jax isn't installed
    raise ImportError(
        "The JAX-accelerated phase-matching sweep path requires the optional 'jax' dependency. "
        "Install it with `pip install photonpairlab[jax]` (or `uv pip install -e '.[jax]'`)."
    ) from e

from .angle_search_jax import find_phase_matching_angle, find_phase_matching_angles
from .group_index_jax import group_index
from .material_bridge import (
    compute_phase_mismatch_sweep,
    delta_k_jax,
    find_phase_matching_angle_sweep,
    group_index_jax,
    refractive_index_jax,
)
from .sellmeier_jax import (
    bbo_like_n,
    effective_index_biaxial,
    effective_index_uniaxial,
    general_sellmeier_n,
    kato_takaoka_n,
    linear_thermal_n,
)

__all__ = [
    "find_phase_matching_angle",
    "find_phase_matching_angles",
    "group_index",
    "delta_k_jax",
    "group_index_jax",
    "refractive_index_jax",
    "find_phase_matching_angle_sweep",
    "compute_phase_mismatch_sweep",
    "bbo_like_n",
    "effective_index_biaxial",
    "effective_index_uniaxial",
    "general_sellmeier_n",
    "kato_takaoka_n",
    "linear_thermal_n",
]
