"""Exact-autodiff replacement for ``BaseMaterial.group_index``'s central-difference derivative."""

from __future__ import annotations

from typing import Callable  # noqa: UP035

import jax
import jax.numpy as jnp


def group_index(n_func: Callable, wavelength):
    """n - wavelength * dn/dwavelength, with dn/dwavelength from ``jax.grad`` instead of a
    finite-difference step. ``wavelength`` may be a scalar or a 1D array (batched via ``jax.vmap``).
    """
    wavelength = jnp.asarray(wavelength, dtype=jnp.float64)
    if wavelength.ndim == 0:
        n = n_func(wavelength)
        dn_dlambda = jax.grad(n_func)(wavelength)
    else:
        n = n_func(wavelength)
        dn_dlambda = jax.vmap(jax.grad(n_func))(wavelength)
    return n - wavelength * dn_dlambda
