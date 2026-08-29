"""JAX (jax.numpy) ports of the closed-form Sellmeier/effective-index formulas implemented in
``crystal/material/model/*.py``. These are pure functions of plain Python floats and JAX arrays
(no ``BaseMaterial``/``MaterialData`` objects) so they can be freely ``jax.jit``-ed, ``jax.grad``-ed,
and ``jax.vmap``-ed. See ``material_bridge.py`` for the glue that extracts coefficients from a real
material model instance and builds these closures.
"""

from __future__ import annotations

import jax.numpy as jnp


def general_sellmeier_n(wl, A: float, B: float, C: float, D: float = 0.0, E: float = 0.0, F: float = 0.0,
                         temp_coeffs: tuple[list[float], list[float]] | None = None, T: float = 25.0):
    """Mirrors ``GeneralSellmeierThermalModel.refractive_index``."""
    wl = jnp.asarray(wl)
    if E == 0.0 and F == 0.0:
        n2 = A + B / (1.0 - C / wl**2) - D * wl**2
    else:
        n2 = A + B / (1.0 - C / wl**2) + D / (1.0 - E / wl**2) - F * wl**2
    n = jnp.sqrt(n2)

    if temp_coeffs is not None:
        n1, n2c = temp_coeffs
        dT = T - 25.0
        deln = (n1[0] + n1[1] / wl + n1[2] / wl**2 + n1[3] / wl**3) * dT + (
            n2c[0] + n2c[1] / wl + n2c[2] / wl**2 + n2c[3] / wl**3
        ) * dT**2
        n = n + deln
    return n


def kato_takaoka_n(wl, A: float, B: float, C: float, D: float = 0.0, E: float = 0.0,
                    temp_coeffs: tuple[float, float, float, float] | None = None, T: float = 25.0):
    """Mirrors ``KatoTakaokaSellmeierThermalModel.refractive_index``."""
    wl = jnp.asarray(wl)
    n2 = A + B / (wl**2 - C) + D / (wl**2 - E)
    n = jnp.sqrt(n2)

    if temp_coeffs is not None:
        tA, tB, tC, tD = temp_coeffs
        n = n + (tA / wl**3 - tB / wl**2 + tC / wl + tD) * 1e-5 * (T - 25.0)
    return n


def linear_thermal_n(wl, A: float, B: float, C: float, D: float, k: float | None = None, T: float = 25.0):
    """Mirrors ``SellmeierLinearThermalModel.refractive_index``."""
    wl = jnp.asarray(wl)
    n2 = A + B / (wl**2 - C) - D * wl**2
    n = jnp.sqrt(n2)

    if k is not None:
        n = n + k * (T - 25.0)
    return n


def bbo_like_n(wl, A: float, B: float, C: float, D: float):
    """Mirrors ``BBO.refractive_index`` / ``BIBO.refractive_index`` (identical closed form)."""
    wl = jnp.asarray(wl)
    return jnp.sqrt(A + B / (wl**2 - C) - D * wl**2)


def effective_index_uniaxial(no, ne, theta_deg):
    """Mirrors ``BBO.effective_refractive_index`` (uniaxial: 'o'/'e' axes, angle theta only)."""
    theta = jnp.radians(theta_deg)
    denom = jnp.cos(theta) ** 2 / no**2 + jnp.sin(theta) ** 2 / ne**2
    return 1.0 / jnp.sqrt(denom)


def effective_index_biaxial(nx, ny, nz, theta_deg, phi_deg):
    """Mirrors the biaxial n_eff formula shared by KatoTakaoka/SellmeierLinear/BIBO models."""
    theta = jnp.radians(theta_deg)
    phi = jnp.radians(phi_deg)
    inv_sq = (
        (jnp.cos(theta) ** 2 * jnp.cos(phi) ** 2) / nx**2
        + (jnp.cos(theta) ** 2 * jnp.sin(phi) ** 2) / ny**2
        + (jnp.sin(theta) ** 2) / nz**2
    )
    # jit-safe stand-in for the numpy path's `if n_eff_sq_inv <= 0: raise ValueError(...)`.
    safe_inv_sq = jnp.where(inv_sq > 0, inv_sq, jnp.nan)
    return jnp.sqrt(1.0 / safe_inv_sq)
