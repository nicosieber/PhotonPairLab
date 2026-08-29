"""Bridges a real ``BaseMaterialModel`` instance (backed by ``resources/materials.json`` via
``MaterialFactory``) to the pure-JAX Sellmeier functions in ``sellmeier_jax.py``, mirroring the
dispatch logic in ``PhaseMatchingStrategy.get_refractive_index``/``get_group_index``
(``base_pm_strategy.py``) and ``PhaseMatchingStrategy.delta_k``.
"""

from __future__ import annotations

import functools
from typing import Callable

import jax
import jax.numpy as jnp

from ...material.base_material import BaseMaterial
from ...material.model.bbo import BBO
from ...material.model.bibo import BIBO
from ...material.model.general_sellmeier_thermal import GeneralSellmeierThermalModel
from ...material.model.kato_takaoka_sellmeier_thermal import KatoTakaokaSellmeierThermalModel
from ...material.model.sellmeier_linear_thermal import SellmeierLinearThermalModel
from .angle_search_jax import find_phase_matching_angles as _find_phase_matching_angles
from .group_index_jax import group_index as _group_index_autodiff
from .sellmeier_jax import (
    bbo_like_n,
    effective_index_biaxial,
    effective_index_uniaxial,
    general_sellmeier_n,
    kato_takaoka_n,
    linear_thermal_n,
)


def _axis_n_func(material_model: BaseMaterial, axis: str, T: float) -> Callable:
    """Build a JAX-traceable n(wavelength_um) closure for one crystal axis, matching whichever
    concrete model class ``material_model`` is (coefficients are read once, eagerly, here — only
    wavelength is traced).
    """
    coeffs = material_model.material.sellmeier.data[axis]  # type: ignore[attr-defined]
    A, B, C = float(coeffs["A"]), float(coeffs["B"]), float(coeffs["C"])

    if isinstance(material_model, (BBO, BIBO)):
        D = float(coeffs["D"])
        return lambda wl: bbo_like_n(wl, A, B, C, D)

    if isinstance(material_model, KatoTakaokaSellmeierThermalModel):
        D = float(coeffs.get("D", 0.0) or 0.0)
        E = float(coeffs.get("E", 0.0) or 0.0)
        temp_coeffs = None
        tc = material_model.material.temperature_corrections  # type: ignore[attr-defined]
        if tc is not None and isinstance(tc.data, dict) and tc.data.get(axis) is not None:
            t = tc.data[axis]
            temp_coeffs = (float(t["A"]), float(t["B"]), float(t["C"]), float(t["D"]))
        return lambda wl: kato_takaoka_n(wl, A, B, C, D, E, temp_coeffs, T)

    if isinstance(material_model, SellmeierLinearThermalModel):
        D = float(coeffs["D"])
        k = None
        tc = material_model.material.temperature_corrections  # type: ignore[attr-defined]
        if tc is not None and isinstance(tc.data, dict) and tc.data.get(axis) is not None:
            k = float(tc.data[axis])
        return lambda wl: linear_thermal_n(wl, A, B, C, D, k, T)

    if isinstance(material_model, GeneralSellmeierThermalModel):
        D = float(coeffs.get("D", 0.0) or 0.0)
        E = float(coeffs.get("E", 0.0) or 0.0)
        F = float(coeffs.get("F", 0.0) or 0.0)
        temp_coeffs = None
        tc = material_model.material.temperature_corrections  # type: ignore[attr-defined]
        if tc is not None and isinstance(tc.data, dict) and tc.data.get(axis) is not None:
            tc_axis = tc.data[axis]
            if isinstance(tc_axis, dict) and "n1" in tc_axis and "n2" in tc_axis:
                temp_coeffs = (tc_axis["n1"], tc_axis["n2"])
        return lambda wl: general_sellmeier_n(wl, A, B, C, D, E, F, temp_coeffs, T)

    raise TypeError(f"No JAX Sellmeier port registered for material model {type(material_model).__name__}.")


def refractive_index_jax(material_model: BaseMaterial, wavelength_um, polarization: str,
                          angle_deg, phi_deg: float, T: float):
    """JAX port of ``PhaseMatchingStrategy.get_refractive_index``."""
    if polarization == "e":
        if isinstance(material_model, GeneralSellmeierThermalModel):
            # Parity with GeneralSellmeierThermalModel.effective_refractive_index, which also
            # always raises (no biaxial 'x' axis data is registered for the materials using this model).
            raise NotImplementedError(
                f"Effective refractive index not implemented for model '{type(material_model).__name__}'."
            )
        if material_model.is_biaxial():
            nx = _axis_n_func(material_model, "x", T)(wavelength_um)
            ny = _axis_n_func(material_model, "y", T)(wavelength_um)
            nz = _axis_n_func(material_model, "z", T)(wavelength_um)
            return effective_index_biaxial(nx, ny, nz, angle_deg, phi_deg)
        no = _axis_n_func(material_model, "o", T)(wavelength_um)
        ne = _axis_n_func(material_model, "e", T)(wavelength_um)
        return effective_index_uniaxial(no, ne, angle_deg)

    axis = material_model.map_polarization_axis(polarization)
    assert axis is not None
    return _axis_n_func(material_model, axis, T)(wavelength_um)


def group_index_jax(material_model: BaseMaterial, wavelength_um, polarization: str,
                     angle_deg, phi_deg: float, T: float):
    """JAX port of ``PhaseMatchingStrategy.get_group_index``, using autodiff instead of the
    central-difference ``derivative()`` in ``base_material.py``.
    """
    def n_func(wl):
        return refractive_index_jax(material_model, wl, polarization, angle_deg, phi_deg, T)

    return _group_index_autodiff(n_func, wavelength_um)


def delta_k_jax(material_model: BaseMaterial, polarizations: tuple[str, str, str],
                 wavelength_pump, wavelength_signal, wavelength_idler, angle_deg, phi_deg: float, T: float):
    """JAX port of ``PhaseMatchingStrategy.delta_k``. Wavelengths are in meters (SI), matching the
    numpy method's external interface; converted to micrometers internally for the Sellmeier forms.
    """
    pol_p, pol_s, pol_i = polarizations
    wl_p = wavelength_pump * 1e6
    wl_s = wavelength_signal * 1e6
    wl_i = wavelength_idler * 1e6

    n_p = refractive_index_jax(material_model, wl_p, pol_p, angle_deg, phi_deg, T)
    n_s = refractive_index_jax(material_model, wl_s, pol_s, angle_deg, phi_deg, T)
    n_i = refractive_index_jax(material_model, wl_i, pol_i, angle_deg, phi_deg, T)

    k_p = 2 * jnp.pi * n_p / wl_p
    k_s = 2 * jnp.pi * n_s / wl_s
    k_i = 2 * jnp.pi * n_i / wl_i
    return (k_p - k_s - k_i) * 1e6


@functools.partial(jax.jit, static_argnames=("material_model", "pol_p", "pol_s", "pol_i", "phi_deg", "T", "bounds", "iters"))
def find_phase_matching_angle_sweep(material_model: BaseMaterial, pol_p: str, pol_s: str, pol_i: str, phi_deg: float,
                                     wavelength_pump, wavelength_signal, wavelength_idler,
                                     T: float, bounds: tuple[float, float] = (0.0, 90.0), iters: int = 50):
    """Jit-compiled batched angle search: one compiled call replaces one
    ``scipy.optimize.minimize_scalar`` per sweep point. All non-array arguments are static, so
    the compiled executable is cached and reused across repeated calls with the same material,
    SPDC-type polarizations, phi/T, and array shape.
    """
    pols = (pol_p, pol_s, pol_i)

    def dk(angle, p, s, i):
        return delta_k_jax(material_model, pols, p, s, i, angle, phi_deg, T)

    return _find_phase_matching_angles(dk, wavelength_pump, wavelength_signal, wavelength_idler, bounds=bounds, iters=iters)


@functools.partial(jax.jit, static_argnames=("material_model", "pol_p", "pol_s", "pol_i", "phi_deg", "T", "bounds", "iters"))
def compute_phase_mismatch_sweep(material_model: BaseMaterial, pol_p: str, pol_s: str, pol_i: str, phi_deg: float,
                                  wavelength_pump, wavelength_signal, wavelength_idler,
                                  T: float, bounds: tuple[float, float] = (0.0, 90.0), iters: int = 50):
    """Jit-compiled batched phase-mismatch sweep: finds the phase-matching angle for every point
    (see ``find_phase_matching_angle_sweep``) then computes n's, N's (via autodiff group index),
    and Δk0 for the whole sweep in the same compiled call. Returns a dict of arrays.
    """
    pols = (pol_p, pol_s, pol_i)
    angle_pm = find_phase_matching_angle_sweep(
        material_model, pol_p, pol_s, pol_i, phi_deg, wavelength_pump, wavelength_signal, wavelength_idler,
        T, bounds, iters,
    )

    def per_point(angle, p, s, i):
        n_p = refractive_index_jax(material_model, p * 1e6, pol_p, angle, phi_deg, T)
        n_s = refractive_index_jax(material_model, s * 1e6, pol_s, angle, phi_deg, T)
        n_i = refractive_index_jax(material_model, i * 1e6, pol_i, angle, phi_deg, T)
        N_p = group_index_jax(material_model, p * 1e6, pol_p, angle, phi_deg, T)
        N_s = group_index_jax(material_model, s * 1e6, pol_s, angle, phi_deg, T)
        N_i = group_index_jax(material_model, i * 1e6, pol_i, angle, phi_deg, T)
        delta_k0 = delta_k_jax(material_model, pols, p, s, i, angle, phi_deg, T)
        return n_p, n_s, n_i, N_p, N_s, N_i, delta_k0

    n_p, n_s, n_i, N_p, N_s, N_i, delta_k0 = jax.vmap(per_point)(
        angle_pm, wavelength_pump, wavelength_signal, wavelength_idler
    )
    return {
        "angle_pm": angle_pm,
        "delta_k0": delta_k0,
        "n": (n_p, n_s, n_i),
        "N": (N_p, N_s, N_i),
    }
