"""Jittable/vmappable bounded scalar minimizer, replacing ``scipy.optimize.minimize_scalar`` for the
phase-matching angle search (which isn't traceable by JAX). A fixed-iteration golden-section search
keeps the loop shape jit-static; ``find_phase_matching_angles`` vmaps it across a whole sweep so an
entire wavelength scan solves in one batched call instead of one scipy call per point.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

_GOLDEN_RATIO = (5.0**0.5 - 1.0) / 2.0  # ~0.618


def find_phase_matching_angle(objective_fn: Callable, bounds: tuple[float, float] = (0.0, 90.0), iters: int = 50):
    """Bounded golden-section search minimizing ``objective_fn(angle)`` (e.g. ``abs(delta_k(angle))``)
    over ``bounds``. ``iters=50`` over a 90 degree bracket comfortably beats scipy's default ``xatol``.
    """
    a0, b0 = bounds

    def body(_, carry):
        a, b = carry
        c = b - _GOLDEN_RATIO * (b - a)
        d = a + _GOLDEN_RATIO * (b - a)
        take_left = objective_fn(c) < objective_fn(d)
        a_new = jnp.where(take_left, a, c)
        b_new = jnp.where(take_left, d, b)
        return (a_new, b_new)

    a, b = jax.lax.fori_loop(
        0, iters, body, (jnp.asarray(a0, dtype=jnp.float64), jnp.asarray(b0, dtype=jnp.float64))
    )
    return (a + b) / 2.0


def find_phase_matching_angles(delta_k_fn: Callable, *batched_args, bounds: tuple[float, float] = (0.0, 90.0),
                                iters: int = 50):
    """Vmapped counterpart of ``find_phase_matching_angle``: ``delta_k_fn(angle, *point_args)`` must
    return a scalar Δk, and ``batched_args`` are same-shape arrays (e.g. wavelength_pump,
    wavelength_signal, wavelength_idler) — one entry per sweep point.
    """

    def solve_one(*point_args):
        return find_phase_matching_angle(lambda angle: jnp.abs(delta_k_fn(angle, *point_args)), bounds, iters)

    return jax.vmap(solve_one)(*batched_args)
