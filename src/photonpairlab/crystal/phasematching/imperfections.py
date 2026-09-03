"""
Pure, array-in/array-out helpers for perturbing a domain-level poling representation
(``domain_signs``, ``domain_widths``) to model manufacturing/fabrication imperfections.

These operate purely on numpy arrays -- no dependency on ``PolingResult`` or any particular
phase-matching strategy -- so any poling algorithm that can express its result as one sign and
one nominal width per physical domain can reuse them (see ``PolingResult.add_wall_position_error``,
``.add_missed_domain_error``, ``.add_duty_cycle_bias`` in ``pm_result.py``, which are the intended
call sites).

References:
    Helmfrid, S. & Arvidsson, G. "Influence of randomly varying domain lengths and nonuniform
    effective index on second-harmonic generation in quasi-phase-matching waveguides,"
    J. Opt. Soc. Am. B 8, 797-804 (1991).

    Graffitti, F., Costa-Filho, J., Kolthammer, W.S. & Brańczyk, A.M. "Design considerations
    for high-purity heralded single-photon sources," Quantum Sci. Technol. 2, 035001 (2017),
    https://doi.org/10.1088/2058-9565/aa78d4 (Sec. III.3.2 "Fabrication imperfections").
"""

import numpy as np


def perturb_widths_cumulative(domain_widths: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Independent-domain-length ("random-walk") wall-position error model (Helmfrid & Arvidsson
    1991): each domain's width is independently scaled by ``1 + N(0, sigma)``, so wall
    positions (the cumulative sum of widths) undergo a random walk whose positional variance
    grows with domain index -- the model H&A found to be the physically dominant error
    mechanism for real devices.

    Widths are clipped to stay strictly positive (guards against pathological sigma).
    """
    domain_widths = np.asarray(domain_widths, dtype=float)
    noise = rng.normal(0.0, sigma, size=domain_widths.shape)
    widths = domain_widths * (1.0 + noise)
    return np.clip(widths, domain_widths * 1e-3, None)


def perturb_widths_independent(domain_widths: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Independent-wall-position ("bounded") error model (Graffitti et al. 2017, Sec. III.3.2):
    each wall is independently offset from its own nominal cumulative position by
    ``N(0, sigma * mean(domain_widths))`` -- unlike ``perturb_widths_cumulative``, this error
    does not accumulate with domain index. ``sigma`` here is "sigma_rvd" in the paper's
    notation, a fraction of the nominal domain width (they sweep 0-0.12; ~0.08 is cited as
    realistic for LN waveguides, ~0.02 measured for KTP in Optica OL 46, 3049 (2021)).

    The outermost two walls (the crystal's physical extent) are never perturbed. Each interior
    offset is clipped to +/-49% of the narrower of its two neighboring nominal domain widths,
    which guarantees every perturbed domain width stays strictly positive regardless of sigma.
    """
    domain_widths = np.asarray(domain_widths, dtype=float)
    n = len(domain_widths)
    edges = np.concatenate(([0.0], np.cumsum(domain_widths)))
    scale = sigma * np.mean(domain_widths)
    offsets = rng.normal(0.0, scale, size=n + 1)
    offsets[0] = 0.0
    offsets[-1] = 0.0
    if n > 1:
        bound = 0.49 * np.minimum(domain_widths[:-1], domain_widths[1:])
        offsets[1:-1] = np.clip(offsets[1:-1], -bound, bound)
    return np.diff(edges + offsets)


def missed_domain_mask(num_domains: int, probability: float, rng: np.random.Generator) -> np.ndarray:
    """Boolean mask, True for each domain independently selected with probability ``probability``."""
    return rng.random(num_domains) < probability


def apply_missed_domains(domain_signs: np.ndarray, probability: float, rng: np.random.Generator) -> np.ndarray:
    """
    Flip the sign of a Bernoulli(``probability``)-selected subset of physical domains, modeling
    poling-voltage failures where a domain fails to invert (Graffitti et al. 2017, Sec.
    III.3.2 "missed domains"). Pure sign-array operation -- independent of domain widths.
    """
    domain_signs = np.asarray(domain_signs).copy()
    mask = missed_domain_mask(len(domain_signs), probability, rng)
    domain_signs[mask] *= -1
    return domain_signs


def apply_duty_cycle_bias(domain_signs: np.ndarray, domain_widths: np.ndarray, factor: float) -> np.ndarray:
    """
    Systematic (deterministic) over-/under-poling: grow ``+1`` domains and shrink ``-1``
    domains by a fixed fraction ``factor`` (or the reverse for ``factor < 0``), modeling a
    consistent fabrication bias in the poling process (Graffitti et al. 2017, Sec. III.3.2).

    ``new_width = domain_widths * (1 + factor * domain_signs)`` -- exact and continuous,
    independent of any fine-grid resolution (unlike recoloring fine simulation cells, this
    works identically regardless of how many samples per domain the fine grid uses).
    """
    return np.asarray(domain_widths, dtype=float) * (1.0 + factor * np.asarray(domain_signs))


def resample_domains_to_fine_grid(domain_signs: np.ndarray, domain_widths: np.ndarray, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Rebuild a ``resolution``-samples-per-domain fine ``(poling_pattern, z)`` pair from a
    (possibly irregular-width) sequence of domain signs/widths. Each domain contributes
    ``resolution`` left-edge-sampled points spanning its own width; the resulting ``z`` is
    recentred so the whole array spans symmetrically around 0, matching the convention used by
    the ideal-poling generators (e.g. ``QPMPhaseMatching._generate_periodic_poling``).

    Fully vectorized and general over non-uniform ``domain_widths`` -- this is what lets a
    perturbation (irregular domain widths) flow back into a fine array of the same shape/
    granularity the simulation and diagnostic-plotting code already expect.
    """
    domain_signs = np.asarray(domain_signs)
    domain_widths = np.asarray(domain_widths, dtype=float)
    edges = np.concatenate(([0.0], np.cumsum(domain_widths)))
    total_length = edges[-1]  # pyright: ignore[reportAny]
    frac = np.arange(resolution) / resolution
    z_local = edges[:-1, None] + domain_widths[:, None] * frac[None, :]
    z = z_local.ravel() - total_length / 2.0
    pattern = np.repeat(domain_signs, resolution)
    return pattern, z
