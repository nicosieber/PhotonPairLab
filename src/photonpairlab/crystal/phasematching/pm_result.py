from dataclasses import dataclass, replace
from typing import Callable, Literal  # noqa: UP035

from photonpairlab.constants import C_VAC

import numpy as np

from . import imperfections

@dataclass(frozen=True)
class PhaseMismatchResult:
    n: tuple[float, float, float]          # (n_p, n_s, n_i)
    N: tuple[float, float, float]          # (N_p, N_s, N_i)
    delta_k0: float                        # Δk0 in m^-1 (SI)
    angle_pm: float | None = None          # useful for APM, optional for QPM
    coherence_length: float | None = None  # useful for QPM

    def get_Np(self) -> float:
        return self.N[0]
    def get_Ns(self) -> float:
        return self.N[1]
    def get_Ni(self) -> float:
        return self.N[2]
    
    def get_angle_pm(self) -> float | None:
        return self.angle_pm
    
    def compute_delta_k1(
            self,
            wavelength_signal_range: np.ndarray,
            wavelength_idler_range: np.ndarray,
            pm_omega_signal: float,
            pm_omega_idler: float
            ) -> np.ndarray:
        """Computes the phase mismatch Δk1 for given signal and idler wavelengths."""
        fs = 2 * np.pi * C_VAC / wavelength_signal_range[None, :]  # Signal frequencies (row vector)
        fi = 2 * np.pi * C_VAC / wavelength_idler_range[:, None]  # Idler frequencies (column vector)

        K_pump = self.get_Np() / C_VAC
        K_signal = self.get_Ns() / C_VAC
        K_idler = self.get_Ni() / C_VAC

        delta_K1 = (K_pump - K_signal) * (fs - pm_omega_signal) + (K_pump - K_idler) * (fi - pm_omega_idler)
        return delta_K1


@dataclass(frozen=True)
class PolingResult:
    """
    Result of a phase-matching strategy's ``generate_poling()``: the domain-sign pattern and its
    z-axis, plus the target vs. actual field-amplitude buildup along the crystal (Graffitti et al.
    2017, Eq. 5 & 9) evaluated for that finalized pattern -- used to compare a designed apodization
    target against what the discrete domain structure actually realizes (e.g. for plotting).
    """
    poling_pattern: np.ndarray
    z: np.ndarray
    temperature_adjusted_length: float
    target_amplitude: np.ndarray
    actual_amplitude: np.ndarray

    # Domain-level metadata, populated only by strategies that support manufacturing-
    # imperfection modeling (see add_wall_position_error/add_missed_domain_error/
    # add_duty_cycle_bias below). All default to None/True so existing PolingResult(...)
    # call sites that don't populate them are unaffected.
    domain_signs: np.ndarray | None = None            # one sign (+1/-1) per physical domain
    domain_widths: np.ndarray | None = None            # that domain's nominal width, meters
    resolution: int | None = None                       # fine samples per domain in poling_pattern/z
    coherence_length: float | None = None
    DeltaK: float | None = None
    target_profile: Callable[[np.ndarray, float, float], np.ndarray] | None = None
    uniform_width: bool = True

    def _require_domain_metadata(self, method_name: str) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Validate that domain_signs/domain_widths/resolution are populated and return them as a
        narrowed (non-None) tuple. Returning them (rather than just validating self.* in place)
        lets callers bind to local variables that static type checkers can actually narrow --
        `self.domain_signs` stays typed as `ndarray | None` after this call returns, since a
        checker can't know a separate method call proves it's no longer None.
        """
        if self.domain_signs is None or self.domain_widths is None or self.resolution is None:
            raise ValueError(
                f"{method_name} requires a PolingResult with domain_signs/domain_widths/resolution "
                "metadata populated (as produced by strategies that support manufacturing-"
                "imperfection modeling, e.g. QPMPhaseMatching's periodic/constant/subcoh poling); "
                f"got domain_signs={self.domain_signs!r}, domain_widths={self.domain_widths!r}, "
                f"resolution={self.resolution!r}."
            )
        return self.domain_signs, self.domain_widths, self.resolution

    def add_wall_position_error(
            self,
            method: Literal["cumulative", "independent"] = "cumulative",
            sigma: float = 0.0,
            rng: np.random.Generator | None = None,
            ) -> "PolingResult":
        """
        Perturb domain wall positions to model random poling-position fabrication error.
        Returns a new ``PolingResult`` (this one is left untouched); a no-op (returns ``self``)
        if ``sigma <= 0``.

        ``method="cumulative"`` (default) draws each domain's *width* independently and lets
        wall positions fall out as the cumulative sum -- the random-walk model of Helmfrid &
        Arvidsson 1991 (JOSA B 8, 797), found to be the physically dominant error mechanism for
        real devices since positional error accumulates with domain index.

        ``method="independent"`` instead perturbs each wall's position independently around its
        own nominal cumulative position -- the bounded, non-accumulating model used in
        Graffitti et al. 2017 (Quantum Sci. Technol. 2, 035001, Sec. III.3.2), where ``sigma``
        is "sigma_rvd", a fraction of the nominal domain width (the paper sweeps 0-0.12; ~0.08
        is cited as realistic for LN waveguides, ~0.02 measured for KTP in Optica OL 46, 3049
        (2021)).

        Sets ``target_amplitude``/``actual_amplitude`` to ``None`` (stale until
        ``Crystal.apply_poling()`` recomputes them) and ``uniform_width=False``.
        """
        domain_signs, domain_widths, resolution = self._require_domain_metadata("add_wall_position_error")
        if sigma <= 0:
            return self
        rng = rng if rng is not None else np.random.default_rng()
        if method == "cumulative":
            new_widths = imperfections.perturb_widths_cumulative(domain_widths, sigma, rng)
        elif method == "independent":
            new_widths = imperfections.perturb_widths_independent(domain_widths, sigma, rng)
        else:
            raise ValueError(f"Unknown method {method!r}; use 'cumulative' or 'independent'.")  # pyright: ignore[reportUnreachable]
        pattern, z = imperfections.resample_domains_to_fine_grid(domain_signs, new_widths, resolution)
        return replace(self, poling_pattern=pattern, z=z,
                        temperature_adjusted_length=float(new_widths.sum()),
                        domain_widths=new_widths, uniform_width=False,
                        target_amplitude=None, actual_amplitude=None)

    def add_missed_domain_error(self, probability: float = 0.0, rng: np.random.Generator | None = None) -> "PolingResult":
        """
        Flip the sign of a Bernoulli(``probability``)-selected subset of physical domains,
        modeling poling-voltage failures where a domain fails to invert (Graffitti et al. 2017,
        Sec. III.3.2 "missed domains"). Returns a new ``PolingResult``; a no-op (returns
        ``self``) if ``probability <= 0``. Pure sign-array operation -- domain widths are
        untouched, so ``uniform_width`` is preserved.
        """
        domain_signs, domain_widths, resolution = self._require_domain_metadata("add_missed_domain_error")
        if probability <= 0:
            return self
        rng = rng if rng is not None else np.random.default_rng()
        new_signs = imperfections.apply_missed_domains(domain_signs, probability, rng)
        pattern, z = imperfections.resample_domains_to_fine_grid(new_signs, domain_widths, resolution)
        return replace(self, poling_pattern=pattern, z=z, domain_signs=new_signs,
                        target_amplitude=None, actual_amplitude=None)

    def add_duty_cycle_bias(self, factor: float = 0.0) -> "PolingResult":
        """
        Systematically grow ``+1`` domains and shrink ``-1`` domains (or the reverse for
        ``factor < 0``) by a fixed fraction, modeling systematic over-/under-poling (Graffitti
        et al. 2017, Sec. III.3.2). ``new_width = domain_widths * (1 + factor*domain_signs)`` --
        exact and continuous, independent of resolution. Returns a new ``PolingResult``; a
        no-op (returns ``self``) if ``factor == 0``.
        """
        domain_signs, domain_widths, resolution = self._require_domain_metadata("add_duty_cycle_bias")
        if factor == 0:
            return self
        new_widths = imperfections.apply_duty_cycle_bias(domain_signs, domain_widths, factor)
        pattern, z = imperfections.resample_domains_to_fine_grid(domain_signs, new_widths, resolution)
        return replace(self, poling_pattern=pattern, z=z,
                        temperature_adjusted_length=float(new_widths.sum()),  # pyright: ignore[reportAny]
                        domain_widths=new_widths, uniform_width=False,
                        target_amplitude=None, actual_amplitude=None)

