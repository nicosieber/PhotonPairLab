from typing import Literal

import numpy as np

from .material import BaseMaterial
from .phasematching import QPMPhaseMatching
from .phasematching import APMPhaseMatching
from .phasematching import SPDCType, PolingMode
from .phasematching import PolingResult

from photonpairlab.laser import BaseLaser

PM_STRATEGY_HANDLER = {
    "quasi": QPMPhaseMatching,
    "angle": APMPhaseMatching,
}

PMStrategyName = Literal["quasi", "angle"]


class Crystal:
    """
    Unified Crystal class that binds a nonlinear optical material with a phase-matching strategy
    (Angle Phase Matching or Quasi Phase Matching).

    Attributes:
        material (BaseMaterial): Nonlinear optical material instance.
        spdc_type (str): SPDC interaction type (e.g., "type-I", "type-II").
        pm_strategy: Instance of the phase-matching strategy.
    """

    def __init__(
            self, crystal_length: float,
            material: BaseMaterial,
            coherence_length: float | None = None,
            T: float = 25.0, pm_strategy: PMStrategyName ="quasi",
            spdc_type: SPDCType ="type-IIoeo",
            phi_deg: float = 0,
            w: float | None = None,
            **kwargs
            ):
        """
        Initializes the Crystal with a given material and phase-matching strategy.

        Args:
            material (BaseMaterial): The nonlinear optical material.
            pm_strategy (str): "quasi" for QPM or "angle" for APM.
            spdc_type (str): SPDC interaction type.
            kwargs: Additional parameters passed to the strategy constructor.
        """
        self.crystal_length = crystal_length
        self.material = material
        self.coherence_length = coherence_length
        self.T = T
        self.spdc_type = spdc_type
        self.phi_deg = phi_deg
        self.w = w

        # Load phase-matching strategy via handler
        try:
            self.pm_strategy = PM_STRATEGY_HANDLER[pm_strategy](material, spdc_type=spdc_type, **kwargs)
        except KeyError as e:
            raise KeyError(f"Unknown phase-matching strategy: {pm_strategy}. Valid options are: {list(PM_STRATEGY_HANDLER.keys())}") from e

        # Temperature Expansion of crystal
        try:
            self.temperature_adjusted_crystal_length = self.material.thermal_expansion(length=self.crystal_length, axis="z", temperature=self.T)
        except ValueError as e:
            raise ValueError(f"Error in thermal expansion: {e}") from e

        
        self.poling_pattern: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.temperature_adjusted_length: float | None = None
        self.target_amplitude: np.ndarray | None = None
        self.actual_amplitude: np.ndarray | None = None

    def compute_phase_mismatch(self, *args, **kwargs):
        """Calculates the phase mismatch using the selected phase-matching strategy."""
        return self.pm_strategy.compute_phase_mismatch(*args, **kwargs)

    def delta_k(self, *args, **kwargs):
        """Calculates the phase mismatch using the selected phase-matching strategy."""
        return self.pm_strategy.delta_k(*args, **kwargs)

    def ideal_coherence_length(self, laser: BaseLaser, wavelength_signal: float, wavelength_idler: float, T: float | None = None) -> float:
        """
        Computes the coherence length Lc = pi / |Delta_k0| for this crystal's material and
        phase-matching strategy at the given laser/target wavelengths (Fejer et al. 1992
        convention: domain width = Lc, QPM grating period = 2*Lc). Uses each strategy's own
        default phase-matching angle (0 for QPM, angle search for APM).
        """
        T = self.T if T is None else T
        pm_result = self.pm_strategy.compute_phase_mismatch(laser, wavelength_signal, wavelength_idler, T=T)
        return np.pi / abs(pm_result.delta_k0)

    def generate_poling(self, laser: BaseLaser, mode: PolingMode, wavelength_signal: float, wavelength_idler:float, **kwargs) -> PolingResult:
        """
        Generates the poling pattern based on the selected phase-matching strategy.

        If `coherence_length` wasn't provided when constructing this Crystal, it is computed
        automatically via `ideal_coherence_length` for the given laser/target wavelengths and
        stored on `self.coherence_length` (an explicitly-provided value is never overwritten).

        Returns the `PolingResult` (also copied onto `self.poling_pattern`/`.z`/etc.), so it can
        be chained into `.add_wall_position_error()`/`.add_missed_domain_error()`/
        `.add_duty_cycle_bias()` and passed to `apply_poling()`.
        """
        if self.coherence_length is None:
            self.coherence_length = self.ideal_coherence_length(laser, wavelength_signal, wavelength_idler, T=self.T)
        poling_result = self.pm_strategy.generate_poling(self.crystal_length, self.T, mode, laser, wavelength_signal, wavelength_idler, self.coherence_length, self.w, **kwargs)
        self.poling_pattern = poling_result.poling_pattern
        self.z = poling_result.z
        self.temperature_adjusted_length = poling_result.temperature_adjusted_length
        self.target_amplitude = poling_result.target_amplitude
        self.actual_amplitude = poling_result.actual_amplitude
        return poling_result

    def apply_poling(self, poling_result: PolingResult) -> None:
        """
        Write a (possibly perturbed) ``PolingResult`` back onto this crystal and recompute
        ``target_amplitude``/``actual_amplitude`` for the new pattern.

        ``poling_result`` must carry ``DeltaK``/``coherence_length``/``resolution`` context --
        i.e. it must be (optionally passed through ``PolingResult.add_wall_position_error()``,
        ``.add_missed_domain_error()``, ``.add_duty_cycle_bias()``) the ``PolingResult``
        originally returned by ``self.generate_poling(...)``. That context travels on
        ``poling_result`` itself, set once at generation time, rather than being re-derived from
        laser/wavelength/T arguments passed again here -- this avoids a class of silent-mismatch
        bugs where the caller might otherwise supply different laser/wavelength/T values than
        were used to originally generate the poling.
        """
        if poling_result.DeltaK is None or poling_result.coherence_length is None or poling_result.resolution is None:
            raise ValueError(
                "apply_poling() requires a PolingResult carrying DeltaK/coherence_length/resolution "
                "metadata, as produced by Crystal.generate_poling(...) (optionally passed through "
                "its .add_wall_position_error()/.add_missed_domain_error()/.add_duty_cycle_bias() methods)."
            )

        if poling_result.uniform_width:
            w = poling_result.domain_widths[0] / poling_result.resolution
            target_amplitude, actual_amplitude = self.pm_strategy.compute_domain_field_arrays(
                poling_result.poling_pattern, w, poling_result.coherence_length,
                poling_result.temperature_adjusted_length, poling_result.DeltaK, poling_result.target_profile)
        else:
            target_amplitude, actual_amplitude = self.pm_strategy.compute_domain_field_arrays_nonuniform(
                poling_result.poling_pattern, poling_result.z, poling_result.coherence_length,
                poling_result.temperature_adjusted_length, poling_result.DeltaK, poling_result.target_profile)

        self.poling_pattern = poling_result.poling_pattern
        self.z = poling_result.z
        self.temperature_adjusted_length = poling_result.temperature_adjusted_length
        self.target_amplitude = target_amplitude
        self.actual_amplitude = actual_amplitude
