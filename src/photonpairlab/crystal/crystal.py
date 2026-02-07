from .material.base_material import BaseMaterial
from .phasematching.qpm_strategy import QPMPhaseMatching
from .phasematching.apm_strategy import APMPhaseMatching
from photonpairlab.laser import *

PM_STRATEGY_HANDLER = {
    "quasi": QPMPhaseMatching,
    "angle": APMPhaseMatching,
}


class Crystal:
    """
    Unified Crystal class that binds a nonlinear optical material with a phase-matching strategy
    (Angle Phase Matching or Quasi Phase Matching).

    Attributes:
        material (BaseMaterial): Nonlinear optical material instance.
        spdc_type (str): SPDC interaction type (e.g., "type-I", "type-II").
        pm_strategy: Instance of the phase-matching strategy.
    """

    def __init__(self, crystal_length: float, material: BaseMaterial, 
                 coherence_length: float | None = None, T: float = 25.0, pm_strategy: str ="quasi", 
                 spdc_type: str ="type-IIoeo", phi_deg: float = 0, w: float | None = None, **kwargs):
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
        except KeyError:
            raise KeyError(f"Unknown phase-matching strategy: {pm_strategy}. Valid options are: {list(PM_STRATEGY_HANDLER.keys())}")

        # Temperature Expansion of crystal
        try:
            self.temperature_adjusted_crystal_length = self.material.thermal_expansion(length=self.crystal_length, axis="z", temperature=self.T)
        except ValueError as e:
            raise ValueError(f"Error in thermal expansion: {e}")

        self.poling_pattern: np.ndarray | None = None
        self.z: np.ndarray | None = None
        self.temperature_adjusted_length: float | None = None

    def compute_phase_mismatch(self, *args, **kwargs):
        """Calculates the phase mismatch using the selected phase-matching strategy."""
        return self.pm_strategy.compute_phase_mismatch(*args, **kwargs)

    def delta_k(self, *args, **kwargs):
        """Calculates the phase mismatch using the selected phase-matching strategy."""
        return self.pm_strategy.delta_k(*args, **kwargs)


    def generate_poling(self, laser: BaseLaser, mode: str, wavelength_signal: float, wavelength_idler:float, **kwargs):
        """Generates the poling pattern based on the selected phase-matching strategy."""
        self.poling_pattern, self.z, self.temperature_adjusted_length = self.pm_strategy.generate_poling(self.crystal_length, self.T, mode, laser, wavelength_signal, wavelength_idler, self.coherence_length, self.w, **kwargs)
