import numpy as np
from scipy.optimize import minimize_scalar

from .base_pm_strategy import PhaseMatchingStrategy
from .pm_result import PhaseMismatchResult
from ..material.base_material import BaseMaterial
from photonpairlab.laser import *

class APMPhaseMatching(PhaseMatchingStrategy):
    """
    Angle Phase-Matching (APM) strategy for nonlinear crystals.
    """

    def __init__(self, material: BaseMaterial, spdc_type: str = "type-II", phi_deg: float = 0.0):
        super().__init__(material, spdc_type)
        self.phi_deg = phi_deg  # Used for biaxial crystals    

    def compute_phase_mismatch(self, laser: BaseLaser, 
                               wavelength_signal: float, 
                               wavelength_idler: float, 
                               angle_pm: float | None = None,
                               T: float = 25.0):
        """
        Compute the phase mismatch Δk = k_p - k_s - k_i for a given laser and wavelengths.
        If no phase-matching angle is provided, it uses the phase-matching angle found by minimizing Δk.

        """

        wavelength_pump = laser.wavelength_pump

        if angle_pm is None:
            angle_pm = self.find_phase_matching_angle(laser, wavelength_signal, wavelength_idler)
        
        # Get polarization states based on SPDC type
        pol_pump, pol_signal, pol_idler = self.get_polarizations()

        # Compute refractive indices using effective_refractive_index for angle-based phase matching
        n_pump = self.get_refractive_index(wavelength_pump, pol_pump, angle_pm, T)
        n_signal = self.get_refractive_index(wavelength_signal, pol_signal, angle_pm, T)
        n_idler = self.get_refractive_index(wavelength_idler, pol_idler, angle_pm, T)
        # Compute group indices
        N_pump = self.get_group_index(wavelength_pump, pol_pump, angle_pm, T)
        N_signal = self.get_group_index(wavelength_signal, pol_signal, angle_pm, T)
        N_idler = self.get_group_index(wavelength_idler, pol_idler, angle_pm, T)

        DeltaK_0 = self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler,T)
        return PhaseMismatchResult(
            n=(n_pump, n_signal, n_idler),
            N=(N_pump, N_signal, N_idler),
            delta_k0=DeltaK_0,
            angle_pm=angle_pm,
            coherence_length=self.coherence_length,
        )


    def generate_poling(self, crystal_length: float, 
                        T: float, 
                        mode: str, 
                        laser: BaseLaser,
                        wavelength_signal: float | None = None,
                        wavelength_idler: float | None = None,
                        coherence_length: float | None = None, 
                        w: float | None = None,
                        resolution: int = 5):

        if mode == "constant":
            return self._generate_constant_poling(crystal_length, T, resolution)
        else:
            raise ValueError(f"Unknown poling mode: {mode}. Only 'constant' is supported for APM.")

    def _generate_constant_poling(self, crystal_length: float, T: float, resolution: int = 5):
        """
        Generates a constant poling structure for angle phase-matched crystals.
        
        """
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        # For APM, poling is typically constant (no periodic poling)
        Lc = 50e-6  # Example coherence length, adjust as needed
        num_domains = int(np.floor(crystal_length / Lc))
        polarizations = np.tile([1, 1], num_domains)
        poling_pattern = np.repeat(polarizations, resolution)
        z = np.linspace(-temperature_adjusted_length / 2, temperature_adjusted_length / 2, len(poling_pattern))
        return poling_pattern, z, temperature_adjusted_length

    def find_phase_matching_angle(self, laser: BaseLaser=CWLaser(405e-9,bandwidth_wavelength=4.3e-9),
                                  wavelength_signal: float=810e-9, wavelength_idler: float=810e-9, T: float | None=None):
        """
        result = minimize_scalar(
            self.delta_k, bounds=(-180, 180), method='bounded',
            args=(wavelength_pump, wavelength_signal, wavelength_idler, 
                    polarization_pump, polarization_signal, polarization_idler))
        """
        result = minimize_scalar(
            lambda angle: abs(self.delta_k(angle, laser, wavelength_signal, wavelength_idler, T)),
            bounds=(0, 90),
            method='bounded',
            #options={'xatol': 1e-5}  # Set the absolute tolerance for the solution
        )
        phase_matching_angle = float(result.x) # type: ignore

        return phase_matching_angle

