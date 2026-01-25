import numpy as np

from ..material.base_material import BaseMaterial
from photonpairlab.laser import *

POLARIZATION_MAP: dict[str, tuple[str, str, str]] = {
    "type-0": ("e", "e", "e"),
    "type-I": ("e", "o", "o"),
    "type-II": ("y", "z", "y"),
    "type-IIeoe": ("e", "o", "e"),
    "type-IIoeo": ("o", "e", "o"),  
}

class PhaseMatchingStrategy:
    """
    Base class for phase-matching strategies (QPM, APM, ...).
    Provides the interface and common placeholders for all strategies.
    """

    def __init__(self, material: BaseMaterial, spdc_type: str="type-II", coherence_length: float | None = None):
        self.material = material
        self.spdc_type = spdc_type
        self.coherence_length = coherence_length

    def get_refractive_index(self, wavelength: float, polarization: str, angle: float | None, T: float | None):
        if polarization == "o":
            axis = self.material.map_polarization_axis("o")
            return self.material.refractive_index(wavelength, axis=axis, temperature=T)
        elif polarization == "e":
            return self.material.effective_refractive_index(wavelength, theta_deg=angle, phi_deg=self.phi_deg) # type: ignore
        else:
            axis = self.material.map_polarization_axis(polarization)
            return self.material.refractive_index(wavelength, axis=axis, temperature=T)

    def get_group_index(self, wavelength: float, polarization: str, angle: float, T: float):
        """
        Returns the group index for the given wavelength, polarization, and angle.
        """
        if polarization == "o":
            axis = self.material.map_polarization_axis("o")
            # Use axis-based group index (QPM or propagation along axis)
            return self.material.group_index(wavelength, axis=axis, temperature=T)
        elif polarization == "e":
            # Use angle-based group index (angle phase-matching)
            return self.material.group_index(wavelength, theta_deg=angle, phi_deg=self.phi_deg) # type: ignore
        else:
            axis = self.material.map_polarization_axis(polarization)
            return self.material.group_index(wavelength, axis=axis, temperature=T)
        
    def get_polarizations(self):
        """
        Return (pol_pump, pol_signal, pol_idler) for the current SPDC type.
        """
        try:
            return POLARIZATION_MAP[self.spdc_type]
        except KeyError as e:
            raise ValueError(f"Invalid SPDC type: {self.spdc_type!r}") from e

    def compute_phase_mismatch(self, *args, **kwargs):
        """
        Compute phase mismatch Δk for the given parameters. 
        This method is used in the simulation.py of the spdc module.
        """
        raise NotImplementedError("Implement in subclass.")

    def delta_k(self, angle: float | None, laser: BaseLaser, 
                wavelength_signal: float | None, wavelength_idler: float | None, T: float | None):
        """
        Calculate Δk = k_p - k_s - k_i for a given angle.

        Args:
            angle_deg (float): Phase-matching angle in degrees.
            laser (CWLaser): Laser object containing pump wavelength.
            wavelength_signal (float): Signal wavelength in meters.
            wavelength_idler (float): Idler wavelength in meters.

        Returns:
            float: Phase mismatch Δk in μm⁻¹.
        """
        # Convert wavelengths to micrometers
        wavelength_pump = laser.wavelength_pump * 1e6
        if wavelength_signal is None or wavelength_idler is None:
            raise ValueError("Both wavelength_signal and wavelength_idler must be provided.")
        else:
            wavelength_signal = wavelength_signal * 1e6
            wavelength_idler = wavelength_idler * 1e6

        # Get polarization states based on SPDC type
        polarzation_pump, polarzation_signal, polarzation_idler = self.get_polarizations()
        
        # Compute refractive indices
        n_p = self.get_refractive_index(wavelength_pump, polarzation_pump, angle, T)
        n_s = self.get_refractive_index(wavelength_signal, polarzation_signal, angle, T)
        n_i = self.get_refractive_index(wavelength_idler, polarzation_idler, angle, T)

        # Compute wavevectors
        k_p = 2 * np.pi * n_p / wavelength_pump
        k_s = 2 * np.pi * n_s / wavelength_signal
        k_i = 2 * np.pi * n_i / wavelength_idler

        return (k_p - k_s - k_i) * 1e6  # Δk in μm⁻¹

    def generate_poling(self, *args, **kwargs):
        """
        Generate poling pattern (periodic, sub-coherence, constant, etc.).
        """
        raise NotImplementedError("Implement in subclass.")