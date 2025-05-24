import numpy as np
from scipy.optimize import minimize_scalar

from .materials_apm import BaseMaterialAPM
from photonpairlab.laser import CWLaser

class CrystalAPM:
    def __init__(self, Lo: float, material: BaseMaterialAPM, spdc: str = "type-I"):
        """
        Initialize the CrystalAPM class.

        Args:
            Lo (float): Pump wavelength in micrometers.
            material (BaseMaterialAPM): Material object containing Sellmeier coefficients.
            spdc (str): Type of SPDC process ('type-I' or 'type-II').
        """
        self.Lo = Lo
        self.material = material
        self.spdc = spdc

        # Constants
        self.nm = 1e-9
        self.um = 1e-6
        self.mm = 1e-3

    def refractive_index(self, wavelength, axis):
        """
        Delegate refractive index calculation to the material object.
        """
        return self.material.refractive_index(wavelength, axis)
    
    def get_polarizations(self):
        """
        Determine the polarization states based on the SPDC type.

        Returns:
            tuple: (pol_p, pol_s, pol_i) for pump, signal, and idler polarizations.
        """
        if self.spdc == 'type-0':
            return 'e', 'e', 'e'
        elif self.spdc == 'type-I':
            return 'e', 'o', 'o'
        elif self.spdc == 'type-IIeoe':
            return 'e', 'o', 'e'
        elif self.spdc == 'type-IIoeo':
            return 'o', 'e', 'o'
        else:
            raise ValueError("Invalid SPDC type.")


    def get_refractive_index(self, wavelength, polarization, angle):
        """
        Get the refractive index for a given wavelength, polarization, and angle.

        Args:
            wavelength (float): Wavelength in micrometers.
            polarization (str): Polarization ('o' or 'e').
            angle (float): Angle in degrees.

        Returns:
            float: Refractive index.
        """
        if polarization == "o":
            return self.refractive_index(wavelength, axis="o")
        elif polarization == "e":
            return self.effective_refractive_index(wavelength, angle)
        else:
            raise ValueError("Polarization must be 'o' or 'e'.")

    def effective_refractive_index(self, wavelength, angle):
        """
        Calculate the effective extraordinary refractive index at a given angle.

        Args:
            lambda_um (float): Wavelength in micrometers.
            theta_deg (float): Angle in degrees.
        Returns:
            float: Effective extraordinary refractive index.
        """
        theta_rad = np.radians(angle)
        no = self.refractive_index(wavelength, axis="o")
        ne = self.refractive_index(wavelength, axis="e")
        return 1 / np.sqrt((np.cos(theta_rad)**2 / ne**2) + (np.sin(theta_rad)**2 / no**2))
    
    def find_phase_matching_angle(self, laser=CWLaser(405e-9,bandwidth_wavelength=4.3e-9), wavelength_signal=810e-9, wavelength_idler=810e-9):
        """
        result = minimize_scalar(
            self.delta_k, bounds=(0, 90), method='bounded',
            args=(wavelength_pump, wavelength_signal, wavelength_idler, 
                    polarization_pump, polarization_signal, polarization_idler))
        """
        result = minimize_scalar(
            lambda angle: abs(self.delta_k(angle, laser, wavelength_signal, wavelength_idler)),
            bounds=(0, 90),
            method='bounded'
        )
        phase_matching_angle = float(result.x)

        return phase_matching_angle
    
    
    def delta_k(self, angle, laser, wavelength_signal, wavelength_idler):
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
        wavelength_pump = laser.lambda_2w * 1e6
        wavelength_signal = wavelength_signal * 1e6
        wavelength_idler = wavelength_idler * 1e6

        # Get polarization states based on SPDC type
        polarzation_pump, polarzation_signal, polarzation_idler = self.get_polarizations()
        
        # Compute refractive indices
        n_p = self.get_refractive_index(wavelength_pump, polarzation_pump, angle)
        n_s = self.get_refractive_index(wavelength_signal, polarzation_signal, angle)
        n_i = self.get_refractive_index(wavelength_idler, polarzation_idler, angle)

        # Compute wavevectors
        k_p = 2 * np.pi * n_p / wavelength_pump
        k_s = 2 * np.pi * n_s / wavelength_signal
        k_i = 2 * np.pi * n_i / wavelength_idler

        return (k_p - k_s - k_i) * 1e6  # Δk in μm⁻¹


    def compute_delta_K(self, laser=CWLaser(405e-9, bandwidth_wavelength=4.3e-9),
                        wavelength_signal=810e-9, wavelength_idler=810e-9,
                        angle_pm=None):
        """
        Calculate Δk = k_p - k_s - k_i. If no phase-matching angle is provided, it uses the
        phase-matching angle found by minimizing Δk.

        Returns:
            float: Phase mismatch Δk in μm⁻¹.
        """
        # If angle is not given, calculate phase matching angle
        if angle_pm is None:
            angle_pm = self.find_phase_matching_angle(laser, wavelength_signal, wavelength_idler)

        # Use the delta_k method to calculate Δk
        return self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler)

    
