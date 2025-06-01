import numpy as np
from scipy.optimize import minimize_scalar

from .materials_apm import BaseMaterialAPM
from photonpairlab.laser import CWLaser

class CrystalAPM:
    def __init__(self, Lo: float, material: BaseMaterialAPM, spdc: str = "type-I", phi_deg: float = 0.0):
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
        self.phi_deg = phi_deg  # used for biaxial crystals

        # Constants
        self.nm = 1e-9
        self.um = 1e-6
        self.mm = 1e-3

        # Poling pattern attributes (to be computed)
        self.sarray = None
        self.z = None

    def generate_poling(self, resolution=5):
        """
        Generates a constant poling structure for angle phase-matched crystals.
        This method creates a constant poling structure (e.g., [1, 1, 1, ...]) over the length of the crystal.
        The resolution determines the number of subdivisions per unit length of the crystal.

        Parameters:
            resolution (int, optional): The number of subdivisions per unit length (Lo).
                                        Default is 5.

        Notes:
            - The total length of the z-axis (z) will match the length of the sarray.
        """
        # Calculate the total number of subdivisions based on resolution and crystal length
        Lc = 50e-6
        Lo = self.Lo
        num_domains = int(np.floor(Lo / Lc))
        # Create the polarizations array using np.tile
        polarizations = np.tile([1, 1], num_domains)
        # Create the sarray using np.repeat
        self.sarray = np.repeat(polarizations, resolution)
        self.Lo = num_domains * Lc
        # Calculate z values directly based on the length of sarray
        self.z = np.linspace(-self.Lo / 2, self.Lo / 2, len(self.sarray))

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
        if polarization == "o":
            axis = self.material.map_polarization_axis("o")
            return self.refractive_index(wavelength, axis=axis)
        elif polarization == "e":
            return self.material.effective_refractive_index(wavelength, angle, phi_deg=self.phi_deg)
        else:
            raise ValueError("Polarization must be 'o' or 'e'.")
    
    def get_group_index(self, wavelength, polarization, angle):
        """
        Returns the group index for the given wavelength, polarization, and angle.
        """
        if polarization == "o":
            axis = self.material.map_polarization_axis("o")
            # Use axis-based group index (QPM or propagation along axis)
            return self.material.group_index(wavelength, axis=axis)
        elif polarization == "e":
            # Use angle-based group index (angle phase-matching)
            return self.material.group_index(wavelength, theta_deg=angle, phi_deg=self.phi_deg)
        else:
            raise ValueError("Polarization must be 'o' or 'e'.")

    def find_phase_matching_angle(self, laser=CWLaser(405e-9,bandwidth_wavelength=4.3e-9), wavelength_signal=810e-9, wavelength_idler=810e-9):
        """
        result = minimize_scalar(
            self.delta_k, bounds=(-180, 180), method='bounded',
            args=(wavelength_pump, wavelength_signal, wavelength_idler, 
                    polarization_pump, polarization_signal, polarization_idler))
        """
        result = minimize_scalar(
            lambda angle: abs(self.delta_k(angle, laser, wavelength_signal, wavelength_idler)),
            bounds=(-180, 180),
            method='bounded',
            #options={'xatol': 1e-5}  # Set the absolute tolerance for the solution
        )
        phase_matching_angle = float(result.x)

        return phase_matching_angle
    
    def compute_phase_mismatch(self, laser, wavelength_signal, wavelength_idler, angle_pm=None):
        """
        Compute the phase mismatch Δk = k_p - k_s - k_i for a given laser and wavelengths.
        If no phase-matching angle is provided, it uses the phase-matching angle found by minimizing Δk.

        Args:
            laser (CWLaser): Laser object containing pump wavelength.
            wavelength_signal (float): Signal wavelength in meters.
            wavelength_idler (float): Idler wavelength in meters.
            angle_pm (float, optional): Phase-matching angle in degrees. If None, it will be calculated.

        Returns:
            float: Phase mismatch Δk in μm⁻¹.
        """

        wavelength_pump = laser.wavelength_pump

        if angle_pm is None:
            angle_pm = self.find_phase_matching_angle(laser, wavelength_signal, wavelength_idler)
        
        # Get polarization states based on SPDC type
        pol_pump, pol_signal, pol_idler = self.get_polarizations()

        # Compute refractive indices using effective_refractive_index for angle-based phase matching
        n_pump = self.get_refractive_index(wavelength_pump, pol_pump, angle_pm)
        n_signal = self.get_refractive_index(wavelength_signal, pol_signal, angle_pm)
        n_idler = self.get_refractive_index(wavelength_idler, pol_idler, angle_pm)
        # Compute group indices
        N_pump = self.get_group_index(wavelength_pump, pol_pump, angle_pm)
        N_signal = self.get_group_index(wavelength_signal, pol_signal, angle_pm)
        N_idler = self.get_group_index(wavelength_idler, pol_idler, angle_pm)

        DeltaK_0 = self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler)

        return (n_pump, n_signal, n_idler), (N_pump, N_signal, N_idler), DeltaK_0
        
    
    
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
        wavelength_pump = laser.wavelength_pump * 1e6
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
        print(f"Phase matching angle: {angle_pm}°")
        # Use the delta_k method to calculate Δk
        return self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler)

    
