import numpy as np
from .base_pm_strategy import PhaseMatchingStrategy
from ..material.base_material import BaseMaterial
from photonpairlab.laser import *

from scipy.optimize import minimize_scalar

class QPMPhaseMatching(PhaseMatchingStrategy):
    """
    Quasi Phase-Matching (QPM) strategy for nonlinear crystals.
    """

    def __init__(self, material: BaseMaterial, spdc_type: str = "type-II", phi_deg: float = 0.0):
        super().__init__(material, spdc_type)
        self.phi_deg = phi_deg  # Used for biaxial crystals

    def compute_phase_mismatch(self, laser: BaseLaser, 
                               wavelength_signal: float, 
                               wavelength_idler: float, 
                               angle_pm: float = 0,
                               T: float = 25.0):
        """
        Computes the phase mismatch (DeltaK_0) based on the SPDC type.

        """

        wavelength_pump = laser.wavelength_pump
      
        # Get polarization states based on SPDC type
        pol_pump, pol_signal, pol_idler = self.get_polarizations()

        # Compute refractive indices using effective_refractive_index for angle-based phase matching
        n_pump = self.get_refractive_index(wavelength_pump * 1e6, pol_pump, angle_pm, T)
        n_signal = self.get_refractive_index(wavelength_signal * 1e6, pol_signal, angle_pm, T)
        n_idler = self.get_refractive_index(wavelength_idler * 1e6, pol_idler, angle_pm, T)
        # Compute group indices
        N_pump = self.get_group_index(wavelength_pump * 1e6, pol_pump, angle_pm, T)
        N_signal = self.get_group_index(wavelength_signal * 1e6, pol_signal, angle_pm, T)
        N_idler = self.get_group_index(wavelength_idler * 1e6, pol_idler, angle_pm, T)

        DeltaK_0 = self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler, T)

        return (n_pump, n_signal, n_idler), (N_pump, N_signal, N_idler), DeltaK_0, angle_pm
    

    def generate_poling(self, crystal_length: float, 
                        T: float, 
                        mode: str, 
                        laser: BaseLaser,
                        wavelength_signal: float = None,
                        wavelength_idler: float = None,
                        coherence_length: float = None, 
                        w: float = None,
                        resolution: int = 5):

        if mode == 'periodic':
            return self._generate_periodic_poling(crystal_length, T, coherence_length, resolution)
        elif mode == 'constant':
            return self._generate_constant_poling(crystal_length, T, coherence_length, resolution)
        elif mode == 'subcoh':
            return self._generate_subcoh_poling(laser, wavelength_signal, wavelength_idler, crystal_length, w, coherence_length, T)
        else:
            raise ValueError(f"Unknown poling mode: {mode}. Use 'periodic' or 'constant'.")
        

    def _generate_periodic_poling(self, crystal_length: float, 
                                  T: float, 
                                  coherence_length: float,
                                  resolution: int = 5):
        """
        Generates a periodic poling structure for the crystal.
        This method creates a periodic poling structure by alternating polarizations
        (e.g., [1, -1, 1, -1, ...]) over the length of the crystal. The resolution
        determines the number of subdivisions per coherence length (coherence_length). The method
        also adjusts the crystal length (crystal_length) to be an integer multiple of the coherence
        length and calculates the corresponding z-axis values.
        Parameters:
            resolution (int, optional): The number of subdivisions per coherence length.
                                        Default is 5.
        Notes:
            - The coherence length (coherence_length) and original crystal length (crystal_length) must be defined
              as attributes of the class before calling this method.
            - The total length of the z-axis (z) will match the length of the poling_pattern.
        """
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        num_domains = int(np.floor(crystal_length / coherence_length))
        # Create the polarizations array using np.tile
        polarizations = np.tile([1, -1], num_domains)
        # Create the poling pattern using np.repeat
        poling_pattern = np.repeat(polarizations, resolution)
        # Adjust crystal_length to be an integer multiple of coherence_length
        crystal_length = num_domains * coherence_length
        # Calculate z values directly based on the length of poling_pattern
        z = np.linspace(-temperature_adjusted_length / 2,
                        temperature_adjusted_length / 2,
                        len(poling_pattern))
        return poling_pattern, z, temperature_adjusted_length
    
    def _generate_constant_poling(self, crystal_length, T, coherence_length, resolution=5):
        """
        Generates a constant poling structure for the crystal.
        This method creates a constant poling structure by using the same polarization
        (e.g., [1, 1, 1, 1, ...]) over the length of the crystal. The resolution
        determines the number of subdivisions per coherence length (coherence_length). The method
        also adjusts the crystal length (crystal_length) to be an integer multiple of the coherence
        length and calculates the corresponding z-axis values.
        Parameters:
            resolution (int, optional): The number of subdivisions per coherence length.
                                        Default is 5.
        Notes:
            - The coherence length (coherence_length) and original crystal length (crystal_length) must be defined
              as attributes of the class before calling this method.
            - The total length of the z-axis (z) will match the length of the poling_pattern.
        """
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        num_domains = int(np.floor(crystal_length / coherence_length))
        # Create the polarizations array using np.tile
        polarizations = np.tile([1, 1], num_domains)
        # Create the poling pattern using np.repeat
        poling_pattern = np.repeat(polarizations, resolution)
        # Adjust crystal_length to be an integer multiple of coherence_length
        crystal_length = num_domains * coherence_length
        # Calculate z values directly based on the length of poling_pattern
        z = np.linspace(-temperature_adjusted_length / 2,
                        temperature_adjusted_length / 2,
                        len(poling_pattern))
        return poling_pattern, z, temperature_adjusted_length

    def _generate_subcoh_poling(self, laser: BaseLaser,
                                wavelength_signal: float,
                                wavelength_idler: float,
                                crystal_length: float,
                                w: float,
                                coherence_length: float,
                                T: float
                                ):
        """
        Generates a sub-coherence length apodized poling pattern for the nonlinear crystal 
        based on the input laser parameters.

        This method computes the poling pattern by iteratively determining the orientation of the 
        nonlinear domains (up or down) that minimizes the error between the target amplitude and 
        the computed amplitude. The resulting poling pattern is stored in the `poling_pattern` attribute, 
        along with additional computed attributes such as `target_amplitudes`, `amuparray`, `amdownarray`, 
        and `altered_z`.

        The algorithm follows the sub-coherence length domain engineering approach, which optimizes 
        the poling pattern to achieve pure down-conversion photons. It uses the refractive indices 
        and group indices of the crystal at the fundamental and second harmonic wavelengths to 
        compute the phase mismatch (DeltaK_0), and applies an iterative apodization algorithm to 
        determine the optimal poling configuration.

        Reference:
            Sub-coherence length apodization algorithm according to:
            Quantum Sci. Technol. 2 (2017) 035001 (https://doi.org/10.1088/2058-9565/aa78d4)
            "Pure down-conversion photons through sub-coherence-length domain engineering"
            Francesco Graffitti, Dmytro Kundys, Derryck T Reid, Agata M Brańczyk, 
            and Alessandro Fedrizzi.

        Notes:
            - The method assumes that the crystal parameters (e.g., `w`, `L`, `coherence_length`) 
              are already defined as attributes of the class.
            - The generated poling pattern (`sarray`) and other computed attributes are stored 
              as class attributes for further use.
        """
        
        # Proceed with the apodization algorithm using self.DeltaK_0

        mstart = 2
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        DeltaK = self.delta_k(angle=None, laser=laser, 
                              wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, T=T)

        num_iterations = int(np.ceil(temperature_adjusted_length / w)) + 1 # Total number of iterations

        # Precompute altered_z
        altered_z = np.linspace(0, num_iterations * w, num_iterations + 1)
        # Initialize poling_pattern
        poling_pattern = np.zeros(num_iterations + 1, dtype=int)
        poling_pattern[0] = -1
        target_amplitudes = np.zeros(num_iterations, dtype=complex)
        amuparray = np.zeros(num_iterations, dtype=complex)
        amdownarray = np.zeros(num_iterations, dtype=complex)

        for idx in range(num_iterations):
            m = mstart + idx

            # Compute target_amplitude once per iteration
            at = self.target_amplitude(w, m, temperature_adjusted_length, coherence_length, DeltaK)

            # Test with poling_pattern[idx + 1] = 1 (up)
            poling_pattern[idx + 1] = 1
            amup = self.Am(w, altered_z[: idx + 2], m, coherence_length, poling_pattern[: idx + 2])

            # Test with poling_pattern[idx + 1] = -1 (down)
            poling_pattern[idx + 1] = -1
            amdown = self.Am(w, altered_z[: idx + 2], m, coherence_length, poling_pattern[: idx + 2])

            # Compute errors
            eup = np.abs(at - amup)
            edown = np.abs(at - amdown)

            # Store results
            target_amplitudes[idx] = at
            amuparray[idx] = amup
            amdownarray[idx] = amdown

            # Decide which orientation minimizes the error
            if eup < edown:
                poling_pattern[idx + 1] = 1  # Keep 'up' orientation
            else:
                poling_pattern[idx + 1] = -1  # Keep 'down' orientation

        poling_pattern = poling_pattern
        target_amplitudes = target_amplitudes
        amuparray = amuparray
        amdownarray = amdownarray
        altered_z = altered_z
        z = altered_z - temperature_adjusted_length / 2
        return poling_pattern, z, temperature_adjusted_length

    def gtarget(self, z, L, coherence_length):
        """
        Computes a Gaussian target function based on the given parameters.

        """
        return np.exp(-((z - L / 2) ** 2) / (L ** 2 / 8)) # L**2 is divided by 8 as suggested by the reference

    def target_amplitude(self, w, m, L, coherence_length, DeltaK):
        """
        Computes the target amplitude for a given set of parameters.
        
        """
        z = np.linspace(0, m * w, num=m)
        g = self.gtarget(z, L, coherence_length / 2)
        cos_term = np.cos(np.pi / (coherence_length / 2) * z)
        exp_term = np.exp(1j * DeltaK * z)
        y = g * cos_term * exp_term
        return -1j * np.trapz(y, z)

    def Am(self, w, altered_z, m, coherence_length, sn):
        """
        Computes the amplitude modulation function Am for a given set of parameters.

        """
        if len(sn) != m:
            raise ValueError("Poling array length wrong.")
        exp_term = np.exp(1j * np.pi / (coherence_length / 2) * altered_z)
        y = np.sum(sn * exp_term)
        return coherence_length / (2 * np.pi) * (np.exp(-1j * np.pi / (coherence_length / 2) * w) - 1) * y
