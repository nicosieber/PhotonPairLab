import numpy as np
from sympy import Symbol, diff
import sympy
import math

from Materials import BaseMaterial

class Crystal:
    def __init__(self, Lc: float, Lo: float, T: float, w: float, material: BaseMaterial):
        """
        Initializes a Crystal object with its physical and material properties.
        Args:
            Lc (float): Coherence length of the crystal in meters.
            Lo (float): Physical length of the crystal in meters.
            T (float): Temperature of the crystal in degrees Celsius.
            w (float): Domain width parameter in meters.
            material (BaseMaterial): Material database object containing the crystal's material properties.
        Attributes:
            Lc (float): Coherence length of the crystal.
            Lo (float): Physical length of the crystal.
            T (float): Temperature of the crystal.
            w (float): Domain width parameter.
            material (BaseMaterial): Material database object.
            nm (float): Conversion factor for nanometers to meters (1e-9).
            um (float): Conversion factor for micrometers to meters (1e-6).
            mm (float): Conversion factor for millimeters to meters (1e-3).
            L (float): Temperature-expanded length of the crystal, calculated using a quadratic expansion formula.
            sarray (None or array-like): Poling pattern attribute to be computed.
            atarray (None or array-like): Poling pattern attribute to be computed.
            amuparray (None or array-like): Poling pattern attribute to be computed.
            amdownarray (None or array-like): Poling pattern attribute to be computed.
            altered_z (None or array-like): Poling pattern attribute to be computed.
            z (None or array-like): Poling pattern attribute to be computed.
        """
        
        self.Lc = Lc        # Coherence length (m)
        self.Lo = Lo        # Physical length of the crystal (m)
        self.T = T          # Temperature (°C)
        self.w = w          # Domain width parameter (m)
        self.material = material  # Materials database object

        # Constants
        self.nm = 1e-9
        self.um = 1e-6
        self.mm = 1e-3

        # Temperature Expansion of crystal
        # (S. Emanueli & A. Arie, App. Opt, vol. 42, No. 33 (2003))
        self.L = self.Lo * (1 + 6.7e-6 * (self.T - 25) + 11e-9 * (self.T - 25) ** 2)

        # Poling pattern attributes (to be computed)
        self.sarray = None
        self.atarray = None
        self.amuparray = None
        self.amdownarray = None
        self.altered_z = None
        self.z = None

    def refractive_index(self, wavelength, axis):
        """
        Delegate refractive index calculation to the material object.
        """
        return self.material.refractive_index(wavelength, axis, temperature=self.T)

    def group_index(self, wavelength, axis):
        """
        Delegate group index calculation to the material object.
        """
        return self.material.group_index(wavelength, axis, temperature=self.T)

    def gtarget(self, z, L, lc):
        """
        Computes a Gaussian target function based on the given parameters.

        Parameters:
            z (float): The position variable.
            L (float): The length parameter, typically representing the crystal length.
            lc (float): Unused parameter in the current implementation.

        Returns:
            float: The value of the Gaussian function evaluated at the given position `z`.
        """
        return np.exp(-((z - L / 2) ** 2) / (L ** 2 / 8)) # L**2 is divided by 8 as suggested by the reference

    def Atarget(self, w, m, L, Lc, DeltaK):
        """
        Computes the target amplitude for a given set of parameters.

        Parameters:
            w (float): The width parameter, typically related to the beam or crystal dimensions.
            m (int): The number of segments or divisions for the integration range.
            L (float): The crystal length or a related parameter.
            Lc (float): The coherence length of the crystal.
            DeltaK (float): The phase mismatch parameter.

        Returns:
            complex: The computed target amplitude as a complex number.
        """
        z = np.linspace(0, m * w, num=m)
        g = self.gtarget(z, L, Lc / 2)
        cos_term = np.cos(np.pi / (Lc / 2) * z)
        exp_term = np.exp(1j * DeltaK * z)
        y = g * cos_term * exp_term
        return -1j * np.trapz(y, z)

    def Am(self, w, altered_z, m, Lc, sn):
        """
        Computes the amplitude modulation function Am for a given set of parameters.

        Parameters:
            w (float): The angular frequency variable.
            altered_z (float): The altered spatial coordinate.
            m (int): The number of poling periods.
            Lc (float): The coherence length of the crystal.
            sn (array-like): The poling array, which must have a length equal to `m`.

        Returns:
            complex: The computed amplitude modulation value.

        Raises:
            ValueError: If the length of the poling array `sn` is not equal to `m`.
        """
        if len(sn) != m:
            raise ValueError("Poling array length wrong.")
        exp_term = np.exp(1j * np.pi / (Lc / 2) * altered_z)
        y = np.sum(sn * exp_term)
        return Lc / (2 * np.pi) * (np.exp(-1j * np.pi / (Lc / 2) * w) - 1) * y

    def generate_poling(self,laser,mode='periodic',resolution=5):
        """
        Generates the poling configuration for the crystal based on the specified mode.

        Parameters:
            laser: object
                The laser object containing relevant parameters for custom poling.
            mode: str, optional
                The mode of poling to generate. Options are:
                - 'periodic': Generates a periodic poling configuration.
                - 'custom': Generates a custom poling configuration based on the laser parameters.
                Default is 'periodic'.
            resolution: int, optional
                The resolution parameter for periodic poling. Higher values result in finer resolution.
                Default is 5.

        Raises:
            ValueError: If an unsupported mode is specified.
        """
        if mode == 'periodic':
            self.generate_periodic_poling(resolution=resolution)
        elif mode == 'custom':
            self.generate_custom_poling(laser)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

    def generate_periodic_poling(self, resolution=5):
        """
        Generates a periodic poling structure for the crystal.
        This method creates a periodic poling structure by alternating polarizations
        (e.g., [1, -1, 1, -1, ...]) over the length of the crystal. The resolution
        determines the number of subdivisions per coherence length (Lc). The method
        also adjusts the crystal length (Lo) to be an integer multiple of the coherence
        length and calculates the corresponding z-axis values.
        Parameters:
            resolution (int, optional): The number of subdivisions per coherence length.
                                        Default is 5.
        Attributes Modified:
            sarray (numpy.ndarray): The array representing the periodic poling structure
                                    with alternating polarizations.
            Lo (float): Adjusted length of the crystal to be an integer multiple of Lc.
            z (numpy.ndarray): The z-axis values corresponding to the periodic poling structure.
        Notes:
            - The coherence length (Lc) and original crystal length (Lo) must be defined
              as attributes of the class before calling this method.
            - The total length of the z-axis (z) will match the length of the sarray.
        """
        
        Lc = self.Lc
        Lo = self.Lo
        num_domains = int(np.floor(Lo / Lc))
        # Create the polarizations array using np.tile
        polarizations = np.tile([1, -1], num_domains)
        # Create the sarray using np.repeat
        self.sarray = np.repeat(polarizations, resolution)
        # Adjust Lo to be an integer multiple of Lc
        self.Lo = num_domains * Lc
        # Calculate z values directly based on the length of sarray
        self.z = np.linspace(-self.L / 2, self.L / 2, len(self.sarray))

    def generate_custom_poling(self, laser):
        """
        Generates a custom poling pattern for the nonlinear crystal based on the input laser parameters.

        This method computes the poling pattern by iteratively determining the orientation of the 
        nonlinear domains (up or down) that minimizes the error between the target amplitude and 
        the computed amplitude. The resulting poling pattern is stored in the `sarray` attribute.

        Reference for this method:
            Sub-coherence length apodization algorithm according to
            Quantum Sci. Technol. 2 (2017)035001 (https://doi.org/10.1088/2058-9565/aa78d4)
            Pure down-conversion photons through sub-coherence-length domain engineering
            Francesco Graffitti, Dmytro Kundys, Derryck T Reid, AgataMBrańczyk
            and Alessandro Fedrizzi.

        Parameters:
            laser (object): An object representing the laser, which must have the following attributes:
                - lambda_w (float): Central wavelength of the fundamental wave (in meters).
                - lambda_2w (float): Central wavelength of the second harmonic wave (in meters).

        Attributes:
            sarray (numpy.ndarray): Array representing the poling pattern (1 for "up", -1 for "down").
            atarray (numpy.ndarray): Array of target amplitudes for each iteration.
            amuparray (numpy.ndarray): Array of computed amplitudes for "up" orientation.
            amdownarray (numpy.ndarray): Array of computed amplitudes for "down" orientation.
            altered_z (numpy.ndarray): Array of altered z-coordinates used in the computation.
            z (numpy.ndarray): Array of z-coordinates shifted by half the crystal length.

        Notes:
            - The method uses the refractive indices and group indices of the crystal at the 
              fundamental and second harmonic wavelengths to compute the phase mismatch (DeltaK_0).
            - The apodization algorithm is applied iteratively to determine the optimal poling pattern.
            - The method assumes that the crystal parameters (e.g., `w`, `L`, `Lc`) 
              are already defined as attributes of the class.
        """
        # Compute refractive indices at central wavelengths
        lambda_w = laser.lambda_w
        lambda_2w = laser.lambda_2w

        nyD = self.refractive_index(lambda_w * 1e6, "y")
        nzD = self.refractive_index(lambda_w * 1e6, "z")
        nyP = self.refractive_index(lambda_2w * 1e6, "y")

        # Compute group indices
        NyP = self.group_index(lambda_2w * 1e6, "y")
        NzD = self.group_index(lambda_w * 1e6, "z")
        NyD = self.group_index(lambda_w * 1e6, "y")
        self.Nmean = (NzD + NyD) / 2.0

        # Compute DeltaK_0
        self.DeltaK_0 = 2 * np.pi * (nyP / lambda_2w - nzD / lambda_w - nyD / lambda_w)

        # Proceed with the apodization algorithm using self.DeltaK_0
        w = self.w
        mstart = 2
        L = self.L
        Lc = self.Lc
        DeltaK = self.DeltaK_0

        num_iterations = math.ceil(L / w) + 1  # Total number of iterations

        # Precompute altered_z
        altered_z = np.linspace(0, num_iterations * w, num_iterations + 1)
        # Initialize sarray
        sarray = np.zeros(num_iterations + 1, dtype=int)
        sarray[0] = -1
        atarray = np.zeros(num_iterations, dtype=complex)
        amuparray = np.zeros(num_iterations, dtype=complex)
        amdownarray = np.zeros(num_iterations, dtype=complex)

        for idx in range(num_iterations):
            m = mstart + idx

            # Compute Atarget once per iteration
            at = self.Atarget(w, m, L, Lc, DeltaK)

            # Test with sarray[idx + 1] = 1 (up)
            sarray[idx + 1] = 1
            amup = self.Am(w, altered_z[: idx + 2], m, Lc, sarray[: idx + 2])

            # Test with sarray[idx + 1] = -1 (down)
            sarray[idx + 1] = -1
            amdown = self.Am(w, altered_z[: idx + 2], m, Lc, sarray[: idx + 2])

            # Compute errors
            eup = np.abs(at - amup)
            edown = np.abs(at - amdown)

            # Store results
            atarray[idx] = at
            amuparray[idx] = amup
            amdownarray[idx] = amdown

            # Decide which orientation minimizes the error
            if eup < edown:
                sarray[idx + 1] = 1  # Keep 'up' orientation
            else:
                sarray[idx + 1] = -1  # Keep 'down' orientation

        self.sarray = sarray
        self.atarray = atarray
        self.amuparray = amuparray
        self.amdownarray = amdownarray
        self.altered_z = altered_z
        self.z = altered_z - L / 2