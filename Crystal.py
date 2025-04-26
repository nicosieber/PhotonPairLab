import numpy as np
from sympy import Symbol, diff
import sympy
import math

class Crystal:
    """
    Represents the properties of the nonlinear crystal used in the simulation.

    For now, the only available material is periodically poled KTP.
    """
    def __init__(self, Lc: float, Lo: float, T: float, w: float, mstart: int, materials_db):
        """
        Initializes the Crystal object with its physical and operational parameters.

        Parameters:
            Lc (float): Coherence length of the crystal in meters.
            Lo (float): Physical length of the crystal in meters.
            T (float): Temperature of the crystal in degrees Celsius.
            w (float): Domain width parameter in meters.
            mstart (int): Starting index for the apodization algorithm.

        Attributes:
            Lc (float): Coherence length (m).
            Lo (float): Physical length of the crystal (m).
            T (float): Temperature (°C).
            w (float): Domain width parameter (m).
            mstart (int): Starting index for the apodization algorithm.
            nm (float): Conversion factor for nanometers to meters (1e-9).
            um (float): Conversion factor for micrometers to meters (1e-6).
            mm (float): Conversion factor for millimeters to meters (1e-3).
            L (float): Temperature-expanded physical length of the crystal (m).
            sarray (None or array-like): Poling pattern attribute (to be computed).
            atarray (None or array-like): Poling pattern attribute (to be computed).
            amuparray (None or array-like): Poling pattern attribute (to be computed).
            amdownarray (None or array-like): Poling pattern attribute (to be computed).
            altered_z (None or array-like): Poling pattern attribute (to be computed).
            z (None or array-like): Poling pattern attribute (to be computed).
        """
        self.Lc = Lc        # Coherence length (m)
        self.Lo = Lo        # Physical length of the crystal (m)
        self.T = T          # Temperature (°C)
        self.w = w          # Domain width parameter (m)
        self.mstart = mstart  # Starting index for the apodization algorithm
        self.materials = materials_db  # Materials database object

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

    def refractive_index(self, wavelength, material, axis):
        """
        Calculate the refractive index for a given wavelength, material, and axis.

        Parameters:
            wavelength (float): Wavelength in micrometers.
            material (str): Name of the material (e.g., "KTP").
            axis (str): Axis for which to calculate the refractive index ("x", "y", or "z").

        Returns:
            float: The refractive index for the specified material and axis.

        Raises:
            ValueError: If the material or coefficients are not found.
        """
        try:
            coeffs = self.materials.get_sellmeier_coefficients(material, axis)
        except ValueError as e:
            raise ValueError(f"Error in refractive_index: {e}")

        # Extract Sellmeier coefficients
        A = coeffs["A"]
        B = coeffs["B"]
        C = coeffs["C"]
        D = coeffs.get("D", 0)
        E = coeffs.get("E", 0)
        F = coeffs.get("F", 0)

        # Compute refractive index using Sellmeier equation
        # Use the appropriate formula based on the number of coefficients
        if E == 0 and F == 0:  # Only four coefficients
            n_squared = A + B / (1 - C / wavelength**2) - D * wavelength**2
        else:  # Six coefficients
            n_squared = (
                A
                + B / (1 - C / wavelength**2)
                + D / (1 - E / wavelength**2)
                - F * wavelength**2
            )
        
        n = np.sqrt(n_squared)

        # Apply temperature corrections if available
        try:
            temp_coeffs = self.materials.get_temperature_corrections(material, axis)
            if temp_coeffs:
                n1 = temp_coeffs["n1"]
                n2 = temp_coeffs["n2"]
                deln = (
                    (n1[0] + n1[1] / wavelength + n1[2] / wavelength**2 + n1[3] / wavelength**3) * (self.T - 25)
                    + (n2[0] + n2[1] / wavelength + n2[2] / wavelength**2 + n2[3] / wavelength**3) * (self.T - 25)**2
                )
                n += deln
        except ValueError:
            # Skip temperature correction if not available
            pass

        return n

    def group_index(self, wavelength, material, axis):
        """
        Calculate the group index for a given wavelength, material, and axis.

        Parameters:
            wavelength (float): Wavelength in micrometers.
            material (str): Name of the material (e.g., "KTP").
            axis (str): Axis for which to calculate the group index ("x", "y", or "z").

        Returns:
            float: The group index for the specified material and axis.

        Raises:
            ValueError: If the material or coefficients are not found.
        """
        x = Symbol("x")
        try:
            coeffs = self.materials.get_sellmeier_coefficients(material, axis)
        except ValueError as e:
            raise ValueError(f"Error in group_index: {e}")

        # Extract Sellmeier coefficients
        A = coeffs["A"]
        B = coeffs["B"]
        C = coeffs["C"]
        D = coeffs.get("D", 0)
        E = coeffs.get("E", 0)
        F = coeffs.get("F", 0)

        # Compute refractive index symbolically
        # Use the appropriate formula based on the number of coefficients
        if E == 0 and F == 0:  # Only four coefficients
            n_squared = A + B / (1 - C / x**2) - D * x**2
        else:  # Six coefficients
            n_squared = (
                A
                + B / (1 - C / x**2)
                + D / (1 - E / x**2)
                - F * x**2
            )
        n = sympy.sqrt(n_squared)
        
        # Compute group index symbolically
        group_index_expr = n - x * diff(n, x)
        group_index_value = group_index_expr.subs({x: wavelength})

        # Apply temperature corrections if available
        try:
            temp_coeffs = self.materials.get_temperature_corrections(material, axis)
            if temp_coeffs:
                n1 = temp_coeffs["n1"]
                n2 = temp_coeffs["n2"]
                deln = (
                    (n1[0] + n1[1] / wavelength + n1[2] / wavelength**2 + n1[3] / wavelength**3) * (self.T - 25)
                    + (n2[0] + n2[1] / wavelength + n2[2] / wavelength**2 + n2[3] / wavelength**3) * (self.T - 25)**2
                )
                group_index_value += deln
        except ValueError:
            # Skip temperature correction if not available
            pass

        return float(group_index_value)

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
        return np.exp(-((z - L / 2) ** 2) / (L ** 2 / 8))

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

    def Am2(self, w, altered_z, m, Lc, sn):
        """
        Computes the amplitude modulation function Am2 for a given set of parameters.

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

        Notes:
            - The function uses a summation over the poling array `sn` multiplied by an
              exponential term to compute the amplitude modulation.
            - The coherence length `Lc` is used to scale the result.
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
        (e.g., [1, -1, 1, -1, ...]) based on the coherence length (Lc) and the 
        overall length (Lo) of the crystal. The resulting structure is stored in 
        the `sarray` attribute, and the `z` attribute is updated to represent the 
        spatial positions corresponding to the poling structure.

        Args:
            resolution (int, optional): The number of points per domain in the 
                periodic poling structure. Defaults to 5.

        Attributes Updated:
            sarray (numpy.ndarray): Array representing the periodic poling structure 
                with alternating polarizations.
            Lo (float): Adjusted overall length of the crystal to be an integer 
                multiple of the coherence length (Lc).
            z (numpy.ndarray): Array of spatial positions corresponding to the 
                periodic poling structure.

        Notes:
            - The method ensures that the overall length (Lo) is adjusted to be an 
              integer multiple of the coherence length (Lc).
            - The `z` array is calculated to span the entire length of the crystal 
              (`L`), with the number of points matching the length of `sarray`.
        """
        # Validate that Lc and Lo are positive numbers
        if not (isinstance(self.Lc, (int, float)) and self.Lc > 0):
            raise ValueError("Coherence length (Lc) must be a positive number.")
        if not (isinstance(self.Lo, (int, float)) and self.Lo > 0):
            raise ValueError("Overall length (Lo) must be a positive number.")

    def generate_periodic_poling(self, resolution=5):
        """
        Generates a periodic poling structure for the crystal.

        This method creates a periodic poling structure by alternating polarizations
        (e.g., [1, -1, 1, -1, ...]) based on the coherence length (Lc) and the 
        overall length (Lo) of the crystal. The resulting structure is stored in 
        the `sarray` attribute, and the `z` attribute is updated to represent the 
        spatial positions corresponding to the poling structure.

        Args:
            resolution (int, optional): The number of points per domain in the 
                periodic poling structure. Defaults to 5.

        Attributes Updated:
            sarray (numpy.ndarray): Array representing the periodic poling structure 
                with alternating polarizations.
            Lo (float): Adjusted overall length of the crystal to be an integer 
                multiple of the coherence length (Lc).
            z (numpy.ndarray): Array of spatial positions corresponding to the 
                periodic poling structure.

        Notes:
            - The method ensures that the overall length (Lo) is adjusted to be an 
              integer multiple of the coherence length (Lc).
            - The `z` array is calculated to span the entire length of the crystal 
              (`L`), with the number of points matching the length of `sarray`.
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
            - The method assumes that the crystal parameters (e.g., `w`, `mstart`, `L`, `Lc`) 
              are already defined as attributes of the class.
        """
        # Compute refractive indices at central wavelengths
        lambda_w = laser.lambda_w
        lambda_2w = laser.lambda_2w

        nyD = self.refractive_index(lambda_w * 1e6, "KTP", "y")
        nzD = self.refractive_index(lambda_w * 1e6, "KTP", "z")
        nyP = self.refractive_index(lambda_2w * 1e6, "KTP", "y")

        # Compute group indices
        NyP = self.group_index(lambda_2w * 1e6, "KTP", "y")
        NzD = self.group_index(lambda_w * 1e6, "KTP", "z")
        NyD = self.group_index(lambda_w * 1e6, "KTP", "y")
        self.Nmean = (NzD + NyD) / 2.0

        # Compute DeltaK_0
        self.DeltaK_0 = 2 * np.pi * (nyP / lambda_2w - nzD / lambda_w - nyD / lambda_w)

        # Proceed with the apodization algorithm using self.DeltaK_0
        w = self.w
        mstart = self.mstart
        L = self.L
        Lc = self.Lc
        DeltaK = self.DeltaK_0

        num_iterations = math.ceil(L / w) + 1  # Total number of iterations

        # Precompute altered_z
        altered_z = np.linspace(0, num_iterations * w, num_iterations + 1)
        # Initialize sarray
        sarray = np.zeros(num_iterations + 1, dtype=int)
        sarray[0] = 1
        atarray = np.zeros(num_iterations, dtype=complex)
        amuparray = np.zeros(num_iterations, dtype=complex)
        amdownarray = np.zeros(num_iterations, dtype=complex)

        for idx in range(num_iterations):
            m = mstart + idx

            # Compute Atarget once per iteration
            at = self.Atarget(w, m, L, Lc, DeltaK)

            # Test with sarray[idx + 1] = 1 (up)
            sarray[idx + 1] = 1
            amup = self.Am2(w, altered_z[: idx + 2], m, Lc, sarray[: idx + 2])

            # Test with sarray[idx + 1] = -1 (down)
            sarray[idx + 1] = -1
            amdown = self.Am2(w, altered_z[: idx + 2], m, Lc, sarray[: idx + 2])

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