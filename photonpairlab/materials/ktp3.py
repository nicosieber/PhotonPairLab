from .base_material import BaseMaterial  
import numpy as np  

class KTP3(BaseMaterial):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References for Sellmeier coefficients and temperature corrections:
    - Kiyoshi Kato and Eiko Takaoka, "Sellmeier and thermo-optic dispersion formulas for KTP," Appl. Opt. 41, 5040-5044 (2002) 

    """
    def __init__(self):
        # Dictionary to store material properties
        self.material = {
            "sellmeier": {
                "x": {"A": 3.29100, "B": 0.04140, "C": 0.03978, "D": 9.35522, "E": 31.45571},
                "y": {"A": 3.45018, "B": 0.04341, "C": 0.04597, "D": 16.98825, "E": 39.43799},
                "z": {"A": 4.59423, "B": 0.06206, "C": 0.04763, "D": 110.80672, "E": 86.12171},
            },
            "temperature_corrections": {
                "x": {"A": 0.1717, "B": 0.5353, "C": 0.8416, "D": 0.1627},
                "y": {"A": 0.1997, "B": 0.4063, "C": 0.5154, "D": 0.5425},
                "z": {"A": 0.9221, "B": 2.9220, "C": 3.6677, "D": 0.1897},
            },
        }
        
    def get_sellmeier_coefficients(self, axis):
        """
        Retrieve the Sellmeier coefficients for a given material and axis.
        """
        try:
            return self.material["sellmeier"][axis]
        except KeyError:
            raise ValueError(f"Sellmeier coefficients for axis '{axis}' not found.")

    def get_temperature_corrections(self, axis):
        """
        Retrieve the temperature correction coefficients for a given material and axis.
        """
        try:
            return self.material["temperature_corrections"][axis]
        except KeyError:
            return None  # Return None if no temperature corrections are available
        
    def refractive_index(self, wavelength, axis, temperature=25):
        """
        Calculate the refractive index of a material using the Sellmeier equation 
        and optionally apply temperature corrections.
        Parameters:
            wavelength (float): The wavelength of light (in micrometers) for which 
                    the refractive index is to be calculated.
            axis (str): The optical axis ('x', 'y', or 'z') for which the refractive 
                index is to be determined.
            temperature (float, optional): The temperature (in degrees Celsius) at 
                           which the refractive index is calculated. 
                           Defaults to 25°C.
        Returns:
            float: The refractive index of the material for the given wavelength, 
               axis, and temperature.
        Raises:
            ValueError: If the Sellmeier coefficients or temperature corrections 
                for the specified axis are not available or invalid.
        """

        try:
            coeffs = self.get_sellmeier_coefficients(axis)
        except ValueError as e:
            raise ValueError(f"Error in refractive_index: {e}")

        # Extract Sellmeier coefficients
        A = coeffs["A"]
        B = coeffs["B"]
        C = coeffs["C"]
        D = coeffs["D"]
        E = coeffs["E"]

        # Compute refractive index using Sellmeier equation
        n_squared = (
            A
            + B / (wavelength**2 - C)
            + D / (wavelength**2 - E)
        )
        
        n = np.sqrt(n_squared)

        # Apply temperature corrections if available
        try:
            temp_coeffs = self.get_temperature_corrections(axis)
            A = temp_coeffs["A"]
            B = temp_coeffs["B"]
            C = temp_coeffs["C"]
            D = temp_coeffs["D"]
            if temp_coeffs:
                n += (A / wavelength**3 - B / wavelength**2 + C / wavelength + D) * 1e-5 * (temperature - 25)
        except ValueError:
            # Skip temperature correction if not available
            pass

        return n