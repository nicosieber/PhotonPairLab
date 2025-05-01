from .base_material import BaseMaterial  
import numpy as np  

class KTP1(BaseMaterial):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References for Sellmeier coefficients and temperature corrections:
    - Sellmeier coefficients: 
        - y-axis: F. Konig et al., APL, 84,1644, 2004
        - z-axis: K. Fradkin et al., APL, 74,914, 1999, https://aip.scitation.org/doi/pdf/10.1063/1.123408
    - Temperature corrections: 
        - Emanueli et al., App. Opt., 42, 33, 2003
    - Thermal expansion:
        - S. Emanueli & A. Arie, App. Opt, vol. 42, No. 33 (2003)
        - No other values / references found
    """
    def __init__(self):
        # Dictionary to store material properties
        self.material = {
            "sellmeier": {
                "y": {"A": 2.09930, "B": 0.922683, "C": 0.0467695, "D": 0.0138408},
                "z": {"A": 2.12725, "B": 1.18431, "C": 0.0514852, "D": 0.6603, "E": 100.00507, "F": 0.00968956},
            },
            "temperature_corrections": {
                "y": {
                    "n1": [6.28977e-6, 6.3061e-6, -6.0629e-6, 2.6486e-6],
                    "n2": [-0.14445e-8, 2.2244e-8, -3.5770e-8, 1.3470e-8],
                },
                "z": {
                    "n1": [9.9587e-6, 9.9228e-6, -8.9603e-6, 4.1010e-6],
                    "n2": [-1.1882e-8, 1.0459e-7, -9.8136e-8, 3.1481e-8],
                },
            },
            "thermal_expansion": {
                "z": {"alpha": 6.7e-6, "beta": 11e-9},
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
            temp_coeffs = self.get_temperature_corrections(axis)
            if temp_coeffs:
                n1 = temp_coeffs["n1"]
                n2 = temp_coeffs["n2"]
                deln = (
                    (n1[0] + n1[1] / wavelength + n1[2] / wavelength**2 + n1[3] / wavelength**3) * (temperature - 25)
                    + (n2[0] + n2[1] / wavelength + n2[2] / wavelength**2 + n2[3] / wavelength**3) * (temperature - 25)**2
                )
                n += deln
        except ValueError:
            # Skip temperature correction if not available
            pass

        return n
    
    def get_thermal_expansion(self, axis):   
        """
        Retrieve the thermal expansion coefficients for a given material and axis.
        """
        try:
            return self.material["thermal_expansion"][axis]
        except KeyError:
            raise ValueError(f"Thermal expansion coefficients for axis '{axis}' not found.")

    def thermal_expansion(self, length, axis, temperature=25):
        """
        Calculate the thermally expanded length of a material along a specified axis.
        This method computes the expanded length of a material based on its thermal 
        expansion coefficients and the change in temperature from a reference value 
        (default is 25°C).
        Parameters:
            length (float): The original length of the material (in meters).
            axis (str): The axis along which the thermal expansion is calculated.
                        This should be a valid axis for which thermal expansion 
                        coefficients are defined.
            temperature (float, optional): The temperature at which the expansion 
                                            is calculated (in °C). Default is 25°C.
        Returns:
            float: The thermally expanded length of the material (in meters).
        Raises:
            ValueError: If the specified axis is invalid or if there is an error 
                        retrieving the thermal expansion coefficients.
        """
        try:
            coeffs = self.get_thermal_expansion(axis)
        except ValueError as e:
            raise ValueError(f"Error in refractive_index: {e}")
        
        # Extract Sellmeier coefficients
        alpha = coeffs["alpha"]
        beta = coeffs["beta"]

        expanded_length = length * (1 + alpha * (temperature - 25) + beta * (temperature - 25)**2)
        return expanded_length