from .base_material import BaseMaterial  
import numpy as np  

class KTP2(BaseMaterial):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References for Sellmeier coefficients and temperature corrections:
    - Sellmeier coefficients: 
        - https://www.unitedcrystals.com/KTPProp.html
    - Temperature corrections: 
        - https://www.unitedcrystals.com/KTPProp.html
    - Thermal expansion:
        - S. Emanueli & A. Arie, App. Opt, vol. 42, No. 33 (2003)
        - No other values / references found
    """
    def __init__(self):
        # Dictionary to store material properties
        self.material = {
            "sellmeier": {
                "x": {"A": 3.0065, "B": 0.03901, "C": 0.04251, "D": 0.01327},
                "y": {"A": 3.0333, "B": 0.04154, "C": 0.04547, "D": 0.01408},
                "z": {"A": 3.3134, "B": 0.05694, "C": 0.05658, "D": 0.01682},
            },
            "temperature_corrections": {
                "x": 1.1e-5,
                "y": 1.3e-5,
                "z": 1.6e-5,
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
        D = coeffs["D"]

        # Compute refractive index using Sellmeier equation
        n_squared = (
            A
            + B / (wavelength**2 - C)
            - D * wavelength**2
        )
        
        n = np.sqrt(n_squared)

        # Apply temperature corrections if available
        try:
            temp_coeff = self.get_temperature_corrections(axis)
            if temp_coeff:
                n += temp_coeff * (temperature - 25)
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