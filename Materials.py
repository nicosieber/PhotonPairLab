import numpy as np
from sympy import Symbol, diff
import sympy

class BaseMaterial:
    """
    Base class for materials. Defines the interface for refractive index and group index calculations.
    """
    def refractive_index(self, wavelength, axis, temperature=None):
        """
        Calculate the refractive index for a given wavelength, axis and temperature.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def group_index(self, wavelength, axis, temperature=None):
        """
        Calculate the group index for a given wavelength, axis and temperature.
        Must be implemented by subclasses.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

class KTP1(BaseMaterial):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References for Sellmeier coefficients and temperature corrections:
    - Sellmeier coefficients: 
        - y-axis: F. Konig et al., APL, 84,1644, 2004
        - z-axis: K. Fradkin et al., APL, 74,914, 1999, https://aip.scitation.org/doi/pdf/10.1063/1.123408
    - Temperature corrections: 
        - Emanueli et al., App. Opt., 42, 33, 2003

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

    def group_index(self, wavelength, axis, temperature=25):
        """
        Calculate the group index of the material for a given wavelength, axis, and temperature.
        The group index is computed symbolically using the Sellmeier coefficients for the material
        and optionally corrected for temperature effects if temperature correction coefficients are available.
        Args:
            wavelength (float): The wavelength (in micrometers) at which to calculate the group index.
            axis (str): The optical axis ("x", "y", or "z") for which the group index is calculated.
            temperature (float, optional): The temperature (in degrees Celsius) at which to calculate
                the group index. Defaults to 25°C.
        Returns:
            float: The calculated group index for the given wavelength, axis, and temperature.
        Raises:
            ValueError: If the Sellmeier coefficients or temperature correction coefficients
                for the specified axis are not available or invalid.
        """
        
        x = Symbol("x")
        try:
            coeffs = self.get_sellmeier_coefficients(axis)
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
        
        # Apply temperature corrections symbolically
        try:
            temp_coeffs = self.get_temperature_corrections(axis)
            if temp_coeffs:
                n1 = temp_coeffs["n1"]
                n2 = temp_coeffs["n2"]
                deln = (
                    (n1[0] + n1[1] / x + n1[2] / x**2 + n1[3] / x**3) * (temperature - 25)
                    + (n2[0] + n2[1] / x + n2[2] / x**2 + n2[3] / x**3) * (temperature - 25)**2
                )
                n += deln
        except ValueError:
            # Skip temperature correction if not available
            pass

        # Compute group index symbolically
        group_index_expr = n - x * diff(n, x)
        group_index_value = group_index_expr.subs({x: wavelength})
        return float(group_index_value)

class KTP2(BaseMaterial):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References for Sellmeier coefficients and temperature corrections:
    - Sellmeier coefficients: 
        - https://www.unitedcrystals.com/KTPProp.html
    - Temperature corrections: 
        - https://www.unitedcrystals.com/KTPProp.html

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

    def group_index(self, wavelength, axis, temperature=25):
        """
        Calculate the group index of the material for a given wavelength, axis, and temperature.
        The group index is computed symbolically using the Sellmeier coefficients for the material
        and optionally corrected for temperature effects if temperature correction coefficients are available.
        Args:
            wavelength (float): The wavelength (in micrometers) at which to calculate the group index.
            axis (str): The optical axis ("x", "y", or "z") for which the group index is calculated.
            temperature (float, optional): The temperature (in degrees Celsius) at which to calculate
                the group index. Defaults to 25°C.
        Returns:
            float: The calculated group index for the given wavelength, axis, and temperature.
        Raises:
            ValueError: If the Sellmeier coefficients or temperature correction coefficients
                for the specified axis are not available or invalid.
        """
        
        x = Symbol("x")
        try:
            coeffs = self.get_sellmeier_coefficients(axis)
        except ValueError as e:
            raise ValueError(f"Error in group_index: {e}")

         # Extract Sellmeier coefficients
        A = coeffs["A"]
        B = coeffs["B"]
        C = coeffs["C"]
        D = coeffs["D"]

        # Compute refractive index using Sellmeier equation
        n_squared = (
            A
            + B / (x**2 - C)
            - D * x**2
        )
        n = sympy.sqrt(n_squared)
        
        # Apply temperature corrections symbolically
        try:
            temp_coeff = self.get_temperature_corrections(axis)
            if temp_coeff:
                n += temp_coeff * (temperature - 25)
        except ValueError:
            # Skip temperature correction if not available
            pass

        # Compute group index symbolically
        group_index_expr = n - x * diff(n, x)
        group_index_value = group_index_expr.subs({x: wavelength})

        return float(group_index_value)
    

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
    
    def group_index(self, wavelength, axis, temperature=25):
        """
        Calculate the group index of the material for a given wavelength, axis, and temperature.
        The group index is computed symbolically using the Sellmeier coefficients for the material
        and optionally corrected for temperature effects if temperature correction coefficients are available.
        Args:
            wavelength (float): The wavelength (in micrometers) at which to calculate the group index.
            axis (str): The optical axis ("x", "y", or "z") for which the group index is calculated.
            temperature (float, optional): The temperature (in degrees Celsius) at which to calculate
                the group index. Defaults to 25°C.
        Returns:
            float: The calculated group index for the given wavelength, axis, and temperature.
        Raises:
            ValueError: If the Sellmeier coefficients or temperature correction coefficients
                for the specified axis are not available or invalid.
        """
        
        x = Symbol("x")
        try:
            coeffs = self.get_sellmeier_coefficients(axis)
        except ValueError as e:
            raise ValueError(f"Error in group_index: {e}")

         # Extract Sellmeier coefficients
        A = coeffs["A"]
        B = coeffs["B"]
        C = coeffs["C"]
        D = coeffs["D"]
        E = coeffs["E"]

        # Compute refractive index using Sellmeier equation
        n_squared = (
            A
            + B / (x**2 - C)
            + D / (x**2 - E)
        )
        
        n = sympy.sqrt(n_squared)
        
        # Apply temperature corrections if available
        try:
            temp_coeffs = self.get_temperature_corrections(axis)
            A = temp_coeffs["A"]
            B = temp_coeffs["B"]
            C = temp_coeffs["C"]
            D = temp_coeffs["D"]
            if temp_coeffs:
                n += (A / x**3 - B / x**2 + C / x + D) * 1e-5 * (temperature - 25)
        except ValueError:
            # Skip temperature correction if not available
            pass

        # Compute group index symbolically
        group_index_expr = n - x * diff(n, x)
        group_index_value = group_index_expr.subs({x: wavelength})

        return float(group_index_value)
    
    