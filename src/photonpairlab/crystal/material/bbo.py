from .base_material import BaseMaterial
import numpy as np  

class BBO(BaseMaterial):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References:
    - Sellmeier coefficients:
        - https://www.unitedcrystals.com/BBOProp.html
    """
    def __init__(self):
        # Dictionary to store material properties
        self.material = {
            "sellmeier": {
                "o": {"A": 2.7359, "B": 0.01878, "C": 0.01822, "D": 0.01354},
                "e": {"A": 2.3753, "B": 0.01224, "C": 0.01667, "D": 0.01516},
            },
            "temperature_corrections": None, # None found so far
            "thermal_expansion": None, # None found so far
            "biaxial": False,
        }
        
    def map_polarization_axis(self, polarization_label):
        return polarization_label  # 'o' and 'e' are native for uniaxial


    def get_sellmeier_coefficients(self, axis):
        """
        Retrieve the Sellmeier coefficients for a given material and axis.
        """
        try:
            return self.material["sellmeier"][axis]
        except KeyError:
            raise ValueError(f"Sellmeier coefficients for axis '{axis}' not found.")

    def refractive_index(self, lambda_um, axis, temperature=None):
        """
        Calculate the refractive index for a given wavelength and axis.

        Args:
            lambda_um (float): Wavelength in micrometers.
            axis (str): Axis of polarization ('o' for ordinary, 'e' for extraordinary).

        Returns:
            float: Refractive index.
        """
        coeffs = self.get_sellmeier_coefficients(axis)
        l2 = lambda_um**2
        return np.sqrt(
            coeffs["A"] + coeffs["B"] / (l2 - coeffs["C"]) - coeffs["D"] * l2
        )
    
    def effective_refractive_index(self, lambda_um, theta_deg, phi_deg=0):
        no = self.refractive_index(lambda_um, axis="o")
        ne = self.refractive_index(lambda_um, axis="e")
        theta = np.radians(theta_deg)
        return 1 / np.sqrt(
            (np.cos(theta)**2 / ne**2) + (np.sin(theta)**2 / no**2)
        )
    
    def thermal_expansion(
        self,
        length, # Same for QPM and APM
        axis, # Same for QPM and APM
        temperature=25, # Used for QPM
        **kwargs # Additional parameters for future extensions
    ):
        # Implement the thermal expansion calculation based on the selected model
        if self.material["thermal_expansion"] is None:
            expanded_length = length
            return expanded_length
        else:
            raise NotImplementedError("Thermal expansion not implemented for BIBO.")