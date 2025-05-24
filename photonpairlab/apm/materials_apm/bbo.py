from .base_material_apm import BaseMaterialAPM
import numpy as np  

class BBO(BaseMaterialAPM):
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
        }

    def get_sellmeier_coefficients(self, axis):
        """
        Retrieve the Sellmeier coefficients for a given material and axis.
        """
        try:
            return self.material["sellmeier"][axis]
        except KeyError:
            raise ValueError(f"Sellmeier coefficients for axis '{axis}' not found.")

    def refractive_index(self, lambda_um, axis):
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