import numpy as np  
from typing import Any, Optional

from .base_material_model import BaseMaterialModel

class BBO(BaseMaterialModel):
    """
    A class to encapsulate and manage material properties for nonlinear crystals.

    References:
    - Sellmeier coefficients:
        - https://www.unitedcrystals.com/BBOProp.html
    """
    
    def is_biaxial(self):
        """
        Check if the crystal is biaxial.
        Returns:
            bool: True if the crystal is biaxial, False if uniaxial.
        """
        return self.material.biaxial
    def map_polarization_axis(self, polarization_label):
        return polarization_label  # 'o' and 'e' are native for uniaxial


    def refractive_index(self, lambda_um, axis, temperature=None):
        """
        Calculate the refractive index for a given wavelength and axis.

        Args:
            lambda_um (float): Wavelength in micrometers.
            axis (str): Axis of polarization ('o' for ordinary, 'e' for extraordinary).

        Returns:
            float: Refractive index.
        """
        try:
            coeffs: dict[str, Any] = self.material.sellmeier.data[axis]
        except Exception as e:
            raise ValueError(f"Sellmeier coefficients for axis '{axis}' not found in '{self.material.name}'.") from e
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

        if self.material.thermal_expansion is None:
            expanded_length = length
            return expanded_length
        else:
            raise NotImplementedError("Thermal expansion not implemented for BIBO.")