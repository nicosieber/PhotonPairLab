import numpy as np
from typing import Any, Optional

from .base_material_model import BaseMaterialModel


class BIBO(BaseMaterialModel):
    """
    BIBO (BiB3O6) crystal with biaxial Sellmeier coefficients.
    References:
    - https://www.newlightphotonics.com/SPDC-Components/BiBO-SPDC-Crystals
    """

    def map_polarization_axis(self, polarization_label):
        """
        Map generic polarization labels to physical crystal axes.
        For example, 'o' → 'y', 'e' → effective index along propagation.
        """
        if polarization_label == 'o':
            return 'y'  # For BiBO, assume ordinary-like wave along y
        elif polarization_label == 'e':
            return None  # 'e' handled by n_eff, no axis
        else:
            raise ValueError(f"Unknown polarization label: '{polarization_label}'")

    def refractive_index(self, lambda_um, axis, temperature=None):
        """
        Compute n_x, n_y, or n_z based on lambda [µm] using:
        n² = A + B / (λ² - C) - D * λ²
        """
        coeffs: dict[str, Any] = self.material.sellmeier.data[axis]
        l2 = lambda_um**2
        A, B, C, D = coeffs["A"], coeffs["B"], coeffs["C"], coeffs["D"]
        n_sq = A + B / (l2 - C) - D * l2
        if n_sq < 0:
            raise ValueError(f"Negative n² computed for λ = {lambda_um} µm on axis '{axis}'")
        return np.sqrt(n_sq)

    def effective_refractive_index(self, lambda_um, theta_deg, phi_deg=0):
        """
        Calculate n_eff for arbitrary propagation direction in a biaxial crystal.
        θ: inclination from optical Z-axis (0° = along z)
        φ: azimuthal angle in XY plane
        """
        theta_rad = np.radians(theta_deg)
        phi_rad = np.radians(phi_deg)

        nx = self.refractive_index(lambda_um, axis="x")
        ny = self.refractive_index(lambda_um, axis="y")
        nz = self.refractive_index(lambda_um, axis="z")

        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        cos_phi = np.cos(phi_rad)
        sin_phi = np.sin(phi_rad)

        n_eff_sq_inv = (
            (cos_theta**2 * cos_phi**2) / nx**2 +
            (cos_theta**2 * sin_phi**2) / ny**2 +
            (sin_theta**2) / nz**2
        )

        if n_eff_sq_inv <= 0:
            raise ValueError(f"Invalid effective index computation: 1/n² ≤ 0 for λ = {lambda_um} µm")

        return np.sqrt(1 / n_eff_sq_inv)
    
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