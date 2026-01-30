from typing import Any

import numpy as np


from .base_material_model import BaseMaterialModel

POLARIZATION_MAP: dict[str, str | None] = {
    "o": "y",
    "e": None,  # 'e' handled by n_eff, no axis
}

class SellmeierLinearThermalModel(BaseMaterialModel):
    """
    Sellmeier model + linear temperature coefficient per axis.
    Expects:
      temperature_corrections.data = { "x": float, "y": float, "z": float }
    where correction is applied as: n(T) = n(25C) + k * (T - 25)
    """

    def map_polarization_axis(self, polarization_label):
        """
        Map generic polarization labels to physical crystal axes.
        For example, 'o' → 'y', 'e' → effective index along propagation.
        """
        return POLARIZATION_MAP.get(polarization_label, polarization_label)

    def refractive_index(self, wavelength, axis, temperature=25, **kwargs):
        wl = np.asarray(wavelength, dtype=float)

        try:
            coeffs: dict[str, Any] = self.material.sellmeier.data[axis]
        except Exception as e:
            raise ValueError(f"Sellmeier coefficients for axis '{axis}' not found in '{self.material.name}'.") from e

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

        tc = self.material.temperature_corrections
        if tc is not None and isinstance(tc.data, dict) and axis in tc.data and tc.data[axis] is not None:
            k = float(tc.data[axis])
            dT = float(temperature) - 25.0
            n = n + k * dT
        return n

    def effective_refractive_index(self, lambda_um, theta_deg=0, phi_deg=0):
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

    def thermal_expansion(self, length, axis, temperature=25, **kwargs):
        te = self.material.thermal_expansion
        if te is None:
            raise ValueError(f"No thermal_expansion data available for '{self.material.name}'")

        if not isinstance(te.data, dict) or axis not in te.data:
            raise ValueError(f"Thermal expansion coefficients for axis '{axis}' not found in '{self.material.name}'")

        coeffs = te.data[axis]
        alpha = float(coeffs["alpha"])
        beta = float(coeffs["beta"])

        dT = float(temperature) - 25.0
        return float(length) * (1.0 + alpha * dT + beta * dT**2)