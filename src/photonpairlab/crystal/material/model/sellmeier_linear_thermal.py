from typing import Any

import numpy as np


from .base_material_model import BaseMaterialModel
from ..material_data import MaterialData


class SellmeierLinearThermalModel(BaseMaterialModel):
    """
    Sellmeier model + linear temperature coefficient per axis.
    Expects:
      temperature_corrections.data = { "x": float, "y": float, "z": float }
    where correction is applied as: n(T) = n(25C) + k * (T - 25)
    """
    def __init__(self, material: MaterialData):
        self.material = material

    def is_biaxial(self):
        return self.material.biaxial

    def map_polarization_axis(self, polarization_label):
        return polarization_label

    def refractive_index(self, wavelength, axis, temperature=25, **kwargs):
        wl = np.asarray(wavelength, dtype=float)

        try:
            coeffs: dict[str, Any] = self.material.sellmeier.data[axis]
        except Exception as e:
            raise ValueError(f"Sellmeier coefficients for axis '{axis}' not found in '{self.material.name}'.") from e

        A = float(coeffs["A"])
        B = float(coeffs["B"])
        C = float(coeffs["C"])
        D = float(coeffs.get("D", 0.0) or 0.0)
        E = float(coeffs.get("E", 0.0) or 0.0)
        F = float(coeffs.get("F", 0.0) or 0.0)

        if (E == 0.0) and (F == 0.0):
            n2 = A + B / (1.0 - C / wl**2) - D * wl**2
        else:
            n2 = (
                A
                + B / (1.0 - C / wl**2)
                + D / (1.0 - E / wl**2)
                - F * wl**2
            )

        n = np.sqrt(n2)

        tc = self.material.temperature_corrections
        if tc is not None and isinstance(tc.data, dict) and axis in tc.data and tc.data[axis] is not None:
            k = float(tc.data[axis])
            dT = float(temperature) - 25.0
            n = n + k * dT

        return n

    def effective_refractive_index(self, wavelength, theta_deg=None, phi_deg=None, **kwargs):
        raise NotImplementedError(
            f"Effective refractive index not implemented for model '{type(self).__name__}'."
        )

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