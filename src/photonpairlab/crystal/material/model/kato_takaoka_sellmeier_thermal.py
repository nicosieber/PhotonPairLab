from typing import Any

import numpy as np

from .base_material_model import BaseMaterialModel
from ..material_data import MaterialData


class KatoTakaokaSellmeierThermalModel(BaseMaterialModel):
    """
    Kato & Takaoka style temperature correction.
    Expects temperature_corrections.data = {
        "x": {"A":..., "B":..., "C":..., "D":...},
        "y": {...},
        "z": {...},
    }

    We don’t assume the exact paper formula here (since you didn’t paste it),
    so this implementation provides a clear placeholder hook:
        n(T) = n(25C) + f(A,B,C,D, wl) * (T-25)

    Replace _delta_n_kato_takaoka() with your exact expression.
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
            tc_axis = tc.data[axis]
            dT = float(temperature) - 25.0
            n = n + self._delta_n_kato_takaoka(wl, tc_axis, dT)

        return n

    def _delta_n_kato_takaoka(self, wl, tc_axis: dict[str, Any], dT: float):
        # TODO: replace this with the exact Kato & Takaoka formula you want.
        # For now, provide a deterministic placeholder so the refactor compiles.
        #
        # Example placeholder: polynomial in wl times dT
        A = float(tc_axis["A"])
        B = float(tc_axis["B"])
        C = float(tc_axis["C"])
        D = float(tc_axis["D"])
        return (A + B * wl + C * wl**2 + D * wl**3) * dT * 1e-5

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