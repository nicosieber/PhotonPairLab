from typing import Any, Optional

import numpy as np


from .base_material_model import BaseMaterialModel


class GeneralSellmeierThermalModel(BaseMaterialModel):
    """
    General Sellmeier model supporting:
      - Sellmeier coefficients per axis
      - Optional temperature correction data per axis in the Emanueli/Arie style:
            data[axis] = {"n1": [...4...], "n2": [...4...]}
      - Optional thermal expansion:
            data[axis] = {"alpha": float, "beta": float}
    """

    def map_polarization_axis(self, polarization_label):
        # Default behavior:
        # - for uniaxial: 'o'/'e' commonly map to 'o'/'e' keys, but your JSON uses 'o' and 'e' for BBO.
        # - for biaxial: expects 'x','y','z'
        #
        # Keep it simple: pass-through unless you want special mappings per material.
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

        # Match your previous logic: decide between 4-coeff vs 6-coeff formula
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

        # Optional temperature corrections
        tc = self.material.temperature_corrections
        if tc is not None and isinstance(tc.data, dict) and axis in tc.data and tc.data[axis] is not None:
            tc_axis = tc.data[axis]

            # Emanueli/Arie style
            if isinstance(tc_axis, dict) and "n1" in tc_axis and "n2" in tc_axis:
                n1 = tc_axis["n1"]
                n2c = tc_axis["n2"]
                dT = float(temperature) - 25.0

                deln = (
                    (n1[0] + n1[1] / wl + n1[2] / wl**2 + n1[3] / wl**3) * dT
                    + (n2c[0] + n2c[1] / wl + n2c[2] / wl**2 + n2c[3] / wl**3) * dT**2
                )
                n = n + deln

            else:
                # If you later add another temperature correction format, handle here.
                pass

        return n

    def effective_refractive_index(self, wavelength, theta_deg=None, phi_deg=None, **kwargs):
        raise NotImplementedError(
            f"Effective refractive index not implemented for model '{type(self).__name__}'."
        )

    def thermal_expansion(self, length: float, axis: str, temperature: float = 25, **kwargs):
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