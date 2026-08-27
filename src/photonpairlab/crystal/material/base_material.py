from __future__ import annotations

import numpy as np


def derivative(f, x: float | np.ndarray, dx: float = 1e-6) -> float | np.ndarray:
    """
    First derivative using central differences.

    Parameters
    ----------
    f  : callable
        Function f(x)
    x  : float or np.ndarray
        Point(s) where the derivative is evaluated
    dx : float
        Step size

    Returns
    -------
    df/dx : float or np.ndarray
    """
    x = np.asarray(x, dtype=float)
    return (f(x + dx) - f(x - dx)) / (2.0 * dx)


class BaseMaterial:
    """
    Base class for materials. Defines the interface for refractive index and group index calculations.
    """
    def is_biaxial(self) -> bool:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def map_polarization_axis(
            self,
            polarization_label: str,  # for uniaxial crystals: 'o', 'e'; for biaxial crystals 'x', 'y', 'z'
        ) -> str | None:
        raise NotImplementedError("This method should be implemented by subclasses.")

    def refractive_index(
            self,
            wavelength: float,  # Same for QPM and APM
            axis: str,  # Same for QPM and APM
            temperature: float | None = None,  # Used for QPM
            **kwargs  # Additional parameters for future extensions
        ) -> float:
        raise NotImplementedError

    def effective_refractive_index(
            self,
            wavelength: float,  # Same for QPM and APM
            theta_deg: float | None = None,  # Used for APM
            phi_deg: float | None = None,  # Used for APM
            **kwargs  # Additional parameters for future extensions
        ) -> float:
        raise NotImplementedError

    def group_index(
            self,
            wavelength: float,  # Same for QPM and APM
            axis: str | None = None,  # Same for QPM and APM
            temperature: float | None = None,  # Used for QPM
            theta_deg: float | None = None,  # Used for APM
            phi_deg: float | None = None,  # Used for APM
            **kwargs  # Additional parameters for future extensions
        ) -> float:
        if theta_deg is not None:
            # Use effective refractive index for angle-based calculation
            n_func = lambda wl: self.effective_refractive_index(wl, theta_deg, phi_deg)
        elif axis is not None:
            # Use axis-based refractive index (for QPM along principal axis)
            n_func = lambda wl: self.refractive_index(wl, axis, temperature)
        else:
            raise ValueError("Either axis or theta_deg must be provided.")

        # Calculate the refractive index at the given wavelength
        n = n_func(wavelength)
        # Use numerical differentiation to calculate dn/dλ
        dn_dlambda = derivative(n_func, wavelength, dx=1e-9)

        # Calculate the group index
        return n - wavelength * dn_dlambda

    def thermal_expansion(
            self,
            length: float,  # Same for QPM and APM
            axis: str,  # Same for QPM and APM
            temperature: float,  # Used for QPM
            **kwargs  # Additional parameters for future extensions
        ) -> float:
        raise NotImplementedError
