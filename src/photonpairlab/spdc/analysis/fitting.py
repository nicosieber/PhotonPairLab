import numpy as np
from scipy.optimize import curve_fit

__all__ = [
    "gaussian",
    "quadratic",
    "quadratic_fit",
    "quadratic_intersection_coordinates",
]


def gaussian(x, amp, cen, wid, off):
    """
    Computes a Gaussian function.

    Parameters:
        x (float or ndarray): The input value(s) where the Gaussian function is evaluated.
        amp (float): The amplitude of the Gaussian peak.
        cen (float): The center position of the Gaussian peak.
        wid (float): The width (variance) of the Gaussian function.
        off (float): The offset added to the Gaussian function.

    Returns:
        float or ndarray: The computed value(s) of the Gaussian function at the given input.
    """
    exponent = -(x - cen) ** 2 / wid
    return amp * np.exp(exponent) + off


def quadratic(x, a, b, c):
    """
    Computes a quadratic function a*x**2 + b*x + c.

    Parameters:
        x (float or ndarray): The input value(s) where the quadratic is evaluated.
        a (float): Quadratic coefficient.
        b (float): Linear coefficient.
        c (float): Constant offset.

    Returns:
        float or ndarray: The computed value(s) of the quadratic function.
    """
    return a * x ** 2 + b * x + c


def quadratic_fit(x, y):
    """
    Perform a quadratic (2nd-order polynomial) fit to the given data points.

    Useful where a straight-line fit does not capture real curvature in the data,
    e.g. temperature-tuning curves where the Sellmeier temperature correction has
    a quadratic term.

    Parameters:
    -----------
    x : numpy.ndarray
        The x-coordinates of the data points.
    y : numpy.ndarray
        The y-coordinates of the data points.
    Returns:
    --------
    tuple
        A tuple containing (a, b, c) and the covariance matrix of the fit.
    """
    popt, pcov = curve_fit(quadratic, x, y)
    return popt, pcov


def quadratic_intersection_coordinates(a1, b1, c1, a2, b2, c2, x_range=None):
    """
    Calculate the intersection point of two quadratic functions.

    Parameters:
    -----------
    a1, b1, c1 : float
        Coefficients of the first quadratic (a1*x**2 + b1*x + c1).
    a2, b2, c2 : float
        Coefficients of the second quadratic (a2*x**2 + b2*x + c2).
    x_range : tuple(float, float), optional
        If given, restricts the returned intersection to a root within
        [min(x_range), max(x_range)]. If both roots (or neither) fall inside the
        range, the one closest to the range's midpoint is returned.

    Returns:
    --------
    tuple
        The (x, y) coordinates of the intersection, or (nan, nan) if the two
        curves don't intersect (for real x) or no root falls within x_range.
    """
    a, b, c = a1 - a2, b1 - b2, c1 - c2

    if np.isclose(a, 0.0):
        if np.isclose(b, 0.0):
            return np.nan, np.nan
        roots = [-c / b]
    else:
        discriminant = b ** 2 - 4 * a * c
        if discriminant < 0:
            return np.nan, np.nan
        sqrt_disc = np.sqrt(discriminant)
        roots = [(-b + sqrt_disc) / (2 * a), (-b - sqrt_disc) / (2 * a)]

    if x_range is not None:
        lo, hi = min(x_range), max(x_range)
        roots = [r for r in roots if lo <= r <= hi]

    if not roots:
        return np.nan, np.nan

    if x_range is not None and len(roots) > 1:
        midpoint = 0.5 * (min(x_range) + max(x_range))
        x = min(roots, key=lambda r: abs(r - midpoint))
    else:
        x = roots[0]

    y = a1 * x ** 2 + b1 * x + c1
    return x, y
