import numpy as np
from numpy.fft import fftshift, fft2
from scipy.optimize import curve_fit

__all__ = [
    "gaussian",
    "sinc",
    "linear",
    "linear_fit",
    "linear_intersection_coordinates",
    "quadratic",
    "quadratic_fit",
    "quadratic_intersection_coordinates",
    "convert_to_time_domain",
    "hom_coincidence_from_rhos",
    "apply_delay_to_rho_freq",
    "hom_dip_vs_delay",
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

def sinc(x, amp, cen, wid, off):
    """
    Computes a sinc function.

    Parameters:
        x (float or ndarray): The input value(s) where the sinc function is evaluated.
        amp (float): The amplitude of the sinc peak.
        cen (float): The center position of the sinc peak.
        wid (float): The width (scaling factor) of the sinc function.
        off (float): The offset added to the sinc function.

    Returns:
        float or ndarray: The computed value(s) of the sinc function at the given input.
    """
    return amp * np.sinc((x - cen) / wid) + off

def linear(x, m, b):
    """
    Computes a linear function.

    Parameters:
        x (float or ndarray): The input value(s) where the linear function is evaluated.
        m (float): The slope of the linear function.
        b (float): The y-intercept of the linear function.

    Returns:
        float or ndarray: The computed value(s) of the linear function at the given input.
    """
    return m * x + b

def linear_fit(x, y):
    """
    Perform a linear fit to the given data points.
    Parameters:
    -----------
    x : numpy.ndarray
        The x-coordinates of the data points.
    y : numpy.ndarray
        The y-coordinates of the data points.
    Returns:
    --------
    tuple
        A tuple containing the slope, intercept, and covariance matrix of the fit.
    """
    popt, pcov = curve_fit(linear, x, y)
    return popt, pcov

def linear_intersection_coordinates(m1, b1, m2, b2):
    """
    Calculate the intersection point of two linear functions.

    Parameters:
    -----------
    m1 : float
        Slope of the first line.
    b1 : float
        Y-intercept of the first line.
    m2 : float
        Slope of the second line.
    b2 : float
        Y-intercept of the second line.

    Returns:
    --------
    tuple
        A tuple containing the x and y coordinates of the intersection point.
    """
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y

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

def convert_to_time_domain(matrix):
    """
    Converts a given matrix from the frequency domain to the time domain.

    This function applies a 2D Fourier transform to the input matrix, 
    using `fft2` from the `numpy.fft` module. The `fftshift` function 
    is used before and after the Fourier transform to center the zero 
    frequency component.

    Parameters:
        matrix (numpy.ndarray): A 2D array representing the input data 
                                in the frequency domain.

    Returns:
        numpy.ndarray: A 2D array representing the transformed data 
                       in the time domain.
    """
    return fftshift(fft2(fftshift(matrix)))


def hom_coincidence_from_rhos(rho1, rho2, R=0.5, T=0.5):
    """
    Coincidence probability for two single-photon states with density matrices rho1 and rho2.
    Uses: Pc = R^2 + T^2 - 2RT * Re(Tr[rho1 rho2])
    Assumes rho1, rho2 are trace-1 density matrices in the same basis.
    """
    overlap = np.trace(rho1 @ rho2)
    return (R**2 + T**2) - 2 * R * T * np.real(overlap)


def apply_delay_to_rho_freq(rho, freqs_hz, tau_s):
    """
    Apply time delay tau_s to a density matrix rho(f,f') in the frequency basis:
        rho_tau(f,f') = exp(-i 2pi (f - f') tau) * rho(f,f')
    freqs_hz: 1D array of frequency-bin centers (Hz), length N.
    rho: NxN density matrix in that same bin basis.
    """
    f = freqs_hz.reshape(-1, 1)
    phase = np.exp(-1j * 2*np.pi * (f - f.T) * tau_s)
    return rho * phase


def hom_dip_vs_delay(rho1, rho2, freqs_hz, taus_s, R=0.5, T=0.5):
    """
    Compute Pc(tau) using frequency-domain delay operator.
    """
    Pc = np.empty_like(taus_s, dtype=float)
    for i, tau in enumerate(taus_s):
        rho2_tau = apply_delay_to_rho_freq(rho2, freqs_hz, tau)
        Pc[i] = hom_coincidence_from_rhos(rho1, rho2_tau, R=R, T=T)
    return Pc