import numpy as np
from scipy.optimize import curve_fit

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
    return amp * np.exp(-(x - cen) ** 2 / wid) + off

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