import numpy as np

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