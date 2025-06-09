import numpy as np
from numpy.fft import fftshift, fft2
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

def interpolate_matrix(matrix, pad_factor):
    
    # --- Zero-padding for interpolation ---
    pad_factor = 10  # Increase resolution
    original_shape = matrix.shape
    pad_x = (pad_factor - 1) * original_shape[0] // 2
    pad_y = (pad_factor - 1) * original_shape[1] // 2

    padded_matrix = np.pad(matrix, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')

    return padded_matrix

def interpolate_array_linspace(array, pad_factor):
    """
    Interpolates a 1D array by increasing the number of points linearly using np.linspace.

    Args:
        array (np.ndarray): 1D input array.
        pad_factor (int): Factor to increase resolution (e.g., 4 = 4x more points).

    Returns:
        np.ndarray: Linearly interpolated array.
    """
    N = len(array)
    x_old = np.linspace(0, 1, N)
    x_new = np.linspace(0, 1, N * pad_factor)
    interpolated = np.interp(x_new, x_old, array)
    return interpolated

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

def compute_purity_and_visibility(rho):
    """
    Computes the purity and visibility for a given density matrix.

    Args:
        rho (numpy.ndarray): The density matrix.

    Returns:
        tuple: Purity and visibility.
    """
    purity = np.real(np.trace(rho @ rho))
    P_min = 0.5 * (1 - purity)
    P_max = 0.5
    visibility = (P_max - P_min) / P_max
    return purity, visibility

def compute_HOM_probability(rho1, rho2, reflection=0.5, transmission=0.5):
    """
    Computes the HOM interference probability based on the overlap of two modes 
    at a beamsplitter with given reflection and transmission coefficients.

    Args:
        overlap (float): Overlap integral between two modes.
        reflection (float): Reflection coefficient (default is 0.5).
        transmission (float): Transmission coefficient (default is 0.5).

    Returns:
        float: HOM interference probability.
    """
    overlap = np.sum(np.conj(rho1) * rho2)
    return 0.5 * (1 - 2 * reflection * transmission * np.abs(overlap))

def compute_cross_correlation(rho1_temporal, rho2_temporal):
    """
    Computes the cross-correlation probabilities between two temporal density matrices.

    Args:
        rho1_temporal (numpy.ndarray): Temporal density matrix for the first mode.
        rho2_temporal (numpy.ndarray): Temporal density matrix for the second mode.

    Returns:
        numpy.ndarray: Cross-correlation probabilities.
    """
    N = rho1_temporal.shape[0]
    t_vals = np.arange(-(N - 1), N)
    P_tau_cross = []

    for t in t_vals:
        """
        For each time delay t, extract the overlapping submatrices from the two temporal density matrices.
        The slicing ensures that only the *common overlapping region* between the two matrices is used:

        - When t > 0, we delay rho1_temporal forward by removing its first t rows/columns,
        and trim rho2_temporal by removing its last t rows/columns to match.
        - When t < 0, we delay rho2_temporal forward (opposite direction), and trim rho1_temporal accordingly.

        This avoids artificial wraparound (as would happen with np.roll) and ensures that
        the overlap is computed only where the two wavepackets truly coincide in time,
        mimicking the physical behavior of delayed wave interference at a beamsplitter.
        """
        if t >= 0:
            A = rho1_temporal[t:, t:]
            B = rho2_temporal[:-t or None, :-t or None]  # handles k=0
        else:
            A = rho1_temporal[:t or None, :t or None]
            B = rho2_temporal[-t:, -t:]
    
        # Only proceed if A and B have the same shape and nonzero size
        if A.shape == B.shape and A.size > 0:
            prob_cross = compute_HOM_probability(A, B) # Use HOM probability calculation
            P_tau_cross.append(prob_cross)
        else:
            P_tau_cross.append(0.5)  # fallback to no interference

    return np.array(P_tau_cross), t_vals


def compute_autocorrelation(rho_temporal):
    """
    Computes the autocorrelation as a function of time delay for a given temporal density matrix.

    Args:
        rho_temporal (numpy.ndarray): Temporal density matrix.

    Returns:
        tuple: A tuple containing:
            - P_tau (numpy.ndarray): Autocorrelation probabilities as a function of time delay.
            - t_vals (numpy.ndarray): Time delay values (arbitrary units).
    """
    N = rho_temporal.shape[0]
    t_vals = np.arange(-(N - 1), N)

    P_tau = []
    for t in t_vals:
        d0 = np.diag(rho_temporal, k=0)
        dk = np.diag(rho_temporal, k=t)
        length = min(len(d0), len(dk))
        overlap = np.sum(np.conj(d0[:length]) * dk[:length])
        prob = 0.5 * (1 - np.abs(overlap)**2)
        P_tau.append(prob)

    return np.array(P_tau), t_vals

def rescale_probabilities(P_tau, visibility):
    """
    Rescales probabilities to match the physical range.

    Args:
        P_tau (numpy.ndarray): The raw probabilities.
        visibility (float): The visibility to use for scaling.

    Returns:
        numpy.ndarray: Rescaled probabilities.
    """
    P_tau_rescaled = (P_tau - np.min(P_tau)) / (np.max(P_tau) - np.min(P_tau))  # normalize to [0,1]
    P_min = 0.5 * (1 - visibility)
    P_max = 0.5
    return P_min + (P_max - P_min) * P_tau_rescaled

 