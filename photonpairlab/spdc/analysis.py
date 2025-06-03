import numpy as np
from scipy.optimize import curve_fit
from inspect import signature

from photonpairlab.spdc.utils import *

class SPDC_Analyzer:
    def __init__(self, results):
        self.results = results

    def schmidt_decomposition(self):
        # Perform Schmidt decomposition (reuse existing logic)
        JSA = self.results["JSA"]
        _, s_vals, _ = np.linalg.svd(JSA / np.amax(JSA), full_matrices=True)
        s_vals = s_vals / np.sqrt(np.sum(s_vals ** 2))  # Normalize
        Purity = np.sum(s_vals ** 4)
        K = 1 / Purity
        return s_vals, Purity, K

    def get_signal_idler_fits(self, fitting_function=gaussian): 
        """
        Computes the signal and idler peaks from the JSI (Joint Spectral Intensity) data using Gaussian fits.
        This method fits a Gaussian to the marginal distributions of the JSI data and returns the fitted peak positions.

        Returns:
            tuple: A tuple containing:
                - signal_fit (tuple): Gaussian fit parameters for the signal peak (amplitude, center, width, offset).
                - idler_fit (tuple): Gaussian fit parameters for the idler peak (amplitude, center, width, offset).
                - signal_data (tuple): Signal wavelengths (nm) and normalized intensities.
                - idler_data (tuple): Idler wavelengths (nm) and normalized intensities.
        """
        # Extract data from results
        JSI = self.results["JSI"]
        signal_wavelengths = self.results["SignalWavelengths"] * 1e9  # Convert to nm
        idler_wavelengths = self.results["IdlerWavelengths"] * 1e9  # Convert to nm

        # Compute marginal distributions
        signal_intensities = np.trapz(JSI, self.results["IdlerWavelengths"], axis=1)
        idler_intensities = np.trapz(JSI, self.results["SignalWavelengths"], axis=0)

        # Normalize intensities
        signal_intensities /= np.amax(signal_intensities)
        idler_intensities /= np.amax(idler_intensities)

        # Fit to the signal and idler marginal distribution
        # Dynamically determine the number of parameters for the fitting function
        num_params = len(signature(fitting_function).parameters) - 1  # Exclude 'x'

        p0_signal = [1] * num_params  # Default initial guess for signal
        p0_idler = [1] * (num_params)   # Default initial guess for idler
        
        # Set the second parameter (center) to the mean of signal and idler wavelengths
        if num_params > 1:
            p0_signal[1] = np.mean(signal_wavelengths)
            p0_idler[1] = np.mean(idler_wavelengths)

        try:
            signal_fit, _ = curve_fit(fitting_function, signal_wavelengths, signal_intensities, p0=p0_signal)
        except Exception as e:
            raise RuntimeError(f"Error fitting signal data: {e}")
        
        try:
            idler_fit, _ = curve_fit(fitting_function, idler_wavelengths, idler_intensities, p0=p0_idler)
        except Exception as e:
            raise RuntimeError(f"Error fitting idler data: {e}")
        
        # Return fit parameters and data
        return signal_fit, idler_fit, (signal_wavelengths, signal_intensities), (idler_wavelengths, idler_intensities)

    def compute_optimal_temp(self):
        pass