import numpy as np
from scipy.optimize import curve_fit

from photonpairlab.spdc.utils import gaussian, linear

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

    def get_signal_idler_fits(self):
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

        # Fit Gaussian to the signal marginal distribution
        p0_signal = [1, np.mean(signal_wavelengths), 1, 0]  # Initial guesses for amp, cen, wid, off
        signal_fit, _ = curve_fit(gaussian, signal_wavelengths, signal_intensities, p0=p0_signal)

        # Fit Gaussian to the idler marginal distribution
        p0_idler = [1, np.mean(idler_wavelengths), 1, 0]  # Initial guesses for amp, cen, wid, off
        idler_fit, _ = curve_fit(gaussian, idler_wavelengths, idler_intensities, p0=p0_idler)

        # Return fit parameters and data
        return signal_fit, idler_fit, (signal_wavelengths, signal_intensities), (idler_wavelengths, idler_intensities)
    
    def compute_optimal_temp(self):
        pass