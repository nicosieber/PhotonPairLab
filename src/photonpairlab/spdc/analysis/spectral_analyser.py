import numpy as np
from scipy.optimize import curve_fit
from inspect import signature

from photonpairlab.spdc.analysis.fitting import gaussian
from photonpairlab.spdc.simulation.results import SPDCResults


class SpectralAnalyzer:
    def __init__(self, results: SPDCResults):
        self.results = results

    def schmidt_decomposition(self):
        JSA = self.results.JSA
        _, s_vals, _ = np.linalg.svd(JSA / np.amax(np.abs(JSA)), full_matrices=True)
        s_vals = s_vals / np.sqrt(np.sum(s_vals ** 2))
        Purity = np.sum(s_vals ** 4)
        K = 1 / Purity
        return s_vals, Purity, K

    def _get_marginals(self):
        """
        Compute normalized signal and idler marginal spectra.

        Returns:
            tuple:
                (signal_wavelengths_nm, signal_intensities),
                (idler_wavelengths_nm, idler_intensities)
        """
        JSI = self.results.JSI
        signal_wavelengths = self.results.SignalWavelengths * 1e9
        idler_wavelengths = self.results.IdlerWavelengths * 1e9

        signal_intensities = np.trapezoid(JSI, self.results.IdlerWavelengths, axis=0)
        idler_intensities = np.trapezoid(JSI, self.results.SignalWavelengths, axis=1)

        signal_max = np.amax(signal_intensities)
        idler_max = np.amax(idler_intensities)

        if signal_max > 0:
            signal_intensities = signal_intensities / signal_max
        if idler_max > 0:
            idler_intensities = idler_intensities / idler_max

        return (signal_wavelengths, signal_intensities), (idler_wavelengths, idler_intensities)

    def _make_gaussian_p0(self, x, y):
        """
        Build a sensible initial guess for a Gaussian-like fit:
        [amplitude, center, width, offset]
        """
        offset = np.min(y)
        amplitude = np.max(y) - offset
        center = x[np.argmax(y)]

        weights = np.clip(y - offset, 0.0, None)
        if np.sum(weights) > 0:
            sigma = np.sqrt(np.sum(weights * (x - center) ** 2) / np.sum(weights))
        else:
            sigma = max((x.max() - x.min()) / 10.0, 1e-6)

        # Matches gaussian(x, amp, cen, wid, off) where exponent uses / wid
        width = max(2.0 * sigma**2, 1e-12)

        return [amplitude, center, width, offset]

    def _fit_peak_region(self, x, y, fitting_function=gaussian, fit_fraction=0.3):
        """
        Fit only the region around the maximum to avoid poor global fits on flat tops.
        """
        num_params = len(signature(fitting_function).parameters) - 1

        peak_idx = np.argmax(y)
        peak_x = x[peak_idx]

        half_window = fit_fraction * (x.max() - x.min())
        mask = np.abs(x - peak_x) <= half_window

        x_fit = x[mask]
        y_fit = y[mask]

        if len(x_fit) < max(5, num_params + 1):
            x_fit = x
            y_fit = y

        if num_params == 4:
            p0 = self._make_gaussian_p0(x_fit, y_fit)
            bounds = (
                [0.0, x.min(), 1e-12, -np.inf],
                [np.inf, x.max(), np.inf, np.inf],
            )
            popt, pcov = curve_fit(
                fitting_function,
                x_fit,
                y_fit,
                p0=p0,
                bounds=bounds,
                maxfev=10000,
            )
        else:
            p0 = [1.0] * num_params
            if num_params > 1:
                p0[1] = peak_x
            popt, pcov = curve_fit(
                fitting_function,
                x_fit,
                y_fit,
                p0=p0,
                maxfev=10000,
            )

        return popt, pcov

    def _quadratic_peak(self, x, y):
        """
        Sub-grid peak estimate using a quadratic fit around the maximum.
        """
        i = np.argmax(y)

        if i == 0 or i == len(y) - 1:
            return x[i]

        x_local = x[i - 1:i + 2]
        y_local = y[i - 1:i + 2]

        a, b, _ = np.polyfit(x_local, y_local, 2)

        if np.isclose(a, 0.0) or a > 0:
            return x[i]

        return -b / (2.0 * a)

    def _centroid_peak(self, x, y, threshold=0.5):
        """
        Weighted centroid of the top part of the peak.
        """
        ymax = np.max(y)
        if ymax <= 0:
            return x[np.argmax(y)]

        mask = y >= threshold * ymax
        if not np.any(mask):
            return x[np.argmax(y)]

        weights = y[mask]
        return np.sum(x[mask] * weights) / np.sum(weights)

    def get_signal_idler_fits(self, fitting_function=gaussian):
        """
        Computes signal and idler fits from the JSI marginal distributions.

        Returns:
            tuple:
                - signal_fit: fitted parameters
                - idler_fit: fitted parameters
                - signal_data: (signal_wavelengths_nm, signal_intensities)
                - idler_data: (idler_wavelengths_nm, idler_intensities)
        """
        (signal_wavelengths, signal_intensities), (idler_wavelengths, idler_intensities) = self._get_marginals()

        try:
            signal_fit, _ = self._fit_peak_region(
                signal_wavelengths,
                signal_intensities,
                fitting_function=fitting_function,
                fit_fraction=0.3,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            raise RuntimeError(f"Error fitting signal data: {e}") from e

        try:
            idler_fit, _ = self._fit_peak_region(
                idler_wavelengths,
                idler_intensities,
                fitting_function=fitting_function,
                fit_fraction=0.3,
            )
        except (RuntimeError, ValueError, TypeError) as e:
            raise RuntimeError(f"Error fitting idler data: {e}") from e

        return (
            signal_fit,
            idler_fit,
            (signal_wavelengths, signal_intensities),
            (idler_wavelengths, idler_intensities),
        )

    def get_signal_idler_peaks(self, method="quadratic", fitting_function=gaussian):
        """
        Robust peak extraction for signal and idler marginals.

        Args:
            method: one of {"argmax", "quadratic", "centroid", "gaussian"}
            fitting_function: used when method == "gaussian"

        Returns:
            tuple:
                - signal_peak_nm
                - idler_peak_nm
                - signal_data: (signal_wavelengths_nm, signal_intensities)
                - idler_data: (idler_wavelengths_nm, idler_intensities)
        """
        (signal_wavelengths, signal_intensities), (idler_wavelengths, idler_intensities) = self._get_marginals()

        if method == "argmax":
            signal_peak = signal_wavelengths[np.argmax(signal_intensities)]
            idler_peak = idler_wavelengths[np.argmax(idler_intensities)]

        elif method == "quadratic":
            signal_peak = self._quadratic_peak(signal_wavelengths, signal_intensities)
            idler_peak = self._quadratic_peak(idler_wavelengths, idler_intensities)

        elif method == "centroid":
            signal_peak = self._centroid_peak(signal_wavelengths, signal_intensities, threshold=0.5)
            idler_peak = self._centroid_peak(idler_wavelengths, idler_intensities, threshold=0.5)

        elif method == "gaussian":
            signal_fit, idler_fit, _, _ = self.get_signal_idler_fits(fitting_function=fitting_function)
            signal_peak = signal_fit[1]
            idler_peak = idler_fit[1]

        else:
            raise ValueError("method must be one of: 'argmax', 'quadratic', 'centroid', 'gaussian'")

        return (
            signal_peak,
            idler_peak,
            (signal_wavelengths, signal_intensities),
            (idler_wavelengths, idler_intensities),
        )