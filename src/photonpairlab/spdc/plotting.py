import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator

from photonpairlab.spdc.hom_utils import *
from photonpairlab.spdc.spdc_results import SPDCResults
from photonpairlab.spdc.spectral_analyser import SpectralAnalyzer

class SPDC_Plotter:
    def __init__(self,  results: SPDCResults):
        self.results = results
    
    def plot_schmidt_coefficients(self, fitting_function=gaussian,font_size=12):
        # Schmidt coefficients
        # Analyze the results
        analyzer = SpectralAnalyzer(self.results)

        # Perform Schmidt decomposition
        s_vals, Purity, _ = analyzer.schmidt_decomposition()
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.bar(np.arange(20), s_vals[0:20], align="center", alpha=0.75)
        ax1.grid(True)
        ax1.set_ylabel("Schmidt Coefficients", fontsize=font_size)
        title = f"Schmidt Decomposition of the JSA - Resulting purity: {round(Purity,2)}"
        ax1.set_title(title, fontsize=font_size)

        # Fitting joint spectral intensity
        # Create subplot for fits and plots for idler and signal
        ax2 = fig.add_subplot(212)
        # Get the signal and idler fits
        signal_fit, idler_fit, (signal_wavelenghts, signal_intensities), (idler_wavelengths, idler_intensities) = analyzer.get_signal_idler_fits(fitting_function)

        # Fit and plot the signal data
        ax2.plot(signal_wavelenghts, signal_intensities, "bo", markersize=4)
        # Use curve_fit to fit the Gaussian function to the data
        ax2.plot(signal_wavelenghts, fitting_function(signal_wavelenghts, *signal_fit), linestyle="--", color="orange")
        # Fit and plot the idler data
        ax2.plot(idler_wavelengths, idler_intensities, "r^", markersize=4)
        # Fit the idler data using curve_fit
        ax2.plot(idler_wavelengths, fitting_function(idler_wavelengths, *idler_fit), linestyle="--", color="green")

        # Formatting the plot
        ax2.grid(True)
        ax2.set_xlim(left=np.amin(signal_wavelenghts), right=np.amax(signal_wavelenghts))
        ax2.set_xlabel("wavelength (nm)")
        ax2.set_ylabel("normalized amplitude", fontsize=font_size)
        ax2.set_title("JSI Profiles", fontsize=font_size)
        ax2.legend(["signal", "fit: signal", "idler", "fit: idler"])
        plt.tight_layout(pad=1.2, w_pad=2, h_pad=2.0)
        
        return fig, (ax1, ax2)
    
    def plot_result(self, key="JSA", fig=None, ax=None, font_size=12, color_map=cm.viridis): # type: ignore
        number_ticklabels = 5

        signal_wavelengths = self.results.SignalWavelengths * 1e9
        idler_wavelengths = self.results.IdlerWavelengths * 1e9

        if fig is None and ax is None:
            fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        elif fig is not None and ax is not None:
            axs = ax
            fig = fig
        else:
            raise ValueError("Both fig and ax must be either None or provided together.")
        
        PLOT_KEY_HANDLER = {
            "Pump": self.results.Pump,
            "Phase": self.results.Phase,
            "JSI": self.results.JSI,
            "JSA": self.results.JSA,
        }
        extent = (
            float(signal_wavelengths.min()),
            float(signal_wavelengths.max()),
            float(idler_wavelengths.min()),
            float(idler_wavelengths.max()),
        )
        im = axs.imshow(PLOT_KEY_HANDLER[key] / np.amax(PLOT_KEY_HANDLER[key]),
                cmap=color_map,
                extent=extent,
                origin='lower')  # or 'upper' if you want to flip y
        im.set_interpolation("bilinear")
        
        axs.set_xlabel("signal wavelength (nm)", fontsize=font_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=font_size)
  
        axs.grid(False)
        axs.xaxis.set_major_locator(MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(MaxNLocator(number_ticklabels))
        #plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs