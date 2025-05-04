import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from photonpairlab.spdc.utils import gaussian, linear
from photonpairlab.spdc.analysis import SPDC_Analyzer

class SPDC_Plotter:
    def __init__(self, results):
        self.results = results
    
    def plot_schmidt_coefficients(self, font_size=12):
        # Schmidt coefficients
        # Analyze the results
        analyzer = SPDC_Analyzer(self.results)

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
        signal_fit, idler_fit, (signal_wavelenghts, signal_intensities), (idler_wavelengths, idler_intensities) = analyzer.get_signal_idler_fits()

        # Fit and plot the signal data
        ax2.plot(signal_wavelenghts, signal_intensities, "bo", markersize=4)
        # Use curve_fit to fit the Gaussian function to the data
        ax2.plot(signal_wavelenghts, gaussian(signal_wavelenghts, *signal_fit), linestyle="--", color="orange")
        # Fit and plot the idler data
        ax2.plot(idler_wavelengths, idler_intensities, "r^", markersize=4)
        # Fit the idler data using curve_fit
        ax2.plot(idler_wavelengths, gaussian(idler_wavelengths, *idler_fit), linestyle="--", color="green")

        # Formatting the plot
        ax2.grid(True)
        ax2.set_xlim(left=np.amin(signal_wavelenghts), right=np.amax(signal_wavelenghts))
        ax2.set_xlabel("wavelength (nm)")
        ax2.set_ylabel("normalized amplitude", fontsize=font_size)
        ax2.set_title("JSI Profiles", fontsize=font_size)
        ax2.legend(["signal", "fit: signal", "idler", "fit: idler"])
        plt.tight_layout(pad=1.2, w_pad=2, h_pad=2.0)
        
        return fig, (ax1, ax2)
    
    def plot_pump(self, font_size=12):
        cmap = cm.viridis
        number_ticklabels = 5

        signal_wavelengths = self.results["SignalWavelengths"] * 1e9
        idler_wavelengths = self.results["IdlerWavelengths"] * 1e9
        Pump = self.results["Pump"]

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im = axs.imshow(Pump / np.amax(Pump), cmap=cmap)
        im.set_interpolation("bilinear")
        im.set_extent([
            signal_wavelengths.min(), signal_wavelengths.max(),
            idler_wavelengths.min(), idler_wavelengths.max()
        ])
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=font_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=font_size)
        axs.set_title("Pump Pulse Envelope (PPE)", fontsize=font_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs

    def plot_phase(self, font_size=12):
        cmap = cm.viridis
        number_ticklabels = 5

        signal_wavelengths = self.results["SignalWavelengths"] * 1e9
        idler_wavelengths = self.results["IdlerWavelengths"] * 1e9
        Phase = self.results["Phase"]

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im = axs.imshow(Phase / np.amax(Phase), cmap=cmap)
        im.set_interpolation("bilinear")
        im.set_extent([
            signal_wavelengths.min(), signal_wavelengths.max(),
            idler_wavelengths.min(), idler_wavelengths.max()
        ])
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=font_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=font_size)
        axs.set_title("Phase Matching Function (PMF)", fontsize=font_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs

    def plot_jsi(self, font_size=12):
        cmap = cm.viridis
        number_ticklabels = 5

        signal_wavelengths = self.results["SignalWavelengths"] * 1e9
        idler_wavelengths = self.results["IdlerWavelengths"] * 1e9
        JSI = self.results["JSI"]

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im = axs.imshow(JSI / np.amax(JSI), cmap=cmap)
        im.set_interpolation("bilinear")
        im.set_extent([
            signal_wavelengths.min(), signal_wavelengths.max(),
            idler_wavelengths.min(), idler_wavelengths.max()
        ])
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=font_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=font_size)
        axs.set_title("Joint Spectral Intensity (JSI)", fontsize=font_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs

    def plot_jsa(self, font_size=12):
        cmap = cm.viridis
        number_ticklabels = 5

        signal_wavelengths = self.results["SignalWavelengths"] * 1e9
        idler_wavelengths = self.results["IdlerWavelengths"] * 1e9
        JSA = self.results["JSA"]

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im = axs.imshow(JSA / np.amax(JSA), cmap=cmap)
        im.set_interpolation("bilinear")
        im.set_extent([
            signal_wavelengths.min(), signal_wavelengths.max(),
            idler_wavelengths.min(), idler_wavelengths.max()
        ])
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=font_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=font_size)
        axs.set_title("Joint Spectral Amplitude (JSA)", fontsize=font_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs