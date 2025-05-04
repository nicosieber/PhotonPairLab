import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.optimize import curve_fit

class SPDC_Simulation:
    def __init__(self, crystal, laser):
        self.crystal = crystal
        self.laser = laser
        # Initialize other parameters
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """
        Initializes the parameters required for the SPDC (Spontaneous Parametric Down-Conversion) simulation.
        This method ensures that the crystal's poling pattern is generated if not already present, extracts
        necessary data from the crystal and laser objects, computes phase mismatch, and calculates various
        parameters such as angular frequencies, group velocities, and bandwidth for the simulation.
        Attributes:
            DeltaK_0 (float): The initial phase mismatch value computed from the crystal and laser properties.
            omega_pump (float): The center angular frequency of the pump wave.
            omega_down (float): The center angular frequency of the down-converted signal and idler waves.
            K_pump (float): The group velocity of the pump wave.
            K_idler (float): The group velocity of the idler wave.
            K_signal (float): The group velocity of the signal wave.
            bandwidth (float): The bandwidth of the laser in angular frequency.
            xi_eff (numpy.ndarray): The effective poling pattern of the crystal, flipped and converted to float64.
            z (numpy.ndarray): The spatial positions along the crystal.
        Notes:
            - This method relies on the `generate_poling` and `compute_phase_mismatch` methods of the `Crystal` class.
            - The laser's FWHM (Full Width at Half Maximum) is used to calculate the bandwidth.
        """
        
        # Ensure that the poling pattern is generated
        if self.crystal.sarray is None:
            self.crystal.generate_poling(self.laser)

        # Use compute_phase_mismatch from the Crystal class
        _, (N_pump, N_signal, N_idler), self.DeltaK_0 = self.crystal.compute_phase_mismatch(self.laser)

        # Center angular frequencies
        self.omega_pump = 2 * np.pi * self.laser.c / self.laser.lambda_2w
        self.omega_down = self.omega_pump / 2

        # Inverse group velocities
        self.K_pump = N_pump / self.laser.c  # k' pump
        self.K_idler = N_idler / self.laser.c  # k' idler
        self.K_signal = N_signal / self.laser.c  # k' signal

        # Bandwidth
        self.bandwidth = (2 * np.pi * self.laser.c) * self.laser.FWHM / (self.laser.lambda_2w ** 2 * 2 * np.sqrt(np.log(2)))

        # xi_eff and z for simulation
        self.xi_eff = np.flip(self.crystal.sarray.astype("float64"))
        self.z = self.crystal.z
    
    # Define the Gaussian function used for fitting
    def gaussian(self, x, amp, cen, wid, off):
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
    
    # Define linear function for fitting
    def linear(self, x, m, b):
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
    
    def compute_phase_integral(self,z, xi_eff, DeltaK):
        """
        Compute the phase integral for a given set of parameters.

        This function calculates the phase integral by integrating over the 
        product of the effective coupling coefficient and the exponential 
        phase factor, using the trapezoidal rule.

        Parameters:
        -----------
        z : numpy.ndarray
            A 1D array representing the spatial positions (e.g., crystal length).
        xi_eff : numpy.ndarray
            A 1D array representing the effective coupling coefficients.
        DeltaK : numpy.ndarray
            A 2D array representing the phase mismatch values.

        Returns:
        --------
        numpy.ndarray
            A 2D array representing the computed phase integral over the spatial 
            positions for the given effective coupling coefficients and phase 
            mismatch values.
        """
        y = xi_eff[:, None, None] * np.exp(-1j * DeltaK[None, :, :] * z[:, None, None])
        return np.trapz(y, z, axis=0)

    def schmidt_decomposition(self, JSA):
        # Uses the SVD to compute Schmidt coefficients, purity, and Schmidt number 
        _, s_vals, _ = np.linalg.svd(JSA / np.amax(JSA), full_matrices=True)
        s_vals = s_vals / np.sqrt(np.sum(s_vals ** 2))  # Normalize
        Purity = np.sum(s_vals ** 4)
        K = 1 / Purity 
        s_vals = s_vals
        return s_vals, Purity, K

    def run_simulation(self, steps=100, dev=5):
        """
        Vectorized SPDC simulation that uses numpy's broadcasting to compute the Joint Spectral Amplitude (JSA),
        pump profile, phase, intensity, and related parameters.
        """
        # Generate signal and idler wavelength arrays
        self.idler_wavelengths = np.linspace(self.laser.lambda_w - dev * 1e-9, self.laser.lambda_w + dev * 1e-9, steps)
        self.signal_wavelengths = np.linspace(self.laser.lambda_w - dev * 1e-9, self.laser.lambda_w + dev * 1e-9, steps)
    
        # Precompute constants
        fs = 2 * np.pi * self.laser.c / self.signal_wavelengths[:, None]  # Signal frequencies (column vector)
        fi = 2 * np.pi * self.laser.c / self.idler_wavelengths[None, :]  # Idler frequencies (row vector)
        DeltaK_1 = (self.K_pump - self.K_signal) * (fs - self.omega_down) + (self.K_pump - self.K_idler) * (fi - self.omega_down)
        DeltaK = self.DeltaK_0 + DeltaK_1
        S = np.exp(-((fi + fs - self.omega_pump) ** 2) / (2 * self.bandwidth ** 2))  # Gaussian pump spectrum
    
        # Compute Pump, Phase, JSI, and JSA using vectorized operations
        self.Pump = S ** 2
        phase = self.compute_phase_integral(self.z, self.xi_eff, DeltaK)
        # Compute phase matching function
        self.Phase = np.abs(phase) ** 2
        Amp = S * phase
    
        self.JSI = np.abs(Amp) ** 2
        self.JSA = np.abs(Amp)
    
        # Schmidt decomposition
        self.s_vals, self.Purity, self.K = self.schmidt_decomposition(self.JSA)
        self.dev = dev
    
    def compute_optimal_temp(self):
        pass

    # Plotting methods remain unchanged
    def plot_pump(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5
        lambda_w_nm = self.laser.lambda_w * 1e9

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(self.Pump / np.amax(self.Pump), cmap=cmap)
        im1.set_interpolation("bilinear")
        im1.set_extent(
            [
                np.floor(lambda_w_nm) - self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) - self.dev,
            ]
        )
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=f_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=f_size)
        axs.set_title("Pump Pulse Envelope (PPE)", fontsize=f_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs


    def plot_phase(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5

        lambda_w_nm = self.laser.lambda_w * 1e9

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(self.Phase / np.amax(self.Phase), cmap=cmap)
        im1.set_interpolation("bilinear")
        im1.set_extent(
            [
                np.floor(lambda_w_nm) - self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) - self.dev,
            ]
        )
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=f_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=f_size)
        axs.set_title("Phase Matching Function (PMF)", fontsize=f_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs

    def plot_jsi(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5

        lambda_w_nm = self.laser.lambda_w * 1e9

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(self.JSI / np.amax(self.JSI), cmap=cmap)
        im1.set_interpolation("bilinear")
        im1.set_extent(
            [
                np.floor(lambda_w_nm) - self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) - self.dev,
            ]
        )
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=f_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=f_size)
        axs.set_title("Joint Spectral Intensity (JSI)", fontsize=f_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs

    def plot_jsa(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5

        lambda_w_nm = self.laser.lambda_w * 1e9

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(self.JSA / np.amax(self.JSA), cmap=cmap)
        im1.set_interpolation("bilinear")
        im1.set_extent(
            [
                np.floor(lambda_w_nm) - self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) + self.dev,
                np.floor(lambda_w_nm) - self.dev,
            ]
        )
        axs.invert_yaxis()
        axs.set_xlabel("signal wavelength (nm)", fontsize=f_size)
        axs.set_ylabel("idler wavelength (nm)", fontsize=f_size)
        axs.set_title("Joint Spectral Amplitude (JSA)", fontsize=f_size)
        axs.grid(False)
        axs.xaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        axs.yaxis.set_major_locator(plt.MaxNLocator(number_ticklabels))
        plt.gcf().set_facecolor((0.960, 0.960, 0.960))
        
        return fig, axs

    def get_signal_idler_fits(self):
        """
        Computes the signal and idler peaks from the JSI (Joint Spectral Intensity) data.
        This method finds the maximum values in the JSI data and returns their corresponding
        signal and idler wavelengths.

        Returns:
            tuple: A tuple containing two elements:
                - signal_peak (float): The wavelength of the signal peak.
                - idler_peak (float): The wavelength of the idler peak.
        """
        # Fit the signal data
        signal_wavelengths = self.signal_wavelengths * 1e9
        signal_itensities = abs(np.trapz(self.JSI, self.signal_wavelengths)) / np.amax(abs(np.trapz(self.JSI, self.signal_wavelengths)))
        # Use curve_fit to fit the Gaussian function to the data
        p0 = [1, np.mean(signal_wavelengths), 1, 0]  # Initial guesses for amp, cen, wid, off
        popt1, _ = curve_fit(self.gaussian, signal_wavelengths, signal_itensities, p0=p0)

        # Fit the idler data
        idler_wavelengths = self.idler_wavelengths * 1e9
        idler_intensities = abs(np.trapz(self.JSI.T, self.idler_wavelengths)) / np.amax(abs(np.trapz(self.JSI.T, self.idler_wavelengths)))
        # Fit the idler data using curve_fit
        p0 = [1, np.mean(idler_wavelengths), 1, 0] # Initial guesses for amp, cen, wid, off
        popt2, _ = curve_fit(self.gaussian, idler_wavelengths, idler_intensities, p0=p0)

        return popt1, popt2, (signal_wavelengths, signal_itensities), (idler_wavelengths, idler_intensities)
    
    def plot_schmidt_coefficients(self, font_size=12):
        # Schmidt coefficients
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.bar(np.arange(20), self.s_vals[0:20], align="center", alpha=0.75)
        ax1.grid(True)
        ax1.set_ylabel("Schmidt Coefficients", fontsize=font_size)
        tlt = f"Schmidt Decomposition of the JSA - Resulting purity: {round(self.Purity,2)}"
        ax1.set_title(tlt, fontsize=font_size)

        # Create subplot for fits and plots for idler and signal
        ax2 = fig.add_subplot(212)
        # Get the signal and idler fits
        popt1, popt2, (signal_wavelenghts, signal_intensities), (idler_wavelengths, idler_intensities) = self.get_signal_idler_fits()

        # Fit and plot the signal data
        ax2.plot(signal_wavelenghts, signal_intensities, "bo", markersize=4)
        # Use curve_fit to fit the Gaussian function to the data
        ax2.plot(signal_wavelenghts, self.gaussian(signal_wavelenghts, *popt1), linestyle="--", color="orange")
        # Fit and plot the idler data
        ax2.plot(idler_wavelengths, idler_intensities, "r^", markersize=4)
        # Fit the idler data using curve_fit
        ax2.plot(idler_wavelengths, self.gaussian(idler_wavelengths, *popt2), linestyle="--", color="green")

        # Formatting the plot
        ax2.grid(True)
        ax2.set_xlim(left=np.amin(signal_wavelenghts), right=np.amax(signal_wavelenghts))
        ax2.set_xlabel("wavelength (nm)")
        ax2.set_ylabel("normalized amplitude", fontsize=font_size)
        ax2.set_title("JSI Profiles", fontsize=font_size)
        ax2.legend(["signal", "fit: signal", "idler", "fit: idler"])
        plt.tight_layout(pad=1.2, w_pad=2, h_pad=2.0)
        
        return fig, (ax1, ax2)