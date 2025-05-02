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
        Pump = self.Pump

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(Pump / np.amax(Pump), cmap=cmap)
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
        plt.show()

    def plot_phase(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5
        dev = 5
        lambda_w_nm = self.laser.lambda_w * 1e9
        Phase = self.Phase

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(Phase / np.amax(Phase), cmap=cmap)
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
        plt.show()

    def plot_jsi(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5
        dev = 5
        lambda_w_nm = self.laser.lambda_w * 1e9
        JSI = self.JSI

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(JSI / np.amax(JSI), cmap=cmap)
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
        plt.show()

    def plot_jsa(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5
        dev = 5
        lambda_w_nm = self.laser.lambda_w * 1e9
        JSA = self.JSA

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(JSA / np.amax(JSA), cmap=cmap)
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
        plt.show()

    def plot_schmidt_coefficients(self):
        # Schmidt coefficients
        s_vals = self.s_vals
        JSI = self.JSI
        idler_wavelengths = self.idler_wavelengths
        signal_wavelengths = self.signal_wavelengths
        f_size = 12

        fig2 = plt.figure()
        ax21 = fig2.add_subplot(211)
        ax21.bar(np.arange(20), s_vals[0:20], align="center", alpha=0.75)
        ax21.grid(True)
        ax21.set_ylabel("Schmidt Coefficients", fontsize=f_size)
        tlt = f"Schmidt Decomposition of the JSA - Resulting purity: {round(self.Purity,2)}"
        ax21.set_title(tlt, fontsize=f_size)

        # Marginal distributions
        ax22 = fig2.add_subplot(212)
        x1 = idler_wavelengths * 1e9
        y1 = abs(np.trapz(JSI, idler_wavelengths)) / np.amax(abs(np.trapz(JSI, idler_wavelengths)))
        ax22.plot(x1, y1, "bo", markersize=4)

        # Define Gaussian function
        def gaussian(x, amp, cen, wid, off):
            return amp * np.exp(-(x - cen) ** 2 / wid) + off

        # Use curve_fit to fit the Gaussian function to the data
        p0 = [1, np.mean(x1), 1, 0]  # Initial guesses for amp, cen, wid, off
        popt1, _ = curve_fit(gaussian, x1, y1, p0=p0)
        ax22.plot(x1, gaussian(x1, *popt1), linestyle="--", color="orange")

        # Fit and plot the idler data
        y2 = abs(np.trapz(JSI.T, idler_wavelengths)) / np.amax(abs(np.trapz(JSI.T, idler_wavelengths)))
        ax22.plot(x1, y2, "r^", markersize=4)

        # Fit the idler data using curve_fit
        popt2, _ = curve_fit(gaussian, x1, y2, p0=p0)
        ax22.plot(x1, gaussian(x1, *popt2), linestyle="--", color="green")

        # Formatting the plot
        ax22.grid(True)
        ax22.set_xlim(left=np.amin(idler_wavelengths * 1e9), right=np.amax(idler_wavelengths * 1e9))
        ax22.set_xlabel("wavelength (nm)")
        ax22.set_ylabel("normalized amplitude", fontsize=f_size)
        ax22.set_title("JSI Profiles", fontsize=f_size)
        ax22.legend(["signal", "fit: signal", "idler", "fit: idler"])
        plt.tight_layout(pad=1.2, w_pad=2, h_pad=2.0)
        plt.show()

    def plot_poling(self):
        z = self.z
        sarray = self.crystal.sarray

        plt.plot(z * 1000, sarray, label="poling")
        plt.xlabel("Position within the crystal (mm)")
        plt.ylabel("Poling value")
        plt.title("Poling Profile")
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_results(self):
        self.plot_pump()
        self.plot_phase()
        self.plot_jsi()
        self.plot_jsa()
        self.plot_schmidt_coefficients()
        #self.plot_poling()