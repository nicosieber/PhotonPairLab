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
            Pf0 (float): The center angular frequency of the pump wave.
            Df0 (float): The center angular frequency of the down-converted signal and idler waves.
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
        
        # Extract necessary data from the crystal and laser
        T = self.crystal.T
        L = self.crystal.L
        Lc = self.crystal.Lc
        z = self.crystal.z
        sarray = self.crystal.sarray

        lambda_w = self.laser.lambda_w
        lambda_2w = self.laser.lambda_2w
        c = self.laser.c

    
        # Use compute_phase_mismatch from the Crystal class
        _, (N_pump, N_signal, N_idler), DeltaK_0 = self.crystal.compute_phase_mismatch(self.laser)
        self.DeltaK_0 = DeltaK_0

        # Center angular frequencies
        self.Pf0 = 2 * np.pi * c / lambda_2w
        self.Df0 = self.Pf0 / 2

        # Group velocities
        self.K_pump = N_pump / c  # k' pump
        self.K_idler = N_idler / c  # k' idler
        self.K_signal = N_signal / c  # k' signal

        # Bandwidth
        self.bandwidth = (2 * np.pi * c) * self.laser.FWHM / (lambda_2w ** 2 * 2 * np.sqrt(np.log(2)))

        # xi_eff and z for simulation
        self.xi_eff = np.flip(sarray.astype("float64"))
        self.z = z

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
    
    def compute_phase(self,z, xi_eff, DeltaK):
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

    def run_simulation_optimized(self, steps=100, dev=5):
        """
        Optimized SPDC simulation to compute the Joint Spectral Amplitude (JSA),
        pump profile, phase, intensity, and related parameters.
        """
        lambda_w = self.laser.lambda_w
        c = self.laser.c
        z = self.z
        xi_eff = self.xi_eff
        DeltaK_0 = self.DeltaK_0
        K_pump = self.K_pump
        K_idler = self.K_idler
        K_signal = self.K_signal
        Df0 = self.Df0
        bandwidth = self.bandwidth
        Pf0 = self.Pf0
    
        # Generate signal and idler wavelength arrays
        idler_wavelengths = np.linspace(lambda_w - dev * 1e-9, lambda_w + dev * 1e-9, steps)
        signal_wavelengths = np.linspace(lambda_w - dev * 1e-9, lambda_w + dev * 1e-9, steps)
    
        # Precompute constants
        fs = 2 * np.pi * c / signal_wavelengths[:, None]  # Signal frequencies (column vector)
        fi = 2 * np.pi * c / idler_wavelengths[None, :]  # Idler frequencies (row vector)
        DeltaK_1 = (K_pump - K_signal) * (fs - Df0) + (K_pump - K_idler) * (fi - Df0)
        DeltaK = DeltaK_0 + DeltaK_1
        S = np.exp(-((fi + fs - Pf0) ** 2) / (2 * bandwidth ** 2))  # Gaussian pump spectrum
    
        # Compute Pump, Phase, JSI, and JSA using vectorized operations
        Pump = S ** 2
        phase = self.compute_phase(z, xi_eff, DeltaK)
        # Compute phase matching function
        Phase = np.abs(phase) ** 2
        Amp = S * phase
        JSI = np.abs(Amp) ** 2
        JSA = np.abs(Amp)
    
        # Store results
        self.signal_wavelengths = signal_wavelengths
        self.idler_wavelengths = idler_wavelengths
        self.Pump = Pump
        self.Phase = Phase
        self.JSI = JSI
        self.JSA = JSA
    
        # Schmidt decomposition
        u, s_vals, vh = np.linalg.svd(JSA / np.amax(JSA), full_matrices=True)
        s_vals = s_vals / np.sqrt(np.sum(s_vals ** 2))  # Normalize
        self.Purity = np.sum(s_vals ** 4)
        self.K = 1 / self.Purity
        self.s_vals = s_vals

    def run_simulation(self, steps=100, dev=5):
        """
        Runs the SPDC (Spontaneous Parametric Down-Conversion) simulation to compute 
        the Joint Spectral Amplitude (JSA), pump profile, phase, intensity, and other 
        related parameters. Additionally, performs Schmidt decomposition to calculate 
        the purity and Schmidt number.

        Parameters:
            steps (int, optional): Number of steps for discretizing the wavelength range. 
                                   Default is 200.
            dev (float, optional): Deviation in nanometers from the central wavelength 
                                   for the wavelength range. Default is 5.

        Attributes Set:
            g (numpy.ndarray): Array of signal wavelengths.
            h (numpy.ndarray): Array of idler wavelengths.
            Pump (numpy.ndarray): 2D array representing the pump profile.
            Phase (numpy.ndarray): 2D array representing the phase profile.
            JSI (numpy.ndarray): 2D array representing the intensity profile.
            JSA (numpy.ndarray): 2D array representing the Joint Spectral Amplitude.
            Purity (float): Purity of the quantum state obtained from Schmidt decomposition.
            K (float): Schmidt number, representing the degree of entanglement.
            s_vals (numpy.ndarray): Singular values from the Schmidt decomposition.

        Notes:
            - The simulation assumes a Gaussian pump spectrum.
            - The Schmidt decomposition is performed using Singular Value Decomposition (SVD).
            - The purity is calculated as the sum of the fourth powers of the normalized 
              singular values.
        """
        lambda_w = self.laser.lambda_w
        c = self.laser.c
        idler_wavelengths = np.linspace(lambda_w - dev * 1e-9, lambda_w + dev * 1e-9, steps)
        signal_wavelengths = np.linspace(lambda_w - dev * 1e-9, lambda_w + dev * 1e-9, steps)
        z = self.z
        xi_eff = self.xi_eff
        DeltaK_0 = self.DeltaK_0
        K_pump = self.K_pump
        K_idler = self.K_idler
        K_signal = self.K_signal
        Df0 = self.Df0
        bandwidth = self.bandwidth
        Pf0 = self.Pf0

        Pump = np.zeros((steps, steps))
        Phase = np.zeros((steps, steps))
        JSI = np.zeros((steps, steps))
        JSA = np.zeros((steps, steps))

        for j in range(steps):
            for s in range(steps):
                fs = 2 * np.pi * c / signal_wavelengths[j]
                fi = 2 * np.pi * c / idler_wavelengths[s]
                DeltaK_1 = (K_pump - K_signal) * (fs - Df0) + (K_pump - K_idler) * (fi - Df0)
                DeltaK = DeltaK_0 + DeltaK_1
                S = np.exp(-((fi + fs - Pf0) ** 2) / (2 * bandwidth ** 2))
                Pump[s, j] = S ** 2
                y = xi_eff * 1000 * np.exp(-1j * DeltaK * z)
                phase = np.trapz(y, z)
                Phase[s, j] = abs(phase ** 2)
                Amp = S * phase
                JSI[s, j] = np.real(np.conj(Amp) * Amp)
                JSA[s, j] = abs(np.real(Amp))
        self.signal_wavelengths = signal_wavelengths
        self.idler_wavelengths = idler_wavelengths
        self.Pump = Pump
        self.Phase = Phase
        self.JSI = JSI
        self.JSA = JSA

        # Schmidt decomposition
        u, s_vals, vh = np.linalg.svd(JSA / np.amax(JSA), full_matrices=True)
        s_vals = s_vals / np.sqrt(np.sum(s_vals ** 2))  # Normalize
        self.Purity = np.sum(s_vals ** 4)
        self.K = 1 / self.Purity
        self.s_vals = s_vals

    # Plotting methods remain unchanged
    def plot_pump(self):
        cmap = cm.viridis
        f_size = 12
        number_ticklabels = 5
        dev = 5
        lambda_w_nm = self.laser.lambda_w * 1e9
        Pump = self.Pump

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(Pump / np.amax(Pump), cmap=cmap)
        im1.set_interpolation("bilinear")
        im1.set_extent(
            [
                np.floor(lambda_w_nm) - dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) - dev,
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
                np.floor(lambda_w_nm) - dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) - dev,
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
                np.floor(lambda_w_nm) - dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) - dev,
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
                np.floor(lambda_w_nm) - dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) + dev,
                np.floor(lambda_w_nm) - dev,
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