import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from lmfit import Model

class SPDC_Simulation:
    """
    Simulates the Spontaneous Parametric Down-Conversion (SPDC) process using the given crystal and laser.
    """
    def __init__(self, crystal, laser):
        self.crystal = crystal
        self.laser = laser
        # Initialize other parameters
        self.initialize_parameters()
    # Define the Gaussian function used for fitting
    def gaussian(self, x, amp, cen, wid, off):
        return amp * np.exp(-(x - cen) ** 2 / wid) + off
    def initialize_parameters(self):
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

        # Refractive indices at central wavelengths
        nyD = self.crystal.refractive_index_y(lambda_w * 1e6)
        nzD = self.crystal.refractive_index_z(lambda_w * 1e6)
        nyP = self.crystal.refractive_index_y(lambda_2w * 1e6)

        # Group indices
        NyP = self.crystal.group_index_y(lambda_2w * 1e6)
        NzD = self.crystal.group_index_z(lambda_w * 1e6)
        NyD = self.crystal.group_index_y(lambda_w * 1e6)

        # Compute DeltaK_0
        self.DeltaK_0 = 2 * np.pi * (nyP / lambda_2w - nzD / lambda_w - nyD / lambda_w)

        # Center angular frequencies
        self.Pf0 = 2 * np.pi * c / lambda_2w
        self.Df0 = self.Pf0 / 2

        # Group velocities
        self.KyP = NyP / c  # k' pump
        self.KzD = NzD / c  # k' idler
        self.KyD = NyD / c  # k' signal

        # Bandwidth
        self.bw = (2 * np.pi * c) * self.laser.FWHM / (lambda_2w ** 2 * 2 * np.sqrt(np.log(2)))

        # xi_eff and z for simulation
        self.xi_eff = np.flip(sarray.astype("float64"))
        self.z = z

    def run_simulation(self, steps=200, dev=5):
        lambda_w = self.laser.lambda_w
        c = self.laser.c
        h = np.linspace(lambda_w - dev * 1e-9, lambda_w + dev * 1e-9, steps)
        g = np.linspace(lambda_w - dev * 1e-9, lambda_w + dev * 1e-9, steps)
        z = self.z
        xi_eff = self.xi_eff
        DeltaK_0 = self.DeltaK_0
        KyP = self.KyP
        KzD = self.KzD
        KyD = self.KyD
        Df0 = self.Df0
        bw = self.bw
        Pf0 = self.Pf0

        Pump = np.zeros((steps, steps))
        Phase = np.zeros((steps, steps))
        II = np.zeros((steps, steps))
        JSA = np.zeros((steps, steps))

        for j in range(steps):
            for s in range(steps):
                fs = 2 * np.pi * c / g[j]
                fi = 2 * np.pi * c / h[s]
                DeltaK_1 = (KyP - KyD) * (fs - Df0) + (KyP - KzD) * (fi - Df0)
                DeltaK = DeltaK_0 + DeltaK_1
                S = np.exp(-((fi + fs - Pf0) ** 2) / (2 * bw ** 2))
                Pump[s, j] = S ** 2
                y = xi_eff * 1000 * np.exp(-1j * DeltaK * z)
                phase = np.trapz(y, z)
                Phase[s, j] = abs(phase ** 2)
                Amp = S * phase
                II[s, j] = np.real(np.conj(Amp) * Amp)
                JSA[s, j] = abs(np.real(Amp))
        self.g = g
        self.h = h
        self.Pump = Pump
        self.Phase = Phase
        self.II = II
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
        delta_labelsize = 4
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
        delta_labelsize = 4
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
        delta_labelsize = 4
        number_ticklabels = 5
        dev = 5
        lambda_w_nm = self.laser.lambda_w * 1e9
        II = self.II

        fig, axs = plt.subplots(1, 1, sharex=True, constrained_layout=False)
        im1 = axs.imshow(II / np.amax(II), cmap=cmap)
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
        delta_labelsize = 4
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
        s_vals = self.s_vals
        II = self.II
        h = self.h
        g = self.g
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
        x1 = h * 1e9
        y1 = abs(np.trapz(II, h)) / np.amax(abs(np.trapz(II, h)))
        ax22.plot(x1, y1, "bo", markersize=4)

        gmodel = Model(self.gaussian)
        params = gmodel.make_params(cen=np.mean(x1), amp=1, wid=1, off=0)
        result1 = gmodel.fit(y1, params, x=x1)
        ax22.plot(x1, result1.best_fit, linestyle="--", color="orange")

        y2 = abs(np.trapz(II.T, h)) / np.amax(abs(np.trapz(II.T, h)))
        ax22.plot(x1, y2, "r^", markersize=4)

        result2 = gmodel.fit(y2, params, x=x1)
        ax22.plot(x1, result2.best_fit, linestyle="--", color="green")

        ax22.grid(True)
        ax22.set_xlim(left=np.amin(h * 1e9), right=np.amax(h * 1e9))
        ax22.set_xlabel("wavelength (nm)")
        ax22.set_ylabel("normalized amplitude", fontsize=f_size)
        ax22.set_title("JSI Profiles", fontsize=f_size)
        ax22.legend(["signal", "fit: signal", "idler", "fit: idler"])
        plt.tight_layout(pad=1.2, w_pad=2, h_pad=2.0)
        plt.show()

    def plot_poling(self):
        # Needs to be rewritten to account for normal periodic poling
        z = self.z
        sarray = self.crystal.sarray
        amuparray = self.crystal.amuparray
        atarray = self.crystal.atarray
        plt.plot(z * 1000, sarray)
        '''
        plt.plot(
            1000 * z[:-1],
            2 * amuparray / (np.amax(amuparray) + np.amin(amuparray)) - 1,
            linewidth=2,
        )
        plt.plot(
            1000 * z[:-1],
            2 * atarray / (np.amax(atarray) + np.amin(atarray)) - 1,
            linewidth=2,
        )
        '''
        plt.legend(["poling"], loc="lower right")
        plt.xlabel("position within the crystal (mm)")
        plt.show()

    def plot_results(self):
        self.plot_pump()
        self.plot_phase()
        self.plot_jsi()
        self.plot_jsa()
        self.plot_schmidt_coefficients()
        #self.plot_poling()