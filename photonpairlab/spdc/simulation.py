import numpy as np
from photonpairlab.spdc.analysis import SPDC_Analyzer

class SPDC_Simulation:
    def __init__(self, crystal, laser):
        # Initialize the SPDC simulation with a crystal and laser object.
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
        
        # Compute Pump, Phase, JSI, and JSA using vectorized operations
        S = np.exp(-((fi + fs - self.omega_pump) ** 2) / (2 * self.bandwidth ** 2))  # Gaussian pump spectrum
        phase = self.compute_phase_integral(self.z, self.xi_eff, DeltaK)
        Amp = S * phase

        self.results = {
            "Pump": S**2,
            "Phase": np.abs(phase) ** 2,
            "JSI": np.abs(Amp) ** 2,
            "JSA": np.abs(Amp),
            "SchmidtCoefficients": None,
            "Purity": None,
            "K": None,
            "SignalWavelengths": self.signal_wavelengths,
            "IdlerWavelengths": self.idler_wavelengths,
            "dev": dev
        }

        return self.results
        