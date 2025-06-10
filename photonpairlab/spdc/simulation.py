import numpy as np
from photonpairlab.spdc.spectral_analyser import SpectralAnalyzer
from photonpairlab.apm.crystal_apm import CrystalAPM
from photonpairlab.qpm.crystal_qpm import CrystalQPM


class SPDC_Simulation:
    def __init__(self, crystal, laser, wavelength_signal=None, wavelength_idler=None, wavelength_signal_range = [None, None], wavelength_idler_range = [None, None]):
        # Initialize the SPDC simulation with a crystal and laser object.
        self.crystal = crystal
        self.laser = laser
        self.wavelength_signal = wavelength_signal
        self.wavelength_signal_start = wavelength_signal_range[0]
        self.wavelength_signal_end = wavelength_signal_range[1]
        self.wavelength_idler = wavelength_idler
        self.wavelength_idler_start = wavelength_idler_range[0]
        self.wavelength_idler_end = wavelength_idler_range[1]
        # Initialize other parameters
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """
        Initializes the simulation parameters based on the type of crystal and laser properties.

        This method sets up various attributes required for the simulation, including phase mismatch,
        angular frequencies, inverse group velocities, bandwidth, and effective nonlinear coefficients.

        Raises:
            ValueError: If the crystal type is unsupported.

        Attributes:
            omega_pump (float): Center angular frequency of the pump beam.
            omega_down (float): Center angular frequency of the down-converted beam (for QPM crystals).
            omega_signal (float): Center angular frequency of the signal beam (for APM crystals).
            omega_idler (float): Center angular frequency of the idler beam (for APM crystals).
            K_pump (float): Inverse group velocity for the pump beam.
            K_signal (float): Inverse group velocity for the signal beam.
            K_idler (float): Inverse group velocity for the idler beam.
            angular_bandwidth (float): Angular bandwidth of the laser.
            xi_eff (numpy.ndarray): Effective nonlinear coefficients for the simulation.
            z (numpy.ndarray): Spatial coordinates along the crystal.

        Notes:
            - For QPM (Quasi-Phase Matching) crystals, the phase mismatch is computed using the `compute_phase_mismatch`
              method of the `CrystalQPM` class.
            - For APM (Angle-Phase Matching) crystals, the phase mismatch is computed using the `compute_phase_mismatch`
              method of the `CrystalAPM` class, with additional wavelength parameters.
        """
        if isinstance(self.crystal, CrystalQPM):
            # Use compute_phase_mismatch from the Crystal class
            _, (N_pump, N_signal, N_idler), self.DeltaK_0 = self.crystal.compute_phase_mismatch(self.laser)
            # Center angular frequencies
            self.omega_pump = 2 * np.pi * self.laser.c / self.laser.wavelength_pump
            self.omega_down = self.omega_pump / 2
        elif isinstance(self.crystal, CrystalAPM):
            # Use compute_phase_mismatch from the Crystal class
            _, (N_pump, N_signal, N_idler), self.DeltaK_0 = self.crystal.compute_phase_mismatch(self.laser, self.wavelength_signal, self.wavelength_idler, angle_pm=None)
            # Center angular frequencies
            self.omega_pump = 2 * np.pi * self.laser.c / self.laser.wavelength_pump
            self.omega_signal = 2 * np.pi * self.laser.c / self.wavelength_signal
            self.omega_idler = 2 * np.pi * self.laser.c / self.wavelength_idler
        else:
            raise ValueError("Unsupported crystal type")
        # Inverse group velocities
        self.K_pump = N_pump / self.laser.c  # k' pump
        self.K_idler = N_idler / self.laser.c  # k' idler
        self.K_signal = N_signal / self.laser.c  # k' signal

        # Bandwidth
        self.angular_bandwidth = self.laser.bandwidth_wavelength_to_angular_bandwidth(self.laser.bandwidth_wavelength)
        # xi_eff and z for simulation
        self.xi_eff = np.flip(self.crystal.poling_pattern.astype("float64"))
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
        Simulates the SPDC (Spontaneous Parametric Down-Conversion) process for a given crystal type.
        This method computes the Joint Spectral Intensity (JSI), Joint Spectral Amplitude (JSA), 
        and other related quantities based on the properties of the laser and crystal. It supports 
        both quasi-phase-matched (QPM) and angle-phase-matched (APM) crystals.
        Args:
            steps (int, optional): Number of steps for wavelength arrays. Default is 100.
            dev (float, optional): Deviation in wavelength (in nanometers) for signal and idler arrays. Default is 5.
        Returns:
            dict: A dictionary containing the following results:
                - "Pump": Gaussian pump spectrum (array).
                - "Phase": Phase integral values (array).
                - "JSI": Joint Spectral Intensity (array).
                - "JSA": Joint Spectral Amplitude (array).
                - "SchmidtCoefficients": Placeholder for Schmidt coefficients (None).
                - "Purity": Placeholder for purity (None).
                - "K": Placeholder for Schmidt number (None).
                - "SignalWavelengths": Signal wavelengths (array).
                - "IdlerWavelengths": Idler wavelengths (array).
                - "dev": Deviation in wavelength used for simulation.
        Raises:
            ValueError: If the crystal type is unsupported for SPDC simulation.
        """
        if self.wavelength_signal_start is None and self.wavelength_signal_end is None and self.wavelength_idler_start is None and self.wavelength_idler_end is None:
            if isinstance(self.crystal, CrystalQPM):
                # Generate signal and idler wavelength arrays
                self.idler_wavelengths = np.linspace(self.laser.wavelength_pump * 2 - dev * 1e-9, self.laser.wavelength_pump * 2 + dev * 1e-9, steps)
                self.signal_wavelengths = np.linspace(self.laser.wavelength_pump * 2 - dev * 1e-9, self.laser.wavelength_pump * 2 + dev * 1e-9, steps)
            elif isinstance(self.crystal, CrystalAPM):
                # Generate signal and idler wavelength arrays
                self.idler_wavelengths = np.linspace(self.wavelength_idler - dev * 1e-9, self.wavelength_idler + dev * 1e-9, steps)
                self.signal_wavelengths = np.linspace(self.wavelength_signal - dev * 1e-9, self.wavelength_signal + dev * 1e-9, steps)
            else:
                raise ValueError("Unsupported crystal type for SPDC simulation") 
        else:
            # Generate signal and idler wavelength arrays based on provided ranges
            self.idler_wavelengths = np.linspace(self.wavelength_idler_start, self.wavelength_idler_end, steps)
            self.signal_wavelengths = np.linspace(self.wavelength_signal_start, self.wavelength_signal_end, steps)
           
        # Precompute constants
        fs = 2 * np.pi * self.laser.c / self.signal_wavelengths[:, None]  # Signal frequencies (column vector)
        fi = 2 * np.pi * self.laser.c / self.idler_wavelengths[None, :]  # Idler frequencies (row vector)

        # Compute DeltaK_0 and DeltaK_1 based on crystal type
        if isinstance(self.crystal, CrystalQPM):
            DeltaK_1 = (self.K_pump - self.K_signal) * (fs - self.omega_down) + (self.K_pump - self.K_idler) * (fi - self.omega_down)
        elif isinstance(self.crystal, CrystalAPM):
            DeltaK_1 = (self.K_pump - self.K_signal) * (fs - self.omega_signal) + (self.K_pump - self.K_idler) * (fi - self.omega_idler)
        else:
            raise ValueError("Unsupported crystal type for SPDC simulation")
        
        DeltaK = self.DeltaK_0 + DeltaK_1
        
        # Compute Pump, Phase, JSI, and JSA using vectorized operations
        S = np.exp(-((fi + fs - self.omega_pump) ** 2) / (2 * self.angular_bandwidth ** 2))  # Gaussian pump spectrum
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
            "dev": dev,
            "c": self.laser.c
        }

        return self.results
        