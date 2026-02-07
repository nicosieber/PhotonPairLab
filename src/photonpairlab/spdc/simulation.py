import numpy as np
from photonpairlab.spdc.spectral_analyser import SpectralAnalyzer
from photonpairlab.spdc.spdc_results import SPDCResults
from photonpairlab.crystal import Crystal
from photonpairlab.laser import BaseLaser
from photonpairlab.laser.utils_laser import bandwidth_wavelength_to_angular_bandwidth

from photonpairlab.constants import C_VAC

class SPDC_Simulation:
    def __init__(self, crystal: Crystal, laser: BaseLaser, 
                 wavelength_signal: float | None=None, wavelength_idler : float | None=None, 
                 wavelength_signal_range: list[float | None] = [None, None], 
                 wavelength_idler_range: list[float | None] = [None, None]):
        # Initialize the SPDC simulation with a crystal and laser object.
        self.crystal = crystal
        self.laser = laser
        if wavelength_signal is None:
            self.wavelength_signal = laser.wavelength_pump * 2  # Default to twice the pump wavelength if not specified
        else:
            self.wavelength_signal = wavelength_signal
        if wavelength_idler is None:
            self.wavelength_idler = laser.wavelength_pump * 2  # Default to twice the pump wavelength if not specified
        else:
            self.wavelength_idler = wavelength_idler
        # Store the wavelengths and their ranges
       
        self.wavelength_signal_start = wavelength_signal_range[0]
        self.wavelength_signal_end = wavelength_signal_range[1]
        self.wavelength_idler_start = wavelength_idler_range[0]
        self.wavelength_idler_end = wavelength_idler_range[1]

        # Initialize other parameters
        self.initialize_parameters()
    
    def initialize_parameters(self):
        """
        Initializes the simulation parameters based on the type of crystal and laser properties.
        This method sets up various attributes required for the simulation, including phase mismatch,
        angular frequencies, inverse group velocities, bandwidth, and effective nonlinear coefficients.
        """
        
        #_, (N_pump, N_signal, N_idler), self.DeltaK_0 = self.crystal.compute_phase_mismatch(self.laser, self.wavelength_signal, self.wavelength_idler, T=self.crystal.T)
        self.pm_result = self.crystal.compute_phase_mismatch(self.laser, self.wavelength_signal, self.wavelength_idler, T=self.crystal.T)
        self.DeltaK_0 = self.pm_result.delta_k0
        # Center angular frequencies
        self.omega_pump = 2 * np.pi * C_VAC / self.laser.wavelength_pump
        self.omega_signal = 2 * np.pi * C_VAC / self.wavelength_signal
        self.omega_idler = 2 * np.pi * C_VAC / self.wavelength_idler

        # Bandwidth
        self.angular_bandwidth = bandwidth_wavelength_to_angular_bandwidth(self.laser.bandwidth_wavelength, self.laser.wavelength_pump)
        
        # xi_eff and z for simulation
        pp = self.crystal.poling_pattern
        if pp is None:
            raise ValueError("Poling pattern is not generated in the crystal.")
        self.xi_eff = np.flip(np.asarray(pp, dtype=np.float64)) 
        self.z = self.crystal.z
    
    def phase_matching_function(self, DeltaK):
        """
        Compute the phase integral for a given set of parameters.

        This function calculates the phase integral by integrating over the 
        product of the effective coupling coefficient and the exponential 
        phase factor, using the trapezoidal rule.
        """
        if self.z is not None:
            y = self.xi_eff[:, None, None] * np.exp(-1j * DeltaK[None, :, :] * self.z[:, None, None])
        return np.trapezoid(y, self.z, axis=0)
    
    def pump_pulse_envelope(self):
        """
        Computes the Gaussian pump spectrum for given signal and idler angular frequencies.

        """
        if self.signal_wavelengths is None or self.idler_wavelengths is None:
            raise ValueError("Signal and idler wavelengths must be defined to compute the pump pulse envelope.")
        fs = 2 * np.pi * C_VAC / self.signal_wavelengths[:, None]  # Signal frequencies (column vector)
        fi = 2 * np.pi * C_VAC / self.idler_wavelengths[None, :]  # Idler frequencies (row vector)

        return np.exp(-((fi + fs - self.omega_pump) ** 2) / (2 * self.angular_bandwidth ** 2))
    
    def run_simulation(self, steps=100, dev=5):
        """
        Simulates the SPDC (Spontaneous Parametric Down-Conversion) process for a given crystal type.
        This method computes the Joint Spectral Intensity (JSI), Joint Spectral Amplitude (JSA), 
        and other related quantities based on the properties of the laser and crystal. It supports 
        both quasi-phase-matched (QPM) and angle-phase-matched (APM) crystals.
 
        """
        if self.wavelength_signal_start is None and self.wavelength_signal_end is None and self.wavelength_idler_start is None and self.wavelength_idler_end is None:
            # Generate signal and idler wavelength arrays
            self.idler_wavelengths = np.linspace(self.wavelength_idler - dev * 1e-9, self.wavelength_idler + dev * 1e-9, steps)
            self.signal_wavelengths = np.linspace(self.wavelength_signal - dev * 1e-9, self.wavelength_signal + dev * 1e-9, steps)
        else:
            # Generate signal and idler wavelength arrays based on provided ranges
            if self.wavelength_signal_start is None or self.wavelength_signal_end is None:
                raise ValueError("Both start and end wavelengths for signal must be provided.")
            if self.wavelength_idler_start is None or self.wavelength_idler_end is None:
                raise ValueError("Both start and end wavelengths for idler must be provided.")
            self.idler_wavelengths = np.linspace(self.wavelength_idler_start, self.wavelength_idler_end, steps)
            self.signal_wavelengths = np.linspace(self.wavelength_signal_start, self.wavelength_signal_end, steps)

        # Compute DeltaK_0 and DeltaK_1 for the defined grid of frequencies
        DeltaK_1 = self.pm_result.compute_delta_k1(self.signal_wavelengths, self.idler_wavelengths, self.omega_signal, self.omega_idler)
        DeltaK = self.DeltaK_0 + DeltaK_1
        
        # Compute Pump, Phase, JSI, and JSA using vectorized operations
        PPE = self.pump_pulse_envelope()        
        PMF = self.phase_matching_function(DeltaK)
        Amp = PPE * PMF

        self.spdc_results = SPDCResults(
            Pump=PPE,
            Phase=np.abs(PMF),
            JSI=np.abs(Amp)**2,
            JSA=np.abs(Amp),
            SchmidtCoefficients=None,
            Purity=None,
            K=None,
            SignalWavelengths=self.signal_wavelengths,
            IdlerWavelengths=self.idler_wavelengths,
            dev=dev,
            c=C_VAC
        )

        return self.spdc_results
        