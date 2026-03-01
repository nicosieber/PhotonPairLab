import numpy as np
from photonpairlab.spdc.spectral_analyser import SpectralAnalyzer
from photonpairlab.spdc.spdc_results import SPDCResults
from photonpairlab.crystal import Crystal
from photonpairlab.laser import BaseLaser
from photonpairlab.laser.utils_laser import bandwidth_wavelength_to_angular_bandwidth

from photonpairlab.spdc.spdc_config import SPDCGridConfig, SPDCCenterConfig, SPDCRunConfig, build_wavelength_axes

from photonpairlab.constants import C_VAC

class SPDC_Simulation:
    def __init__(
            self, crystal: Crystal, laser: BaseLaser, 
            wavelength_signal: float | None=None, wavelength_idler : float | None=None, 
            wavelength_signal_range: list[float | None] = [None, None], 
            wavelength_idler_range: list[float | None] = [None, None],
            grid: SPDCGridConfig | None = None,
                 ):
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

        self.grid = grid or SPDCGridConfig()

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
    
    def pump_pulse_envelope(self, signal_wavelengths: np.ndarray, idler_wavelengths: np.ndarray) -> np.ndarray:
        """
        Computes the Gaussian pump spectrum for given signal/idler wavelength axes.
        """
        fs = 2 * np.pi * C_VAC / signal_wavelengths[None, :]  # (1, Ns)
        fi = 2 * np.pi * C_VAC / idler_wavelengths[:, None]   # (Ni, 1)
        return np.exp(-((fi + fs - self.omega_pump) ** 2) / (2 * self.angular_bandwidth ** 2))
    
    def run(self) -> SPDCResults:
        """
        Runs the SPDC simulation and returns the results.
        This method builds the wavelength axes, computes the phase mismatch on the grid,
        calculates the pump envelope and phase matching function, and then compiles the 
        results into an SPDCResults object.
        """
        cfg = SPDCRunConfig(
            center=SPDCCenterConfig(
                wavelength_signal=self.wavelength_signal,
                wavelength_idler=self.wavelength_idler,
            ),
            grid=self.grid,
        )

        signal_wavelengths, idler_wavelengths, wl_s0, wl_i0 = build_wavelength_axes(
            self.laser.wavelength_pump, cfg
        )
        
        # Compute delta_k on the grid
        DeltaK_1 = self.pm_result.compute_delta_k1(
            signal_wavelengths, idler_wavelengths, self.omega_signal, self.omega_idler
        )
        DeltaK = self.DeltaK_0 + DeltaK_1

        # Compute pump envelope and phase matching function
        PPE = self.pump_pulse_envelope(signal_wavelengths, idler_wavelengths)
        PMF = self.phase_matching_function(DeltaK)
        Amp = PPE * PMF


        return SPDCResults(
            Pump=PPE,
            Phase=np.abs(PMF),
            JSI=np.abs(Amp) ** 2,
            JSA=np.abs(Amp), 
            SchmidtCoefficients=None,
            Purity=None,
            K=None,
            SignalWavelengths=signal_wavelengths,
            IdlerWavelengths=idler_wavelengths,
            dev=cfg.grid.dev_nm,
            c=C_VAC,
        )