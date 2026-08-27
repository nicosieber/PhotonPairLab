from dataclasses import dataclass

from photonpairlab.constants import C_VAC

import numpy as np

@dataclass(frozen=True)
class PhaseMismatchResult:
    n: tuple[float, float, float]          # (n_p, n_s, n_i)
    N: tuple[float, float, float]          # (N_p, N_s, N_i)
    delta_k0: float                        # Δk0 in m^-1 (SI)
    angle_pm: float | None = None          # useful for APM, optional for QPM
    coherence_length: float | None = None  # useful for QPM

    def get_Np(self) -> float:
        return self.N[0]
    def get_Ns(self) -> float:
        return self.N[1]
    def get_Ni(self) -> float:
        return self.N[2]
    
    def get_angle_pm(self) -> float | None:
        return self.angle_pm
    
    def compute_delta_k1(
            self, 
            wavelength_signal_range: np.ndarray, 
            wavelength_idler_range: np.ndarray, 
            pm_omega_signal: float, 
            pm_omega_idler: float
            ) -> np.ndarray:
        """Computes the phase mismatch Δk1 for given signal and idler wavelengths."""
        fs = 2 * np.pi * C_VAC / wavelength_signal_range[None, :]  # Signal frequencies (row vector)
        fi = 2 * np.pi * C_VAC / wavelength_idler_range[:, None]  # Idler frequencies (column vector)

        K_pump = self.get_Np() / C_VAC
        K_signal = self.get_Ns() / C_VAC
        K_idler = self.get_Ni() / C_VAC

        delta_K1 = (K_pump - K_signal) * (fs - pm_omega_signal) + (K_pump - K_idler) * (fi - pm_omega_idler)
        return delta_K1
    
