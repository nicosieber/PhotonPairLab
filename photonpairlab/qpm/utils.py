import numpy as np

def compute_delta_k0(wavelength_pump, wavelength_signal, wavelength_idler, n_signal, n_idler, n_pump):
        """
        Computes the phase mismatch (deltak_0: float) based on the provided inputs.        
        """
        return 2 * np.pi * (n_signal / wavelength_signal + n_idler / wavelength_idler - n_pump / wavelength_pump)
