import numpy as np

__all__ = [
    "apply_delay_to_rho_freq",
    "hom_coincidence_from_rhos",
    "hom_dip_vs_delay",
]


def hom_coincidence_from_rhos(rho1, rho2, R=0.5, T=0.5):
    """
    Coincidence probability for two single-photon states with density matrices rho1 and rho2.
    Uses: Pc = R^2 + T^2 - 2RT * Re(Tr[rho1 rho2])
    Assumes rho1, rho2 are trace-1 density matrices in the same basis.
    """
    overlap = np.trace(rho1 @ rho2)
    return (R**2 + T**2) - 2 * R * T * np.real(overlap)


def apply_delay_to_rho_freq(rho, freqs_hz, tau_s):
    """
    Apply time delay tau_s to a density matrix rho(f,f') in the frequency basis:
        rho_tau(f,f') = exp(-i 2pi (f - f') tau) * rho(f,f')
    freqs_hz: 1D array of frequency-bin centers (Hz), length N.
    rho: NxN density matrix in that same bin basis.
    """
    f = freqs_hz.reshape(-1, 1)
    phase = np.exp(-1j * 2*np.pi * (f - f.T) * tau_s)
    return rho * phase


def hom_dip_vs_delay(rho1, rho2, freqs_hz, taus_s, R=0.5, T=0.5):
    """
    Compute Pc(tau) using frequency-domain delay operator.
    """
    Pc = np.empty_like(taus_s, dtype=float)
    for i, tau in enumerate(taus_s):
        rho2_tau = apply_delay_to_rho_freq(rho2, freqs_hz, tau)
        Pc[i] = hom_coincidence_from_rhos(rho1, rho2_tau, R=R, T=T)
    return Pc
