import numpy as np

__all__ = [
    "apply_delay_to_rho_freq",
    "hom_coincidence_from_rhos",
    "hom_dip_vs_delay",
    "linear_polarization",
    "polarization_overlap",
]


def linear_polarization(theta_rad):
    """
    Jones vector of linear polarization at angle theta_rad from horizontal: [cos θ, sin θ].
    """
    return np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=complex)


def polarization_overlap(pol1=None, pol2=None):
    """
    Polarization indistinguishability |<p1|p2>|^2 of two (normalized) Jones vectors.
    None means "same polarization as the other photon", i.e. returns 1.0 if either is None.
    """
    if pol1 is None or pol2 is None:
        return 1.0
    p1 = np.asarray(pol1, dtype=complex)
    p2 = np.asarray(pol2, dtype=complex)
    if p1.shape != (2,) or p2.shape != (2,):
        raise ValueError("Jones vectors must have shape (2,).")
    n1, n2 = np.linalg.norm(p1), np.linalg.norm(p2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Jones vectors must be non-zero.")
    return float(np.abs(np.vdot(p1, p2)) ** 2 / (n1 * n2) ** 2)


def hom_coincidence_from_rhos(rho1, rho2, R=0.5, T=0.5, pol_overlap=1.0):
    """
    Coincidence probability for two single-photon states with density matrices rho1 and rho2.
    Uses: Pc = R^2 + T^2 - 2RT * |<p1|p2>|^2 * Re(Tr[rho1 rho2])
    Assumes rho1, rho2 are trace-1 density matrices in the same basis; pol_overlap = |<p1|p2>|^2.
    """
    overlap = np.trace(rho1 @ rho2)
    return (R**2 + T**2) - 2 * R * T * pol_overlap * np.real(overlap)


def apply_delay_to_rho_freq(rho, freqs_hz, tau_s):
    """
    Apply time delay tau_s to a density matrix rho(f,f') in the frequency basis:
        rho_tau(f,f') = exp(-i 2pi (f - f') tau) * rho(f,f')
    freqs_hz: 1D array of frequency-bin centers (Hz), length N, in the same order as rho's rows.
    rho: NxN density matrix in that same bin basis.
    """
    f = freqs_hz.reshape(-1, 1)
    phase = np.exp(-1j * 2*np.pi * (f - f.T) * tau_s)
    return rho * phase


def hom_dip_vs_delay(rho1, rho2, freqs_hz, taus_s, R=0.5, T=0.5, pol_overlap=1.0):
    """
    Compute Pc(tau) using frequency-domain delay operator.
    """
    Pc = np.empty_like(taus_s, dtype=float)
    for i, tau in enumerate(taus_s):
        rho2_tau = apply_delay_to_rho_freq(rho2, freqs_hz, tau)
        Pc[i] = hom_coincidence_from_rhos(rho1, rho2_tau, R=R, T=T, pol_overlap=pol_overlap)
    return Pc
