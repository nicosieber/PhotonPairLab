"""
Pure HOM math, independent of ``SPDCResults``.

Every HOM experiment modeled here has the coincidence probability

    Pc(tau) = R^2 + T^2 - 2RT * |<p1|p2>|^2 * X(tau)

and the models differ only in the spectral interference term X(tau):

- independent photons (two sources): X(tau) = Re Tr[rho1 rho2(tau)]
  -> ``rho_overlap_vs_delay``
- signal and idler of the same pair: X(tau) = Re <psi| SWAP_tau |psi>
  -> ``exchange_overlap_vs_delay``

``coincidence_probability`` is the only place the three factors are combined.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike

from photonpairlab._typing import ComplexArray, FloatArray, Mode

__all__ = [
    "apply_delay_to_rho_freq",
    "coincidence_probability",
    "default_delays",
    "exchange_overlap_vs_delay",
    "hom_coincidence_from_rhos",
    "hom_dip_vs_delay",
    "linear_polarization",
    "polarization_overlap",
    "reduced_density_matrix",
    "rho_overlap_vs_delay",
]


# --- polarization -------------------------------------------------------------------------------


def linear_polarization(theta_rad: float) -> ComplexArray:
    """
    Jones vector of linear polarization at angle theta_rad from horizontal: [cos θ, sin θ].
    """
    return np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=complex)


def polarization_overlap(pol1: ArrayLike | None = None, pol2: ArrayLike | None = None) -> float:
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


# --- spectral states ----------------------------------------------------------------------------


def reduced_density_matrix(jsa: ComplexArray, mode: Mode) -> ComplexArray:
    """
    Trace-one reduced spectral density matrix of one photon of the pair.

    ``jsa`` has shape (N_idler, N_signal): axis 0 is idler, axis 1 is signal. Tracing out the
    idler (summing over axis 0) leaves the signal's density matrix, and vice versa.
    """
    if mode == "signal":
        rho = jsa.T @ jsa.conj()
    elif mode == "idler":
        rho = jsa @ jsa.T.conj()
    else:
        raise ValueError(f"Invalid mode: {mode!r}. Choose 'signal' or 'idler'.")
    return rho / np.trace(rho)


def default_delays(freqs_hz: FloatArray) -> FloatArray:
    """A delay axis spanning a few inverse bin spacings, used when the caller gives none."""
    n = len(freqs_hz)
    df = abs(freqs_hz[-1] - freqs_hz[0]) / (n - 1)
    t_max = 5 / df
    return np.linspace(-t_max, t_max, (2 * n - 1) * 100)


# --- spectral interference terms X(tau) ---------------------------------------------------------


def apply_delay_to_rho_freq(rho: ComplexArray, freqs_hz: FloatArray, tau_s: float) -> ComplexArray:
    """
    Apply time delay tau_s to a density matrix rho(f,f') in the frequency basis:
        rho_tau(f,f') = exp(-i 2pi (f - f') tau) * rho(f,f')
    freqs_hz: 1D array of frequency-bin centers (Hz), length N, in the same order as rho's rows.
    rho: NxN density matrix in that same bin basis.
    """
    f = freqs_hz.reshape(-1, 1)
    phase = np.exp(-1j * 2 * np.pi * (f - f.T) * tau_s)
    return rho * phase


def rho_overlap_vs_delay(
    rho1: ComplexArray, rho2: ComplexArray, freqs_hz: FloatArray, taus_s: Iterable[float]
) -> FloatArray:
    """
    Independent-photon interference term X(tau) = Re Tr[rho1 rho2(tau)], photon 2 delayed by tau.

    Exact for uncorrelated photons (e.g. heralded photons from two sources). For photons of the
    same pair use ``exchange_overlap_vs_delay`` instead.
    """
    # Tr[rho1 rho2(tau)] = sum_{f,f'} rho1(f',f) rho2(f,f') e^{-i 2pi (f-f') tau}
    weights = rho1.T * rho2
    df = freqs_hz[:, None] - freqs_hz[None, :]
    return np.array([np.real(np.sum(weights * np.exp(-1j * 2 * np.pi * df * t))) for t in taus_s])


def exchange_overlap_vs_delay(
    jsa: ComplexArray, freqs_hz: FloatArray, taus_s: Iterable[float]
) -> FloatArray:
    """
    Same-pair interference term (signal in port 1, idler in port 2, idler delayed by tau):

        X(tau) = Re sum f*(w_s, w_i) f(w_i, w_s) e^{-i (w_s - w_i) tau} / sum |f|^2

    i.e. the overlap of the JSA with its signal<->idler-exchanged copy. ``jsa`` has shape
    (N_idler, N_signal) on one shared frequency grid ``freqs_hz``, so the exchange is a transpose.
    Reduces to ``rho_overlap_vs_delay`` only when the JSA factorizes.
    """
    if jsa.shape[0] != jsa.shape[1] or jsa.shape[0] != len(freqs_hz):
        raise ValueError("Exchange overlap needs a square JSA on one shared signal/idler grid.")
    weights = jsa.conj() * jsa.T / np.sum(np.abs(jsa) ** 2)
    df = freqs_hz[None, :] - freqs_hz[:, None]  # f_signal - f_idler on the (idler, signal) grid
    return np.array([np.real(np.sum(weights * np.exp(-1j * 2 * np.pi * df * t))) for t in taus_s])


# --- combining the factors ----------------------------------------------------------------------


def coincidence_probability(
    spectral_overlap: FloatArray | float, R: float = 0.5, T: float = 0.5, pol_overlap: float = 1.0
) -> FloatArray:
    """Pc = R^2 + T^2 - 2RT * |<p1|p2>|^2 * X: the only place the HOM factors are combined."""
    return (R**2 + T**2) - 2 * R * T * pol_overlap * np.asarray(spectral_overlap, dtype=float)


# --- backwards-compatible wrappers --------------------------------------------------------------


def hom_coincidence_from_rhos(
    rho1: ComplexArray, rho2: ComplexArray, R: float = 0.5, T: float = 0.5, pol_overlap: float = 1.0
) -> float:
    """
    Coincidence probability for two single-photon states with density matrices rho1 and rho2.
    Uses: Pc = R^2 + T^2 - 2RT * |<p1|p2>|^2 * Re(Tr[rho1 rho2])
    Assumes rho1, rho2 are trace-1 density matrices in the same basis; pol_overlap = |<p1|p2>|^2.
    """
    overlap = float(np.real(np.trace(rho1 @ rho2)))
    return float(coincidence_probability(overlap, R=R, T=T, pol_overlap=pol_overlap))


def hom_dip_vs_delay(
    rho1: ComplexArray,
    rho2: ComplexArray,
    freqs_hz: FloatArray,
    taus_s: FloatArray,
    R: float = 0.5,
    T: float = 0.5,
    pol_overlap: float = 1.0,
) -> FloatArray:
    """Independent-photon Pc(tau), using the frequency-domain delay operator."""
    X = rho_overlap_vs_delay(rho1, rho2, freqs_hz, taus_s)
    return coincidence_probability(X, R=R, T=T, pol_overlap=pol_overlap)
