"""
Hong-Ou-Mandel interference between photons described by ``SPDCResults``.

Two different experiments need two different spectral models, so the caller picks one explicitly:

- ``HOMAnalyzer.same_pair(results)``: signal and idler of the *same* pair interfere. The photons are
  spectrally correlated, and the exact interference term is the JSA exchange overlap. It is 1 for
  any exchange-symmetric JSA, however impure.
- ``HOMAnalyzer.independent(results_a, results_b)``: photons from two independent pair events
  (e.g. two heralded sources) interfere. The input is a product state, and the exact interference
  term is Re Tr[rho_a rho_b(tau)], which for identical sources is bounded by the spectral purity.

Using the independent model for a same-pair experiment underestimates the visibility. The
analyzer can't tell the two apart from the results alone (two identical crystals give identical
results), which is why the model is a choice and not inferred.
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from photonpairlab._typing import ComplexArray, FloatArray, Mode
from photonpairlab.spdc.analysis.hom_math import (
    coincidence_probability,
    default_delays,
    exchange_overlap_vs_delay,
    polarization_overlap,
    reduced_density_matrix,
    rho_overlap_vs_delay,
)
from photonpairlab.spdc.analysis.two_mode_hom_results import HOMResults
from photonpairlab.spdc.simulation.results import SPDCResults


def _wavelengths(results: SPDCResults, mode: Mode) -> FloatArray:
    if mode == "signal":
        return np.asarray(results.SignalWavelengths, dtype=float)
    if mode == "idler":
        return np.asarray(results.IdlerWavelengths, dtype=float)
    raise ValueError(f"Invalid mode: {mode!r}. Choose 'signal' or 'idler'.")


def _purity(rho: ComplexArray) -> float:
    return float(np.real(np.trace(rho @ rho)))


class HOMAnalyzer:
    def __init__(self, *results: SPDCResults) -> None:
        """
        Holds one or more ``SPDCResults``. Prefer the static methods ``same_pair`` and
        ``independent``, which take the results they need directly.
        """
        self.results_list = list(results)

    # --- the two experiments --------------------------------------------------------------------

    @staticmethod
    def same_pair(
        results: SPDCResults,
        taus_s: FloatArray | None = None,
        R: float = 0.5,
        T: float = 0.5,
        pol_signal: ArrayLike | None = None,
        pol_idler: ArrayLike | None = None,
    ) -> HOMResults:
        """
        HOM interference between the signal and idler of the same pair (idler delayed by tau):

            Pc(tau) = R^2 + T^2 - 2RT * |<p_s|p_i>|^2 * X_exch(tau)

        with X_exch the JSA exchange overlap (see ``hom_math.exchange_overlap_vs_delay``).
        Requires identical signal and idler wavelength grids (``signal_range == idler_range``).

        Args:
            results: one simulation.
            taus_s: delays in seconds (default: a few inverse bin spacings).
            R, T: beam-splitter reflectivity and transmissivity.
            pol_signal, pol_idler: Jones vectors at the beam splitter. None means identical
                polarizations, i.e. |<p_s|p_i>|^2 = 1. See ``hom_math.linear_polarization``.
        """
        wl_s = _wavelengths(results, "signal")
        wl_i = _wavelengths(results, "idler")
        if wl_s.shape != wl_i.shape or not np.allclose(wl_s, wl_i):
            raise ValueError(
                "same_pair needs identical signal and idler wavelength grids; "
                "use SPDCGridConfig(signal_range=r, idler_range=r)."
            )
        freqs = results.c / wl_s
        taus = default_delays(freqs) if taus_s is None else np.asarray(taus_s, dtype=float)
        jsa = np.asarray(results.JSA)

        X = exchange_overlap_vs_delay(jsa, freqs, taus)
        pol = polarization_overlap(pol_signal, pol_idler)
        return HOMResults(
            model="same_pair",
            tau_s=taus,
            coincidence_probabilities=coincidence_probability(X, R=R, T=T, pol_overlap=pol),
            spectral_overlap=X,
            overlap_at_zero_delay=float(exchange_overlap_vs_delay(jsa, freqs, [0.0])[0]),
            polarization_overlap=pol,
            R=R,
            T=T,
            purity1=_purity(reduced_density_matrix(jsa, "signal")),
            purity2=_purity(reduced_density_matrix(jsa, "idler")),
        )

    @staticmethod
    def independent(
        results_a: SPDCResults,
        results_b: SPDCResults,
        mode_a: Mode = "signal",
        mode_b: Mode = "signal",
        taus_s: FloatArray | None = None,
        R: float = 0.5,
        T: float = 0.5,
        pol_a: ArrayLike | None = None,
        pol_b: ArrayLike | None = None,
    ) -> HOMResults:
        """
        HOM interference between photons from two independent pair events (photon b delayed by tau),
        each heralded by a frequency-unresolving detection of its partner:

            Pc(tau) = R^2 + T^2 - 2RT * |<p_a|p_b>|^2 * Re Tr[rho_a rho_b(tau)]

        For two identical sources the visibility is bounded by the spectral purity.

        Args:
            results_a, results_b: the two sources (may be equal for identical crystals).
            mode_a, mode_b: which photon of each pair interferes ("signal" or "idler").
            taus_s: delays in seconds (default: a few inverse bin spacings).
            R, T: beam-splitter reflectivity and transmissivity.
            pol_a, pol_b: Jones vectors at the beam splitter. None means identical polarizations.
        """
        if results_a is results_b and mode_a != mode_b:
            warnings.warn(
                "independent() was given the signal and idler of the same results object. If these "
                "photons come from the same pair, the density-matrix model is wrong for them; use "
                "HOMAnalyzer.same_pair(results). (Ignore this for two identical but independent "
                "sources.)",
                UserWarning,
                stacklevel=2,
            )
        wl_a = _wavelengths(results_a, mode_a)
        wl_b = _wavelengths(results_b, mode_b)
        if wl_a.shape != wl_b.shape or not np.allclose(wl_a, wl_b):
            raise ValueError(
                "Wavelength grids do not match; resample/interpolate onto a common grid first."
            )

        # Keep rho's row/column order: sorting would misalign the frequency bins with rho.
        freqs = results_a.c / wl_a
        taus = default_delays(freqs) if taus_s is None else np.asarray(taus_s, dtype=float)
        rho_a = reduced_density_matrix(np.asarray(results_a.JSA), mode_a)
        rho_b = reduced_density_matrix(np.asarray(results_b.JSA), mode_b)

        X = rho_overlap_vs_delay(rho_a, rho_b, freqs, taus)
        pol = polarization_overlap(pol_a, pol_b)
        return HOMResults(
            model="independent",
            tau_s=taus,
            coincidence_probabilities=coincidence_probability(X, R=R, T=T, pol_overlap=pol),
            spectral_overlap=X,
            overlap_at_zero_delay=float(np.real(np.trace(rho_a @ rho_b))),
            polarization_overlap=pol,
            R=R,
            T=T,
            purity1=_purity(rho_a),
            purity2=_purity(rho_b),
        )

    # --- helpers / backwards compatibility ------------------------------------------------------

    def get_reduced_density_matrix(
        self, mode: Mode = "signal", results_index: int = 0
    ) -> tuple[ComplexArray, ComplexArray]:
        """
        Reduced density matrix of one photon of ``results_list[results_index]``.

        Returns:
            (rho, JSA): the trace-one density matrix and the JSA it was computed from.
        """
        if results_index >= len(self.results_list):
            raise IndexError(
                f"Invalid results_index: {results_index}. "
                f"Only {len(self.results_list)} results available."
            )
        jsa = np.asarray(self.results_list[results_index].JSA)
        return reduced_density_matrix(jsa, mode), jsa

    def compute_two_mode_HOM(
        self,
        mode1: Mode = "signal",
        mode2: Mode = "signal",
        taus_s: FloatArray | None = None,
        R: float = 0.5,
        T: float = 0.5,
        pol1: ArrayLike | None = None,
        pol2: ArrayLike | None = None,
    ) -> HOMResults:
        """
        Deprecated: use ``HOMAnalyzer.independent(results_a, results_b, ...)`` for photons from
        independent sources, or ``HOMAnalyzer.same_pair(results, ...)`` for signal and idler of
        one pair. This method is ``independent(results_list[0], results_list[1], ...)``.
        """
        warnings.warn(
            "compute_two_mode_HOM is deprecated; use HOMAnalyzer.independent(...) for independent "
            "sources or HOMAnalyzer.same_pair(...) for signal and idler of one pair.",
            DeprecationWarning,
            stacklevel=2,
        )
        if len(self.results_list) < 2:
            raise ValueError(
                "At least two results are required to compute two-mode HOM interference."
            )
        return HOMAnalyzer.independent(
            self.results_list[0],
            self.results_list[1],
            mode_a=mode1,
            mode_b=mode2,
            taus_s=taus_s,
            R=R,
            T=T,
            pol_a=pol1,
            pol_b=pol2,
        )
