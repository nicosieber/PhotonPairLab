from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from photonpairlab._typing import FloatArray, HOMModel


@dataclass(frozen=True)
class HOMResults:
    """
    Result of a HOM calculation, keeping the three factors of

        Pc(tau) = R^2 + T^2 - 2RT * |<p1|p2>|^2 * X(tau)

    separate: the beam-splitter balance and the polarization overlap only scale the dip depth,
    while the spectral interference term X(tau) also sets its shape.

    Attributes
    ----------
    model : "same_pair" or "independent"
        Which spectral model produced ``spectral_overlap`` (JSA exchange overlap or Tr[rho1 rho2]).
    tau_s : delays in seconds.
    coincidence_probabilities : Pc(tau).
    spectral_overlap : X(tau).
    overlap_at_zero_delay : X(0), evaluated exactly at tau = 0 (not read off the delay grid).
    polarization_overlap : |<p1|p2>|^2.
    R, T : beam-splitter reflectivity and transmissivity.
    purity1, purity2 : spectral purities Tr[rho^2] of the two interfering photons.
    """

    model: HOMModel
    tau_s: FloatArray
    coincidence_probabilities: FloatArray
    spectral_overlap: FloatArray
    overlap_at_zero_delay: float
    polarization_overlap: float
    R: float
    T: float
    purity1: float
    purity2: float

    @property
    def tau_fs(self) -> FloatArray:
        return self.tau_s * 1e15

    @property
    def baseline(self) -> float:
        """Coincidence probability far outside the dip (distinguishable photons): R^2 + T^2."""
        return self.R**2 + self.T**2

    @property
    def beam_splitter_factor(self) -> float:
        """2RT / (R^2 + T^2): the visibility ceiling set by the beam-splitter balance."""
        return 2 * self.R * self.T / self.baseline

    @property
    def spectral_visibility(self) -> float:
        """
        max_tau X(tau) over the sampled delays: the visibility a perfect beam splitter and
        polarization would give.
        """
        return float(np.max(self.spectral_overlap))

    @property
    def visibility(self) -> float:
        """
        (baseline - min Pc) / baseline
        = beam_splitter_factor * polarization_overlap * spectral_visibility.
        """
        return float((self.baseline - np.min(self.coincidence_probabilities)) / self.baseline)

    def visibility_breakdown(self) -> dict[str, float]:
        """The factors whose product is ``visibility``."""
        return {
            "beam_splitter": self.beam_splitter_factor,
            "polarization": self.polarization_overlap,
            "spectral": self.spectral_visibility,
            "total": self.visibility,
        }


#: Backwards-compatible name.
TwoModeHOMResults = HOMResults
