import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class TwoModeHOMResults:
    """Data class to store HOM2 simulation results."""
    coincidence_probabilities: np.ndarray
    tau_s: np.ndarray
    tau_fs: np.ndarray
    overlap_at_zero_delay: float
    purity1: float
    purity2: float
