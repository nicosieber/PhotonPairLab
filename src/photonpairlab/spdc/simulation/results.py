import numpy as np

from dataclasses import dataclass


@dataclass(frozen=True)
class SPDCResults:
    """Data class to store SPDC simulation results."""
    Pump: np.ndarray
    Phase: np.ndarray
    JSI: np.ndarray
    JSA: np.ndarray
    SchmidtCoefficients: np.ndarray | None
    Purity: float | None
    K: float | None
    SignalWavelengths: np.ndarray
    IdlerWavelengths: np.ndarray
    dev: float
    c: float