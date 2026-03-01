from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

WavelengthRange = Tuple[float, float]

@dataclass(frozen=True)
class SPDCCenterConfig:
    """
    Center wavelengths for signal/idler used to compute phase mismatch and (if no explicit ranges)
    to build the grid around.
    """
    wavelength_signal: Optional[float] = None
    wavelength_idler: Optional[float] = None

@dataclass(frozen=True)
class SPDCGridConfig:
    """
    Defines the wavelength grid for SPDC simulations.

    Use either:
      - explicit ranges (signal_range / idler_range), OR
      - a center + dev_nm.
    """
    steps: int = 100
    dev_nm: float = 5.0

    signal_range: Optional[WavelengthRange] = None
    idler_range: Optional[WavelengthRange] = None

    def uses_explicit_ranges(self) -> bool:
        return (self.signal_range is not None) or (self.idler_range is not None)

    def validate(self) -> None:
        if self.steps < 2:
            raise ValueError("steps must be >= 2")

        if (self.signal_range is None) ^ (self.idler_range is None):
            # You *can* relax this if you want, but it avoids accidental mismatched grids.
            raise ValueError("Provide both signal_range and idler_range, or neither.")
        
        if self.signal_range is not None:
            s0, s1 = self.signal_range
            i0, i1 = self.idler_range  # type: ignore
            if not (s0 < s1):
                raise ValueError("signal_range must be (start, end) with start < end")
            if not (i0 < i1):
                raise ValueError("idler_range must be (start, end) with start < end")
            
@dataclass(frozen=True)
class SPDCRunConfig:
    """
    Everything needed to run an SPDC simulation on a grid.
    """
    center: SPDCCenterConfig = SPDCCenterConfig()
    grid: SPDCGridConfig = SPDCGridConfig()

    def validate(self) -> None:
        self.grid.validate()


def build_wavelength_axes(
    laser_wavelength_pump: float,
    cfg: SPDCRunConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns (signal_wavelengths, idler_wavelengths, wl_signal_center, wl_idler_center).
    """
    cfg.validate()

    wl_s0 = cfg.center.wavelength_signal or (2.0 * laser_wavelength_pump)
    wl_i0 = cfg.center.wavelength_idler or (2.0 * laser_wavelength_pump)

    steps = cfg.grid.steps

    if cfg.grid.signal_range is None:
        dev = cfg.grid.dev_nm * 1e-9
        s = np.linspace(wl_s0 - dev, wl_s0 + dev, steps)
        i = np.linspace(wl_i0 - dev, wl_i0 + dev, steps)
    else:
        assert cfg.grid.idler_range is not None, "idler_range must be provided when signal_range is provided"
        s0, s1 = cfg.grid.signal_range
        i0, i1 = cfg.grid.idler_range  # guaranteed by validate()
        s = np.linspace(s0, s1, steps)
        i = np.linspace(i0, i1, steps)

    return s, i, wl_s0, wl_i0