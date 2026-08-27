import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

WavelengthRange = tuple[float, float]


class SPDCCenterConfig(BaseModel):
    """
    Center wavelengths for signal/idler used to compute phase mismatch and (if no explicit ranges)
    to build the grid around.
    """
    model_config = ConfigDict(frozen=True)

    wavelength_signal: float | None = Field(default=None, gt=0)
    wavelength_idler: float | None = Field(default=None, gt=0)


class SPDCGridConfig(BaseModel):
    """
    Defines the wavelength grid for SPDC simulations.

    Use either:
      - explicit ranges (signal_range / idler_range), OR
      - a center + dev_nm.
    """
    model_config = ConfigDict(frozen=True)

    steps: int = Field(default=100, ge=2)
    dev_nm: float = Field(default=5.0, gt=0)

    signal_range: WavelengthRange | None = None
    idler_range: WavelengthRange | None = None

    def uses_explicit_ranges(self) -> bool:
        return (self.signal_range is not None) or (self.idler_range is not None)

    @model_validator(mode="after")
    def _check_ranges(self) -> "SPDCGridConfig":
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

        return self


class SPDCRunConfig(BaseModel):
    """
    Everything needed to run an SPDC simulation on a grid.
    """
    model_config = ConfigDict(frozen=True)

    center: SPDCCenterConfig = Field(default_factory=SPDCCenterConfig)
    grid: SPDCGridConfig = Field(default_factory=SPDCGridConfig)


def build_wavelength_axes(
    laser_wavelength_pump: float,
    cfg: SPDCRunConfig,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Returns (signal_wavelengths, idler_wavelengths, wl_signal_center, wl_idler_center).
    """
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
        i0, i1 = cfg.grid.idler_range  # guaranteed by validation
        s = np.linspace(s0, s1, steps)
        i = np.linspace(i0, i1, steps)

    return s, i, wl_s0, wl_i0
