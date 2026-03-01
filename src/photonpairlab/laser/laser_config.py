from dataclasses import dataclass

@dataclass
class LaserConfig:
    wavelength_pump: float = 775e-9
    pulse_duration: float = 1.7e-12
