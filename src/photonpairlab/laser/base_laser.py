import numpy as np

class BaseLaser:
    """
    Base class for lasers. Contains common attributes and methods.
    """
    def __init__(self, wavelength_pump):
        """
        Initializes the LaserBase object.

        Args:
            wavelength (float): Central wavelength of the laser in meters (m).
        """
        self.wavelength_pump: float = wavelength_pump  # Central wavelength of the pump (m)
        self.c: float = 299792458  # Speed of light in meters per second (m/s)
        self.bandwidth_wavelength: float | None = None  # Bandwidth in wavelength (m)
        self.angular_bandwidth: float | None = None  # Bandwidth in angular frequency (rad/s)