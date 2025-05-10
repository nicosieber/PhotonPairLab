from photonpairlab.laser.laser_base import LaserBase

class CWLaser(LaserBase):
    """
    Represents a continuous-wave (CW) laser.
    """
    def __init__(self, wavelength, bandwidth):
        """
        Initializes the CWLaser object.

        Args:
            wavelength (float): Central wavelength of the laser in meters (m).
            bandwidth (float): Bandwidth of the laser in meters (m).
        """
        super().__init__(wavelength)
        self.bandwidth = bandwidth  # Bandwidth (m)