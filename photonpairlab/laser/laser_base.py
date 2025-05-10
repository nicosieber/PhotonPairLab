import numpy as np

class LaserBase:
    """
    Base class for lasers. Contains common attributes and methods.
    """
    def __init__(self, wavelength):
        """
        Initializes the LaserBase object.

        Args:
            wavelength (float): Central wavelength of the laser in meters (m).
        """
        self.lambda_2w = wavelength  # Central wavelength of the pump (m)
        self.lambda_w = 2 * wavelength  # Central wavelength of down-converted photons (m)
        self.c = 299792458  # Speed of light in meters per second (m/s)

    def bandwidth_to_pulse_width(self, bandwidth, lambda_0):
        """
        Convert bandwidth to pulse width.

        Args:
            bandwidth (float): Bandwidth of the laser in meters.
            lambda_0 (float): Central wavelength of the laser in meters.

        Returns:
            float: Pulse width in seconds.
        """
        pulse_width = 2 * np.log(2) / np.pi * lambda_0 ** 2 / (bandwidth * self.c)
        return pulse_width

    def pulse_width_to_bandwidth(self, pulse_width, lambda_0):
        """
        Convert pulse width to bandwidth.

        Args:
            pulse_width (float): Pulse width of the laser in seconds.
            lambda_0 (float): Central wavelength of the laser in meters.

        Returns:
            float: Bandwidth in meters.
        """
        bandwidth = 2 * np.log(2) / np.pi * lambda_0 ** 2 / (pulse_width * self.c)
        return bandwidth