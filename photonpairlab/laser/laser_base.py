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

    def bandwidth_wavelength_to_pulse_width(self, bandwidth):
        """
        Convert bandwidth to pulse width.

        Args:
            bandwidth (float): Bandwidth of the laser in meters.
            lambda_0 (float): Central wavelength of the laser in meters.

        Returns:
            float: Pulse width in seconds.
        """
        pulse_width = 2 * np.log(2) / np.pi * self.lambda_2w ** 2 / (bandwidth * self.c)
        return pulse_width

    def pulse_duration_to_bandwidth_wavelength(self, pulse_width):
        """
        Convert pulse width to bandwidth.

        Args:
            pulse_width (float): Pulse width of the laser in seconds.
            lambda_0 (float): Central wavelength of the laser in meters.

        Returns:
            float: Bandwidth in meters.
        """
        bandwidth = 2 * np.log(2) / np.pi * self.lambda_2w ** 2 / (pulse_width * self.c)
        return bandwidth
    
    def bandwidth_wavelength_to_angular_bandwidth(self, bandwidth_wavelength):
        """
        Convert bandwidth in wavelength to angular bandwidth.

        Args:
            bandwidth_wavelength (float): Bandwidth of the laser in meters.

        Returns:
            float: Angular bandwidth in radians per second.
        """
        angular_bandwidth = (2 * np.pi * self.c) * bandwidth_wavelength/ (self.lambda_2w ** 2 * 2 * np.sqrt(np.log(2)))
        return angular_bandwidth
    
    def angular_bandwidth_to_bandwidth_wavelength(self, angular_bandwidth):
        """
        Convert angular bandwidth to bandwidth in wavelength.
        Args:
            angular_bandwidth (float): Angular bandwidth in radians per second.
        Returns:
            float: Bandwidth in meters.
        """
        bandwidth_wavelength = (self.lambda_2w ** 2 * 2 * np.sqrt(np.log(2))) / (2 * np.pi * self.c) * angular_bandwidth
        return bandwidth_wavelength
    
