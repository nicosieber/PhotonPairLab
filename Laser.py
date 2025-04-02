import numpy as np
class Laser:
    """
    Represents the laser (pump source) used in the simulation.
    """
    def __init__(self, wavelength, pulse_duration):
        self.lambda_2w = wavelength  # Central wavelength of the pump (m)
        self.lambda_w = 2 * wavelength  # Central wavelength of down-converted photons (m)
        self.pulse_duration = pulse_duration  # Pulse duration (s)
        # Constants
        self.c = 3e8  # Speed of light (m/s)
        self.FWHM = self.pulse_width_to_bandwidth(self.pulse_duration, self.lambda_2w)

    def pulse_width_to_bandwidth(self, PulseWidth, lambda_0):
        """
        Calculate the bandwidth of a laser pulse given its pulse width and central wavelength.

        Parameters:
        -----------
        PulseWidth : float
            The temporal pulse width of the laser in seconds.
        lambda_0 : float
            The central wavelength of the laser in meters.

        Returns:
        --------
        float
            The bandwidth of the laser pulse in meters.

        Notes:
        ------
        The calculation assumes a Gaussian pulse shape and uses the relationship:
            bandwidth = (2 * ln(2) / π) * (lambda_0^2 / (PulseWidth * c))
        where `c` is the speed of light in vacuum.
        """
        bandwidth = 2 * np.log(2) / np.pi * lambda_0 ** 2 / (PulseWidth * self.c)
        return bandwidth