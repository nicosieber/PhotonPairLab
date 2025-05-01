import numpy as np
class Laser:
    """
    Represents the laser (pump source) used in the simulation.
    """
    def __init__(self, wavelength, pulse_duration):
        """
        Initializes the Laser object with the given parameters.

        Args:
            wavelength (float): Central wavelength of the pump in meters (m).
            pulse_duration (float): Pulse duration in seconds (s).

        Attributes:
            lambda_2w (float): Central wavelength of the pump (m).
            lambda_w (float): Central wavelength of the down-converted photons (m), 
                              calculated as twice the pump wavelength.
            pulse_duration (float): Pulse duration (s).
            c (float): Speed of light in meters per second (m/s).
            FWHM (float): Full Width at Half Maximum (FWHM) of the pulse bandwidth, 
                          calculated using the pulse duration and pump wavelength.
        """
        self.lambda_2w = wavelength  
        self.lambda_w = 2 * wavelength  
        self.pulse_duration = pulse_duration  
        self.c = 299792458  
        self.FWHM = self.pulse_width_to_bandwidth(self.pulse_duration, self.lambda_2w)

    def pulse_width_to_bandwidth(self, pulse_width, lambda_0):
        """
        Calculate the bandwidth of a laser pulse given its pulse width and central wavelength.

        Parameters:
        -----------
        pulse_width : float
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
        bandwidth = 2 * np.log(2) / np.pi * lambda_0 ** 2 / (pulse_width * self.c)
        return bandwidth
