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
        bandwidth = 2 * np.log(2) / np.pi * lambda_0 ** 2 / (PulseWidth * self.c)
        return bandwidth