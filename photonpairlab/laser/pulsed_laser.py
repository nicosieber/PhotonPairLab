from photonpairlab.laser.base_laser import BaseLaser

class PulsedLaser(BaseLaser):
    """
    Represents a pulsed laser.
    """
    def __init__(self, wavelength, pulse_duration=None, repetition_rate=None):
        """
        Initializes the PulsedLaser object.

        Args:
            wavelength (float): Central wavelength of the laser in meters (m).
            pulse_duration (float): Pulse duration in seconds (s).
        """
        super().__init__(wavelength)

        self.pulse_duration = pulse_duration  # Pulse duration (s)
        self.bandwidth_wavelength = self.pulse_duration_to_bandwidth_wavelength(self.pulse_duration)  # Bandwidth (m)

        self.repetition_rate = repetition_rate  # Repetition rate (Hz)
