from photonpairlab.laser.base_laser import BaseLaser

from .utils_laser import (
    bandwidth_wavelength_to_angular_bandwidth,
    angular_bandwidth_to_bandwidth_wavelength,
)

class CWLaser(BaseLaser):
    """
    Represents a continuous-wave (CW) laser.
    """
    def __init__(self, wavelength, **kwargs):
        """
        Initializes the CWLaser object.

        Args:
            wavelength (float): Central wavelength of the laser in meters (m).
            **kwargs: Optional keyword arguments:
                - bandwidth_wavelength (float): Bandwidth in wavelength (meters).
                - angular_bandwidth (float): Bandwidth in angular frequency (radians/second).

        Raises:
            ValueError: If neither or both `bandwidth_wavelength` and `angular_bandwidth` are provided.
        """
        super().__init__(wavelength)
        
        # Extract optional parameters
        bandwidth_wavelength = kwargs.get("bandwidth_wavelength")
        angular_bandwidth = kwargs.get("angular_bandwidth")

        # Ensure only one of the parameters is provided
        if bandwidth_wavelength is not None and angular_bandwidth is not None:
            raise ValueError("Provide only one of `bandwidth_wavelength` or `angular_bandwidth`, not both.")
        if bandwidth_wavelength is None and angular_bandwidth is None:
            raise ValueError("You must provide either `bandwidth_wavelength` or `angular_bandwidth`.")

        # Handle the provided parameter
        if bandwidth_wavelength is not None:
            self.bandwidth_wavelength = bandwidth_wavelength
            self.angular_bandwidth = bandwidth_wavelength_to_angular_bandwidth(bandwidth_wavelength, self.wavelength_pump)
        elif angular_bandwidth is not None:
            self.angular_bandwidth = angular_bandwidth
            self.bandwidth_wavelength = angular_bandwidth_to_bandwidth_wavelength(angular_bandwidth, self.wavelength_pump)