import numpy as np

c_vac: float = 299792458

def bandwidth_wavelength_to_pulse_width(bandwidth, wavelength_pump):
    """
    Convert bandwidth to pulse width.

    Args:
        bandwidth (float): Bandwidth of the laser in meters.
        lambda_0 (float): Central wavelength of the laser in meters.

    Returns:
        float: Pulse width in seconds.
    """
    pulse_width = 2 * np.log(2) / np.pi * wavelength_pump ** 2 / (bandwidth * c_vac)
    return pulse_width

def pulse_duration_to_bandwidth_wavelength(pulse_width, wavelength_pump):
    """
    Convert pulse width to bandwidth.

    Args:
        pulse_width (float): Pulse width of the laser in seconds.
        lambda_0 (float): Central wavelength of the laser in meters.

    Returns:
        float: Bandwidth in meters.
    """
    bandwidth = 2 * np.log(2) / np.pi * wavelength_pump ** 2 / (pulse_width * c_vac)
    return bandwidth

def bandwidth_wavelength_to_angular_bandwidth(bandwidth_wavelength, wavelength_pump):
    """
    Convert bandwidth in wavelength to angular bandwidth.

    Args:
        bandwidth_wavelength (float): Bandwidth of the laser in meters.

    Returns:
        float: Angular bandwidth in radians per second.
    """
    angular_bandwidth = (2 * np.pi * c_vac) * bandwidth_wavelength/ (wavelength_pump ** 2 * 2 * np.sqrt(np.log(2)))
    return angular_bandwidth

def angular_bandwidth_to_bandwidth_wavelength(angular_bandwidth, wavelength_pump):
    """
    Convert angular bandwidth to bandwidth in wavelength.
    Args:
        angular_bandwidth (float): Angular bandwidth in radians per second.
    Returns:
        float: Bandwidth in meters.
    """
    bandwidth_wavelength = (wavelength_pump ** 2 * 2 * np.sqrt(np.log(2))) / (2 * np.pi * c_vac) * angular_bandwidth
    return bandwidth_wavelength
