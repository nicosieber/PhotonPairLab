import pytest

from photonpairlab.laser import CWLaser
from photonpairlab.laser.utils_laser import (
    bandwidth_wavelength_to_angular_bandwidth,
    angular_bandwidth_to_bandwidth_wavelength,
)


def test_cw_laser_from_bandwidth_wavelength():
    laser = CWLaser(775e-9, bandwidth_wavelength=1e-9)
    assert laser.bandwidth_wavelength == 1e-9
    expected = bandwidth_wavelength_to_angular_bandwidth(1e-9, 775e-9)
    assert laser.angular_bandwidth == pytest.approx(expected)


def test_cw_laser_from_angular_bandwidth():
    laser = CWLaser(775e-9, angular_bandwidth=1e9)
    assert laser.angular_bandwidth == 1e9
    expected = angular_bandwidth_to_bandwidth_wavelength(1e9, 775e-9)
    assert laser.bandwidth_wavelength == pytest.approx(expected)


def test_cw_laser_requires_exactly_one_bandwidth_spec():
    with pytest.raises(ValueError):
        CWLaser(775e-9)
    with pytest.raises(ValueError):
        CWLaser(775e-9, bandwidth_wavelength=1e-9, angular_bandwidth=1e9)


def test_bandwidth_wavelength_angular_roundtrip():
    bandwidth_wavelength = 2e-9
    wavelength_pump = 800e-9
    angular = bandwidth_wavelength_to_angular_bandwidth(bandwidth_wavelength, wavelength_pump)
    back = angular_bandwidth_to_bandwidth_wavelength(angular, wavelength_pump)
    assert back == pytest.approx(bandwidth_wavelength)
