import pytest

from photonpairlab.laser import PulsedLaser
from photonpairlab.laser.utils_laser import (
    pulse_duration_to_bandwidth_wavelength,
    bandwidth_wavelength_to_pulse_width,
)


def test_pulsed_laser_derives_bandwidth_from_pulse_duration():
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    expected = pulse_duration_to_bandwidth_wavelength(1.7e-12, 775e-9)
    assert laser.bandwidth_wavelength == pytest.approx(expected)
    assert laser.pulse_duration == 1.7e-12


def test_pulsed_laser_stores_repetition_rate():
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12, repetition_rate=80e6)
    assert laser.repetition_rate == 80e6


def test_pulse_duration_bandwidth_roundtrip():
    pulse_duration = 1.7e-12
    wavelength_pump = 775e-9
    bandwidth = pulse_duration_to_bandwidth_wavelength(pulse_duration, wavelength_pump)
    back = bandwidth_wavelength_to_pulse_width(bandwidth, wavelength_pump)
    assert back == pytest.approx(pulse_duration)
