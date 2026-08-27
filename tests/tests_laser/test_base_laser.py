from photonpairlab.laser.base_laser import BaseLaser


def test_base_laser_stores_wavelength_and_defaults():
    laser = BaseLaser(775e-9)
    assert laser.wavelength_pump == 775e-9
    assert laser.c == 299792458
    assert laser.bandwidth_wavelength is None
    assert laser.angular_bandwidth is None
