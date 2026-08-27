import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser


@pytest.fixture
def material():
    return MaterialFactory.create("ktp1")


def _realized_domain_width(crystal):
    resolution = 5
    signs = crystal.poling_pattern[::resolution]
    total_length = crystal.z[-1] - crystal.z[0]
    return total_length / len(signs)


def test_periodic_poling_domain_width_matches_coherence_length(material):
    coherence_length = 46.2e-6
    crystal = Crystal(
        crystal_length=30e-3, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=coherence_length, w=18e-6, T=25,
    )
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    crystal.generate_poling(
        laser=laser, mode="periodic",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5,
    )
    assert _realized_domain_width(crystal) == pytest.approx(coherence_length, rel=1e-3)


def test_periodic_poling_alternates_sign(material):
    coherence_length = 46.2e-6
    crystal = Crystal(
        crystal_length=30e-3, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=coherence_length, w=18e-6, T=25,
    )
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    crystal.generate_poling(
        laser=laser, mode="periodic",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5,
    )
    signs = crystal.poling_pattern[::5]
    assert set(np.unique(signs)) == {1, -1}
    assert np.all(np.diff(signs) != 0)


def test_subcoh_poling_does_not_crash_for_e_polarization_spdc_type():
    # type-I maps to ('e', 'o', 'o') in POLARIZATION_MAP; this used to raise TypeError
    # because _generate_subcoh_poling passed angle=None into an angle-dependent index call.
    # ktp1's model doesn't implement effective_refractive_index at all, so use ktp2
    # (SellmeierLinearThermalModel), which does support 'e'-polarization lookups.
    material = MaterialFactory.create("ktp2")
    crystal = Crystal(
        crystal_length=2e-3, material=material, pm_strategy="quasi",
        spdc_type="type-I", coherence_length=10e-6, w=2e-6, T=25,
    )
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    crystal.generate_poling(
        laser=laser, mode="subcoh",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9,
    )
    assert crystal.poling_pattern is not None
    assert set(np.unique(crystal.poling_pattern)).issubset({1, -1})
