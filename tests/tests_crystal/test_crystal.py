import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.crystal.phasematching.qpm_strategy import QPMPhaseMatching
from photonpairlab.laser import PulsedLaser


@pytest.fixture
def material():
    return MaterialFactory.create("ktp1")


@pytest.fixture
def laser():
    return PulsedLaser(775e-9, pulse_duration=1.7e-12)


def test_crystal_applies_thermal_expansion(material):
    crystal = Crystal(crystal_length=30e-3, material=material, pm_strategy="quasi", T=25.0)
    assert crystal.temperature_adjusted_crystal_length == pytest.approx(30e-3, rel=1e-2)


def test_crystal_rejects_unknown_pm_strategy(material):
    with pytest.raises(KeyError):
        Crystal(crystal_length=30e-3, material=material, pm_strategy="not-a-strategy")


def test_crystal_delta_k_matches_strategy(material):
    crystal = Crystal(crystal_length=30e-3, material=material, pm_strategy="quasi", spdc_type="type-II", T=25.0)
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    delta_k = crystal.delta_k(0, laser, 1550e-9, 1550e-9, 25.0)
    assert isinstance(delta_k, float)


def test_ideal_coherence_length_matches_pi_over_delta_k(material, laser):
    crystal = Crystal(crystal_length=30e-3, material=material, pm_strategy="quasi", spdc_type="type-II", T=20.0)
    Lc = crystal.ideal_coherence_length(laser, 1550e-9, 1550e-9)

    qpm = QPMPhaseMatching(material, spdc_type="type-II")
    delta_k0 = qpm.compute_phase_mismatch(laser, 1550e-9, 1550e-9, angle_pm=0, T=20.0).delta_k0
    assert Lc == pytest.approx(np.pi / abs(delta_k0))


def test_generate_poling_auto_fills_missing_coherence_length(material, laser):
    crystal = Crystal(crystal_length=30e-3, material=material, pm_strategy="quasi", spdc_type="type-II", w=18e-6, T=20.0)
    assert crystal.coherence_length is None

    crystal.generate_poling(laser=laser, mode="periodic", wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5)

    expected_Lc = crystal.ideal_coherence_length(laser, 1550e-9, 1550e-9)
    assert crystal.coherence_length == pytest.approx(expected_Lc)

    # Realized domain width should match the auto-filled coherence_length (see qpm_strategy tests
    # for the general domain-width regression check).
    signs = crystal.poling_pattern[::5]
    realized_width = (crystal.z[-1] - crystal.z[0]) / len(signs)
    assert realized_width == pytest.approx(crystal.coherence_length, rel=1e-3)


def test_generate_poling_does_not_overwrite_explicit_coherence_length(material, laser):
    crystal = Crystal(
        crystal_length=30e-3, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=99e-6, w=18e-6, T=20.0,
    )
    crystal.generate_poling(laser=laser, mode="periodic", wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5)
    assert crystal.coherence_length == 99e-6
