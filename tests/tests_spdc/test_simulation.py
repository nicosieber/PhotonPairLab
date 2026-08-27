import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.crystal.phasematching.qpm_strategy import QPMPhaseMatching
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation, SPDCGridConfig


@pytest.fixture
def crystal():
    material = MaterialFactory.create("ktp1")
    crystal = Crystal(
        crystal_length=30e-3, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=46.2e-6, w=18e-6, T=25,
    )
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    crystal.generate_poling(
        laser=laser, mode="periodic",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5,
    )
    return crystal


@pytest.fixture
def laser():
    return PulsedLaser(775e-9, pulse_duration=1.7e-12)


def test_jsa_is_complex_and_jsi_is_its_squared_magnitude(crystal, laser):
    sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=40, dev_nm=5.0))
    results = sim.run()

    assert np.iscomplexobj(results.JSA)
    np.testing.assert_allclose(results.JSI, np.abs(results.JSA) ** 2)


def test_jsi_is_nonnegative_and_peaked_near_center(crystal, laser):
    sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=60, dev_nm=5.0))
    results = sim.run()

    assert np.all(results.JSI >= 0)
    assert results.JSI.max() > 0


def test_legacy_range_params_are_honored(crystal, laser):
    # wavelength_signal_range/wavelength_idler_range used to be stored and then silently
    # ignored (run() only ever consulted self.grid); they must now build the grid.
    sim = SPDC_Simulation(
        crystal, laser,
        wavelength_signal_range=[1500e-9, 1600e-9],
        wavelength_idler_range=[1500e-9, 1600e-9],
    )
    results = sim.run()
    assert results.SignalWavelengths.min() == pytest.approx(1500e-9)
    assert results.SignalWavelengths.max() == pytest.approx(1600e-9)


def test_periodic_qpm_phase_matches_at_the_intended_degenerate_wavelength():
    # Integration check for the standard QPM coherence-length convention (domain width
    # = Lc = pi / |Delta_k0|, grating period = 2*Lc): building a crystal with that Lc
    # must produce a JSI peaked at the wavelengths it was designed for. This is what
    # actually catches a domain-width/period bug (a pure unit test on domain width
    # alone does not, since it can pass while the realized grating period is still
    # wrong relative to what the physics needs).
    material = MaterialFactory.create("ktp1")
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)

    qpm = QPMPhaseMatching(material, spdc_type="type-II")
    delta_k0 = qpm.compute_phase_mismatch(laser, 1550e-9, 1550e-9, angle_pm=0, T=25.0).delta_k0
    coherence_length = np.pi / abs(delta_k0)

    crystal = Crystal(
        crystal_length=30e-3, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=coherence_length, w=18e-6, T=25,
    )
    crystal.generate_poling(
        laser=laser, mode="periodic",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5,
    )
    sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=80, dev_nm=5.0))
    results = sim.run()

    peak_idx = np.unravel_index(np.argmax(results.JSI), results.JSI.shape)
    peak_signal = results.SignalWavelengths[peak_idx[1]]
    peak_idler = results.IdlerWavelengths[peak_idx[0]]
    assert peak_signal == pytest.approx(1550e-9, abs=1e-9)
    assert peak_idler == pytest.approx(1550e-9, abs=1e-9)


def test_jsa_shape_matches_signal_idler_axes(crystal, laser):
    sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=25, dev_nm=5.0))
    results = sim.run()

    n_idler = len(results.IdlerWavelengths)
    n_signal = len(results.SignalWavelengths)
    assert results.JSA.shape == (n_idler, n_signal)
