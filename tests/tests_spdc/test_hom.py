import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.spdc_config import SPDCGridConfig
from photonpairlab.spdc.simulation import SPDC_Simulation
from photonpairlab.spdc.hom_analyser import HOMAnalyzer


def _make_results(wavelength_signal, wavelength_idler, steps=20, dev_nm=3.0):
    material = MaterialFactory.create("ktp1")
    crystal = Crystal(
        crystal_length=30e-3, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=46.2e-6, w=18e-6, T=25,
    )
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    crystal.generate_poling(
        laser=laser, mode="periodic",
        wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, resolution=5,
    )
    sim = SPDC_Simulation(
        crystal, laser,
        wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler,
        grid=SPDCGridConfig(steps=steps, dev_nm=dev_nm),
    )
    return sim.run()


@pytest.fixture
def results():
    # Non-degenerate: signal and idler centered on different wavelengths, so the
    # signal/idler reduced density matrices live on genuinely different frequency axes.
    return _make_results(1500e-9, 1610e-9)


def test_reduced_density_matrix_shapes_match_their_own_mode_axis(results):
    hom = HOMAnalyzer(results)
    rho_signal, _ = hom.get_reduced_density_matrix(mode="signal")
    rho_idler, _ = hom.get_reduced_density_matrix(mode="idler")

    n_signal = len(results.SignalWavelengths)
    n_idler = len(results.IdlerWavelengths)
    assert rho_signal.shape == (n_signal, n_signal)
    assert rho_idler.shape == (n_idler, n_idler)


def test_reduced_density_matrices_are_trace_one(results):
    hom = HOMAnalyzer(results)
    rho_signal, _ = hom.get_reduced_density_matrix(mode="signal")
    rho_idler, _ = hom.get_reduced_density_matrix(mode="idler")

    assert np.trace(rho_signal) == pytest.approx(1.0)
    assert np.trace(rho_idler) == pytest.approx(1.0)


def test_two_mode_hom_dip_matches_standard_formula_at_zero_delay():
    results = _make_results(1550e-9, 1550e-9)
    hom = HOMAnalyzer(results, results)

    hom_res = hom.compute_two_mode_HOM(mode1="signal", mode2="signal", R=0.5, T=0.5)

    zero_idx = np.argmin(np.abs(hom_res.tau_s))
    expected_min = 0.5 * (1.0 - hom_res.overlap_at_zero_delay)
    assert hom_res.coincidence_probabilities[zero_idx] == pytest.approx(expected_min, abs=1e-3)
