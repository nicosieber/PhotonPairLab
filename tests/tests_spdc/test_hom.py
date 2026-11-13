import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation, SPDCGridConfig
from photonpairlab.spdc.analysis import HOMAnalyzer, hom_dip_vs_delay, linear_polarization, polarization_overlap


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


def test_default_polarization_reproduces_identical_polarization_result():
    results = _make_results(1550e-9, 1550e-9)
    hom = HOMAnalyzer(results, results)
    taus = np.linspace(-3e-12, 3e-12, 101)

    default = hom.compute_two_mode_HOM(taus_s=taus)
    same_pol = hom.compute_two_mode_HOM(taus_s=taus, pol1=linear_polarization(0.3), pol2=linear_polarization(0.3))

    np.testing.assert_allclose(default.coincidence_probabilities, same_pol.coincidence_probabilities)
    assert default.polarization_overlap == 1.0
    assert same_pol.polarization_overlap == pytest.approx(1.0)


def test_orthogonal_polarizations_give_no_dip():
    results = _make_results(1550e-9, 1550e-9)
    hom = HOMAnalyzer(results, results)

    hom_res = hom.compute_two_mode_HOM(
        taus_s=np.linspace(-3e-12, 3e-12, 101), R=0.4, T=0.6,
        pol1=linear_polarization(0.0), pol2=linear_polarization(np.pi / 2),
    )

    np.testing.assert_allclose(hom_res.coincidence_probabilities, 0.4**2 + 0.6**2, atol=1e-12)


@pytest.mark.parametrize("theta", [0.1, 0.4, 0.65, 1.2])
def test_dip_depth_scales_with_cos_squared_of_polarization_angle(theta):
    results = _make_results(1550e-9, 1550e-9)
    hom = HOMAnalyzer(results, results)
    taus = np.array([0.0])

    ref = hom.compute_two_mode_HOM(taus_s=taus)
    rot = hom.compute_two_mode_HOM(taus_s=taus, pol1=linear_polarization(0.0), pol2=linear_polarization(theta))

    depth_ref = 0.5 - ref.coincidence_probabilities[0]
    depth_rot = 0.5 - rot.coincidence_probabilities[0]
    assert depth_rot == pytest.approx(np.cos(theta) ** 2 * depth_ref, rel=1e-10)
    assert rot.polarization_overlap == pytest.approx(np.cos(theta) ** 2)


def test_polarization_overlap_handles_elliptical_states():
    h = np.array([1, 0])
    diag_circ = np.array([1, 1j]) / np.sqrt(2)  # circular: equal H/V weight, 90° relative phase
    assert polarization_overlap(h, diag_circ) == pytest.approx(0.5)
    assert polarization_overlap(np.array([1, 1j]), np.array([1, -1j])) == pytest.approx(0.0)
    assert polarization_overlap(None, h) == 1.0


def test_delay_axis_uses_frequencies_in_rho_order():
    # Photon 2 carries a linear spectral phase, i.e. it arrives t0 later. The dip must then sit at
    # tau = -t0 (the delay that compensates it). Sorting the frequencies would mirror it to +t0.
    wl = np.linspace(1545e-9, 1555e-9, 81)  # ascending wavelength -> descending frequency
    c = 299792458.0
    f = c / wl
    f0 = c / 1550e-9
    t0 = 1.5e-12
    amp = np.exp(-((f - f0) / 0.2e12) ** 2)
    psi1 = amp / np.linalg.norm(amp)
    psi2 = psi1 * np.exp(-1j * 2 * np.pi * f * t0)
    rho1 = np.outer(psi1, psi1.conj())
    rho2 = np.outer(psi2, psi2.conj())

    taus = np.linspace(-3e-12, 3e-12, 601)
    Pc = hom_dip_vs_delay(rho1, rho2, f, taus)

    assert taus[np.argmin(Pc)] == pytest.approx(-t0, abs=2e-14)
    assert Pc.min() == pytest.approx(0.0, abs=1e-6)
