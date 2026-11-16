import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation, SPDCGridConfig
from photonpairlab.spdc.analysis import (
    HOMAnalyzer,
    SpectralAnalyzer,
    exchange_overlap_vs_delay,
    hom_dip_vs_delay,
    linear_polarization,
    polarization_overlap,
    reduced_density_matrix,
    rho_overlap_vs_delay,
)
from photonpairlab.spdc.simulation.results import SPDCResults


def _make_results(wavelength_signal, wavelength_idler, steps=20, dev_nm=3.0):
    material = MaterialFactory.create("ktp1")
    crystal = Crystal(
        crystal_length=30e-3,
        material=material,
        pm_strategy="quasi",
        spdc_type="type-II",
        coherence_length=46.2e-6,
        w=18e-6,
        T=25,
    )
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)
    crystal.generate_poling(
        laser=laser,
        mode="periodic",
        wavelength_signal=wavelength_signal,
        wavelength_idler=wavelength_idler,
        resolution=5,
    )
    sim = SPDC_Simulation(
        crystal,
        laser,
        wavelength_signal=wavelength_signal,
        wavelength_idler=wavelength_idler,
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
    hom_res = HOMAnalyzer.independent(results, results, "signal", "signal", R=0.5, T=0.5)

    zero_idx = np.argmin(np.abs(hom_res.tau_s))
    expected_min = 0.5 * (1.0 - hom_res.overlap_at_zero_delay)
    assert hom_res.coincidence_probabilities[zero_idx] == pytest.approx(expected_min, abs=1e-3)


def test_default_polarization_reproduces_identical_polarization_result():
    results = _make_results(1550e-9, 1550e-9)
    taus = np.linspace(-3e-12, 3e-12, 101)

    default = HOMAnalyzer.independent(results, results, taus_s=taus)
    same_pol = HOMAnalyzer.independent(
        results,
        results,
        taus_s=taus,
        pol_a=linear_polarization(0.3),
        pol_b=linear_polarization(0.3),
    )

    np.testing.assert_allclose(
        default.coincidence_probabilities, same_pol.coincidence_probabilities
    )
    assert default.polarization_overlap == 1.0
    assert same_pol.polarization_overlap == pytest.approx(1.0)


def test_orthogonal_polarizations_give_no_dip():
    results = _make_results(1550e-9, 1550e-9)
    hom_res = HOMAnalyzer.independent(
        results,
        results,
        taus_s=np.linspace(-3e-12, 3e-12, 101),
        R=0.4,
        T=0.6,
        pol_a=linear_polarization(0.0),
        pol_b=linear_polarization(np.pi / 2),
    )

    np.testing.assert_allclose(hom_res.coincidence_probabilities, 0.4**2 + 0.6**2, atol=1e-12)


@pytest.mark.parametrize("theta", [0.1, 0.4, 0.65, 1.2])
def test_dip_depth_scales_with_cos_squared_of_polarization_angle(theta):
    results = _make_results(1550e-9, 1550e-9)
    taus = np.array([0.0])

    ref = HOMAnalyzer.independent(results, results, taus_s=taus)
    rot = HOMAnalyzer.independent(
        results,
        results,
        taus_s=taus,
        pol_a=linear_polarization(0.0),
        pol_b=linear_polarization(theta),
    )

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
    amp = np.exp(-(((f - f0) / 0.2e12) ** 2))
    psi1 = amp / np.linalg.norm(amp)
    psi2 = psi1 * np.exp(-1j * 2 * np.pi * f * t0)
    rho1 = np.outer(psi1, psi1.conj())
    rho2 = np.outer(psi2, psi2.conj())

    taus = np.linspace(-3e-12, 3e-12, 601)
    Pc = hom_dip_vs_delay(rho1, rho2, f, taus)

    assert taus[np.argmin(Pc)] == pytest.approx(-t0, abs=2e-14)
    assert Pc.min() == pytest.approx(0.0, abs=1e-6)


# --- same-pair vs. independent models ------------------------------------------------------------

C = 299792458.0


def _synthetic_results(jsa, wl):
    jsa = jsa / np.linalg.norm(jsa)
    return SPDCResults(
        Pump=np.ones_like(jsa, dtype=float),
        Phase=np.ones_like(jsa),
        JSI=np.abs(jsa) ** 2,
        JSA=jsa,
        SchmidtCoefficients=None,
        Purity=None,
        K=None,
        SignalWavelengths=wl,
        IdlerWavelengths=wl,
        dev=0.0,
        c=C,
    )


@pytest.fixture
def grid():
    wl = np.linspace(1545e-9, 1555e-9, 61)
    return wl, C / wl


def _gauss(f, f0, width, t0=0.0):
    return np.exp(-(((f - f0) / width) ** 2)) * np.exp(-1j * 2 * np.pi * f * t0)


def test_exchange_overlap_equals_rho_overlap_for_factorizable_jsa(grid):
    # Different, chirped signal/idler spectra so the overlap is nontrivial at every delay.
    wl, f = grid
    phi_s = _gauss(f, C / 1549.5e-9, 0.15e12, t0=0.3e-12)
    phi_i = _gauss(f, C / 1550.5e-9, 0.2e12)
    jsa = np.outer(phi_i, phi_s)  # (idler, signal)
    taus = np.linspace(-3e-12, 3e-12, 121)

    X_exch = exchange_overlap_vs_delay(jsa, f, taus)
    X_rho = rho_overlap_vs_delay(
        reduced_density_matrix(jsa, "signal"), reduced_density_matrix(jsa, "idler"), f, taus
    )

    np.testing.assert_allclose(X_exch, X_rho, atol=1e-12)


def test_exchange_symmetric_correlated_jsa_interferes_perfectly(grid):
    # Strongly anti-correlated but signal<->idler symmetric: same-pair X(0) = 1, while the
    # density-matrix model would be capped near the (low) purity.
    wl, f = grid
    F, Fp = np.meshgrid(f, f)  # F: signal axis (1), Fp: idler axis (0)
    f0 = C / 1550e-9
    jsa = np.exp(-(((F + Fp - 2 * f0) / 0.05e12) ** 2)) * np.exp(-(((F - Fp) / 0.6e12) ** 2))
    res = _synthetic_results(jsa.astype(complex), wl)

    same = HOMAnalyzer.same_pair(res, taus_s=np.array([0.0]))
    with pytest.warns(UserWarning, match="same_pair"):
        wrong = HOMAnalyzer.independent(res, res, "signal", "idler", taus_s=np.array([0.0]))

    assert same.overlap_at_zero_delay == pytest.approx(1.0, abs=1e-12)
    assert same.visibility == pytest.approx(1.0, abs=1e-12)
    assert same.purity1 < 0.5
    assert wrong.overlap_at_zero_delay < 0.6


def test_identical_independent_sources_have_visibility_equal_to_purity():
    results = _make_results(1550e-9, 1550e-9)
    taus = np.linspace(-1e-12, 1e-12, 201)  # contains tau = 0
    hom_res = HOMAnalyzer.independent(results, results, "signal", "signal", taus_s=taus)
    _, purity, _ = SpectralAnalyzer(results).schmidt_decomposition()

    assert hom_res.model == "independent"
    assert hom_res.visibility == pytest.approx(purity, rel=1e-6)
    assert hom_res.purity1 == pytest.approx(purity, rel=1e-6)


def test_polarization_scales_depth_but_not_spectral_shape():
    results = _make_results(1550e-9, 1550e-9)
    taus = np.linspace(-3e-12, 3e-12, 101)
    theta = 0.5
    ref = HOMAnalyzer.independent(results, results, taus_s=taus)
    rot = HOMAnalyzer.independent(
        results,
        results,
        taus_s=taus,
        pol_a=linear_polarization(0.0),
        pol_b=linear_polarization(theta),
    )

    np.testing.assert_allclose(rot.spectral_overlap, ref.spectral_overlap)
    np.testing.assert_allclose(
        0.5 - rot.coincidence_probabilities,
        np.cos(theta) ** 2 * (0.5 - ref.coincidence_probabilities),
        atol=1e-14,
    )
    assert rot.visibility == pytest.approx(np.cos(theta) ** 2 * ref.visibility)


@pytest.mark.parametrize("R", [0.5, 0.3])
def test_visibility_breakdown_multiplies_to_total(grid, R):
    wl, f = grid
    jsa = np.outer(_gauss(f, C / 1550e-9, 0.2e12), _gauss(f, C / 1550.3e-9, 0.2e12))
    res = _synthetic_results(jsa, wl)
    hom_res = HOMAnalyzer.same_pair(
        res,
        taus_s=np.linspace(-2e-12, 2e-12, 81),
        R=R,
        T=1 - R,
        pol_signal=linear_polarization(0.0),
        pol_idler=linear_polarization(0.4),
    )
    b = hom_res.visibility_breakdown()

    assert b["beam_splitter"] == pytest.approx(2 * R * (1 - R) / (R**2 + (1 - R) ** 2))
    assert b["total"] == pytest.approx(b["beam_splitter"] * b["polarization"] * b["spectral"])
    assert hom_res.model == "same_pair"


def test_same_pair_rejects_different_signal_idler_grids(results):
    with pytest.raises(ValueError, match="identical signal and idler"):
        HOMAnalyzer.same_pair(results)


def test_deprecated_wrapper_matches_independent():
    results = _make_results(1550e-9, 1550e-9)
    taus = np.linspace(-2e-12, 2e-12, 41)
    with pytest.deprecated_call():
        old = HOMAnalyzer(results, results).compute_two_mode_HOM("signal", "signal", taus_s=taus)
    new = HOMAnalyzer.independent(results, results, taus_s=taus)

    np.testing.assert_array_equal(old.coincidence_probabilities, new.coincidence_probabilities)
    np.testing.assert_allclose(old.tau_fs, taus * 1e15)


def test_marginal_centroids_track_signal_idler_detuning(grid):
    wl, f = grid
    wl_s, wl_i = 1549.0e-9, 1551.5e-9
    jsa = np.outer(_gauss(f, C / wl_i, 0.1e12), _gauss(f, C / wl_s, 0.1e12))
    s_nm, i_nm = SpectralAnalyzer(_synthetic_results(jsa, wl)).marginal_centroids_nm()

    assert s_nm == pytest.approx(wl_s * 1e9, abs=0.01)
    assert i_nm == pytest.approx(wl_i * 1e9, abs=0.01)
