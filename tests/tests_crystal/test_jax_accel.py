import numpy as np
import pytest

pytest.importorskip("jax")

from photonpairlab.crystal.material.material_factory import MaterialFactory  # noqa: E402
from photonpairlab.crystal.phasematching.apm_strategy import APMPhaseMatching  # noqa: E402
from photonpairlab.crystal.phasematching.jax_accel import (  # noqa: E402
    bbo_like_n,
    general_sellmeier_n,
    group_index_jax,
    kato_takaoka_n,
    linear_thermal_n,
    refractive_index_jax,
)
from photonpairlab.laser import CWLaser  # noqa: E402


@pytest.mark.parametrize("wavelength_um", [0.4, 0.532, 0.8, 1.064])
def test_bbo_sellmeier_matches_numpy(wavelength_um):
    bbo = MaterialFactory.create("bbo")
    for axis in ("o", "e"):
        expected = bbo.refractive_index(wavelength_um, axis=axis)
        coeffs = bbo.material.sellmeier.data[axis]
        got = bbo_like_n(wavelength_um, coeffs["A"], coeffs["B"], coeffs["C"], coeffs["D"])
        assert float(got) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize("wavelength_um,T", [(0.8, 25.0), (1.064, 40.0), (1.55, 60.0)])
def test_ktp3_kato_takaoka_matches_numpy(wavelength_um, T):
    ktp3 = MaterialFactory.create("ktp3")
    for axis in ("x", "y", "z"):
        expected = ktp3.refractive_index(wavelength_um, axis=axis, temperature=T)
        coeffs = ktp3.material.sellmeier.data[axis]
        tc = ktp3.material.temperature_corrections.data[axis]
        got = kato_takaoka_n(
            wavelength_um,
            coeffs["A"], coeffs["B"], coeffs["C"], coeffs.get("D", 0.0) or 0.0, coeffs.get("E", 0.0) or 0.0,
            (tc["A"], tc["B"], tc["C"], tc["D"]),
            T,
        )
        assert float(got) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("wavelength_um,T", [(0.8, 25.0), (1.064, 45.0)])
def test_ktp2_linear_thermal_matches_numpy(wavelength_um, T):
    ktp2 = MaterialFactory.create("ktp2")
    for axis in ("x", "y", "z"):
        expected = ktp2.refractive_index(wavelength_um, axis=axis, temperature=T)
        coeffs = ktp2.material.sellmeier.data[axis]
        k = ktp2.material.temperature_corrections.data[axis]
        got = linear_thermal_n(wavelength_um, coeffs["A"], coeffs["B"], coeffs["C"], coeffs["D"], k, T)
        assert float(got) == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("wavelength_um", [0.8, 1.55])
def test_ktp1_general_sellmeier_matches_numpy(wavelength_um):
    ktp1 = MaterialFactory.create("ktp1")
    for axis in ("y", "z"):
        expected = ktp1.refractive_index(wavelength_um, axis=axis, temperature=25.0)
        coeffs = ktp1.material.sellmeier.data[axis]
        got = general_sellmeier_n(
            wavelength_um,
            coeffs["A"], coeffs["B"], coeffs["C"],
            coeffs.get("D", 0.0) or 0.0, coeffs.get("E", 0.0) or 0.0, coeffs.get("F", 0.0) or 0.0,
        )
        assert float(got) == pytest.approx(expected, rel=1e-8)


def test_refractive_index_jax_matches_numpy_for_bbo_effective_index():
    bbo = MaterialFactory.create("bbo")
    for theta in (0.0, 30.0, 45.0, 90.0):
        expected = bbo.effective_refractive_index(0.532, theta_deg=theta)
        got = refractive_index_jax(bbo, 0.532, "e", theta, 0.0, 25.0)
        assert float(got) == pytest.approx(expected, rel=1e-10)


def test_group_index_jax_matches_finite_difference_group_index():
    bbo = MaterialFactory.create("bbo")
    expected = bbo.group_index(0.532, axis="o")
    got = group_index_jax(bbo, 0.532, "o", angle_deg=0.0, phi_deg=0.0, T=25.0)
    assert float(got) == pytest.approx(expected, rel=1e-6)


def test_find_phase_matching_angles_jax_matches_scipy_loop():
    # Reproduces notebooks/demo.ipynb cell 22: BBO, type-IIeoe, degenerate SPDC pump scan.
    bbo = MaterialFactory.create("bbo")
    apm = APMPhaseMatching(bbo, spdc_type="type-IIeoe")

    pump_candidates_nm = np.arange(560.0, 610.0, 2.0)
    T = 25.0

    expected_angles = []
    for pump_nm in pump_candidates_nm:
        scan_pump = pump_nm * 1e-9
        scan_laser = CWLaser(scan_pump, bandwidth_wavelength=1e-9)
        expected_angles.append(apm.find_phase_matching_angle(scan_laser, 2 * scan_pump, 2 * scan_pump, T))
    expected_angles = np.array(expected_angles)

    pump_m = pump_candidates_nm * 1e-9
    got_angles = np.asarray(
        apm.find_phase_matching_angles_jax(pump_m, 2 * pump_m, 2 * pump_m, T=T)
    )

    np.testing.assert_allclose(got_angles, expected_angles, atol=1e-3)

    # Both should genuinely phase-match: Δk(angle) ≈ 0 at the found angle.
    for pump_nm, angle in zip(pump_candidates_nm, got_angles):
        scan_pump = pump_nm * 1e-9
        scan_laser = CWLaser(scan_pump, bandwidth_wavelength=1e-9)
        dk = apm.delta_k(float(angle), scan_laser, 2 * scan_pump, 2 * scan_pump, T)
        assert abs(dk) < 1e2  # near-zero on the 1e6-1e7 m^-1 scale of a mismatched Δk


def test_compute_phase_mismatch_jax_matches_scipy_loop():
    bbo = MaterialFactory.create("bbo")
    apm = APMPhaseMatching(bbo, spdc_type="type-IIeoe")

    pump_candidates_nm = np.arange(700.0, 720.0, 4.0)
    T = 25.0

    expected_N_pump = []
    for pump_nm in pump_candidates_nm:
        scan_pump = pump_nm * 1e-9
        scan_laser = CWLaser(scan_pump, bandwidth_wavelength=1e-9)
        result = apm.compute_phase_mismatch(scan_laser, 2 * scan_pump, 2 * scan_pump, T=T)
        expected_N_pump.append(result.N[0])
    expected_N_pump = np.array(expected_N_pump)

    pump_m = pump_candidates_nm * 1e-9
    batch = apm.compute_phase_mismatch_jax(pump_m, 2 * pump_m, 2 * pump_m, T=T)
    got_N_pump = np.asarray(batch["N"][0])

    np.testing.assert_allclose(got_N_pump, expected_N_pump, rtol=1e-5)
