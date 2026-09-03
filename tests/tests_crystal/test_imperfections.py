import numpy as np
import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.crystal.phasematching.pm_result import PolingResult
from photonpairlab.crystal.phasematching.qpm_strategy import QPMPhaseMatching
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation, SPDCGridConfig

COHERENCE_LENGTH = 46.2e-6
CRYSTAL_LENGTH = 30e-3
SUBCOH_CRYSTAL_LENGTH = 2e-3
SUBCOH_W = 18e-6
RESOLUTION = 20


@pytest.fixture
def material():
    return MaterialFactory.create("ktp1")


@pytest.fixture
def laser():
    return PulsedLaser(775e-9, pulse_duration=1.7e-12)


def _periodic_crystal(material, laser, resolution=RESOLUTION):
    crystal = Crystal(
        crystal_length=CRYSTAL_LENGTH, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=COHERENCE_LENGTH, T=25,
    )
    poling = crystal.generate_poling(
        laser=laser, mode="periodic",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=resolution,
    )
    return crystal, poling


def _constant_crystal(material, laser, resolution=RESOLUTION):
    crystal = Crystal(
        crystal_length=CRYSTAL_LENGTH, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=COHERENCE_LENGTH, T=25,
    )
    poling = crystal.generate_poling(
        laser=laser, mode="constant",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=resolution,
    )
    return crystal, poling


def _subcoh_crystal(material, laser):
    crystal = Crystal(
        crystal_length=SUBCOH_CRYSTAL_LENGTH, material=material, pm_strategy="quasi",
        spdc_type="type-II", coherence_length=COHERENCE_LENGTH, w=SUBCOH_W, T=25,
    )
    poling = crystal.generate_poling(
        laser=laser, mode="subcoh",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9,
    )
    return crystal, poling


def _apm_constant_crystal(material, laser):
    crystal = Crystal(
        crystal_length=CRYSTAL_LENGTH, material=material, pm_strategy="angle",
        spdc_type="type-II", coherence_length=50e-6, T=25,
    )
    poling = crystal.generate_poling(
        laser=laser, mode="constant",
        wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5,
    )
    return crystal, poling


BUILDERS = [_periodic_crystal, _constant_crystal, _subcoh_crystal]
BUILDER_IDS = ["periodic", "constant", "subcoh"]


# ---------------------------------------------------------------------------
# Domain metadata is populated correctly by every in-scope QPM poling mode
# ---------------------------------------------------------------------------

def test_periodic_poling_populates_domain_metadata(material, laser):
    _, poling = _periodic_crystal(material, laser)
    num_domains = int(np.floor(CRYSTAL_LENGTH / COHERENCE_LENGTH))
    assert poling.domain_signs is not None and len(poling.domain_signs) == num_domains
    assert np.all(poling.domain_widths == pytest.approx(COHERENCE_LENGTH))
    assert poling.resolution == RESOLUTION
    assert poling.coherence_length == pytest.approx(COHERENCE_LENGTH)
    assert poling.DeltaK is not None
    assert poling.uniform_width is True


def test_subcoh_poling_populates_domain_metadata(material, laser):
    _, poling = _subcoh_crystal(material, laser)
    num_domains = len(poling.poling_pattern)
    assert poling.domain_signs is not None and len(poling.domain_signs) == num_domains
    np.testing.assert_array_equal(poling.domain_signs, poling.poling_pattern)
    assert np.all(poling.domain_widths == pytest.approx(SUBCOH_W))
    assert poling.resolution == 1


def test_apm_constant_poling_does_not_populate_domain_metadata(material, laser):
    # APM support is explicitly out of scope for v1 -- confirms the guard below has a real
    # target: a PolingResult that legitimately lacks the metadata contract.
    _, poling = _apm_constant_crystal(material, laser)
    assert poling.domain_signs is None
    assert poling.domain_widths is None
    assert poling.resolution is None


# ---------------------------------------------------------------------------
# Genericity: the metadata contract, not a strategy whitelist, gates support
# ---------------------------------------------------------------------------

def test_hand_built_poling_result_supports_imperfections_without_any_strategy():
    domain_signs = np.resize([1, -1], 50)
    domain_widths = np.full(50, 10e-6)
    poling = PolingResult(
        poling_pattern=np.repeat(domain_signs, 3), z=np.arange(150) * (10e-6 / 3),
        temperature_adjusted_length=500e-6, target_amplitude=None, actual_amplitude=None,
        domain_signs=domain_signs, domain_widths=domain_widths, resolution=3,
        coherence_length=10e-6, DeltaK=0.0, target_profile=None, uniform_width=True,
    )
    rng = np.random.default_rng(0)
    perturbed = poling.add_missed_domain_error(probability=1.0, rng=rng)
    np.testing.assert_array_equal(perturbed.domain_signs, -domain_signs)


# ---------------------------------------------------------------------------
# No-ops
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", BUILDERS, ids=BUILDER_IDS)
def test_add_wall_position_error_noop_at_sigma_zero(material, laser, build):
    _, poling = build(material, laser)
    assert poling.add_wall_position_error(sigma=0.0) is poling
    assert poling.add_wall_position_error(method="independent", sigma=0.0) is poling


@pytest.mark.parametrize("build", BUILDERS, ids=BUILDER_IDS)
def test_add_missed_domain_error_noop_at_probability_zero(material, laser, build):
    _, poling = build(material, laser)
    assert poling.add_missed_domain_error(probability=0.0) is poling


@pytest.mark.parametrize("build", BUILDERS, ids=BUILDER_IDS)
def test_add_duty_cycle_bias_noop_at_factor_zero(material, laser, build):
    _, poling = build(material, laser)
    assert poling.add_duty_cycle_bias(factor=0.0) is poling


# ---------------------------------------------------------------------------
# Out-of-scope guard
# ---------------------------------------------------------------------------

def test_add_star_raises_for_poling_result_without_domain_metadata(material, laser):
    _, poling = _apm_constant_crystal(material, laser)
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        poling.add_wall_position_error(sigma=0.02, rng=rng)
    with pytest.raises(ValueError):
        poling.add_missed_domain_error(probability=0.1, rng=rng)
    with pytest.raises(ValueError):
        poling.add_duty_cycle_bias(factor=0.05)


# ---------------------------------------------------------------------------
# Missed domains
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", BUILDERS, ids=BUILDER_IDS)
def test_add_missed_domain_error_flips_whole_physical_domains(material, laser, build):
    _, poling = build(material, laser)
    rng = np.random.default_rng(42)
    perturbed = poling.add_missed_domain_error(probability=1.0, rng=rng)
    np.testing.assert_array_equal(perturbed.domain_signs, -poling.domain_signs)
    blocks = perturbed.poling_pattern.reshape(-1, perturbed.resolution)
    assert np.all(blocks == blocks[:, :1])  # every physical domain is sign-uniform


def test_add_missed_domain_error_reproducible_with_seed(material, laser):
    _, poling = _periodic_crystal(material, laser)
    a = poling.add_missed_domain_error(probability=0.3, rng=np.random.default_rng(7))
    b = poling.add_missed_domain_error(probability=0.3, rng=np.random.default_rng(7))
    np.testing.assert_array_equal(a.domain_signs, b.domain_signs)


# ---------------------------------------------------------------------------
# Duty-cycle bias
# ---------------------------------------------------------------------------

def test_add_duty_cycle_bias_matches_analytic_formula(material, laser):
    _, poling = _periodic_crystal(material, laser)
    factor = 0.05
    perturbed = poling.add_duty_cycle_bias(factor=factor)
    expected = poling.domain_widths * (1.0 + factor * poling.domain_signs)
    np.testing.assert_allclose(perturbed.domain_widths, expected)
    assert perturbed.uniform_width is False
    assert perturbed.temperature_adjusted_length == pytest.approx(expected.sum())
    # +1 domains grew, -1 domains shrank
    plus = poling.domain_signs == 1
    assert np.all(perturbed.domain_widths[plus] > poling.domain_widths[plus])
    assert np.all(perturbed.domain_widths[~plus] < poling.domain_widths[~plus])


# ---------------------------------------------------------------------------
# Wall-position error
# ---------------------------------------------------------------------------

def test_add_wall_position_error_sets_uniform_width_false(material, laser):
    _, poling = _periodic_crystal(material, laser)
    perturbed = poling.add_wall_position_error(sigma=0.02, rng=np.random.default_rng(1))
    assert perturbed.uniform_width is False
    assert perturbed.poling_pattern.shape == poling.poling_pattern.shape


def test_add_wall_position_error_rejects_unknown_method(material, laser):
    _, poling = _periodic_crystal(material, laser)
    with pytest.raises(ValueError):
        poling.add_wall_position_error(method="bogus", sigma=0.02, rng=np.random.default_rng(1))


def test_add_wall_position_error_independent_widths_stay_positive(material, laser):
    _, poling = _periodic_crystal(material, laser)
    perturbed = poling.add_wall_position_error(method="independent", sigma=0.1, rng=np.random.default_rng(2))
    assert np.all(perturbed.domain_widths > 0)


def test_add_wall_position_error_cumulative_widths_stay_positive(material, laser):
    _, poling = _periodic_crystal(material, laser)
    perturbed = poling.add_wall_position_error(method="cumulative", sigma=0.1, rng=np.random.default_rng(3))
    assert np.all(perturbed.domain_widths > 0)


def test_add_wall_position_error_cumulative_shifts_total_length(material, laser):
    _, poling = _periodic_crystal(material, laser)
    perturbed = poling.add_wall_position_error(method="cumulative", sigma=0.05, rng=np.random.default_rng(4))
    assert perturbed.temperature_adjusted_length != pytest.approx(poling.temperature_adjusted_length, rel=1e-6)


def test_add_wall_position_error_reproducible_with_seed(material, laser):
    _, poling = _periodic_crystal(material, laser)
    a = poling.add_wall_position_error(sigma=0.02, rng=np.random.default_rng(9))
    b = poling.add_wall_position_error(sigma=0.02, rng=np.random.default_rng(9))
    np.testing.assert_allclose(a.z, b.z)


# ---------------------------------------------------------------------------
# Chain composability
# ---------------------------------------------------------------------------

def test_chain_order_affects_result(material, laser):
    # missed-domain-error flips some signs; duty-cycle-bias's width perturbation depends on
    # *which* signs are active when it runs -- so growing/shrinking widths before vs. after the
    # flips gives genuinely different results, unlike e.g. two purely-multiplicative width
    # perturbations (which commute regardless of order).
    _, poling = _periodic_crystal(material, laser)

    def chain_a(p):
        return (p.add_missed_domain_error(probability=0.3, rng=np.random.default_rng(5))
                 .add_duty_cycle_bias(factor=0.1))

    def chain_b(p):
        return (p.add_duty_cycle_bias(factor=0.1)
                 .add_missed_domain_error(probability=0.3, rng=np.random.default_rng(5)))

    result_a = chain_a(poling)
    result_b = chain_b(poling)
    np.testing.assert_array_equal(result_a.domain_signs, result_b.domain_signs)  # same flip mask
    assert not np.allclose(result_a.domain_widths, result_b.domain_widths)  # different widths


def test_chain_composes_all_three_mechanisms(material, laser):
    _, poling = _periodic_crystal(material, laser)
    rng = np.random.default_rng(11)
    perturbed = (
        poling
        .add_wall_position_error(method="cumulative", sigma=0.02, rng=rng)
        .add_missed_domain_error(probability=0.05, rng=rng)
        .add_duty_cycle_bias(factor=0.05)
    )
    assert perturbed.uniform_width is False
    assert len(perturbed.poling_pattern) == len(poling.poling_pattern)
    assert perturbed.target_amplitude is None and perturbed.actual_amplitude is None


# ---------------------------------------------------------------------------
# Non-uniform-width fallback matches the closed form on an untouched grid
# ---------------------------------------------------------------------------

def test_general_fallback_matches_closed_form_on_uniform_grid(material, laser):
    # Build z/pattern via the same resampler add_*() uses after a perturbation, rather than the
    # ideal generator's own z (which spans temperature_adjusted_length, not exactly
    # num_domains*coherence_length -- an immaterial ~1-domain rounding difference for the
    # simulation physics, but one that matters for this K-resonant parity check specifically:
    # a tiny fractional grid-width mismatch times the O(1e3) total phase K*L is a non-negligible
    # absolute phase drift). This keeps the two formulas being compared on a self-consistent grid.
    from photonpairlab.crystal.phasematching import imperfections
    _, poling = _periodic_crystal(material, laser)
    strategy = QPMPhaseMatching(material, spdc_type="type-II")
    w = poling.domain_widths[0] / poling.resolution
    pattern, z = imperfections.resample_domains_to_fine_grid(poling.domain_signs, poling.domain_widths, poling.resolution)

    _, actual_exact = strategy.compute_domain_field_arrays(
        pattern, w, poling.coherence_length, poling.domain_widths.sum(), poling.DeltaK)
    _, actual_general = strategy.compute_domain_field_arrays_nonuniform(
        pattern, z, poling.coherence_length, poling.domain_widths.sum(), poling.DeltaK)

    np.testing.assert_allclose(np.abs(actual_general[-1]), np.abs(actual_exact[-1]), rtol=1e-2)


# ---------------------------------------------------------------------------
# Crystal.apply_poling
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [_periodic_crystal, _subcoh_crystal], ids=["periodic", "subcoh"])
def test_apply_poling_updates_crystal_attributes(material, laser, build):
    crystal, poling = build(material, laser)
    perturbed = poling.add_missed_domain_error(probability=0.1, rng=np.random.default_rng(6))
    crystal.apply_poling(perturbed)
    assert crystal.poling_pattern is perturbed.poling_pattern
    assert crystal.z is perturbed.z
    assert crystal.target_amplitude is not None
    assert crystal.actual_amplitude is not None
    assert crystal.target_amplitude.shape == crystal.poling_pattern.shape


def test_apply_poling_raises_without_metadata(material, laser):
    crystal, poling = _apm_constant_crystal(material, laser)
    with pytest.raises(ValueError):
        crystal.apply_poling(poling)


# ---------------------------------------------------------------------------
# End-to-end regression through the physics engine
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("build", [_periodic_crystal, _subcoh_crystal], ids=["periodic", "subcoh"])
def test_apply_poling_end_to_end_through_simulation(material, laser, build):
    crystal, poling = build(material, laser)
    rng = np.random.default_rng(13)
    perturbed = (
        poling
        .add_wall_position_error(method="cumulative", sigma=0.02, rng=rng)
        .add_missed_domain_error(probability=0.02, rng=rng)
        .add_duty_cycle_bias(factor=0.05)
    )
    crystal.apply_poling(perturbed)

    simulation = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=30, dev_nm=5.0))
    results = simulation.run()

    assert np.all(np.isfinite(results.JSA))
    assert np.any(results.JSI > 0)
