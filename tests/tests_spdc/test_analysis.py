import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.spdc_config import SPDCGridConfig
from photonpairlab.spdc.simulation import SPDC_Simulation
from photonpairlab.spdc.spectral_analyser import SpectralAnalyzer


@pytest.fixture
def results():
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
    sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=60, dev_nm=5.0))
    return sim.run()


def test_schmidt_purity_is_in_valid_range(results):
    analyzer = SpectralAnalyzer(results)
    s_vals, purity, K = analyzer.schmidt_decomposition()

    assert 0.0 < purity <= 1.0 + 1e-9
    assert K >= 1.0 - 1e-9
    assert s_vals.sum() > 0


def test_signal_idler_peaks_are_near_expected_center(results):
    analyzer = SpectralAnalyzer(results)
    signal_peak, idler_peak, _, _ = analyzer.get_signal_idler_peaks(method="quadratic")

    # Degenerate type-II SPDC around 1550 nm signal/idler for a 775 nm pump.
    assert signal_peak == pytest.approx(1550.0, abs=5.0)
    assert idler_peak == pytest.approx(1550.0, abs=5.0)


def test_subcoh_apodization_gives_higher_purity_than_periodic_poling():
    # Sub-coherence-length domain engineering (Graffitti et al. 2017) apodizes the
    # nonlinearity profile to suppress the sinc sidelobes a plain periodic grating
    # produces, which should give a meaningfully higher heralded-photon purity for
    # the same crystal length/coherence length. This is a regression check for a
    # bug where the greedy domain-by-domain optimizer's Am/target_amplitude formulas
    # didn't match Eq. 9 of the paper (wrong z-indexing, wrong prefactor, a hardcoded
    # first domain, and a fast-oscillating term that leaked into the target instead
    # of being dropped analytically) -- with those bugs, subcoh performed no better
    # than a plain periodic grating.
    material = MaterialFactory.create("ktp1")
    coherence_length = 23.12e-6
    crystal_length = 2e-3
    w = 2e-6
    laser = PulsedLaser(775e-9, pulse_duration=0.1e-12)

    def purity_for(mode):
        crystal = Crystal(
            crystal_length=crystal_length, material=material, pm_strategy="quasi",
            spdc_type="type-II", coherence_length=coherence_length, w=w, T=20,
        )
        kwargs = {"resolution": 5} if mode == "periodic" else {}
        crystal.generate_poling(
            laser=laser, mode=mode,
            wavelength_signal=1550e-9, wavelength_idler=1550e-9, **kwargs,
        )
        sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=200, dev_nm=15.0))
        results = sim.run()
        _, purity, _ = SpectralAnalyzer(results).schmidt_decomposition()
        return purity

    purity_periodic = purity_for("periodic")
    purity_subcoh = purity_for("subcoh")

    assert purity_subcoh > purity_periodic + 0.05
    assert purity_subcoh > 0.99
