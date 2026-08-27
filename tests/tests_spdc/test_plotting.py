import matplotlib
matplotlib.use("Agg")

import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation, SPDCGridConfig
from photonpairlab.spdc.plotting import SPDC_Plotter


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
    sim = SPDC_Simulation(crystal, laser, grid=SPDCGridConfig(steps=40, dev_nm=5.0))
    return sim.run()


@pytest.mark.parametrize("key", ["Pump", "Phase", "JSI", "JSA"])
def test_plot_result_renders_for_each_key(results, key):
    # JSA is complex; imshow can't handle complex arrays directly, so plot_result
    # must plot its magnitude (a regression check for that conversion).
    plotter = SPDC_Plotter(results)
    fig, ax = plotter.plot_result(key=key)
    assert fig is not None
    assert ax is not None


def test_plot_schmidt_coefficients_renders(results):
    plotter = SPDC_Plotter(results)
    fig, ax = plotter.plot_schmidt_coefficients()
    assert fig is not None
    assert ax is not None


def test_plot_schmidt_coefficients_respects_n_coefficients(results):
    plotter = SPDC_Plotter(results)
    fig, ax = plotter.plot_schmidt_coefficients(n_coefficients=5)
    assert len(ax.patches) == 5


def test_plot_signal_idler_spectra_renders(results):
    plotter = SPDC_Plotter(results)
    fig, ax = plotter.plot_signal_idler_spectra()
    assert fig is not None
    assert ax is not None


def test_schmidt_and_spectra_plots_compose_into_one_figure(results):
    # The two plots used to be forced into one fixed 2-panel figure; they must still be
    # freely composable into any layout via the shared fig/ax pattern.
    import matplotlib.pyplot as plt

    plotter = SPDC_Plotter(results)
    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plotter.plot_schmidt_coefficients(fig=fig, ax=ax1)
    plotter.plot_signal_idler_spectra(fig=fig, ax=ax2)
    assert fig.axes == [ax1, ax2]
