import matplotlib
matplotlib.use("Agg")

import pytest

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.spdc_config import SPDCGridConfig
from photonpairlab.spdc.simulation import SPDC_Simulation
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
    fig = plotter.plot_schmidt_coefficients()
    assert fig is not None
