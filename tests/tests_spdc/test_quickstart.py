import numpy as np
import pytest

from photonpairlab.quickstart import simulate_spdc
from photonpairlab.spdc.simulation import SPDCGridConfig


def test_simulate_spdc_degenerate_default_peaks_at_twice_pump_wavelength():
    results = simulate_spdc(
        material_name="ktp1",
        crystal_length=30e-3,
        wavelength_pump=775e-9,
        pulse_duration=1.7e-12,
    )
    peak_idx = np.unravel_index(np.argmax(results.JSI), results.JSI.shape)
    peak_signal = results.SignalWavelengths[peak_idx[1]]
    peak_idler = results.IdlerWavelengths[peak_idx[0]]

    assert peak_signal == pytest.approx(1550e-9, abs=1e-9)
    assert peak_idler == pytest.approx(1550e-9, abs=1e-9)


def test_simulate_spdc_respects_explicit_signal_idler_wavelengths():
    results = simulate_spdc(
        material_name="ktp1",
        crystal_length=30e-3,
        wavelength_pump=775e-9,
        pulse_duration=1.7e-12,
        wavelength_signal=1500e-9,
        wavelength_idler=1610e-9,
    )
    assert results.SignalWavelengths.min() < 1500e-9 < results.SignalWavelengths.max()
    assert results.IdlerWavelengths.min() < 1610e-9 < results.IdlerWavelengths.max()


def test_simulate_spdc_respects_explicit_coherence_length():
    # A fixed coherence_length (e.g. for a temperature-tuning sweep, where the grating
    # shouldn't be re-optimized at every temperature) must be forwarded, not silently
    # overridden by Crystal.generate_poling's auto-fill. A deliberately detuned
    # coherence_length should visibly shift the JSI peak away from the target
    # wavelength; the auto-computed (ideal) one always peaks exactly on target.
    common_kwargs = dict(
        material_name="ktp1", crystal_length=30e-3,
        wavelength_pump=775e-9, pulse_duration=1.7e-12,
        grid=SPDCGridConfig(steps=100, dev_nm=15.0),
    )
    results_ideal = simulate_spdc(**common_kwargs)
    results_detuned = simulate_spdc(**common_kwargs, coherence_length=30e-6)

    def peak_signal_nm(results):
        idx = np.unravel_index(np.argmax(results.JSI), results.JSI.shape)
        return results.SignalWavelengths[idx[1]] * 1e9

    assert peak_signal_nm(results_ideal) == pytest.approx(1550.0, abs=0.5)
    assert peak_signal_nm(results_detuned) != pytest.approx(1550.0, abs=0.5)


def test_simulate_spdc_subcoh_mode_requires_domain_width_like_generate_poling():
    with pytest.raises(ValueError):
        simulate_spdc(
            material_name="ktp1",
            crystal_length=2e-3,
            wavelength_pump=775e-9,
            pulse_duration=0.1e-12,
            poling_mode="subcoh",
        )
