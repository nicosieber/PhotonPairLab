import pytest

from photonpairlab.crystal.material.material_factory import MaterialFactory
from photonpairlab.crystal.phasematching.apm_strategy import APMPhaseMatching
from photonpairlab.crystal.phasematching.qpm_strategy import QPMPhaseMatching
from photonpairlab.laser import PulsedLaser


def test_apm_and_qpm_agree_on_dispersive_refractive_indices():
    # APM used to compute n/N from raw-meter wavelengths instead of micrometers,
    # producing near-constant (non-dispersive) indices. QPM's code path was already
    # correct, so the two should now agree closely for the same material/wavelengths.
    material = MaterialFactory.create("ktp1")
    laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)

    apm = APMPhaseMatching(material, spdc_type="type-II")
    qpm = QPMPhaseMatching(material, spdc_type="type-II")

    apm_result = apm.compute_phase_mismatch(laser, 1550e-9, 1550e-9, angle_pm=0, T=25.0)
    qpm_result = qpm.compute_phase_mismatch(laser, 1550e-9, 1550e-9, angle_pm=0, T=25.0)

    for apm_n, qpm_n in zip(apm_result.n, qpm_result.n):
        assert apm_n == pytest.approx(qpm_n, rel=1e-6)
    for apm_N, qpm_N in zip(apm_result.N, qpm_result.N):
        assert apm_N == pytest.approx(qpm_N, rel=1e-6)


def test_apm_refractive_index_is_dispersive():
    # A non-dispersive (unit-conversion) bug would make n(pump) == n(idler) even though
    # they're on the same crystal axis ('y' for type-II) but at very different wavelengths.
    material = MaterialFactory.create("ktp1")
    laser = PulsedLaser(400e-9, pulse_duration=1.7e-12)
    apm = APMPhaseMatching(material, spdc_type="type-II")

    result = apm.compute_phase_mismatch(laser, 800e-9, 800e-9, angle_pm=45, T=25.0)
    n_pump, n_signal, n_idler = result.n
    assert n_pump != pytest.approx(n_idler, rel=1e-3)
