"""
Convenience entry point bundling the common SPDC simulation workflow (material lookup,
crystal + laser construction, poling generation, and running the simulation) into a
single call, for the common case: degenerate type-II QPM with periodic poling.

For anything beyond that -- angle phase-matching, sub-coherence apodization, multiple
crystals for HOM comparisons, custom wavelength grids -- build the pipeline directly
from `Crystal`, `PulsedLaser`/`CWLaser`, and `SPDC_Simulation` instead.
"""

from photonpairlab.crystal import Crystal
from photonpairlab.crystal.material import MaterialFactory
from photonpairlab.crystal.phasematching import SPDCType, PolingMode
from photonpairlab.crystal.crystal import PMStrategyName
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation, SPDCGridConfig, SPDCResults


def simulate_spdc(
    material_name: str,
    crystal_length: float,
    wavelength_pump: float,
    pulse_duration: float,
    wavelength_signal: float | None = None,
    wavelength_idler: float | None = None,
    spdc_type: SPDCType = "type-II",
    pm_strategy: PMStrategyName = "quasi",
    poling_mode: PolingMode = "periodic",
    T: float = 25.0,
    w: float | None = None,
    coherence_length: float | None = None,
    grid: SPDCGridConfig | None = None,
    resolution: int = 5,
) -> SPDCResults:
    """
    Build a crystal + pulsed pump laser, generate its poling pattern, run the SPDC
    simulation, and return the results -- for the common degenerate type-II QPM case.

    Parameters
    ----------
    material_name : str
        Passed to :func:`~photonpairlab.crystal.material.MaterialFactory.create`
        (e.g. ``"ktp1"``, ``"bbo"``, ``"bibo"``).
    crystal_length : float
        Physical crystal length (m).
    wavelength_pump : float
        Pump center wavelength (m).
    pulse_duration : float
        Pump pulse duration (s), used to derive the pump bandwidth.
    wavelength_signal, wavelength_idler : float, optional
        Target signal/idler wavelengths (m). Default to ``2 * wavelength_pump``
        (degenerate SPDC) if not given.
    spdc_type : str
        SPDC interaction type, e.g. ``"type-II"``.
    pm_strategy : str
        ``"quasi"`` for QPM or ``"angle"`` for APM.
    poling_mode : str
        Poling generation mode: ``"periodic"``, ``"constant"``, or ``"subcoh"``
        (QPM only for ``"subcoh"``).
    T : float
        Crystal temperature (deg C).
    w : float, optional
        Domain width; required for ``poling_mode="subcoh"`` or APM's ``"constant"``.
    coherence_length : float, optional
        QPM domain width. If not given, it is computed automatically from the
        crystal's own dispersion at ``T``/``wavelength_signal``/``wavelength_idler``
        via :meth:`~photonpairlab.crystal.Crystal.ideal_coherence_length` (see
        ``Crystal.generate_poling``). Pass this explicitly when you need a *fixed*
        grating evaluated away from its design point -- e.g. a temperature-tuning
        sweep, where re-deriving the "ideal" coherence length at every temperature
        would re-optimize the poling for each point instead of showing how one fixed
        grating's phase-matching drifts with temperature.
    grid : SPDCGridConfig, optional
        Wavelength grid for the simulation. Defaults to
        ``SPDCGridConfig(steps=100, dev_nm=5.0)`` if not given.
    resolution : int
        Subdivisions per domain for periodic/constant poling (ignored by ``subcoh``).

    Returns
    -------
    SPDCResults
        The simulation results (JSA/JSI and wavelength axes).
    """
    if wavelength_signal is None:
        wavelength_signal = 2 * wavelength_pump
    if wavelength_idler is None:
        wavelength_idler = 2 * wavelength_pump

    material = MaterialFactory.create(material_name)
    crystal = Crystal(
        crystal_length=crystal_length,
        material=material,
        pm_strategy=pm_strategy,
        spdc_type=spdc_type,
        T=T,
        w=w,
        coherence_length=coherence_length,
    )
    laser = PulsedLaser(wavelength_pump, pulse_duration=pulse_duration)
    crystal.generate_poling(
        laser=laser,
        mode=poling_mode,
        wavelength_signal=wavelength_signal,
        wavelength_idler=wavelength_idler,
        resolution=resolution,
    )

    grid = grid or SPDCGridConfig(steps=100, dev_nm=5.0)
    simulation = SPDC_Simulation(
        crystal, laser,
        wavelength_signal=wavelength_signal,
        wavelength_idler=wavelength_idler,
        grid=grid,
    )
    return simulation.run()
