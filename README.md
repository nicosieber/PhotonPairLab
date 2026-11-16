# PhotonPairLab

[![Tests](https://github.com/nicosieber/PhotonPairLab/actions/workflows/tests.yml/badge.svg)](https://github.com/nicosieber/PhotonPairLab/actions/workflows/tests.yml)

**PhotonPairLab** is a Python toolkit for simulating photon pair generation via spontaneous parametric down-conversion (SPDC) in nonlinear crystals.
It supports both angle phase-matching (APM) and quasi-phase-matching (QPM) using a unified, extensible object-oriented architecture.

![PhotonPairLab: joint spectral amplitude and Hong-Ou-Mandel interference](assets/hero.png)

---

## Features

- Unified `Crystal` class supporting both QPM and APM via the strategy pattern
- Material data (KTP, BBO, BiBO, ...) loaded by name via `MaterialFactory.create(name)`
- Pluggable phase-matching strategies (`QPMPhaseMatching`, `APMPhaseMatching`), with `Literal`-typed
  parameters for IDE autocomplete on valid `pm_strategy`/`spdc_type`/poling-`mode` values
- QPM poling in periodic, constant, or Gaussian-apodized (`mode="subcoh"`, sub-coherence-length
  domain engineering) form, plus automatic coherence-length calculation
  (`Crystal.ideal_coherence_length`) so the grating always matches the crystal's own dispersion
- Simulation of the joint spectral amplitude/intensity (JSA/JSI) and Schmidt decomposition (heralded
  photon purity)
- Hong-Ou-Mandel (HOM) interference with two explicit spectral models:
  `HOMAnalyzer.same_pair(results)` for signal and idler of the *same* pair (JSA exchange overlap) and
  `HOMAnalyzer.independent(results_a, results_b)` for photons from independent sources
  (Re Tr[ρ_a ρ_b(τ)], bounded by the spectral purity). Both return a `HOMResults` whose visibility
  factors into beam splitter × polarization × spectral overlap (`visibility_breakdown()`), with
  optional Jones-vector polarizations at the beam splitter
- A one-call `simulate_spdc(...)` convenience entry point for the common case, alongside the full
  `Crystal`/`SPDC_Simulation` pipeline for anything more specific
- Manufacturing-imperfection modeling for QPM poling — wall-position error (random-walk or
  bounded), missed domains, and duty-cycle bias — chainable on any `PolingResult` via
  `.add_wall_position_error()`/`.add_missed_domain_error()`/`.add_duty_cycle_bias()` and fed back
  into the crystal with `Crystal.apply_poling(...)` to see the effect on JSA/JSI purity and HOM
  interference
- Optional JAX-accelerated sweeps (phase-matching angle search, group index, Sellmeier) over many
  wavelengths/temperatures at once, via the `jax` extra — never imported unless you ask for it
- Consistent plot styling: `SPDC_Plotter.use_style()` applies the plotter's chart style to any
  matplotlib figure, and `SPDC_Plotter.style_legend(ax)` places legends below the axes
- Easily extensible for new materials and phase-matching types

---

## Installation

Requires Python 3.12+. Clone the repository and install in editable mode:

```sh
git clone https://github.com/nicosieber/PhotonPairLab.git
cd PhotonPairLab
pip install -e .
```

If you use [uv](https://docs.astral.sh/uv/), `uv pip install -e .` does the same.

For the optional JAX-accelerated phase-matching sweeps, install the `jax` extra:

```sh
pip install -e ".[jax]"
```

---

## Usage Example

The quickest path — material, crystal, laser, poling, and simulation in one call, with the crystal's
coherence length computed automatically from its own dispersion:

```python
from photonpairlab import simulate_spdc

results = simulate_spdc(
    material_name="ktp1",       # or "bbo", "bibo", "ktp2", "ktp3"
    crystal_length=30e-3,
    wavelength_pump=775e-9,
    pulse_duration=1.7e-12,
    spdc_type="type-II",
    poling_mode="periodic",     # or "constant", "subcoh"
)
```

For more control — building the crystal and laser yourself, choosing an explicit wavelength grid, or
reusing the same crystal for multiple simulations:

```python
from photonpairlab import Crystal, MaterialFactory, PulsedLaser, SPDC_Simulation, SPDCGridConfig

material = MaterialFactory.create("ktp1")
crystal = Crystal(
    crystal_length=30e-3,
    material=material,
    pm_strategy="quasi",   # or "angle"
    spdc_type="type-II",
    T=30,
)

laser = PulsedLaser(775e-9, pulse_duration=1.7e-12)

# coherence_length wasn't given above, so it's computed automatically here from the crystal's own
# dispersion at these target wavelengths (pass one explicitly to override, e.g. for detuning studies)
crystal.generate_poling(laser=laser, mode="periodic", wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5)

grid = SPDCGridConfig(steps=100, signal_range=(1545e-9, 1560e-9), idler_range=(1545e-9, 1560e-9))
simulation = SPDC_Simulation(crystal, laser, grid=grid)
results = simulation.run()
```

Analyzing the result — purity and HOM interference:

```python
from photonpairlab import HOMAnalyzer, SpectralAnalyzer, SPDC_Plotter

# HOM needs identical signal and idler grids (signal_range == idler_range, as above)
spectral = SpectralAnalyzer(results)  # Schmidt decomposition, marginals, centroids

# signal and idler of the same pair (e.g. type-II, idler rotated to match the signal)
hom = HOMAnalyzer.same_pair(results)
# photons from two independent sources instead (e.g. two heralded signals)
hom_indep = HOMAnalyzer.independent(results, results, mode_a="signal", mode_b="signal")

print(hom.visibility_breakdown())  # {'beam_splitter': ..., 'polarization': ..., 'spectral': ..., 'total': ...}
SPDC_Plotter.plot_hom_dip({"same pair": hom, "independent": hom_indep}, show_visibility=True)
```

`HOMAnalyzer.compute_two_mode_HOM` and the `TwoModeHOMResults` name still work but are deprecated
aliases of `independent`/`HOMResults`.

**For a full, worked tutorial** — theory interleaved with code, covering QPM, sub-coherence
apodization, Schmidt purity, HOM interference, temperature tuning, and angle phase-matching/GVM,
with references to the literature — see [`notebooks/demo.ipynb`](notebooks/demo.ipynb).

Modeling a real, imperfectly-fabricated crystal instead of an idealized one:

```python
result = crystal.generate_poling(laser=laser, mode="periodic",
                                  wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5)

# chain any subset of the three fabrication-error mechanisms (Graffitti et al. 2017, Sec. III.3.2)
result = (result
          .add_wall_position_error(method="cumulative", sigma=0.02)
          .add_missed_domain_error(probability=0.01)
          .add_duty_cycle_bias(factor=0.02))

crystal.apply_poling(result)  # recomputes target_amplitude/actual_amplitude for the perturbed pattern
```

See [`notebooks/manufacturing_imperfections.ipynb`](notebooks/manufacturing_imperfections.ipynb)
for the full tutorial on all three mechanisms and their effect on purity and HOM interference.

For diagnosing a low HOM visibility — separating polarization from spectral mismatch, why the
same-pair and independent-source models differ, and a fit template for measured dips — see
[`notebooks/hom_visibility.ipynb`](notebooks/hom_visibility.ipynb).

---

## Running the tests

```sh
python -m pytest
```

Tests are organized by subpackage under `tests/` (`tests_crystal/`, `tests_laser/`, `tests_spdc/`).

---

## Simulation Workflow 

The diagram below shows the actual simulation workflow — from choosing a material through
phase-matching, the optional manufacturing-imperfection step, and on to simulation/analysis — rather
than the package layout.

```mermaid
flowchart TD
    MAT["Choose material<br/>MaterialFactory.create(name)"]
    LAS["Choose laser<br/>CWLaser or PulsedLaser"]
    CRY["Build Crystal<br/>(length, temperature, spdc_type, pm_strategy)"]

    MAT --> CRY
    CRY --> STRAT{"pm_strategy"}

    STRAT -->|quasi| QPM{"generate_poling(mode)"}
    QPM -->|periodic| QP["Alternating +/- domains"]
    QPM -->|constant| QC["Unpoled reference"]
    QPM -->|subcoh| QS["Apodized domains<br/>(Graffitti et al. 2017)"]
    QP --> POL["PolingResult<br/>(domain_signs, domain_widths)"]
    QC --> POL
    QS --> POL

    STRAT -->|angle| APM["generate_poling(mode='constant')<br/>search phase-matching angle"]
    APM --> POLA["PolingResult<br/>(no domain metadata)"]
    APM -.->|"optional, [jax] extra"| JAX["find_phase_matching_angles_jax<br/>batched angle / Δk sweeps"]

    POL --> IMP{"Model manufacturing<br/>imperfections?"}
    IMP -->|no| READY["Ideal poling on Crystal"]
    IMP -->|yes| CHAIN["add_wall_position_error /<br/>add_missed_domain_error /<br/>add_duty_cycle_bias"]
    CHAIN --> APPLY["crystal.apply_poling(perturbed)"]
    APPLY --> READYP["Perturbed poling on Crystal<br/>actual amplitude diverges from target"]

    POLA --> READYA["Ideal poling on Crystal<br/>(imperfections unsupported)"]

    READY --> SIM["SPDC_Simulation(crystal, laser).run()"]
    READYP --> SIM
    READYA --> SIM
    LAS --> SIM

    SIM --> RES["SPDCResults<br/>Pump, Phase, JSI, JSA"]

    RES --> SPEC["SpectralAnalyzer<br/>Schmidt decomposition, purity,<br/>marginal centroids"]
    RES --> HOMQ{"Which photons<br/>interfere?"}
    HOMQ -->|"signal + idler<br/>of one pair"| SAME["HOMAnalyzer.same_pair(results)<br/>JSA exchange overlap"]
    HOMQ -->|"two independent<br/>sources"| INDEP["HOMAnalyzer.independent(a, b)<br/>Re Tr[ρ_a ρ_b(τ)]"]
    SAME --> HOMR["HOMResults<br/>Pc(τ), visibility =<br/>BS × polarization × spectral"]
    INDEP --> HOMR
    HOMR --> HOMPLOT["SPDC_Plotter.plot_hom_dip<br/>(show_visibility=True)"]
    RES --> PLOT["SPDC_Plotter<br/>JSI / JSA heatmap"]
    READY --> PROFILE["SPDC_Plotter.plot_poling_profile<br/>target vs actual amplitude"]
    READYP --> PROFILE
```

- **Materials** (`crystal/material/`): properties looked up by name via `MaterialFactory.create(name)`
  from `src/photonpairlab/resources/materials.json`.
- **Phase-matching strategies** (`crystal/phasematching/`): QPM and APM via the strategy pattern;
  manufacturing-imperfection modeling (`imperfections.py`) chains onto QPM `PolingResult`s only, since
  APM's constant poling carries no domain metadata to perturb. The optional `jax_accel/` subpackage
  (lazily imported, `jax` extra) batches angle searches and Δk/group-index sweeps.
- **Laser** (`laser/`): `CWLaser`/`PulsedLaser` spectral envelope.
- **SPDC** (`spdc/`): simulation (`simulation/`), analysis (`analysis/` — `SpectralAnalyzer`,
  `HOMAnalyzer`/`HOMResults`, and the `SPDCResults`-independent HOM math in `hom_math.py`), plotting
  (`plotting.py`).
- **HOM model choice:** the analyzer can't infer from the results whether two photons share a pair
  (two identical crystals give identical results), so you pick the model. Using `independent` for a
  same-pair experiment caps the visibility at the purity and underestimates it.

---

## Extending

- **Add a new material:** Add an entry to `src/photonpairlab/resources/materials.json` and, if needed, a new model class in
  `crystal/material/model/` inheriting from `BaseMaterialModel`, registered in `material_factory.py`'s
  `MODEL_MAPPER`.
- **Add a new phase-matching strategy:** Create a new class in `crystal/phasematching/` inheriting from
  `PhaseMatchingStrategy`, and register it in `crystal.py`'s `PM_STRATEGY_HANDLER`.
- **Use your new classes** by passing them to the `Crystal` constructor.

---

## License

MIT

---

## Acknowledgments

PhotonPairLab is inspired by the needs of quantum optics research and is open for contributions and extensions.

---

## Demo

**For a full walkthrough with theory and worked examples, see [`notebooks/demo.ipynb`](notebooks/demo.ipynb).**

**For manufacturing-imperfection modeling specifically, see
[`notebooks/manufacturing_imperfections.ipynb`](notebooks/manufacturing_imperfections.ipynb).**

**For diagnosing HOM visibility (polarization vs. spectral mismatch), see
[`notebooks/hom_visibility.ipynb`](notebooks/hom_visibility.ipynb).**

---

## Getting in contact
If you want to reach out to me, you can do so by contacting me on [LinkedIn](https://www.linkedin.com/in/nico-sieber-0a7204156/).
