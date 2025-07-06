# PhotonPairLab

**PhotonPairLab** is a Python toolkit for simulating photon pair generation via spontaneous parametric down-conversion (SPDC) in nonlinear crystals.  
It supports both angle phase-matching (APM) and quasi-phase-matching (QPM) using a unified, extensible object-oriented architecture.

---

## Features

- Unified `Crystal` class supporting both QPM and APM via strategy pattern
- Modular material classes (`KTP1`, `BBO`, `BIBO`, etc.)
- Pluggable phase-matching strategies (`QPMPhaseMatching`, `APMPhaseMatching`)
- Simulation of joint spectral amplitude/intensity (JSA/JSI)
- HOM dip and spectral analysis tools
- Easily extensible for new materials and phase-matching types

---

## Installation

Clone the repository and install with pip:

```sh
git clone https://github.com/yourusername/photonpairlab.git
cd photonpairlab
pip install -e .
```

---

## Usage Example

```python
from photonpairlab.crystal import Crystal, KTP1, BBO
from photonpairlab.laser import PulsedLaser
from photonpairlab.spdc.simulation import SPDC_Simulation

# Choose material and phase-matching strategy
material = KTP1()
crystal = Crystal(
    crystal_length=30e-3,
    material=material,
    pm_strategy="quasi",  # or "angle"
    spdc_type="type-II",
    coherence_length=46.2e-6,
    w=18e-6,
    T=30
)

# Define laser
wavelength_pump = 775e-9  # Pump wavelength in meters  
pulse_duration = 1.7e-12  # Pulse duration in seconds
laser = PulsedLaser(wavelength_pump, pulse_duration=pulse_duration)

# Generate poling (for QPM)
crystal.generate_poling(laser=laser, mode="periodic", wavelength_signal=1550e-9, wavelength_idler=1550e-9, resolution=5)

# Run SPDC simulation
simulation = SPDC_Simulation(crystal, laser, wavelength_signal_range=[1545e-9, 1560e-9], wavelength_idler_range=[1545e-9, 1560e-9])
results = simulation.run_simulation(steps=100)
```

---

## Architecture Overview

```
photonpairlab/
│
├── crystal/
│   ├── materials/
│   │   ├── base_material.py
│   │   ├── ktp1.py
│   │   ├── bbo.py
│   │   └── ...
│   ├── pmstrategy/
│   │   ├── base_pm_strategy.py
│   │   ├── qpm_strategy.py
│   │   ├── apm_strategy.py
│   │   └── ...
│   ├── crystal.py
│   └── ...
├── spdc/
│   ├── simulation.py
│   ├── analysis.py
│   └── ...
└── ...
```

- **Materials:** All nonlinear crystal properties (refractive index, thermal expansion, etc.)
- **Phase-Matching Strategies:** QPM and APM logic, interchangeable via the strategy pattern
- **Crystal:** Unified interface, delegates phase-matching to the chosen strategy
- **SPDC:** Simulation and analysis tools

---

## Extending

- **Add a new material:** Create a new class in `crystal/materials/` inheriting from `BaseMaterial`.
- **Add a new phase-matching strategy:** Create a new class in `crystal/pmstrategy/` inheriting from `PhaseMatchingStrategy`.
- **Use your new classes** by passing them to the `Crystal` constructor.

---

## License

MIT

---

## Acknowledgments

PhotonPairLab is inspired by the needs of quantum optics research and is open for contributions and extensions.

---

**For detailed examples, see the `demo.ipynb` notebook.**