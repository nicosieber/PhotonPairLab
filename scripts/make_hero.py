"""Regenerate assets/hero.png: JSA of a periodically poled KTP crystal next to the HOM dips of two
independent, identical sources for periodic vs. sub-coherence (apodized) poling.

Run from the repo root:  uv run python scripts/make_hero.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from photonpairlab import Crystal, HOMAnalyzer, MaterialFactory, PulsedLaser, SPDC_Plotter, SPDC_Simulation, SPDCGridConfig  # noqa: E402

LAMBDA_P, LAMBDA_SI = 775e-9, 1550e-9
GRID = SPDCGridConfig(steps=200, signal_range=(1545e-9, 1555e-9), idler_range=(1545e-9, 1555e-9))


# Same source parameters as notebooks/demo.ipynb.
LASER = PulsedLaser(LAMBDA_P, pulse_duration=1.7e-12)


def simulate(mode, **crystal_kwargs):
    crystal = Crystal(crystal_length=30e-3, material=MaterialFactory.create("ktp1"), pm_strategy="quasi", spdc_type="type-II", T=20.0, **crystal_kwargs)
    poling_kwargs = {"resolution": 5} if mode == "periodic" else {}
    crystal.generate_poling(laser=LASER, mode=mode, wavelength_signal=LAMBDA_SI, wavelength_idler=LAMBDA_SI, **poling_kwargs)
    return SPDC_Simulation(crystal, LASER, grid=GRID).run()


results = {"periodic": simulate("periodic"), "subcoh (apodized)": simulate("subcoh", w=18e-6)}
taus_s = np.linspace(-7500e-15, 7500e-15, 601)
hom = {label: HOMAnalyzer.independent(r, r, mode_a="signal", mode_b="signal", taus_s=taus_s) for label, r in results.items()}

SPDC_Plotter.use_style()
fig, (ax_jsa, ax_hom) = plt.subplots(1, 2, figsize=(10, 4), gridspec_kw={"width_ratios": [1, 1.25]})
SPDC_Plotter(results["periodic"]).plot_result("JSA", fig=fig, ax=ax_jsa)
ax_jsa.set_title("Joint spectral amplitude\n(30 mm KTP, periodic poling)")
SPDC_Plotter.plot_hom_dip(hom, fig=fig, ax=ax_hom)
ax_hom.set_title("Hong-Ou-Mandel interference\n(two independent sources)")
SPDC_Plotter.style_legend(ax_hom, ncol=2)
fig.tight_layout()

out = Path(__file__).resolve().parents[1] / "assets" / "hero.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
for label, h in hom.items():
    print(f"{label}: visibility {h.visibility:.3f}, purity {h.purity1:.3f}")
print(f"saved {out}")
