from typing import Callable, Literal

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..material.base_material import BaseMaterial
from .pm_result import PolingResult
from photonpairlab.laser import BaseLaser

POLARIZATION_MAP: dict[str, tuple[str, str, str]] = {
    "type-0": ("e", "e", "e"),
    "type-I": ("e", "o", "o"),
    "type-II": ("y", "z", "y"),
    "type-IIeoe": ("e", "o", "e"),
    "type-IIoeo": ("o", "e", "o"),
}

# Must stay in sync with POLARIZATION_MAP above.
SPDCType = Literal["type-0", "type-I", "type-II", "type-IIeoe", "type-IIoeo"]
# Union of the poling modes supported across all PhaseMatchingStrategy subclasses
# (QPM: periodic/constant/subcoh; APM: constant only).
PolingMode = Literal["periodic", "constant", "subcoh"]

class PhaseMatchingStrategy:
    """
    Base class for phase-matching strategies (QPM, APM, ...).
    Provides the interface and common placeholders for all strategies.
    """

    def __init__(self, material: BaseMaterial, spdc_type: SPDCType="type-II", coherence_length: float | None = None):
        self.material = material
        self.spdc_type = spdc_type
        self.coherence_length = coherence_length

    def get_refractive_index(self, wavelength: float, polarization: str, angle: float | None, T: float | None):
        if polarization == "o":
            axis = self.material.map_polarization_axis("o")
            return self.material.refractive_index(wavelength, axis=axis, temperature=T)
        elif polarization == "e":
            return self.material.effective_refractive_index(wavelength, theta_deg=angle, phi_deg=self.phi_deg) # type: ignore
        else:
            axis = self.material.map_polarization_axis(polarization)
            return self.material.refractive_index(wavelength, axis=axis, temperature=T)

    def get_group_index(self, wavelength: float, polarization: str, angle: float, T: float):
        """
        Returns the group index for the given wavelength, polarization, and angle.
        """
        if polarization == "o":
            axis = self.material.map_polarization_axis("o")
            # Use axis-based group index (QPM or propagation along axis)
            return self.material.group_index(wavelength, axis=axis, temperature=T)
        elif polarization == "e":
            # Use angle-based group index (angle phase-matching)
            return self.material.group_index(wavelength, theta_deg=angle, phi_deg=self.phi_deg) # type: ignore
        else:
            axis = self.material.map_polarization_axis(polarization)
            return self.material.group_index(wavelength, axis=axis, temperature=T)
        
    def get_polarizations(self):
        """
        Return (pol_pump, pol_signal, pol_idler) for the current SPDC type.
        """
        try:
            return POLARIZATION_MAP[self.spdc_type]
        except KeyError as e:
            raise ValueError(f"Invalid SPDC type: {self.spdc_type!r}") from e

    def compute_phase_mismatch(self, *args, **kwargs):
        """
        Compute phase mismatch Δk for the given parameters. 
        This method is used in the simulation.py of the spdc module.
        """
        raise NotImplementedError("Implement in subclass.")

    def delta_k(self, angle: float | None, laser: BaseLaser, 
                wavelength_signal: float | None, wavelength_idler: float | None, T: float | None):
        """
        Calculate Δk = k_p - k_s - k_i for a given angle.

        Args:
            angle_deg (float): Phase-matching angle in degrees.
            laser (CWLaser): Laser object containing pump wavelength.
            wavelength_signal (float): Signal wavelength in meters.
            wavelength_idler (float): Idler wavelength in meters.

        Returns:
            float: Phase mismatch Δk in m⁻¹ (SI).
        """
        # Convert wavelengths to micrometers
        wavelength_pump = laser.wavelength_pump * 1e6
        if wavelength_signal is None or wavelength_idler is None:
            raise ValueError("Both wavelength_signal and wavelength_idler must be provided.")
        else:
            wavelength_signal = wavelength_signal * 1e6
            wavelength_idler = wavelength_idler * 1e6

        # Get polarization states based on SPDC type
        polarzation_pump, polarzation_signal, polarzation_idler = self.get_polarizations()
        
        # Compute refractive indices
        n_p = self.get_refractive_index(wavelength_pump, polarzation_pump, angle, T)
        n_s = self.get_refractive_index(wavelength_signal, polarzation_signal, angle, T)
        n_i = self.get_refractive_index(wavelength_idler, polarzation_idler, angle, T)

        # Compute wavevectors
        k_p = 2 * np.pi * n_p / wavelength_pump
        k_s = 2 * np.pi * n_s / wavelength_signal
        k_i = 2 * np.pi * n_i / wavelength_idler

        return (k_p - k_s - k_i) * 1e6  # k_x were in µm⁻¹ (µm-scaled wavelengths); *1e6 converts to m⁻¹

    def generate_poling(self, *args, **kwargs) -> PolingResult:
        """
        Generate poling pattern (periodic, sub-coherence, constant, etc.). Returns a
        ``PolingResult`` bundling the domain-sign pattern/z-axis with its target vs. actual
        field-amplitude buildup (see ``compute_domain_field_arrays``).
        """
        raise NotImplementedError("Implement in subclass.")

    @staticmethod
    def uniform_target(z: np.ndarray, L: float, coherence_length: float) -> np.ndarray:
        """
        Ideal, fully-efficient target profile (g(z) = 1) -- the default for non-apodized
        (periodic/constant) poling, where there is no intentional envelope to track.

        Note this represents an idealized *continuous* sinusoidal nonlinearity cos(Kz), not the
        realizable ±1 square wave a plain periodic grating actually is. For a non-apodized (plain
        periodic) grating, expect the realized ``actual_amplitude`` to run ~4/pi above the
        ``target_amplitude`` this produces: a square wave's fundamental Fourier component is 4/pi
        times a pure sinusoid of the same peak amplitude (the same rectangular-window effect
        responsible for a plain grating's phase-matching-function side-lobes). This is expected,
        not a bug -- it's exactly the gap sub-coherence-length apodization (``subcoh``) closes by
        choosing domain signs to track a target instead of ignoring it.
        """
        return np.ones_like(z)

    def compute_domain_field_arrays(
            self,
            domain_signs: np.ndarray,
            w: float,
            coherence_length: float,
            L: float,
            DeltaK: float,
            target_profile: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Vectorized target vs. actual field-amplitude buildup (Graffitti et al. 2017, Eq. 5 & 9,
        https://doi.org/10.1088/2058-9565/aa78d4) for a *finalized* sequence of domain signs of
        width w, one value per domain (length = len(domain_signs)).

        ``actual_amplitude`` is the field amplitude the discrete domain pattern actually realizes
        (Eq. 9); ``target_amplitude`` is the designed/ideal field amplitude for a continuous target
        nonlinearity g_target(z) = target_profile(z, L, coherence_length) * cos(pi z/coherence_length)
        (Eq. 5), defaulting to a uniform g(z) = 1 envelope when no ``target_profile`` is given.
        Both are generic to any domain-sign sequence, not just ones chosen to track a target --
        e.g. for periodic/constant poling, ``actual_amplitude`` still shows how well (or poorly, for
        an unpoled/non-inverted pattern) the realized structure builds up against the ideal ramp.
        """
        target_profile = target_profile or self.uniform_target
        n = np.arange(1, len(domain_signs) + 1)
        z = n * w
        K = np.pi / coherence_length

        exp_term = np.exp(1j * K * z)
        actual_amplitude = (coherence_length / np.pi) * (np.exp(-1j * K * w) - 1) * np.cumsum(np.asarray(domain_signs) * exp_term)

        g = target_profile(z, L, coherence_length)
        freq_plus, freq_minus = DeltaK + K, DeltaK - K
        freq = freq_plus if abs(freq_plus) < abs(freq_minus) else freq_minus
        y = 0.5 * g * np.exp(1j * freq * z)
        target_amplitude = -1j * cumulative_trapezoid(y, z, initial=0)

        return target_amplitude, actual_amplitude

    def compute_domain_field_arrays_nonuniform(
            self,
            domain_signs: np.ndarray,
            z: np.ndarray,
            coherence_length: float,
            L: float,
            DeltaK: float,
            target_profile: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
            ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generalization of ``compute_domain_field_arrays`` for a domain-sign sequence sampled on
        an arbitrary (non-uniform-width) ``z`` grid -- e.g. after
        ``PolingResult.add_wall_position_error()`` or ``.add_duty_cycle_bias()`` have perturbed
        domain widths, which breaks the single-scalar-``w`` assumption behind
        ``compute_domain_field_arrays``'s closed-form Eq. 9 recursion (Graffitti et al. 2017).

        Both ``target_amplitude`` and ``actual_amplitude`` are evaluated by direct cumulative
        trapezoidal integration against the (possibly irregular) ``z`` -- the same numerical
        technique ``compute_domain_field_arrays`` already uses for ``target_amplitude``, applied
        here to ``actual_amplitude`` too, since Eq. 9's closed form has no direct
        non-uniform-width analogue without re-deriving it from scratch. This introduces an
        O(1/resolution) discretization error at each domain boundary relative to the exact
        closed form (negligible for resolution >~ 10-20); on a uniform grid it converges to
        ``compute_domain_field_arrays``'s result as resolution grows. Used only for the
        ``target_amplitude``/``actual_amplitude`` diagnostic (``plot_poling_profile``) --
        ``SPDC_Simulation.phase_matching_function`` integrates the actual physics independently
        and is already exact/general over irregular ``z``.
        """
        target_profile = target_profile or self.uniform_target
        K = np.pi / coherence_length

        y_actual = np.asarray(domain_signs) * np.exp(1j * K * z)
        actual_amplitude = -1j * cumulative_trapezoid(y_actual, z, initial=0)

        g = target_profile(z, L, coherence_length)
        freq_plus, freq_minus = DeltaK + K, DeltaK - K
        freq = freq_plus if abs(freq_plus) < abs(freq_minus) else freq_minus
        y_target = 0.5 * g * np.exp(1j * freq * z)
        target_amplitude = -1j * cumulative_trapezoid(y_target, z, initial=0)

        return target_amplitude, actual_amplitude