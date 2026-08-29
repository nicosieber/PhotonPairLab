import numpy as np
from scipy.optimize import minimize_scalar

from .base_pm_strategy import PhaseMatchingStrategy, SPDCType, PolingMode
from .pm_result import PhaseMismatchResult, PolingResult
from ..material.base_material import BaseMaterial
from photonpairlab.laser import BaseLaser

class APMPhaseMatching(PhaseMatchingStrategy):
    """
    Angle Phase-Matching (APM) strategy for nonlinear crystals.
    """

    def __init__(self, material: BaseMaterial, spdc_type: SPDCType = "type-II", phi_deg: float = 0.0):
        super().__init__(material, spdc_type)
        self.phi_deg = phi_deg  # Used for biaxial crystals    

    def compute_phase_mismatch(self, laser: BaseLaser, 
                               wavelength_signal: float, 
                               wavelength_idler: float, 
                               angle_pm: float | None = None,
                               T: float = 25.0):
        """
        Compute the phase mismatch Δk = k_p - k_s - k_i for a given laser and wavelengths.
        If no phase-matching angle is provided, it uses the phase-matching angle found by minimizing Δk.

        """

        wavelength_pump = laser.wavelength_pump

        if angle_pm is None:
            angle_pm = self.find_phase_matching_angle(laser, wavelength_signal, wavelength_idler)
        
        # Get polarization states based on SPDC type
        pol_pump, pol_signal, pol_idler = self.get_polarizations()

        # Compute refractive indices using effective_refractive_index for angle-based phase matching
        # (Sellmeier coefficients are calibrated for wavelength in micrometers)
        n_pump = self.get_refractive_index(wavelength_pump * 1e6, pol_pump, angle_pm, T)
        n_signal = self.get_refractive_index(wavelength_signal * 1e6, pol_signal, angle_pm, T)
        n_idler = self.get_refractive_index(wavelength_idler * 1e6, pol_idler, angle_pm, T)
        # Compute group indices
        N_pump = self.get_group_index(wavelength_pump * 1e6, pol_pump, angle_pm, T)
        N_signal = self.get_group_index(wavelength_signal * 1e6, pol_signal, angle_pm, T)
        N_idler = self.get_group_index(wavelength_idler * 1e6, pol_idler, angle_pm, T)

        DeltaK_0 = self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler,T)
        return PhaseMismatchResult(
            n=(n_pump, n_signal, n_idler),
            N=(N_pump, N_signal, N_idler),
            delta_k0=DeltaK_0,
            angle_pm=angle_pm,
            coherence_length=self.coherence_length,
        )


    def generate_poling(self, crystal_length: float,
                        T: float,
                        mode: PolingMode,
                        laser: BaseLaser,
                        wavelength_signal: float | None = None,
                        wavelength_idler: float | None = None,
                        coherence_length: float | None = None,
                        w: float | None = None,
                        resolution: int = 5):

        if mode == "constant":
            if wavelength_signal is None or wavelength_idler is None:
                raise ValueError("Both wavelength_signal and wavelength_idler must be provided for APM poling generation.")
            return self._generate_constant_poling(crystal_length, T, resolution, laser, wavelength_signal, wavelength_idler)
        else:
            raise ValueError(f"Unknown poling mode: {mode}. Only 'constant' is supported for APM.")

    def _generate_constant_poling(self, crystal_length: float, T: float, resolution: int,
                                  laser: BaseLaser,
                                  wavelength_signal: float,
                                  wavelength_idler: float):
        """
        Generates a constant (unpoled) structure for angle phase-matched crystals.

        """
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        # For APM, poling is typically constant (no periodic poling)
        Lc = 50e-6  # Example coherence length, adjust as needed
        num_domains = int(np.floor(crystal_length / Lc))
        polarizations = np.tile([1, 1], num_domains)
        poling_pattern = np.repeat(polarizations, resolution)
        z = np.linspace(-temperature_adjusted_length / 2, temperature_adjusted_length / 2, len(poling_pattern))
        DeltaK = self.delta_k(angle=0, laser=laser, wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, T=T)
        # See QPMPhaseMatching._generate_periodic_poling: use the fine, resolution-repeated
        # poling_pattern so the field arrays line up point-for-point with z/poling_pattern.
        target_amplitude, actual_amplitude = self.compute_domain_field_arrays(
            poling_pattern, Lc / resolution, Lc, temperature_adjusted_length, DeltaK)
        return PolingResult(poling_pattern, z, temperature_adjusted_length, target_amplitude, actual_amplitude)

    def find_phase_matching_angle(
            self, laser: BaseLaser,
            wavelength_signal: float, 
            wavelength_idler: float, 
            T: float | None = None
            ) -> float:
        """
        result = minimize_scalar(
            self.delta_k, bounds=(-180, 180), method='bounded',
            args=(wavelength_pump, wavelength_signal, wavelength_idler, 
                    polarization_pump, polarization_signal, polarization_idler))
        """
        result = minimize_scalar(
            lambda angle: abs(self.delta_k(angle, laser, wavelength_signal, wavelength_idler, T)),
            bounds=(0, 90),
            method='bounded',
            #options={'xatol': 1e-5}  # Set the absolute tolerance for the solution
        )
        phase_matching_angle = float(result.x) # type: ignore

        return phase_matching_angle

    def find_phase_matching_angles_jax(
            self,
            wavelength_pump,
            wavelength_signal,
            wavelength_idler,
            T: float = 25.0,
            bounds: tuple[float, float] = (0.0, 90.0),
            iters: int = 50,
        ):
        """
        Vectorized counterpart to ``find_phase_matching_angle``: solves an entire sweep of
        (wavelength_pump, wavelength_signal, wavelength_idler) triples in a single batched JAX
        call instead of looping ``scipy.optimize.minimize_scalar`` once per point.

        Each argument may be a scalar or a 1D array; scalars are broadcast to the array shape.
        Wavelengths are in meters (SI), matching ``find_phase_matching_angle``'s laser-wavelength
        convention. Requires the optional 'jax' dependency (``pip install photonpairlab[jax]``).

        Returns a JAX array of phase-matching angles (degrees), one per sweep point.
        """
        from .jax_accel import find_phase_matching_angle_sweep
        import jax.numpy as jnp

        pol_p, pol_s, pol_i = self.get_polarizations()
        wl_s = jnp.asarray(wavelength_signal, dtype=jnp.float64)
        wl_p = jnp.broadcast_to(jnp.asarray(wavelength_pump, dtype=jnp.float64), wl_s.shape)
        wl_i = jnp.broadcast_to(jnp.asarray(wavelength_idler, dtype=jnp.float64), wl_s.shape)

        return find_phase_matching_angle_sweep(
            self.material, pol_p, pol_s, pol_i, self.phi_deg, wl_p, wl_s, wl_i, T, bounds, iters
        )

    def compute_phase_mismatch_jax(
            self,
            wavelength_pump,
            wavelength_signal,
            wavelength_idler,
            T: float = 25.0,
        ) -> dict:
        """
        Vectorized counterpart to ``compute_phase_mismatch``: for an entire sweep of
        (wavelength_pump, wavelength_signal, wavelength_idler) triples, finds the phase-matching
        angle for each point (see ``find_phase_matching_angles_jax``) and then computes n's, N's,
        and Δk0 for all points in one batched JAX call — N's use exact autodiff instead of the
        numpy path's finite-difference derivative.

        Each argument may be a scalar or a 1D array; scalars are broadcast to the array shape.
        Wavelengths are in meters (SI). Requires the optional 'jax' dependency
        (``pip install photonpairlab[jax]``).

        Returns a dict of JAX arrays (one entry per sweep point): ``angle_pm``, ``delta_k0``,
        ``n`` (tuple of n_pump/n_signal/n_idler arrays), ``N`` (tuple of N_pump/N_signal/N_idler
        arrays).
        """
        from .jax_accel import compute_phase_mismatch_sweep
        import jax.numpy as jnp

        pol_p, pol_s, pol_i = self.get_polarizations()
        wl_s = jnp.asarray(wavelength_signal, dtype=jnp.float64)
        wl_p = jnp.broadcast_to(jnp.asarray(wavelength_pump, dtype=jnp.float64), wl_s.shape)
        wl_i = jnp.broadcast_to(jnp.asarray(wavelength_idler, dtype=jnp.float64), wl_s.shape)

        return compute_phase_mismatch_sweep(
            self.material, pol_p, pol_s, pol_i, self.phi_deg, wl_p, wl_s, wl_i, T
        )

