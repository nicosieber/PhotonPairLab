from typing import Callable

import numpy as np
from scipy.special import erf
from .base_pm_strategy import PhaseMatchingStrategy, SPDCType, PolingMode
from .pm_result import PhaseMismatchResult, PolingResult
from ..material.base_material import BaseMaterial
from photonpairlab.laser import BaseLaser

class QPMPhaseMatching(PhaseMatchingStrategy):
    """
    Quasi Phase-Matching (QPM) strategy for nonlinear crystals.
    """

    def __init__(self, material: BaseMaterial, spdc_type: SPDCType = "type-II", phi_deg: float = 0.0):
        super().__init__(material, spdc_type)
        self.phi_deg = phi_deg  # Used for biaxial crystals

    def compute_phase_mismatch(self, laser: BaseLaser, 
                               wavelength_signal: float, 
                               wavelength_idler: float, 
                               angle_pm: float = 0,
                               T: float = 25.0):
        """
        Computes the phase mismatch (DeltaK_0) based on the SPDC type.

        """

        wavelength_pump = laser.wavelength_pump
      
        # Get polarization states based on SPDC type
        pol_pump, pol_signal, pol_idler = self.get_polarizations()

        # Compute refractive indices using effective_refractive_index for angle-based phase matching
        n_pump = self.get_refractive_index(wavelength_pump * 1e6, pol_pump, angle_pm, T)
        n_signal = self.get_refractive_index(wavelength_signal * 1e6, pol_signal, angle_pm, T)
        n_idler = self.get_refractive_index(wavelength_idler * 1e6, pol_idler, angle_pm, T)
        # Compute group indices
        N_pump = self.get_group_index(wavelength_pump * 1e6, pol_pump, angle_pm, T)
        N_signal = self.get_group_index(wavelength_signal * 1e6, pol_signal, angle_pm, T)
        N_idler = self.get_group_index(wavelength_idler * 1e6, pol_idler, angle_pm, T)

        DeltaK_0 = self.delta_k(angle_pm, laser, wavelength_signal, wavelength_idler, T)
        
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
                        wavelength_signal: float,
                        wavelength_idler: float,
                        coherence_length: float | None = None,
                        w: float | None = None,
                        resolution: int = 5,
                        target_profile: Callable[[np.ndarray, float, float], np.ndarray] | None = None):
        if coherence_length is None:
            raise ValueError("coherence_length must be provided for QPM poling generation.")
        if wavelength_idler is None or wavelength_signal is None:
            raise ValueError("Both wavelength_signal and wavelength_idler must be provided for QPM poling generation.")


        if mode == 'periodic':
            return self._generate_periodic_poling(crystal_length, T, coherence_length, resolution, laser, wavelength_signal, wavelength_idler)
        elif mode == 'constant':
            return self._generate_constant_poling(crystal_length, T, coherence_length, resolution, laser, wavelength_signal, wavelength_idler)
        elif mode == 'subcoh':
            if wavelength_signal is None or wavelength_idler is None:
                raise ValueError("Both wavelength_signal and wavelength_idler must be provided for sub-coherence poling generation.")
            if w is None:
                raise ValueError("Domain width 'w' must be provided for sub-coherence poling generation.")
            return self._generate_subcoh_poling(laser, wavelength_signal, wavelength_idler, crystal_length, w, coherence_length, T, target_profile)
        else:
            raise ValueError(f"Unknown poling mode: {mode}. Use 'periodic' or 'constant'.")
        

    def _generate_periodic_poling(self, crystal_length: float,
                                  T: float,
                                  coherence_length: float,
                                  resolution: int,
                                  laser: BaseLaser,
                                  wavelength_signal: float,
                                  wavelength_idler: float):
        """
        Generates a periodic poling structure for the crystal.
        This method creates a periodic poling structure by alternating polarizations
        (e.g., [1, -1, 1, -1, ...]) over the length of the crystal. The resolution
        determines the number of subdivisions per coherence length (coherence_length). The method
        also adjusts the crystal length (crystal_length) to be an integer multiple of the coherence
        length and calculates the corresponding z-axis values.
        Parameters:
            resolution (int, optional): The number of subdivisions per coherence length.
                                        Default is 5.
        Notes:
            - The coherence length (coherence_length) and original crystal length (crystal_length) must be defined
              as attributes of the class before calling this method.
            - The total length of the z-axis (z) will match the length of the poling_pattern.
        """
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        num_domains = int(np.floor(crystal_length / coherence_length))
        # Create the polarizations array with one alternating-sign entry per domain
        # (np.resize repeats [1, -1] and truncates to exactly num_domains entries;
        # np.tile would double the domain count and halve the realized domain width).
        polarizations = np.resize([1, -1], num_domains)
        # Create the poling pattern using np.repeat
        poling_pattern = np.repeat(polarizations, resolution)
        # Calculate z values directly based on the length of poling_pattern
        z = np.linspace(-temperature_adjusted_length / 2,
                        temperature_adjusted_length / 2,
                        len(poling_pattern))
        DeltaK = self.delta_k(angle=0, laser=laser, wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, T=T)
        # Field arrays are computed on poling_pattern (resolution-repeated), not the coarse
        # pre-repeat polarizations, so they line up point-for-point with z/poling_pattern --
        # each fine step is 1/resolution of a coherence length wide.
        target_amplitude, actual_amplitude = self.compute_domain_field_arrays(
            poling_pattern, coherence_length / resolution, coherence_length, temperature_adjusted_length, DeltaK)
        return PolingResult(poling_pattern, z, temperature_adjusted_length, target_amplitude, actual_amplitude)

    def _generate_constant_poling(self, crystal_length, T, coherence_length, resolution,
                                  laser: BaseLaser,
                                  wavelength_signal: float,
                                  wavelength_idler: float):
        """
        Generates a constant (unpoled) structure for the crystal.
        This method creates a constant poling structure by using the same polarization
        (e.g., [1, 1, 1, 1, ...]) over the length of the crystal. The resolution
        determines the number of subdivisions per coherence length (coherence_length). The method
        also adjusts the crystal length (crystal_length) to be an integer multiple of the coherence
        length and calculates the corresponding z-axis values.
        Parameters:
            resolution (int, optional): The number of subdivisions per coherence length.
                                        Default is 5.
        Notes:
            - The coherence length (coherence_length) and original crystal length (crystal_length) must be defined
              as attributes of the class before calling this method.
            - The total length of the z-axis (z) will match the length of the poling_pattern.
        """
        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        num_domains = int(np.floor(crystal_length / coherence_length))
        # Create the polarizations array with one entry per domain (see _generate_periodic_poling)
        polarizations = np.resize([1, 1], num_domains)
        # Create the poling pattern using np.repeat
        poling_pattern = np.repeat(polarizations, resolution)
        # Calculate z values directly based on the length of poling_pattern
        z = np.linspace(-temperature_adjusted_length / 2,
                        temperature_adjusted_length / 2,
                        len(poling_pattern))
        DeltaK = self.delta_k(angle=0, laser=laser, wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, T=T)
        # See _generate_periodic_poling: use the fine, resolution-repeated poling_pattern so the
        # field arrays line up point-for-point with z/poling_pattern.
        target_amplitude, actual_amplitude = self.compute_domain_field_arrays(
            poling_pattern, coherence_length / resolution, coherence_length, temperature_adjusted_length, DeltaK)
        return PolingResult(poling_pattern, z, temperature_adjusted_length, target_amplitude, actual_amplitude)

    def _generate_subcoh_poling(self, laser: BaseLaser,
                                wavelength_signal: float,
                                wavelength_idler: float,
                                crystal_length: float,
                                w: float,
                                coherence_length: float,
                                T: float,
                                target_profile: Callable[[np.ndarray, float, float], np.ndarray] | None = None,
                                ):
        """
        Generates a sub-coherence length apodized poling pattern for the nonlinear crystal 
        based on the input laser parameters.

        This method computes the poling pattern by iteratively determining the orientation of the 
        nonlinear domains (up or down) that minimizes the error between the target amplitude and 
        the computed amplitude. The resulting poling pattern is stored in the `poling_pattern` attribute, 
        along with additional computed attributes such as `target_amplitudes`, `amuparray`, `amdownarray`, 
        and `altered_z`.

        The algorithm follows the sub-coherence length domain engineering approach, which optimizes 
        the poling pattern to achieve pure down-conversion photons. It uses the refractive indices 
        and group indices of the crystal at the fundamental and second harmonic wavelengths to 
        compute the phase mismatch (DeltaK_0), and applies an iterative apodization algorithm to 
        determine the optimal poling configuration.

        Reference:
            Sub-coherence length apodization algorithm according to:
            Quantum Sci. Technol. 2 (2017) 035001 (https://doi.org/10.1088/2058-9565/aa78d4)
            "Pure down-conversion photons through sub-coherence-length domain engineering"
            Francesco Graffitti, Dmytro Kundys, Derryck T Reid, Agata M Brańczyk, 
            and Alessandro Fedrizzi.

        Notes:
            - The method assumes that the crystal parameters (e.g., `w`, `L`, `coherence_length`) 
              are already defined as attributes of the class.
            - The generated poling pattern (`sarray`) and other computed attributes are stored 
              as class attributes for further use.
        """
        if wavelength_signal is None or wavelength_idler is None:
            raise ValueError("Both wavelength_signal and wavelength_idler must be provided for sub-coherence poling generation.")

        target_profile = target_profile or self.gtarget

        # Proceed with the apodization algorithm using self.DeltaK_0

        temperature_adjusted_length = self.material.thermal_expansion(length=crystal_length, axis="z", temperature=T)
        # angle=0 matches the default angle_pm used elsewhere in QPM (collinear, on-axis
        # propagation); passing None crashes for any SPDC type using 'e' polarization,
        # since effective_refractive_index requires a numeric angle.
        DeltaK = self.delta_k(angle=0, laser=laser,
                              wavelength_signal=wavelength_signal, wavelength_idler=wavelength_idler, T=T)

        num_domains = int(np.ceil(temperature_adjusted_length / w))
        poling_pattern = np.zeros(num_domains, dtype=int)

        # Arbitrary-small-domains method (Graffitti et al. 2017, Appendix A): starting
        # from an empty domain list, greedily choose each domain's sign (up/down) by
        # minimizing e_m = |A_target(m*w) - A_m({s_n})| (Eq. 10), one domain at a time
        # from the very first domain onward -- there is no "seed" domain fixed in advance,
        # since every later decision depends on the accumulated history of prior choices.
        for m in range(1, num_domains + 1):
            idx = m - 1
            at = self.target_amplitude(w, m, temperature_adjusted_length, coherence_length, DeltaK, target_profile)

            poling_pattern[idx] = 1
            amup = self.Am(w, m, coherence_length, poling_pattern[:m])

            poling_pattern[idx] = -1
            amdown = self.Am(w, m, coherence_length, poling_pattern[:m])

            eup = np.abs(at - amup)
            edown = np.abs(at - amdown)

            poling_pattern[idx] = 1 if eup < edown else -1

        # z-position of each domain's right edge (w, 2w, ..., num_domains*w), centered on 0.
        z = np.arange(1, num_domains + 1) * w - temperature_adjusted_length / 2
        target_amplitude, actual_amplitude = self.compute_domain_field_arrays(
            poling_pattern, w, coherence_length, temperature_adjusted_length, DeltaK, target_profile)
        return PolingResult(poling_pattern, z, temperature_adjusted_length, target_amplitude, actual_amplitude)

    def gtarget(self, z, L, coherence_length):
        """
        Computes a Gaussian target function based on the given parameters.

        """
        return np.exp(-((z - L / 2) ** 2) / (L ** 2 / 8)) # L**2 is divided by 8 as suggested by the reference

    def sigmoid_target(self, z, L, coherence_length, sigma):
        """
        Monotonic sigmoid (error-function) target profile rising from 0 to 1 across the crystal,
        centered at L/2 with transition width `sigma`. Useful as an alternative apodization target
        to the symmetric Gaussian bump in `gtarget` -- e.g. to reproduce figures where the target
        nonlinear profile is plotted against a "profile width sigma/L".

        Note: unlike `gtarget`, this takes an extra `sigma` argument, so it cannot be passed directly
        as `target_profile` -- wrap it first, e.g. `functools.partial(strategy.sigmoid_target, sigma=...)`.
        """
        return 0.5 * (1 + erf((z - L / 2) / (np.sqrt(2) * sigma)))

    def target_amplitude(self, w, m, L, coherence_length, DeltaK, target_profile: Callable[[np.ndarray, float, float], np.ndarray] | None = None):
        """
        Target field amplitude after m domains of width w (Graffitti et al. 2017, Eq. 5),
        for a target nonlinearity g_target(z) = g(z)*cos(pi z/coherence_length), where g(z) is
        given by `target_profile` (defaults to the Gaussian `gtarget`).

        Expanding cos(Kz)*exp(i*DeltaK*z) = 1/2 exp(i(DeltaK+K)z) + 1/2 exp(i(DeltaK-K)z)
        (K = pi/coherence_length), only the near-resonant term (whichever combination is
        closest to zero frequency) is kept -- this is what Eq. 7 does analytically
        ("ignoring the quickly oscillating terms"). The other term oscillates with
        period ~2*coherence_length, i.e. on the same scale as the domain width itself
        for sub-coherence-length domains, so a discrete sum over z = w, 2w, ..., m*w
        cannot reliably average it out numerically; it must be dropped analytically
        instead, or it leaks into the target as spurious high-frequency structure.

        z runs over domain *right edges* w, 2w, ..., m*w, matching the convention
        used in Am's Eq. 9 sum (domain n occupies ((n-1)w, nw]).
        """
        target_profile = target_profile or self.gtarget
        z = np.arange(1, m + 1) * w
        g = target_profile(z, L, coherence_length)
        K = np.pi / coherence_length
        freq_plus = DeltaK + K
        freq_minus = DeltaK - K
        freq = freq_plus if abs(freq_plus) < abs(freq_minus) else freq_minus
        y = 0.5 * g * np.exp(1j * freq * z)
        return -1j * np.trapezoid(y, z)

    def Am(self, w, m, coherence_length, sn):
        """
        Field amplitude generated by m domains of width w with signs sn, evaluated
        at Delta k = pi/coherence_length (Graffitti et al. 2017, Eq. 9):

            A_m = (coherence_length / pi) * (exp(-i*K*w) - 1) * sum_n sn[n] * exp(i*K*n*w)

        with K = pi/coherence_length and n = 1, ..., m (domain n's right edge is n*w).
        """
        if len(sn) != m:
            raise ValueError("Poling array length wrong.")
        K = np.pi / coherence_length
        n = np.arange(1, m + 1)
        exp_term = np.exp(1j * K * n * w)
        y = np.sum(np.asarray(sn) * exp_term)
        return coherence_length / np.pi * (np.exp(-1j * K * w) - 1) * y
