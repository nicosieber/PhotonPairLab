============
SPDC: Theory
============

PhotonPairLab is a Python-based simulation toolkit for modeling the generation of photon pairs via spontaneous parametric down-conversion (SPDC) in nonlinear crystals.

Spontaneous Parametric Down-Conversion is a second-order nonlinear process in which a pump photon spontaneously splits into two photons (called *signal* and *idler* photons) by interacting with its surrounding medium.

The effect can be grouped into three categories:

- **Type 0**: Pump, signal and idler photons have the same polarization. Typically, this is ordinary polarization (denoted ``o``) for all three photons.

- **Type I**: Signal and idler photons have the same polarization, but it is orthogonal to the pump photon’s polarization. Typically, the pump photon is extraordinary polarized (``e``), while the signal and idler photons are both ordinary polarized (``o``), or vice versa.

- **Type II**: Signal and idler photons have orthogonal polarizations to each other. For example, one photon is ordinary polarized (``o``), and the other is extraordinary polarized (``e``).

The process itself is described by the Hamiltonian

.. math::

   \mathbf{H}
   =
   c \int d\omega_s \, d\omega_i \,
   \epsilon(\omega_s,\omega_i)
   \varphi(\omega_s,\omega_i)
   \mathbf{a}^\dagger(\omega_s)
   \mathbf{a}^\dagger(\omega_i)
   + h.c.

with :math:`c` being the vacuum speed of light and :math:`\mathbf{a}^\dagger` bosonic creation operators.
:math:`\epsilon(\omega_s,\omega_i)` represents the **pump pulse envelope (PPE)** and
:math:`\varphi(\omega_s,\omega_i)` is the **phase matching function (PMF)**, which is determined by the crystal properties.

The SPDC process is constrained by conservation laws.

Energy conservation
-------------------

.. math::

   \omega_p = \omega_s + \omega_i

This ensures that the total energy of the generated signal :math:`\omega_s` and idler :math:`\omega_i`
photons equals the energy of the pump photon :math:`\omega_p`.

Momentum conservation
---------------------

.. math::

   \vec{k}_p = \vec{k}_s + \vec{k}_i

This relation must be satisfied for efficient photon pair generation. However, in many practical scenarios this condition cannot be fulfilled exactly due to material dispersion or birefringence. This mismatch leads to reduced efficiency and must be corrected using **phase matching techniques**.

Pump Pulse Envelope (PPE)
-------------------------

We assume a Gaussian profile for the pump pulse. Its spectral envelope is given by

.. math::

   \epsilon(\omega_s, \omega_i)
   =
   \exp\left[
   -
   \left(
   \frac{\omega_s + \omega_i - \omega_p}{2\omega_{\text{fwhm}}}
   \right)^2
   \right]

This term reflects the spectral amplitude distribution of the pump field.

The full-width-half-maximum (FWHM) of the pulse in time and frequency are linked via

.. math::

   \Delta\nu \cdot \Delta\tau
   =
   \frac{2\ln(2)}{\pi}

This yields a conversion from pulse duration :math:`\Delta\tau`
to wavelength bandwidth :math:`\Delta\lambda` at a given central wavelength :math:`\lambda_0`:

.. math::

   \Delta\lambda
   =
   \frac{2\ln(2)}{\pi}
   \cdot
   \frac{\lambda_0^2}{c \Delta\tau}

Phase Matching Function (PMF)
-----------------------------

The phase matching function determines how well the interacting waves satisfy momentum conservation within the nonlinear crystal.

.. math::

   \varphi(\omega_s, \omega_i)
   =
   \int
   \chi(z)
   \exp[-i\Delta k(\omega_s, \omega_i) z]
   \, dz

where the phase mismatch is defined as

.. math::

   \Delta k(\omega_s, \omega_i)
   =
   k_p(\omega_s + \omega_i)
   -
   k_s(\omega_s)
   -
   k_i(\omega_i)

This mismatch accumulates along the crystal length and suppresses efficient down-conversion when non-zero. The shape of the PMF is typically **sinc-like** and strongly affects the joint spectral amplitude (JSA).

Phase Matching Techniques
-------------------------

The central goal of any phase-matching technique is to minimize the **phase mismatch**, ensuring that the interacting waves remain in phase as they propagate through the nonlinear crystal.

- In **angle phase matching**, this is achieved by optimizing the internal propagation angle(s) — :math:`\theta` for uniaxial crystals, or both :math:`\theta` and :math:`\phi` for biaxial crystals — to find the direction where :math:`\Delta k \approx 0`.

- In **quasi-phase matching (QPM)**, the crystal is periodically or aperiodically poled to engineer a compensating grating vector. The poling period :math:`\Lambda` is tuned so that :math:`\Delta k_{\text{eff}} \approx 0`.

Angle Phase Matching (APM)
--------------------------

Angle phase matching is a method to fulfill the phase-matching condition in birefringent nonlinear crystals by exploiting their angularly dependent refractive indices. Unlike quasi-phase matching, this technique does not require periodic poling, but instead involves orienting the crystal at a specific angle to the pump beam.

Phase-Matching Condition
~~~~~~~~~~~~~~~~~~~~~~~~

The momentum conservation (phase-matching) condition for collinear SPDC is

.. math::

   \vec{k}_p = \vec{k}_s + \vec{k}_i

This implies that the refractive indices of the interacting waves must satisfy

.. math::

   n_p(\theta) \cdot \omega_p
   =
   n_s \cdot \omega_s
   +
   n_i \cdot \omega_i

Since birefringent crystals have polarization- and angle-dependent refractive indices, adjusting the internal angle :math:`\theta` can allow this condition to be satisfied.

Effective Refractive Index in Uniaxial Crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In uniaxial crystals like **BBO**, the refractive index for extraordinary-polarized light depends on the angle :math:`\theta` between the optical axis and the propagation direction:

.. math::

   \frac{1}{n_e^2(\theta)}
   =
   \frac{\cos^2(\theta)}{n_o^2}
   +
   \frac{\sin^2(\theta)}{n_e^2}

where

- :math:`n_o` — ordinary refractive index (independent of angle)
- :math:`n_e` — extraordinary refractive index (along the optical axis)
- :math:`n_e(\theta)` — effective refractive index for the extraordinary ray

This allows precise tuning of the pump's effective index to match the phase-matching condition by rotating the crystal.

Effective Refractive Index in Biaxial Crystals
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In biaxial crystals like **BiBO**, the situation is more complex due to two optic axes and three principal refractive indices: :math:`n_x, n_y, n_z`.

.. math::

   \frac{\cos^2(\theta)\cos^2(\phi)}{n_x^2}
   +
   \frac{\cos^2(\theta)\sin^2(\phi)}{n_y^2}
   +
   \frac{\sin^2(\theta)}{n_z^2}
   =
   \frac{1}{n_{\text{eff}}^2(\theta,\phi)}

where

- :math:`n_x, n_y, n_z` are the **principal refractive indices** of the biaxial crystal along the orthogonal crystal axes.
- :math:`\theta` is the **internal polar angle**, measured from the crystal :math:`z`-axis.
- :math:`\phi` is the **internal azimuthal angle**, measured in the plane perpendicular to the :math:`z`-axis.

Together, :math:`\theta` and :math:`\phi` specify the **propagation direction of light inside the crystal**, and this direction determines the effective refractive index experienced by each polarization mode.

Angle phase matching thus provides a tunable method for satisfying the momentum conservation condition by exploiting the intrinsic birefringence of nonlinear crystals.

Quasi-Phase Matching (QPM)
--------------------------

In materials like periodically-poled KTP (ppKTP), exact phase matching is not naturally possible due to fixed dispersion properties. Instead, QPM uses periodic inversion of the nonlinear susceptibility :math:`\chi^{(2)}` every coherence length :math:`l_c`.

This effectively flips the phase of the downconverted wave and maintains constructive interference over long distances.

In this case, the phase mismatch is compensated by an engineered grating vector

.. math::

   \Delta K \approx \frac{2\pi}{\Lambda}

where :math:`\Lambda` is the poling period.

The modified phase matching condition becomes

.. math::

   k_p \approx k_s + k_i + \Delta K

This enables efficient SPDC even in materials without natural birefringent phase matching.