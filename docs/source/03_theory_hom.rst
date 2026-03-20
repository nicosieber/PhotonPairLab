Hong–Ou–Mandel Interference
===========================

Introduction
------------

Hong–Ou–Mandel (HOM) interference is one of the most fundamental quantum optical
two-photon interference effects. It occurs when two indistinguishable photons
arrive simultaneously at the two input ports of a balanced beamsplitter.
Instead of exiting separately, the photons *bunch* and always leave together
through the same output port.

As a consequence, the coincidence probability between detectors placed at the
two output ports vanishes when the photons are perfectly indistinguishable.
This phenomenon manifests as the well-known **HOM dip** in the coincidence
rate as a function of the relative delay between the photons.

The HOM dip is widely used to quantify photon indistinguishability,
spectral purity, and temporal coherence in quantum photonic experiments.


Two-Photon Interference at a Beamsplitter
-----------------------------------------

Consider two photons entering a balanced beamsplitter through input modes
:math:`a` and :math:`b`.

The beamsplitter transforms the field creation operators according to

.. math::

    a^\dagger \rightarrow \frac{1}{\sqrt{2}} (c^\dagger + d^\dagger)

.. math::

    b^\dagger \rightarrow \frac{1}{\sqrt{2}} (c^\dagger - d^\dagger)

where :math:`c` and :math:`d` denote the output modes.

If the input state contains one photon in each input port,

.. math::

    |\psi\rangle = a^\dagger b^\dagger |0\rangle,

the transformation leads to

.. math::

    |\psi_{\text{out}}\rangle =
    \frac{1}{2}
    \left(
    c^{\dagger 2} - d^{\dagger 2}
    \right)|0\rangle.

The cross term responsible for coincidences cancels exactly.
As a result, both photons always exit through the same port and the
coincidence probability becomes zero.


Role of Distinguishability
--------------------------

Perfect photon bunching requires the photons to be indistinguishable
in all degrees of freedom:

* frequency
* polarization
* spatial mode
* arrival time

If the photons become distinguishable, the destructive interference
is reduced and coincidences reappear.

Experimentally this is observed by introducing a relative time delay
:math:`\tau` between the photons and measuring the coincidence rate.
The result is the characteristic **HOM dip**.


Spectral Description of Single Photons
--------------------------------------

Single photons generated in nonlinear processes such as spontaneous
parametric down-conversion (SPDC) are generally described in the
frequency domain.

The two-photon state can be written as

.. math::

    |\Psi\rangle =
    \int d\omega_s d\omega_i \,
    \Phi(\omega_s,\omega_i)
    a_s^\dagger(\omega_s)
    a_i^\dagger(\omega_i)
    |0\rangle

where

:math:`\Phi(\omega_s,\omega_i)` is the **Joint Spectral Amplitude (JSA)**.

The reduced spectral state of one photon is obtained by tracing out
the partner photon,

.. math::

    \rho_s(\omega,\omega')
    =
    \int d\omega_i\,
    \Phi(\omega,\omega_i)
    \Phi^*(\omega',\omega_i).

This reduced density matrix describes the spectral properties of the
single photon.


Coincidence Probability
-----------------------

For two photons described by density matrices
:math:`\rho_1` and :math:`\rho_2`,
the coincidence probability at a beamsplitter is

.. math::

    P_c(\tau)
    =
    R^2 + T^2
    -
    2RT\,\mathrm{Re}
    \left[
    \mathrm{Tr}
    \left(
    \rho_1
    \rho_2(\tau)
    \right)
    \right],

where

* :math:`R` and :math:`T` are the reflection and transmission coefficients
  of the beamsplitter
* :math:`\rho_2(\tau)` represents the delayed photon state.

For a balanced beamsplitter (:math:`R=T=1/2`) this reduces to

.. math::

    P_c(\tau)
    =
    \frac{1}{2}
    \left(
    1 -
    \mathrm{Re}
    \left[
    \mathrm{Tr}(\rho_1 \rho_2(\tau))
    \right]
    \right).

The overlap between the two density matrices determines the depth
of the HOM dip.


Time Delay in the Frequency Domain
----------------------------------

A temporal delay corresponds to a phase shift in the frequency basis.
If the spectral density matrix is written as

.. math::

    \rho(\omega,\omega'),

a delay :math:`\tau` transforms the state as

.. math::

    \rho_\tau(\omega,\omega')
    =
    e^{-i(\omega-\omega')\tau}
    \rho(\omega,\omega').

This representation is particularly convenient for numerical simulations,
because the delay can be applied directly as a phase factor to the
density matrix.


Numerical Evaluation
--------------------

Numerically, the HOM dip can be evaluated by performing the following steps:

1. Compute the reduced density matrices of the photons from the JSA.

2. Apply a time delay to one photon by multiplying the density matrix
   with the phase factor

   .. math::

       e^{-i2\pi(f-f')\tau}.

3. Compute the overlap

   .. math::

       \mathrm{Tr}(\rho_1 \rho_2(\tau)).

4. Insert the overlap into the coincidence probability formula.

This approach allows the coincidence probability to be evaluated
for arbitrary spectral states.

The implementation in this package follows precisely this procedure.
The reduced density matrices are obtained from the simulated JSA
and the delay is applied in the frequency domain. The coincidence
probability is then computed from the density-matrix overlap.

See :class:`HOMAnalyzer` for the implementation. :contentReference[oaicite:0]{index=0}

The resulting coincidence probabilities and delay axis are stored
in the :class:`TwoModeHOMResults` container. :contentReference[oaicite:1]{index=1}


Connection to Photon Purity
---------------------------

If two photons originate from identical sources and interfere
with themselves, the HOM visibility becomes

.. math::

    V = \mathrm{Tr}(\rho^2),

which corresponds to the **spectral purity** of the photon.

Consequently, HOM interference provides an experimentally accessible
method to measure the purity of single-photon states.


Applicability of Numerical HOM Calculations
-------------------------------------------

The density-matrix formulation of HOM interference is particularly
well suited for numerical simulations because

* arbitrary spectral correlations can be included,
* mixed photon states can be treated naturally,
* partial distinguishability can be modeled,
* delays can be applied analytically in the frequency basis.

These properties make the approach especially useful for simulations
based on joint spectral amplitudes produced by SPDC sources,
where spectral correlations play a central role.


References
----------

Hong, C. K., Ou, Z. Y., & Mandel, L. (1987).  
*Measurement of subpicosecond time intervals between two photons by interference.*  
Physical Review Letters **59**, 2044.

Mandel, L., & Wolf, E. (1995).  
*Optical Coherence and Quantum Optics.*  
Cambridge University Press.

Mosley, P. J. et al. (2008).  
*Heralded generation of ultrafast single photons in pure quantum states.*  
Physical Review Letters **100**, 133601.

Brańczyk, A. M. (2017).  
*Hong–Ou–Mandel interference.*  
In *Quantum Photonics* (Springer).

Rohde, P. P., & Ralph, T. C. (2005).  
*Modelling photon distinguishability in quantum interference.*  
Physical Review A **71**, 032320.