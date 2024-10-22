# objSPDC
## Object oriented implementation of the simulation of joint spectral amplitudes and other related properties for SPDC.

Spontaneous Parametric Down-Conversion (SPDC) is a second order non-linear process in which a pump photon spontaneously is split into two photons (call signal and idler photons) by interacting with its surrounding medium. 

The effect can be grouped into three categories:
- Type 0: Pump, signal and idler photons have the same polarization. Typically, this is ordinary polarization (denoted "o") for all three photons.
- Type I: Signal and idler photons have the **same polarization**, but it is **orthogonal** to the pump photon’s polarization. Typically, the pump photon is extraordinary polarized ("e"), while the signal and idler photons are both ordinary polarized ("o") or vice versa.
- Type II: Signal and idler photons have **orthogonal polarizations** to each other. For example, one photon is ordinary polarized ("o"), and the other is extraordinary polarized ("e").

During SPDC, energy and momentum is conserved. This can be described by the following equations:

$$\omega_p=\omega_s+\omega_i$$

$$\vec{k}_p=\vec{k}_s+\vec{k}_i+\dfrac{2\pi}{\Lambda}\vec{z}$$

Here $\omega_{p,s,i}$ and $\vec{k}_{p,s,i}$ represent for the pump, signal and idler frequencies / wavenumbers and $\Lambda$ being the poling period, the total distance of two domains of equal length but opposing second order non-linearity $\chi^{(2)}$. The process itself is described by the Hamiltonian

$$\hat{H}=c\int d\omega_s d\omega_i\epsilon(\omega_s,\omega_i)\varphi(\omega_s,\omega_i)\hat{a}^\dagger(\omega_s)\hat{a}^\dagger(\omega_i)+h.c.$$