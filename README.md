# objSPDC
## Description
Object oriented implementation of the simulation of joint spectral amplitudes and other related properties for SPDC.

## Theoretical background
Spontaneous Parametric Down-Conversion (SPDC) is a second order non-linear process in which a pump photon spontaneously is split into two photons (call signal and idler photons) by interacting with its surrounding medium. 

The effect can be grouped into three categories:
- Type 0: Pump, signal and idler photons have the same polarization. Typically, this is ordinary polarization (denoted "o") for all three photons.
- Type I: Signal and idler photons have the **same polarization**, but it is **orthogonal** to the pump photon’s polarization. Typically, the pump photon is extraordinary polarized ("e"), while the signal and idler photons are both ordinary polarized ("o") or vice versa.
- Type II: Signal and idler photons have **orthogonal polarizations** to each other. For example, one photon is ordinary polarized ("o"), and the other is extraordinary polarized ("e").

During SPDC, energy and momentum is conserved. This can be described by the following equations:

$$\omega_p=\omega_s+\omega_i$$

$$\vec{k}_p=\vec{k}_s+\vec{k}_i+\dfrac{2\pi}{\Lambda}\vec{z}$$

Here $\omega_{p,s,i}$ and $\vec{k}_{p,s,i}$ represent for the pump, signal and idler frequencies / wavenumbers and $\Lambda$ being the poling period, the total distance of two domains of equal length but opposing second order non-linearity $\chi^{(2)}$. The process itself is described by the Hamiltonian

$$\mathbf{H}=c\int d\omega_s d\omega_i\epsilon(\omega_s,\omega_i)\varphi(\omega_s,\omega_i)\mathbf{a}^\dagger(\omega_s)\mathbf{a}^\dagger(\omega_i)+h.c.$$

with $c$ being the vacuum speed of light and $\mathbf{a}^\dagger$ as bosonic creation operators. $\epsilon(\omega_s,\omega_i)$ represents the pump pulse envelope (PPE) and $\varphi(\omega_s,\omega_i)$ is the phase matching function (PMF), which is a characteristic function determined by the crystals properties. 

### Pump pulse envelope
Assuming a Gaussian pump pulse, the function describing the PPE can be written as

$$\epsilon(\omega_s,\omega_i)=e^{-\left(\dfrac{\omega_i+\omega_s-\omega_p}{2\omega_{\text{fwhm}}}\right)^2}.$$

Conversion from pulse duration to bandwidth can be calculated as follows: With $c=\lambda\nu$ ($\lambda$ as wavelength, $\nu$ as frequency), the expression

$$\dfrac{d\lambda}{d\nu}=-\dfrac{c}{\nu^2}$$

can be obtained. By further referring to absolute values and changing the differential to a difference, the expression becomes

$$\Delta\lambda=\dfrac{c}{\nu^2}\Delta\nu.$$

In case for Gaussian pulses, the relation $\Delta\nu\Delta\tau=2\ln(2)/\pi$ can be used to obtain the formula for converting pulsdurations $\Delta\tau$ to bandwidths $\Delta\lambda$ for given central wavelengths $\lambda_0$:

$$\Delta\lambda=\dfrac{2\ln(2)}{\pi}\dfrac{\lambda^2}{c\Delta\tau}$$

### Phase matching function




