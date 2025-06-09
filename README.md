# PhotonPairLab
## Description
**PhotonPairLab** is a Python-based simulation toolkit for modeling the generation of photon pairs via spontaneous parametric down-conversion (SPDC) in nonlinear crystals. This project is designed with a clean, modular, and object-oriented architecture, making it extensible for further development or integration into larger quantum optics simulations.

## Features

* Object-oriented architecture with clean separation of concerns.
* Models type-0, type-I, and type-II SPDC processes.
* Supports both quasi-phase matching (QPM) and angle phase matching (APM) processes.
* Visualization tools for key physical quantities, including joint spectral properties, enabling intuitive analysis and interpretation of SPDC processes.
* Easily extendable to support different crystal types and pump configurations.
* Includes methods for generating poling patterns for QPM crystals and constant poling structures for APM crystals.
* Optimized phase-matching angle calculations for APM crystals using numerical minimization techniques.
* Accurate computation of refractive indices, group indices, and phase mismatch for both QPM and APM crystals.

## Architecture Overview

The codebase is structured using well-defined classes:

* `qpm`: Handles quasi-phase matching (QPM) processes. This module includes:
  - `materials_qpm`: Provides models for nonlinear optical materials used in QPM, including their Sellmeier coefficients, temperature corrections, and thermal expansion properties.
  - `crystal_qpm`: Encapsulates physical properties of QPM crystals, such as poling period, temperature, and dispersion, and provides methods for generating alternating poling patterns.
* `apm`: Handles angle phase matching (APM) processes. This module includes:
  - `materials_apm`: Provides models for nonlinear optical materials used in APM, including their effective refractive indices and group indices based on propagation angles.
  - `crystal_apm`: Encapsulates physical properties of APM crystals, such as phase-matching angles, constant poling structures, and dispersion, and provides methods for generating constant poling patterns.
* `laser`: Models the pump laser, supporting both continuous-wave (CW) and pulsed lasers. This module includes:
  - `base_laser`: A base class containing shared functionality, such as wavelength and utility methods for bandwidth and pulse width conversions.
  - `pulsed_laser`: Represents pulsed lasers, allowing for the calculation of bandwidth from pulse duration and vice versa.
  - `cw_laser`: Represents continuous-wave lasers, where the bandwidth is directly specified.
* `spdc`: Handles the simulation, analysis, and visualization of SPDC processes, including computing the JSA and related quantities. This module includes:
  - `simulation`: Provides tools for simulating SPDC processes, including generating the Joint Spectral Amplitude (JSA) and related quantities.
  - `spectral_analyzer`: Contains methods for analyzing spectral properties, such as signal and idler peaks, Schmidt decomposition, and purity calculations.
  - `hom_analyzer`: Enables the computation of Hong-Ou-Mandel (HOM) interference, including cross-correlation and autocorrelation probabilities.
  - `plotting`: Provides visualization tools for SPDC-related quantities, such as the JSA, Schmidt coefficients, and HOM dips.
  - `utils`: Includes utility functions for interpolation, matrix manipulation, and general-purpose calculations used across the SPDC module.

This separation makes the code easy to read, maintain, and expand.

## How to use
For a demonstration on how to use **PhotonPairLab**, have a look at the [demo notebook](./demo.ipynb).

## Disclaimer
This project is a work in progress, and while I strive for accuracy, there may still be areas that need improvement or refinement. I encourage experts in the field to contribute by:

* Adding new materials, including their Sellmeier coefficients, temperature corrections, and thermal expansion properties.
* Reviewing the current implementation to ensure correctness from a physics perspective.
* Suggesting improvements to existing features or providing feedback on better approaches.
* Proposing or implementing new capabilities that could enhance the project's functionality.

Your expertise and contributions would be greatly appreciated to make this project more robust and reliable!