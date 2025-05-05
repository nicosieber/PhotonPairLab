# PhotonPairLab
## Description
**PhotonPairLab** is a Python-based simulation toolkit for modeling the generation of photon pairs via spontaneous parametric down-conversion (SPDC) in nonlinear crystals. This project is designed with a clean, modular, and object-oriented architecture, making it extensible for further development or integration into larger quantum optics simulations.

## Features

* Object-oriented architecture with clean separation of concerns.
* Models type-0, type-I and type-II SPDC processes.
* Visualization tools for key physical quantities, including joint spectral properties, enabling intuitive analysis and interpretation of SPDC processes.
* Easily extendable to support different crystal types and pump configurations.

## Architecture Overview

The codebase is structured using well-defined classes:

* `materials`: Provides models for nonlinear optical materials, including their Sellmeier coefficients, temperature corrections, and thermal expansion properties.
* `crystal`: Encapsulates physical properties like poling period, temperature, and dispersion. 
* `laser`: Models the pump laser, including its wavelength and bandwidth.
* `spdc`: Handles the simulation, analysis, and visualization of SPDC processes, including computing the JSA and related quantities.

This separation makes the code easy to read, maintain, and expand.

## Disclaimer
This project is a work in progress, and while I strive for accuracy, there may still be areas that need improvement or refinement. I encourage experts in the field to contribute by:

* Adding new materials, including their Sellmeier coefficients, temperature corrections, and thermal expansion properties.
* Reviewing the current implementation to ensure correctness from a physics perspective.
* Suggesting improvements to existing features or providing feedback on better approaches.
* Proposing or implementing new capabilities that could enhance the project's functionality.

Your expertise and contributions would be greatly appreciated to make this project more robust and reliable!

## References for ...
... Sellmeier coefficients and temperature corrections
1. F. Konig et al., APL, 84,1644, 2004
2. K. Fradkin et al., APL, 74,914, 1999, https://aip.scitation.org/doi/pdf/10.1063/1.123408
3. Emanueli et al., App. Opt., 42, 33, 2003
4. https://www.unitedcrystals.com/KTPProp.html
5. Kato et al., Appl. Opt. 41, 5040-5044 (2002) 

... thermal expansion:
1. S. Emanueli & A. Arie, App. Opt, vol. 42, No. 33 (2003)
2. ... Unfortunately I could not find other values / references for (KTP) so far

... implementation of the sub-coherence-length apodization algorithm
1. Francesco Graffitti, Dmytro Kundys, Derryck T. Reid, Agata M. Brańczyk and Alessandro Fedrizzi, Quantum Sci. Technol. 2 (2017)035001 (https://doi.org/10.1088/2058-9565/aa78d4)
2. Tambasco et al., Vol. 24, No. 17 | 22 Aug 2016 | OPTICS EXPRESS 19616 (https://opg.optica.org/oe/fulltext.cfm?uri=oe-24-17-19616&id=348856)

