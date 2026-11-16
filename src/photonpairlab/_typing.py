"""Shared type aliases for PhotonPairLab."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
ComplexArray = NDArray[np.complex128]

#: Which photon of a pair: signal (JSA axis 1) or idler (JSA axis 0).
Mode = Literal["signal", "idler"]

#: Which HOM experiment a result describes (see ``HOMAnalyzer``).
HOMModel = Literal["same_pair", "independent"]
