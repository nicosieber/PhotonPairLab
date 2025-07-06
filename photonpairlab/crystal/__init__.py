from .crystal import Crystal

# Optionally, expose materials and strategies for convenience
from .material.bbo import BBO
from .material.bibo import BIBO
from .material.ktp1 import KTP1
from .material.ktp2 import KTP2
from .material.ktp3 import KTP3
from .material.base_material import BaseMaterial
from .phasematching.apm_strategy import APMPhaseMatching

__all__ = [
    "Crystal",
    "BBO",
    "BIBO",
    "KTP1",
    "KTP2",
    "KTP3",
    "APMPhaseMatching",
]