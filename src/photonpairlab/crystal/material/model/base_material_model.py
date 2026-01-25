from __future__ import annotations

from ..base_material import BaseMaterial
from ..material_data import MaterialData


class BaseMaterialModel(BaseMaterial):
    """
    Base class for all material models that are backed by MaterialData from JSON.
    This is purely to give type checkers a consistent __init__ signature.
    """
    def __init__(self, material: MaterialData) -> None:
        self.material = material
