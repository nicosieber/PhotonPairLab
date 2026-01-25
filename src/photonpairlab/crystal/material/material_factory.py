
from typing import Type

from .material_loader import load_material_data
from .model.base_material_model import BaseMaterialModel

from .model import (
    GeneralSellmeierThermalModel,
    SellmeierLinearThermalModel,
    KatoTakaokaSellmeierThermalModel,
)


QPM_MODEL_MAPPER: dict[str, Type[BaseMaterialModel]] = {
    "ktp1": GeneralSellmeierThermalModel,
    "ktp2": SellmeierLinearThermalModel,
    "ktp3": KatoTakaokaSellmeierThermalModel,
}


class MaterialFactory:
    @staticmethod
    def create(name: str) -> BaseMaterialModel:
        material_data = load_material_data(name)

        try:
            model_cls = QPM_MODEL_MAPPER[name]
        except KeyError as e:
            raise ValueError(
                f"No model registered for material '{name}'. Add it to QPM_MODEL_MAPPER."
            ) from e

        return model_cls(material_data)