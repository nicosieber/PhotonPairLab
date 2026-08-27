from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Section:
    sources: list[str]
    data: Any

@dataclass(frozen=True)
class MaterialData:
    name: str
    biaxial: bool
    sellmeier: Section
    temperature_corrections: Section | None
    thermal_expansion: Section | None


