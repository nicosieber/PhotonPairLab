from dataclasses import dataclass
from typing import Any, Mapping, Optional

@dataclass(frozen=True)
class Section:
    sources: list[str]
    data: Any

@dataclass(frozen=True)
class MaterialData:
    name: str
    biaxial: bool
    sellmeier: Section
    temperature_corrections: Optional[Section]
    thermal_expansion: Optional[Section]


