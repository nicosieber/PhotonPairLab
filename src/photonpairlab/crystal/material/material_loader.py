from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from photonpairlab.config import RESOURCES_DIR
from .material_data import MaterialData, Section


_MATERIALS_PATH = Path(RESOURCES_DIR) / "materials.json"


def _parse_section(raw: Mapping[str, Any] | None) -> Section | None:
    if raw is None:
        return None
    return Section(
        sources=list(raw["sources"]),
        data=raw["data"],
    )


@lru_cache
def _load_all_raw() -> dict[str, Any]:
    with open(_MATERIALS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("materials.json must contain a JSON object at the top level.")
    return data


def load_material_data(name: str) -> MaterialData:
    raw_all = _load_all_raw()

    try:
        raw = raw_all[name]
    except KeyError as e:
        available = ", ".join(sorted(raw_all.keys()))
        raise KeyError(f"Unknown material '{name}'. Available: {available}") from e

    # sellmeier is required
    sellmeier_raw = raw["sellmeier"]

    return MaterialData(
        name=name,
        biaxial=bool(raw["biaxial"]),
        sellmeier=Section(
            sources=list(sellmeier_raw["sources"]),
            data=sellmeier_raw["data"],
        ),
        temperature_corrections=_parse_section(raw.get("temperature_corrections")),
        thermal_expansion=_parse_section(raw.get("thermal_expansion")),
    )
