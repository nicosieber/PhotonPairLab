from dataclasses import dataclass

@dataclass
class CrystalConfig:
    material_name: str = "ktp1"
    crystal_length: float = 30e-3
    coherence_length: float = 46.22e-6
    domain_width: float = 18e-6
    temperature: float = 20
    pm_strategy: str = "quasi"
    spdc_type: str = "type-II"