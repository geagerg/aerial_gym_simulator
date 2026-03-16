from dataclasses import dataclass
from typing import Tuple


RECORD_TARGET_XYZ: Tuple[float, float, float] = (0.0, 0.8, 0.0)


@dataclass(frozen=True)
class ThirdPersonPreset:
    width: int = 480
    height: int = 270
    min_range: float = 0.2
    max_range: float = 30.0
    translation_xyz: Tuple[float, float, float] = (-1.5, 0.0, 0.25)
    euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)


THIRD_PERSON_PRESET = ThirdPersonPreset()
