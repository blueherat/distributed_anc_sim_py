from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class RoomConfig:
    """Acoustic room and simulation parameters."""

    size: Tuple[float, float, float] = (5.0, 5.0, 5.0)
    fs: int = 4000
    sound_speed: float = 343.0
    image_source_order: int = 2
    material_absorption: float = 0.5
    compensate_fractional_delay: bool = True
    fractional_delay_shift: int | None = None


@dataclass
class NodeRadialLayout:
    """One node layout around the primary source in polar form."""

    azimuth_deg: float
    ref_radius: float
    sec_radius: float
    err_radius: float
    z_offset: float = 0.0


@dataclass
class ScenarioConfig:
    """Full scenario definition used to build one simulation environment."""

    room: RoomConfig = field(default_factory=RoomConfig)
    source_position: Tuple[float, float, float] = (2.2, 2.1, 1.6)
    primary_source_ids: Tuple[int, ...] = (101,)
    node_layouts: List[NodeRadialLayout] = field(
        default_factory=lambda: [
            NodeRadialLayout(azimuth_deg=18.0, ref_radius=0.42, sec_radius=0.66, err_radius=0.93, z_offset=0.03),
            NodeRadialLayout(azimuth_deg=127.0, ref_radius=0.50, sec_radius=0.78, err_radius=1.06, z_offset=-0.02),
            NodeRadialLayout(azimuth_deg=211.0, ref_radius=0.46, sec_radius=0.74, err_radius=1.01, z_offset=0.01),
            NodeRadialLayout(azimuth_deg=323.0, ref_radius=0.39, sec_radius=0.63, err_radius=0.91, z_offset=0.04),
        ]
    )
    reference_start_id: int = 401
    secondary_start_id: int = 201
    error_start_id: int = 301
    boundary_margin: float = 0.15
