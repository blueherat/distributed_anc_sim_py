from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from .rir_manager import RIRManager


@dataclass
class CircularArray:
    radius: float
    num_elements: int
    center: np.ndarray
    start_id: int

    element_angles: np.ndarray = field(init=False)
    element_positions: np.ndarray = field(init=False)
    element_ids: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.center = np.asarray(self.center, dtype=float)
        self.compute_geometry()

    def compute_geometry(self) -> None:
        self.element_angles = 2.0 * np.pi * np.arange(self.num_elements, dtype=float) / float(self.num_elements)
        self.element_ids = self.start_id + np.arange(self.num_elements, dtype=int)

        x = self.center[0] + self.radius * np.cos(self.element_angles)
        y = self.center[1] + self.radius * np.sin(self.element_angles)
        z = np.full(self.num_elements, self.center[2], dtype=float)
        self.element_positions = np.column_stack([x, y, z])

    def register_to_manager(self, mgr: RIRManager, device_type: Literal["mic", "secondary", "reference"]) -> None:
        if device_type == "mic":
            for i in range(self.num_elements):
                mgr.add_error_microphone(int(self.element_ids[i]), self.element_positions[i])
        elif device_type == "secondary":
            for i in range(self.num_elements):
                mgr.add_secondary_speaker(int(self.element_ids[i]), self.element_positions[i])
        elif device_type == "reference":
            for i in range(self.num_elements):
                mgr.add_reference_microphone(int(self.element_ids[i]), self.element_positions[i])
        else:
            raise ValueError(f"Unsupported device_type: {device_type}")
