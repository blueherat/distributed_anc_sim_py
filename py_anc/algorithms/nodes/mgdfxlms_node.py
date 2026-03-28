from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from ...topology import Node


@dataclass
class MGDFxLMSNode(Node):
    step_size: float = 0.01
    lc: int = 32
    w: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    xf_taps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    gradient: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    direction: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    ckm_taps: Dict[int, np.ndarray] = field(default_factory=dict)

    def init(self, filter_length: int) -> None:
        self.w = np.zeros(filter_length, dtype=float)
        self.xf_taps = self.w.copy()
        self.gradient = self.w.copy()
        self.direction = self.w.copy()
        self.ckm_taps = {}
