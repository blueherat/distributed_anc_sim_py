from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...topology import Node


@dataclass
class DiffFxLMSNode(Node):
    step_size: float = 0.01
    w: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    psi: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    xf_taps: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))

    def init(self, filter_length: int) -> None:
        self.w = np.zeros(filter_length, dtype=float)
        self.psi = self.w.copy()
        self.xf_taps = self.w.copy()
