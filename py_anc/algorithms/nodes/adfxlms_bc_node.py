from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ...topology import Node


@dataclass
class ADFxLMSBCNode(Node):
    step_size: float = 0.01
    phi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    psi: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))
    xf_taps: np.ndarray = field(default_factory=lambda: np.zeros((0, 0), dtype=float))

    def init(self, filter_length: int) -> None:
        num_neighbors = len(self.neighbor_ids)
        self.phi = np.zeros((filter_length, num_neighbors), dtype=float)
        self.psi = self.phi.copy()
        self.xf_taps = self.phi.copy()
