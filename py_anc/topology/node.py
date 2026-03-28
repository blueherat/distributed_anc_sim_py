from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    """Physical node mapping one reference mic, one secondary speaker and one error mic."""

    node_id: int
    ref_mic_id: int | None = None
    sec_spk_id: int | None = None
    err_mic_id: int | None = None
    neighbor_ids: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.neighbor_ids:
            self.neighbor_ids = [self.node_id]

    def add_ref_mic(self, mic_id: int) -> None:
        self.ref_mic_id = mic_id

    def add_sec_spk(self, spk_id: int) -> None:
        self.sec_spk_id = spk_id

    def add_err_mic(self, mic_id: int) -> None:
        self.err_mic_id = mic_id
