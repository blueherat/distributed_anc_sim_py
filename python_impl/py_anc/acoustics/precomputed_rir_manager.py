from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np


RIRKey = Tuple[int, int]


@dataclass
class PrecomputedRIRManager:
    """Lightweight manager for running ANC algorithms on precomputed paths."""

    reference_microphones: Dict[int, np.ndarray] = field(default_factory=dict)
    secondary_speakers: Dict[int, np.ndarray] = field(default_factory=dict)
    error_microphones: Dict[int, np.ndarray] = field(default_factory=dict)
    _secondary_rirs: Dict[RIRKey, np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_padded_arrays(
        cls,
        ref_ids: Iterable[int],
        sec_ids: Iterable[int],
        err_ids: Iterable[int],
        sec_rirs: np.ndarray,
        sec_rir_lengths: np.ndarray,
    ) -> "PrecomputedRIRManager":
        manager = cls()

        ref_ids = [int(v) for v in np.asarray(ref_ids).reshape(-1)]
        sec_ids = [int(v) for v in np.asarray(sec_ids).reshape(-1)]
        err_ids = [int(v) for v in np.asarray(err_ids).reshape(-1)]

        sec_rirs = np.asarray(sec_rirs, dtype=float)
        sec_rir_lengths = np.asarray(sec_rir_lengths, dtype=int)

        if sec_rirs.ndim != 3:
            raise ValueError(f"sec_rirs must be 3D, got shape {sec_rirs.shape}.")
        if sec_rir_lengths.shape != sec_rirs.shape[:2]:
            raise ValueError(
                "sec_rir_lengths shape must match first two dimensions of sec_rirs, "
                f"got {sec_rir_lengths.shape} vs {sec_rirs.shape[:2]}."
            )
        if sec_rirs.shape[0] != len(sec_ids) or sec_rirs.shape[1] != len(err_ids):
            raise ValueError(
                "sec_rirs dimensions must match sec_ids/err_ids lengths, "
                f"got sec_rirs={sec_rirs.shape}, sec_ids={len(sec_ids)}, err_ids={len(err_ids)}."
            )

        # Positions are not used by algorithms in strict mode; placeholders keep API compatibility.
        zero = np.zeros(3, dtype=float)
        manager.reference_microphones = {rid: zero.copy() for rid in ref_ids}
        manager.secondary_speakers = {sid: zero.copy() for sid in sec_ids}
        manager.error_microphones = {eid: zero.copy() for eid in err_ids}

        for i, sid in enumerate(sec_ids):
            for j, eid in enumerate(err_ids):
                n_taps = int(sec_rir_lengths[i, j])
                if n_taps <= 0:
                    manager._secondary_rirs[(sid, eid)] = np.zeros(1, dtype=float)
                    continue
                manager._secondary_rirs[(sid, eid)] = sec_rirs[i, j, :n_taps].astype(float, copy=True)

        return manager

    def get_secondary_rir(self, spk_id: int, mic_id: int) -> np.ndarray:
        key = (int(spk_id), int(mic_id))
        if key not in self._secondary_rirs:
            raise KeyError(f"Secondary RIR for path {key} does not exist.")
        return self._secondary_rirs[key]
