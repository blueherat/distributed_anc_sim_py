from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Tuple

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve


RIRKey = Tuple[int, int]


@dataclass
class RIRManager:
    room: np.ndarray = field(default_factory=lambda: np.array([7.0, 5.0, 3.0], dtype=float))
    fs: int = 48_000
    sound_speed: float = 343.0
    image_source_order: int = 3
    material_absorption: float = 0.5
    compensate_fractional_delay: bool = True
    fractional_delay_shift: int | None = None

    primary_speakers: Dict[int, np.ndarray] = field(default_factory=dict)
    reference_microphones: Dict[int, np.ndarray] = field(default_factory=dict)
    secondary_speakers: Dict[int, np.ndarray] = field(default_factory=dict)
    error_microphones: Dict[int, np.ndarray] = field(default_factory=dict)

    primary_rirs: Dict[RIRKey, np.ndarray] = field(default_factory=dict)
    reference_rirs: Dict[RIRKey, np.ndarray] = field(default_factory=dict)
    secondary_rirs: Dict[RIRKey, np.ndarray] = field(default_factory=dict)

    def add_primary_speaker(self, spk_id: int, position: Iterable[float]) -> None:
        self.primary_speakers[int(spk_id)] = np.asarray(position, dtype=float)

    def add_reference_microphone(self, mic_id: int, position: Iterable[float]) -> None:
        self.reference_microphones[int(mic_id)] = np.asarray(position, dtype=float)

    def add_secondary_speaker(self, spk_id: int, position: Iterable[float]) -> None:
        self.secondary_speakers[int(spk_id)] = np.asarray(position, dtype=float)

    def add_error_microphone(self, mic_id: int, position: Iterable[float]) -> None:
        self.error_microphones[int(mic_id)] = np.asarray(position, dtype=float)

    def _create_room(self) -> pra.ShoeBox:
        try:
            pra.constants.set("c", float(self.sound_speed))
        except Exception:
            pass

        return pra.ShoeBox(
            p=self.room.astype(float),
            fs=int(self.fs),
            max_order=int(self.image_source_order),
            absorption=float(self.material_absorption),
        )

    def _compute_rir(self, tx_pos: np.ndarray, rx_positions: np.ndarray) -> np.ndarray:
        if rx_positions.size == 0:
            return np.zeros((0, 0), dtype=float)

        room = self._create_room()
        room.add_source(np.asarray(tx_pos, dtype=float))

        mic_positions = np.asarray(rx_positions, dtype=float).T
        room.add_microphone_array(pra.MicrophoneArray(mic_positions, fs=self.fs))
        room.compute_rir()

        rirs = [np.asarray(room.rir[i][0], dtype=float) for i in range(mic_positions.shape[1])]
        max_len = max(len(rir) for rir in rirs)

        rir_mat = np.zeros((len(rirs), max_len), dtype=float)
        for i, rir in enumerate(rirs):
            rir_mat[i, : len(rir)] = rir
        return self._apply_fractional_delay_compensation(rir_mat)

    def _resolve_fractional_delay_shift(self) -> int:
        if self.fractional_delay_shift is not None:
            return max(int(self.fractional_delay_shift), 0)

        try:
            frac_len = int(pra.constants.get("frac_delay_length"))
        except Exception:
            frac_len = 81

        return max(frac_len // 2, 0)

    def _apply_fractional_delay_compensation(self, rir_mat: np.ndarray) -> np.ndarray:
        if not self.compensate_fractional_delay or rir_mat.size == 0:
            return rir_mat

        n_shift = self._resolve_fractional_delay_shift()
        if n_shift <= 0:
            return rir_mat

        if n_shift >= rir_mat.shape[1]:
            return np.zeros_like(rir_mat)

        compensated = np.zeros_like(rir_mat)
        compensated[:, : rir_mat.shape[1] - n_shift] = rir_mat[:, n_shift:]
        return compensated

    def build(self, verbose: bool = True) -> None:
        if not self.primary_speakers:
            raise ValueError("No primary speakers have been added.")
        if not self.secondary_speakers:
            raise ValueError("No secondary speakers have been added.")
        if not self.error_microphones:
            raise ValueError("No error microphones have been added.")

        err_ids = list(self.error_microphones.keys())
        err_positions = np.vstack([self.error_microphones[i] for i in err_ids])

        ref_ids = list(self.reference_microphones.keys())
        ref_positions = (
            np.vstack([self.reference_microphones[i] for i in ref_ids])
            if ref_ids
            else np.zeros((0, 3), dtype=float)
        )

        self.primary_rirs.clear()
        self.reference_rirs.clear()
        self.secondary_rirs.clear()

        for spk_id, spk_pos in self.primary_speakers.items():
            ir_to_err = self._compute_rir(spk_pos, err_positions)
            for i, mic_id in enumerate(err_ids):
                self.primary_rirs[(spk_id, mic_id)] = ir_to_err[i, :].copy()
            if verbose:
                print(f"Primary paths for Speaker {spk_id} computed. RIR length: {ir_to_err.shape[1]}")

            if ref_ids:
                ir_to_ref = self._compute_rir(spk_pos, ref_positions)
                for i, ref_id in enumerate(ref_ids):
                    self.reference_rirs[(spk_id, ref_id)] = ir_to_ref[i, :].copy()
                if verbose:
                    print(f"Reference paths for Speaker {spk_id} computed. RIR length: {ir_to_ref.shape[1]}")

        for spk_id, spk_pos in self.secondary_speakers.items():
            ir_to_err = self._compute_rir(spk_pos, err_positions)
            for i, mic_id in enumerate(err_ids):
                self.secondary_rirs[(spk_id, mic_id)] = ir_to_err[i, :].copy()
            if verbose:
                print(f"Secondary paths for Speaker {spk_id} computed. RIR length: {ir_to_err.shape[1]}")

    def get_primary_rir(self, spk_id: int, mic_id: int) -> np.ndarray:
        key = (int(spk_id), int(mic_id))
        if key not in self.primary_rirs:
            raise KeyError(f"Primary RIR for path {key} does not exist.")
        return self.primary_rirs[key]

    def get_reference_rir(self, spk_id: int, ref_mic_id: int) -> np.ndarray:
        key = (int(spk_id), int(ref_mic_id))
        if key not in self.reference_rirs:
            raise KeyError(f"Reference RIR for path {key} does not exist.")
        return self.reference_rirs[key]

    def get_secondary_rir(self, spk_id: int, mic_id: int) -> np.ndarray:
        key = (int(spk_id), int(mic_id))
        if key not in self.secondary_rirs:
            raise KeyError(f"Secondary RIR for path {key} does not exist.")
        return self.secondary_rirs[key]

    def calculate_desired_signal(self, source_signal: np.ndarray, n_samples: int) -> np.ndarray:
        key_pri = list(self.primary_speakers.keys())
        key_err = list(self.error_microphones.keys())

        if source_signal.shape[1] != len(key_pri):
            raise ValueError(
                f"source_signal columns ({source_signal.shape[1]}) must match number of primary speakers ({len(key_pri)})."
            )

        d = np.zeros((n_samples, len(key_err)), dtype=float)
        for m, err_id in enumerate(key_err):
            for j, spk_id in enumerate(key_pri):
                p = self.get_primary_rir(spk_id, err_id)
                d_jm = fftconvolve(source_signal[:, j], p)
                d[:, m] += d_jm[:n_samples]
        return d

    def calculate_reference_signal(self, source_signal: np.ndarray, n_samples: int) -> np.ndarray:
        key_pri = list(self.primary_speakers.keys())
        key_ref = list(self.reference_microphones.keys())

        if not key_ref:
            raise ValueError("No reference microphones have been added.")

        if source_signal.shape[1] != len(key_pri):
            raise ValueError(
                f"source_signal columns ({source_signal.shape[1]}) must match number of primary speakers ({len(key_pri)})."
            )

        x_ref = np.zeros((n_samples, len(key_ref)), dtype=float)
        for r, ref_id in enumerate(key_ref):
            for j, spk_id in enumerate(key_pri):
                p_ref = self.get_reference_rir(spk_id, ref_id)
                x_jr = fftconvolve(source_signal[:, j], p_ref)
                x_ref[:, r] += x_jr[:n_samples]
        return x_ref

    def plot_layout(self, ax=None):
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(8, 6))
            ax = fig.add_subplot(111, projection="3d")

        ax.set_title("Room Layout Configuration")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")

        room_x, room_y, room_z = self.room
        corners = np.array(
            [
                [0, 0, 0],
                [room_x, 0, 0],
                [room_x, room_y, 0],
                [0, room_y, 0],
                [0, 0, room_z],
                [room_x, 0, room_z],
                [room_x, room_y, room_z],
                [0, room_y, room_z],
            ],
            dtype=float,
        )
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        for a, b in edges:
            ax.plot(
                [corners[a, 0], corners[b, 0]],
                [corners[a, 1], corners[b, 1]],
                [corners[a, 2], corners[b, 2]],
                "k--",
                linewidth=1.0,
            )

        self._plot_devices(ax, self.primary_speakers, "r", "s", "Primary Source")
        self._plot_devices(ax, self.secondary_speakers, "g", "D", "Secondary Source")
        self._plot_devices(ax, self.reference_microphones, "m", "^", "Reference Mic")
        self._plot_devices(ax, self.error_microphones, "b", "o", "Error Mic")

        ax.set_xlim([-0.5, room_x + 0.5])
        ax.set_ylim([-0.5, room_y + 0.5])
        ax.set_zlim([0.0, room_z + 0.5])
        ax.legend(loc="best")
        return ax

    def plot_layout_2d(self, ax=None):
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))

        ax.set_title("Room Layout Configuration (Top View)")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")

        room_x, room_y, _ = self.room
        room_rect = Rectangle((0.0, 0.0), float(room_x), float(room_y), fill=False, linestyle="--", linewidth=1.0, edgecolor="k", label="Room Boundary")
        ax.add_patch(room_rect)

        self._plot_devices_2d(ax, self.primary_speakers, "r", "s", "Primary Source")
        self._plot_devices_2d(ax, self.secondary_speakers, "g", "D", "Secondary Source")
        self._plot_devices_2d(ax, self.reference_microphones, "m", "^", "Reference Mic")
        self._plot_devices_2d(ax, self.error_microphones, "b", "o", "Error Mic")

        ax.set_xlim([-0.5, float(room_x) + 0.5])
        ax.set_ylim([-0.5, float(room_y) + 0.5])
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        return ax

    @staticmethod
    def _plot_devices(ax, device_dict: Dict[int, np.ndarray], color: str, marker: str, label: str) -> None:
        if not device_dict:
            return
        ids = list(device_dict.keys())
        positions = np.vstack([device_dict[i] for i in ids])
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], color=color, marker=marker, label=label)

    @staticmethod
    def _plot_devices_2d(ax, device_dict: Dict[int, np.ndarray], color: str, marker: str, label: str) -> None:
        if not device_dict:
            return
        ids = list(device_dict.keys())
        positions = np.vstack([device_dict[i] for i in ids])
        ax.scatter(positions[:, 0], positions[:, 1], color=color, marker=marker, label=label)
