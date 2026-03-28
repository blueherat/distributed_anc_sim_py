from __future__ import annotations

from dataclasses import asdict
from typing import Tuple

import numpy as np

from ..acoustics import RIRManager
from .config import NodeRadialLayout, RoomConfig, ScenarioConfig


def _clip_position(position: np.ndarray, room_size: np.ndarray, margin: float) -> np.ndarray:
    clipped = np.asarray(position, dtype=float).copy()
    clipped[0] = np.clip(clipped[0], margin, room_size[0] - margin)
    clipped[1] = np.clip(clipped[1], margin, room_size[1] - margin)
    clipped[2] = np.clip(clipped[2], margin, room_size[2] - margin)
    return clipped


def _min_pairwise_distance(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return float("inf")

    min_dist = float("inf")
    for i in range(points.shape[0] - 1):
        dist = np.linalg.norm(points[i + 1 :] - points[i], axis=1)
        if dist.size:
            min_dist = min(min_dist, float(np.min(dist)))
    return min_dist


def validate_scenario_config(config: ScenarioConfig) -> None:
    if not config.node_layouts:
        raise ValueError("ScenarioConfig.node_layouts cannot be empty.")

    if len(config.primary_source_ids) == 0:
        raise ValueError("ScenarioConfig.primary_source_ids cannot be empty.")

    for i, layout in enumerate(config.node_layouts):
        if not (layout.ref_radius < layout.sec_radius < layout.err_radius):
            raise ValueError(
                f"Node {i + 1}: expected ref_radius < sec_radius < err_radius, got "
                f"{layout.ref_radius}, {layout.sec_radius}, {layout.err_radius}."
            )


def build_manager_from_config(
    config: ScenarioConfig,
) -> Tuple[RIRManager, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build one RIRManager from a high-level ScenarioConfig."""

    validate_scenario_config(config)

    mgr = RIRManager()
    mgr.room = np.asarray(config.room.size, dtype=float)
    mgr.fs = int(config.room.fs)
    mgr.sound_speed = float(config.room.sound_speed)
    mgr.image_source_order = int(config.room.image_source_order)
    mgr.material_absorption = float(config.room.material_absorption)
    mgr.compensate_fractional_delay = bool(config.room.compensate_fractional_delay)
    mgr.fractional_delay_shift = (
        None if config.room.fractional_delay_shift is None else int(config.room.fractional_delay_shift)
    )

    room_size = mgr.room
    margin = float(config.boundary_margin)

    source_position = _clip_position(np.asarray(config.source_position, dtype=float), room_size, margin)

    source_ids = []
    for i, source_id in enumerate(config.primary_source_ids):
        # If multiple primary sources are used, place them near the center source point.
        offset = np.array([0.05 * i, -0.04 * i, 0.0], dtype=float)
        pos = _clip_position(source_position + offset, room_size, margin)
        mgr.add_primary_speaker(int(source_id), pos)
        source_ids.append(int(source_id))

    ref_ids = []
    sec_ids = []
    err_ids = []

    for i, layout in enumerate(config.node_layouts):
        theta = np.deg2rad(layout.azimuth_deg)
        direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)

        ref_pos = source_position + direction * float(layout.ref_radius) + np.array([0.0, 0.0, layout.z_offset], dtype=float)
        sec_pos = source_position + direction * float(layout.sec_radius) + np.array([0.0, 0.0, layout.z_offset], dtype=float)
        err_pos = source_position + direction * float(layout.err_radius) + np.array([0.0, 0.0, layout.z_offset], dtype=float)

        ref_pos = _clip_position(ref_pos, room_size, margin)
        sec_pos = _clip_position(sec_pos, room_size, margin)
        err_pos = _clip_position(err_pos, room_size, margin)

        ref_id = int(config.reference_start_id + i)
        sec_id = int(config.secondary_start_id + i)
        err_id = int(config.error_start_id + i)

        mgr.add_reference_microphone(ref_id, ref_pos)
        mgr.add_secondary_speaker(sec_id, sec_pos)
        mgr.add_error_microphone(err_id, err_pos)

        ref_ids.append(ref_id)
        sec_ids.append(sec_id)
        err_ids.append(err_id)

    return (
        mgr,
        np.asarray(source_ids, dtype=int),
        np.asarray(ref_ids, dtype=int),
        np.asarray(sec_ids, dtype=int),
        np.asarray(err_ids, dtype=int),
    )


def sample_asymmetric_scenario(
    seed: int = 42,
    num_nodes: int = 4,
    room: RoomConfig | None = None,
    boundary_margin: float = 0.15,
    min_inter_node_distance: float = 0.35,
    max_sampling_attempts: int = 250,
) -> ScenarioConfig:
    """Create one random but physically reasonable non-symmetric scenario."""

    if room is None:
        room = RoomConfig()

    rng = np.random.default_rng(seed)

    room_size = np.asarray(room.size, dtype=float)
    source_margin = 1.2
    source_position = np.array(
        [
            rng.uniform(source_margin, room_size[0] - source_margin),
            rng.uniform(source_margin, room_size[1] - source_margin),
            rng.uniform(1.1, max(1.15, room_size[2] - 0.8)),
        ],
        dtype=float,
    )

    for _ in range(max_sampling_attempts):
        # Start from spread angles and add jitter so the layout is intentionally non-symmetric.
        base = np.linspace(20.0, 340.0, num_nodes)
        jitter = rng.uniform(-28.0, 28.0, size=num_nodes)
        azimuths = (base + jitter) % 360.0

        node_layouts = []
        ref_positions = []
        sec_positions = []
        err_positions = []

        for az in azimuths:
            ref_radius = rng.uniform(0.35, 0.60)
            sec_radius = ref_radius + rng.uniform(0.15, 0.30)
            err_radius = sec_radius + rng.uniform(0.12, 0.30)
            z_offset = rng.uniform(-0.08, 0.10)

            layout = NodeRadialLayout(
                azimuth_deg=float(az),
                ref_radius=float(ref_radius),
                sec_radius=float(sec_radius),
                err_radius=float(err_radius),
                z_offset=float(z_offset),
            )
            node_layouts.append(layout)

            theta = np.deg2rad(layout.azimuth_deg)
            direction = np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float)
            z_vec = np.array([0.0, 0.0, layout.z_offset], dtype=float)

            ref_positions.append(_clip_position(source_position + direction * layout.ref_radius + z_vec, room_size, boundary_margin))
            sec_positions.append(_clip_position(source_position + direction * layout.sec_radius + z_vec, room_size, boundary_margin))
            err_positions.append(_clip_position(source_position + direction * layout.err_radius + z_vec, room_size, boundary_margin))

        if min_inter_node_distance <= 0:
            return ScenarioConfig(
                room=room,
                source_position=tuple(source_position.tolist()),
                node_layouts=node_layouts,
                boundary_margin=float(boundary_margin),
            )

        min_ref = _min_pairwise_distance(np.vstack(ref_positions))
        min_sec = _min_pairwise_distance(np.vstack(sec_positions))
        min_err = _min_pairwise_distance(np.vstack(err_positions))

        if min(min_ref, min_sec, min_err) >= float(min_inter_node_distance):
            return ScenarioConfig(
                room=room,
                source_position=tuple(source_position.tolist()),
                node_layouts=node_layouts,
                boundary_margin=float(boundary_margin),
            )

    raise RuntimeError(
        "Failed to sample scenario with required inter-node spacing. "
        f"Try lowering min_inter_node_distance (current={min_inter_node_distance}) "
        f"or reducing num_nodes (current={num_nodes})."
    )


def plot_layout_with_labels(
    mgr: RIRManager,
    source_ids: np.ndarray,
    ref_ids: np.ndarray,
    sec_ids: np.ndarray,
    err_ids: np.ndarray,
    title: str = "Scenario Layout Preview (Top View)",
    save_path: str | None = None,
):
    """Plot 2D room frame and annotate all device IDs for visual inspection."""

    import matplotlib.pyplot as plt

    ax = mgr.plot_layout_2d()
    ax.set_title(title)

    def _annotate(ids: np.ndarray, table: dict[int, np.ndarray], prefix: str, color: str) -> None:
        for did in ids:
            pos = np.asarray(table[int(did)], dtype=float)
            ax.text(pos[0], pos[1], f"{prefix}{int(did)}", color=color, fontsize=8)

    _annotate(source_ids, mgr.primary_speakers, "P", "darkred")
    _annotate(ref_ids, mgr.reference_microphones, "R", "purple")
    _annotate(sec_ids, mgr.secondary_speakers, "S", "darkgreen")
    _annotate(err_ids, mgr.error_microphones, "E", "navy")

    if len(source_ids) > 0:
        source_pos = np.asarray(mgr.primary_speakers[int(source_ids[0])], dtype=float)
        for i in range(min(len(ref_ids), len(sec_ids), len(err_ids))):
            ref_pos = np.asarray(mgr.reference_microphones[int(ref_ids[i])], dtype=float)
            sec_pos = np.asarray(mgr.secondary_speakers[int(sec_ids[i])], dtype=float)
            err_pos = np.asarray(mgr.error_microphones[int(err_ids[i])], dtype=float)
            ax.plot([source_pos[0], ref_pos[0]], [source_pos[1], ref_pos[1]], color="purple", alpha=0.35, linewidth=1.0)
            ax.plot([source_pos[0], sec_pos[0]], [source_pos[1], sec_pos[1]], color="darkgreen", alpha=0.35, linewidth=1.0)
            ax.plot([source_pos[0], err_pos[0]], [source_pos[1], err_pos[1]], color="navy", alpha=0.35, linewidth=1.0)

    fig = ax.figure
    if save_path:
        fig.savefig(save_path, dpi=140, bbox_inches="tight")

    return fig, ax


def scenario_to_dict(config: ScenarioConfig) -> dict:
    return asdict(config)
