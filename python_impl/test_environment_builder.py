from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.scenarios import RoomConfig, build_manager_from_config, plot_layout_with_labels, sample_asymmetric_scenario, scenario_to_dict
from py_anc.utils import wn_gen


if __name__ == "__main__":
    # =========================
    # User configuration block
    # =========================
    num_scenarios = 3
    base_seed = 2026

    room_cfg = RoomConfig(
        size=(6.0, 5.5, 3.2),
        fs=4000,
        sound_speed=343.0,
        image_source_order=2,
        material_absorption=0.45,
    )

    # quick synthetic check only, not data generation
    test_duration_s = 2.0
    f_low = 100.0
    f_high = 1500.0

    output_dir = ROOT_DIR / "python_scripts"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_scenarios = []

    for i in range(num_scenarios):
        scenario = sample_asymmetric_scenario(seed=base_seed + i, num_nodes=4, room=room_cfg)
        mgr, source_ids, ref_ids, sec_ids, err_ids = build_manager_from_config(scenario)

        print(f"Scenario {i + 1}: source={scenario.source_position}, nodes={len(scenario.node_layouts)}")
        mgr.build(verbose=False)

        # quick validity check for multi-reference signal synthesis
        source_cols = []
        time_axis = None
        for j in range(len(source_ids)):
            noise_col, t = wn_gen(mgr.fs, test_duration_s, f_low, f_high, rng=np.random.default_rng(base_seed + i * 17 + j))
            source_cols.append(noise_col[:, 0])
            if time_axis is None:
                time_axis = t

        source_signal = np.column_stack(source_cols)
        d = mgr.calculate_desired_signal(source_signal, len(time_axis))
        x = mgr.calculate_reference_signal(source_signal, len(time_axis))

        print(
            f"  desired shape={d.shape}, reference shape={x.shape}, "
            f"ref mics={len(ref_ids)}, sec spks={len(sec_ids)}, err mics={len(err_ids)}"
        )

        if i == 0:
            preview_path = output_dir / "layout_preview_environment_builder.png"
            fig, _ = plot_layout_with_labels(
                mgr,
                source_ids=source_ids,
                ref_ids=ref_ids,
                sec_ids=sec_ids,
                err_ids=err_ids,
                title="Environment Builder Layout Preview (Scenario 1)",
                save_path=str(preview_path),
            )
            plt.close(fig)
            print(f"  layout preview saved: {preview_path}")

        all_scenarios.append(scenario_to_dict(scenario))

    scenarios_path = output_dir / "scenario_preview_configs.json"
    scenarios_path.write_text(json.dumps(all_scenarios, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Scenario config preview saved: {scenarios_path}")
    print("Environment builder test finished.")
