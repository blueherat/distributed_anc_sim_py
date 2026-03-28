from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.algorithms import adfxlms, adfxlms_bc, cdfxlms, cfxlms, dcfxlms, diff_fxlms, mgdfxlms
from py_anc.algorithms.nodes import (
    ADFxLMSBCNode,
    ADFxLMSNode,
    CDFxLMSNode,
    DCFxLMSNode,
    DiffFxLMSNode,
    MGDFxLMSNode,
)
from py_anc.scenarios import NodeRadialLayout, RoomConfig, ScenarioConfig, build_manager_from_config, plot_layout_with_labels, sample_asymmetric_scenario
from py_anc.topology import Network
from py_anc.utils import wn_gen
from py_anc.viz import plot_results, plot_tap_weights


ALGORITHM_MAP = {
    "CFxLMS": cfxlms,
    "ADFxLMS": adfxlms,
    "ADFxLMS-BC": adfxlms_bc,
    "Diff-FxLMS": diff_fxlms,
    "DCFxLMS": dcfxlms,
    "CDFxLMS": cdfxlms,
    "MGDFxLMS": mgdfxlms,
}


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x / denom


def _create_network(node_cls, mu: float, ref_ids: np.ndarray, sec_ids: np.ndarray, err_ids: np.ndarray, lc: int | None = None) -> Network:
    net = Network()
    for idx in range(len(ref_ids)):
        if lc is None:
            node = node_cls(node_id=idx + 1, step_size=mu)
        else:
            node = node_cls(node_id=idx + 1, step_size=mu, lc=lc)
        node.add_ref_mic(int(ref_ids[idx]))
        node.add_sec_spk(int(sec_ids[idx]))
        node.add_err_mic(int(err_ids[idx]))
        net.add_node(node)
    return net


def _connect_for_algorithm(net: Network, alg_name: str) -> None:
    if alg_name == "ADFxLMS":
        net.connect_nodes(1, 2)
        net.connect_nodes(1, 4)
        net.connect_nodes(2, 3)
        net.connect_nodes(2, 4)
        net.connect_nodes(3, 4)
    elif alg_name in {"ADFxLMS-BC", "Diff-FxLMS"}:
        net.connect_nodes(1, 3)
        net.connect_nodes(1, 4)
        net.connect_nodes(2, 3)
        net.connect_nodes(2, 4)
    elif alg_name in {"CDFxLMS", "MGDFxLMS"}:
        net.connect_nodes(1, 3)
        net.connect_nodes(1, 4)
        net.connect_nodes(2, 3)
        net.connect_nodes(2, 4)
        net.connect_nodes(1, 2)
        net.connect_nodes(3, 4)


def _run_algorithm(
    alg_name: str,
    time_axis: np.ndarray,
    mgr,
    x: np.ndarray,
    d: np.ndarray,
    ref_ids: np.ndarray,
    sec_ids: np.ndarray,
    err_ids: np.ndarray,
    l: int,
    mu: float,
):
    params = {
        "time": time_axis,
        "rir_manager": mgr,
        "L": l,
        "reference_signal": x,
        "desired_signal": d,
    }

    if alg_name == "CFxLMS":
        params["mu"] = mu
        return ALGORITHM_MAP[alg_name](params)

    if alg_name == "ADFxLMS":
        net = _create_network(ADFxLMSNode, mu, ref_ids, sec_ids, err_ids)
    elif alg_name == "ADFxLMS-BC":
        net = _create_network(ADFxLMSBCNode, mu, ref_ids, sec_ids, err_ids)
    elif alg_name == "Diff-FxLMS":
        net = _create_network(DiffFxLMSNode, mu, ref_ids, sec_ids, err_ids)
    elif alg_name == "DCFxLMS":
        net = _create_network(DCFxLMSNode, mu, ref_ids, sec_ids, err_ids)
    elif alg_name == "CDFxLMS":
        net = _create_network(CDFxLMSNode, mu, ref_ids, sec_ids, err_ids)
    elif alg_name == "MGDFxLMS":
        net = _create_network(MGDFxLMSNode, mu, ref_ids, sec_ids, err_ids, lc=16)
    else:
        raise ValueError(f"Unsupported algorithm: {alg_name}")

    _connect_for_algorithm(net, alg_name)
    params["network"] = net
    return ALGORITHM_MAP[alg_name](params)


def _compute_nse_db(d: np.ndarray, e: np.ndarray, fs: int, window_seconds: float = 1.0) -> list[float]:
    win = max(1, int(round(window_seconds * fs)))
    out = []
    for m in range(d.shape[1]):
        d_seg = d[-win:, m]
        e_seg = e[-win:, m]
        d_pow = np.mean(d_seg**2) + np.finfo(float).eps
        e_pow = np.mean(e_seg**2) + np.finfo(float).eps
        out.append(float(10.0 * np.log10(e_pow / d_pow)))
    return out


def _compute_nr_db(d: np.ndarray, e: np.ndarray, fs: int, window_seconds: float = 1.0) -> list[float]:
    nse = _compute_nse_db(d, e, fs, window_seconds)
    return [-v for v in nse]


if __name__ == "__main__":
    # =========================
    # User configuration block
    # =========================
    selected_algorithms = ["CFxLMS"]

    duration_s = 10.0
    f_low = 100.0
    f_high = 1500.0
    filter_length = 1024
    mu = 1e-4
    random_seed = 42

    save_plots = True
    output_dir = ROOT_DIR / "python_scripts"

    use_random_scenario = False

    if use_random_scenario:
        scenario_cfg = sample_asymmetric_scenario(
            seed=random_seed,
            num_nodes=4,
            room=RoomConfig(size=(5.0, 5.0, 5.0), fs=4000, image_source_order=2, material_absorption=0.5),
        )
    else:
        scenario_cfg = ScenarioConfig(
            room=RoomConfig(size=(5.0, 5.0, 5.0), fs=4000, image_source_order=2, material_absorption=0.5),
            source_position=(2.0, 2.4, 1.55),
            node_layouts=[
                NodeRadialLayout(azimuth_deg=20.0, ref_radius=0.40, sec_radius=0.63, err_radius=0.90, z_offset=0.02),
                NodeRadialLayout(azimuth_deg=138.0, ref_radius=0.52, sec_radius=0.79, err_radius=1.08, z_offset=-0.02),
                NodeRadialLayout(azimuth_deg=226.0, ref_radius=0.46, sec_radius=0.73, err_radius=1.00, z_offset=0.03),
                NodeRadialLayout(azimuth_deg=314.0, ref_radius=0.38, sec_radius=0.61, err_radius=0.88, z_offset=0.00),
            ],
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected algorithms: {', '.join(selected_algorithms)}")
    mgr, source_ids, ref_ids, sec_ids, err_ids = build_manager_from_config(scenario_cfg)

    print("Building room impulse responses...")
    mgr.build(verbose=False)

    layout_png = output_dir / "layout_preview_test.png"
    fig_layout, _ = plot_layout_with_labels(
        mgr,
        source_ids=source_ids,
        ref_ids=ref_ids,
        sec_ids=sec_ids,
        err_ids=err_ids,
        title="Test Scenario Layout",
        save_path=str(layout_png),
    )
    plt.close(fig_layout)
    print(f"Layout preview saved to: {layout_png}")

    # Generate one source channel per primary source.
    source_columns = []
    time_axis = None
    for idx in range(len(source_ids)):
        noise_col, t = wn_gen(mgr.fs, duration_s, f_low, f_high, rng=np.random.default_rng(random_seed + idx))
        source_columns.append(noise_col[:, 0])
        if time_axis is None:
            time_axis = t
    source_signal = np.column_stack(source_columns)
    source_signal = _normalize_columns(source_signal)

    d = mgr.calculate_desired_signal(source_signal, len(time_axis))
    x = mgr.calculate_reference_signal(source_signal, len(time_axis))
    x = _normalize_columns(x)

    runtimes = {}
    nse_db_last_1s = {}
    nr_db_last_1s = {}
    all_results = []

    for alg_name in selected_algorithms:
        if alg_name not in ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm: {alg_name}")

        t0 = time.perf_counter()
        results = _run_algorithm(
            alg_name,
            time_axis,
            mgr,
            x,
            d,
            ref_ids,
            sec_ids,
            err_ids,
            filter_length,
            mu,
        )
        dt = time.perf_counter() - t0

        runtimes[alg_name] = dt
        nse_db_last_1s[alg_name] = _compute_nse_db(d, results["err_hist"], mgr.fs)
        nr_db_last_1s[alg_name] = _compute_nr_db(d, results["err_hist"], mgr.fs)
        all_results.append(results)
        print(f"{alg_name} runtime: {dt:.6f} s")
        print(f"{alg_name} mean NR(d/e) over last 1s: {float(np.mean(nr_db_last_1s[alg_name])):.4f} dB")

    # Optional diagnostic plots.
    if save_plots and all_results:
        tap_fig = plot_tap_weights([r["filter_coeffs"] for r in all_results], selected_algorithms, list(sec_ids))
        tap_fig.savefig(output_dir / "tap_weights_test.png", dpi=140, bbox_inches="tight")
        plt.close(tap_fig)

        for ch, err_id in enumerate(err_ids):
            err_signals_ch = [r["err_hist"][:, ch] for r in all_results]
            fig = plot_results(time_axis[:, 0], d[:, ch], err_signals_ch, selected_algorithms, int(err_id), float(mgr.fs))
            fig.savefig(output_dir / f"error_compare_mic_{int(err_id)}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)

    summary = {
        "algorithms": selected_algorithms,
        "duration_s": duration_s,
        "fs": int(mgr.fs),
        "f_low": f_low,
        "f_high": f_high,
        "L": int(filter_length),
        "mu": float(mu),
        "runtimes_s": runtimes,
        "nse_db_last_1s": nse_db_last_1s,
        "nr_db_last_1s": nr_db_last_1s,
        "layout_preview": str(layout_png),
    }

    summary_path = output_dir / "last_run_summary_from_test.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to: {summary_path}")
    print("Done.")
