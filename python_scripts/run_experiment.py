from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.acoustics import RIRManager
from py_anc.algorithms import adfxlms, adfxlms_bc, cdfxlms, cfxlms, dcfxlms, diff_fxlms, mgdfxlms
from py_anc.algorithms.nodes import (
    ADFxLMSBCNode,
    ADFxLMSNode,
    CDFxLMSNode,
    DCFxLMSNode,
    DiffFxLMSNode,
    MGDFxLMSNode,
)
from py_anc.topology import Network
from py_anc.utils import wn_gen


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


def _build_manager(fs: int) -> tuple[RIRManager, np.ndarray]:
    mgr = RIRManager()
    mgr.fs = fs
    mgr.room = np.array([5.0, 5.0, 5.0], dtype=float)
    mgr.image_source_order = 2
    mgr.material_absorption = 0.5

    center = mgr.room / 2.0

    mgr.add_primary_speaker(101, center + np.array([1.0, 0.0, 0.0]))

    r_ref = 0.9
    ref_mic_ids = np.array([401, 402, 403, 404], dtype=int)
    mgr.add_reference_microphone(ref_mic_ids[0], center + np.array([r_ref, 0.0, 0.0]))
    mgr.add_reference_microphone(ref_mic_ids[1], center - np.array([r_ref, 0.0, 0.0]))
    mgr.add_reference_microphone(ref_mic_ids[2], center + np.array([0.0, r_ref, 0.0]))
    mgr.add_reference_microphone(ref_mic_ids[3], center - np.array([0.0, r_ref, 0.0]))

    r1 = 0.6
    mgr.add_secondary_speaker(201, center + np.array([r1, 0.0, 0.0]))
    mgr.add_secondary_speaker(202, center - np.array([r1, 0.0, 0.0]))
    mgr.add_secondary_speaker(203, center + np.array([0.0, r1, 0.0]))
    mgr.add_secondary_speaker(204, center - np.array([0.0, r1, 0.0]))

    r2 = 0.3
    mgr.add_error_microphone(301, center + np.array([r2, 0.0, 0.0]))
    mgr.add_error_microphone(302, center - np.array([r2, 0.0, 0.0]))
    mgr.add_error_microphone(303, center + np.array([0.0, r2, 0.0]))
    mgr.add_error_microphone(304, center - np.array([0.0, r2, 0.0]))

    return mgr, ref_mic_ids


def _create_network(node_cls, mu: float, ref_mic_ids: np.ndarray, lc: int | None = None) -> Network:
    net = Network()

    constructor_args = []
    if lc is not None:
        constructor_args = [lc]

    nodes = []
    for idx in range(4):
        if lc is None:
            node = node_cls(node_id=idx + 1, step_size=mu)
        else:
            node = node_cls(node_id=idx + 1, step_size=mu, lc=lc)
        node.add_ref_mic(int(ref_mic_ids[idx]))
        node.add_sec_spk(201 + idx)
        node.add_err_mic(301 + idx)
        nodes.append(node)
        net.add_node(node)

    return net


def _connect_for_algorithm(net: Network, alg_name: str) -> None:
    if alg_name == "ADFxLMS":
        net.connect_nodes(1, 2)
        net.connect_nodes(1, 4)
        net.connect_nodes(2, 3)
        net.connect_nodes(2, 4)
        net.connect_nodes(1, 2)
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


def _run_algorithm(alg_name: str, time_axis, mgr, x, d, ref_mic_ids, l, mu):
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
        net = _create_network(ADFxLMSNode, mu, ref_mic_ids)
    elif alg_name == "ADFxLMS-BC":
        net = _create_network(ADFxLMSBCNode, mu, ref_mic_ids)
    elif alg_name == "Diff-FxLMS":
        net = _create_network(DiffFxLMSNode, mu, ref_mic_ids)
    elif alg_name == "DCFxLMS":
        net = _create_network(DCFxLMSNode, mu, ref_mic_ids)
    elif alg_name == "CDFxLMS":
        net = _create_network(CDFxLMSNode, mu, ref_mic_ids)
    elif alg_name == "MGDFxLMS":
        net = _create_network(MGDFxLMSNode, mu, ref_mic_ids, lc=16)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Python ANC simulations.")
    parser.add_argument(
        "--algorithms",
        type=str,
        default="CFxLMS",
        help="Comma-separated algorithm list, e.g. CFxLMS,ADFxLMS",
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--fs", type=int, default=4000)
    parser.add_argument("--f-low", type=float, default=100.0)
    parser.add_argument("--f-high", type=float, default=1500.0)
    parser.add_argument("--L", type=int, default=1024)
    parser.add_argument("--mu", type=float, default=1e-4)
    parser.add_argument(
        "--output-json",
        type=str,
        default="python_scripts/last_run_summary.json",
    )
    args = parser.parse_args()

    selected_algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    print(f"当前选择运行的算法: {', '.join(selected_algorithms)}")

    mgr, ref_mic_ids = _build_manager(args.fs)

    print("正在构建声学环境...")
    mgr.build(verbose=False)

    rng = np.random.default_rng(42)
    noise, time_axis = wn_gen(args.fs, args.duration, args.f_low, args.f_high, rng=rng)
    source_signal = _normalize_columns(noise)

    d = mgr.calculate_desired_signal(source_signal, len(time_axis))
    x = mgr.calculate_reference_signal(source_signal, len(time_axis))
    x = _normalize_columns(x)

    runtimes = {}
    nse = {}

    for alg_name in selected_algorithms:
        if alg_name not in ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm: {alg_name}")

        t0 = time.perf_counter()
        results = _run_algorithm(alg_name, time_axis, mgr, x, d, ref_mic_ids, args.L, args.mu)
        dt = time.perf_counter() - t0

        runtimes[alg_name] = dt
        nse[alg_name] = _compute_nse_db(d, results["err_hist"], args.fs)

        print(f"{alg_name} 仿真耗时 {dt:.6f} 秒。")

    summary = {
        "algorithms": selected_algorithms,
        "duration_s": args.duration,
        "fs": args.fs,
        "f_low": args.f_low,
        "f_high": args.f_high,
        "L": args.L,
        "mu": args.mu,
        "runtimes_s": runtimes,
        "nse_db_last_1s": nse,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"结果已写入: {out_path}")
    print("全部完成。")


if __name__ == "__main__":
    main()
