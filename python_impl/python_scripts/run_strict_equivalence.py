from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import loadmat

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.acoustics import PrecomputedRIRManager
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


DEFAULT_DATASET = ROOT_DIR / "python_scripts" / "strict_equiv_dataset.mat"
DEFAULT_SUMMARY = ROOT_DIR / "python_scripts" / "strict_py_summary.json"
DEFAULT_CURVES = ROOT_DIR / "python_scripts" / "strict_py_curves.npz"


ALGORITHM_MAP = {
    "CFxLMS": cfxlms,
    "ADFxLMS": adfxlms,
    "ADFxLMS-BC": adfxlms_bc,
    "Diff-FxLMS": diff_fxlms,
    "DCFxLMS": dcfxlms,
    "CDFxLMS": cdfxlms,
    "MGDFxLMS": mgdfxlms,
}


def _as_2d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        return arr[:, None]
    return arr


def _to_ids(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=int).reshape(-1)


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


def _compute_curves(d: np.ndarray, e: np.ndarray, fs: int, window_ms: float) -> tuple[np.ndarray, np.ndarray]:
    win = max(1, int(round((window_ms / 1000.0) * fs)))
    kernel = np.ones(win, dtype=float) / float(win)

    d_inst = np.mean(d**2, axis=1)
    e_inst = np.mean(e**2, axis=1)

    d_pow = np.convolve(d_inst, kernel, mode="same") + np.finfo(float).eps
    e_pow = np.convolve(e_inst, kernel, mode="same") + np.finfo(float).eps

    nse = 10.0 * np.log10(e_pow / d_pow)
    nr = -nse
    return nse, nr


def _create_network(node_cls, mu: float, ref_ids: np.ndarray, sec_ids: np.ndarray, err_ids: np.ndarray, lc: int | None = None) -> Network:
    net = Network()
    n_nodes = len(ref_ids)

    for idx in range(n_nodes):
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
    mgr: PrecomputedRIRManager,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict-equivalence Python ANC simulations.")
    parser.add_argument("--dataset", type=str, default=str(DEFAULT_DATASET))
    parser.add_argument(
        "--algorithms",
        type=str,
        default="CFxLMS,ADFxLMS,ADFxLMS-BC,Diff-FxLMS,DCFxLMS,CDFxLMS,MGDFxLMS",
        help="Comma-separated algorithm list.",
    )
    parser.add_argument("--curve-window-ms", type=float, default=50.0)
    parser.add_argument("--summary-out", type=str, default=str(DEFAULT_SUMMARY))
    parser.add_argument("--curves-out", type=str, default=str(DEFAULT_CURVES))
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    if not dataset_path.is_absolute():
        dataset_path = (Path.cwd() / dataset_path).resolve()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Strict dataset not found: {dataset_path}")

    mat = loadmat(str(dataset_path))

    time_axis = np.asarray(mat["time"], dtype=float).reshape(-1)
    x = _as_2d(mat["reference_signal"])
    d = _as_2d(mat["desired_signal"])
    ref_ids = _to_ids(mat["ref_ids"])
    sec_ids = _to_ids(mat["sec_ids"])
    err_ids = _to_ids(mat["err_ids"])
    sec_rirs = np.asarray(mat["sec_rirs"], dtype=float)
    sec_rir_lengths = np.asarray(mat["sec_rir_lengths"], dtype=int)

    fs = int(np.asarray(mat["fs"]).reshape(-1)[0])
    l = int(np.asarray(mat["L"]).reshape(-1)[0])
    mu = float(np.asarray(mat["mu"]).reshape(-1)[0])

    mgr = PrecomputedRIRManager.from_padded_arrays(ref_ids, sec_ids, err_ids, sec_rirs, sec_rir_lengths)

    selected_algorithms = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    print(f"Strict mode selected algorithms: {', '.join(selected_algorithms)}")

    runtimes: dict[str, float] = {}
    nse_last: dict[str, list[float]] = {}
    nr_last: dict[str, list[float]] = {}
    nse_curves: list[np.ndarray] = []
    nr_curves: list[np.ndarray] = []

    for alg_name in selected_algorithms:
        if alg_name not in ALGORITHM_MAP:
            raise ValueError(f"Unsupported algorithm: {alg_name}")

        t0 = time.perf_counter()
        results = _run_algorithm(alg_name, time_axis, mgr, x, d, ref_ids, sec_ids, err_ids, l, mu)
        dt = float(time.perf_counter() - t0)

        e = np.asarray(results["err_hist"], dtype=float)

        runtimes[alg_name] = dt
        nse_last[alg_name] = _compute_nse_db(d, e, fs)
        nr_last[alg_name] = _compute_nr_db(d, e, fs)
        nse_curve, nr_curve = _compute_curves(d, e, fs, args.curve_window_ms)
        nse_curves.append(nse_curve)
        nr_curves.append(nr_curve)

        print(f"{alg_name} strict-python runtime: {dt:.6f}s")

    summary = {
        "dataset": str(dataset_path),
        "algorithms": selected_algorithms,
        "fs": fs,
        "L": l,
        "mu": mu,
        "curve_window_ms": float(args.curve_window_ms),
        "runtimes_s": runtimes,
        "nse_db_last_1s": nse_last,
        "nr_db_last_1s": nr_last,
    }

    summary_path = Path(args.summary_out)
    if not summary_path.is_absolute():
        summary_path = (Path.cwd() / summary_path).resolve()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    curves_path = Path(args.curves_out)
    if not curves_path.is_absolute():
        curves_path = (Path.cwd() / curves_path).resolve()
    curves_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(curves_path),
        time=time_axis.astype(float),
        algorithms=np.asarray(selected_algorithms, dtype="U32"),
        nse_curves=np.stack(nse_curves, axis=1),
        nr_curves=np.stack(nr_curves, axis=1),
    )

    print(f"Strict Python summary saved to: {summary_path}")
    print(f"Strict Python curves saved to: {curves_path}")


if __name__ == "__main__":
    main()
