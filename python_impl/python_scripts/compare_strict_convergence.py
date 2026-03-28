from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PY_SUMMARY = ROOT_DIR / "python_scripts" / "strict_py_summary.json"
DEFAULT_MAT_SUMMARY = ROOT_DIR / "python_scripts" / "strict_mat_summary.json"
DEFAULT_PY_CURVES = ROOT_DIR / "python_scripts" / "strict_py_curves.npz"
DEFAULT_MAT_CURVES = ROOT_DIR / "python_scripts" / "strict_mat_curves.mat"
DEFAULT_REPORT = ROOT_DIR / "python_scripts" / "strict_convergence_report.json"
DEFAULT_OVERLAY_PNG = ROOT_DIR / "python_scripts" / "strict_convergence_overlay.png"
DEFAULT_DELTA_PNG = ROOT_DIR / "python_scripts" / "strict_convergence_delta.png"


def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        p = (Path.cwd() / p).resolve()
    return p


def _to_algorithm_list(arr: np.ndarray) -> list[str]:
    a = np.asarray(arr)

    if a.dtype.kind in {"U", "S"}:
        return [str(v).strip() for v in a.reshape(-1)]

    out: list[str] = []
    for item in a.reshape(-1):
        if isinstance(item, np.ndarray):
            if item.dtype.kind in {"U", "S"}:
                if item.ndim == 0:
                    out.append(str(item.item()).strip())
                else:
                    out.append("".join(item.reshape(-1).tolist()).strip())
            else:
                out.append(str(item.reshape(-1)[0]).strip())
        else:
            out.append(str(item).strip())
    return out


def _safe_field_name(alg_name: str) -> str:
    return alg_name.replace("-", "_").replace(" ", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare strict-equivalence convergence curves (Python vs MATLAB).")
    parser.add_argument("--py-summary", type=str, default=str(DEFAULT_PY_SUMMARY))
    parser.add_argument("--mat-summary", type=str, default=str(DEFAULT_MAT_SUMMARY))
    parser.add_argument("--py-curves", type=str, default=str(DEFAULT_PY_CURVES))
    parser.add_argument("--mat-curves", type=str, default=str(DEFAULT_MAT_CURVES))
    parser.add_argument("--report-out", type=str, default=str(DEFAULT_REPORT))
    parser.add_argument("--overlay-png", type=str, default=str(DEFAULT_OVERLAY_PNG))
    parser.add_argument("--delta-png", type=str, default=str(DEFAULT_DELTA_PNG))
    args = parser.parse_args()

    py_summary_path = _resolve(args.py_summary)
    mat_summary_path = _resolve(args.mat_summary)
    py_curves_path = _resolve(args.py_curves)
    mat_curves_path = _resolve(args.mat_curves)
    report_path = _resolve(args.report_out)
    overlay_png = _resolve(args.overlay_png)
    delta_png = _resolve(args.delta_png)

    for p in [py_summary_path, mat_summary_path, py_curves_path, mat_curves_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required file not found: {p}")

    py_summary = json.loads(py_summary_path.read_text(encoding="utf-8"))
    mat_summary = json.loads(mat_summary_path.read_text(encoding="utf-8"))

    py_curves_npz = np.load(str(py_curves_path), allow_pickle=False)
    py_time = np.asarray(py_curves_npz["time"], dtype=float).reshape(-1)
    py_algorithms = [str(v) for v in np.asarray(py_curves_npz["algorithms"]).reshape(-1)]
    py_nr = np.asarray(py_curves_npz["nr_curves"], dtype=float)

    mat_curves = loadmat(str(mat_curves_path))
    mat_time = np.asarray(mat_curves["time"], dtype=float).reshape(-1)
    mat_algorithms = _to_algorithm_list(mat_curves["algorithms"])

    nr_key = "nr_curves" if "nr_curves" in mat_curves else "nrCurves"
    if nr_key not in mat_curves:
        raise KeyError("MATLAB curves file must contain 'nr_curves' or 'nrCurves'.")
    mat_nr = np.asarray(mat_curves[nr_key], dtype=float)

    if py_nr.ndim != 2 or mat_nr.ndim != 2:
        raise ValueError("Both py/mat nr_curves must be 2D arrays of shape [time, algorithms].")

    n_common = min(len(py_time), len(mat_time), py_nr.shape[0], mat_nr.shape[0])
    py_time = py_time[:n_common]
    py_nr = py_nr[:n_common, :]
    mat_nr = mat_nr[:n_common, :]

    mat_idx = {name: i for i, name in enumerate(mat_algorithms)}

    rows: list[dict] = []
    deltas: list[np.ndarray] = []
    labels: list[str] = []

    for i, alg in enumerate(py_algorithms):
        if alg not in mat_idx:
            continue
        j = mat_idx[alg]

        curve_delta = py_nr[:, i] - mat_nr[:, j]
        deltas.append(curve_delta)
        labels.append(alg)

        safe_name = _safe_field_name(alg)
        py_rt = float(py_summary.get("runtimes_s", {}).get(alg, float("nan")))
        mat_rt = float(mat_summary.get("runtimes_s", {}).get(safe_name, float("nan")))

        py_last = np.asarray(py_summary.get("nr_db_last_1s", {}).get(alg, []), dtype=float)
        mat_last = np.asarray(mat_summary.get("nr_db_last_1s", {}).get(safe_name, []), dtype=float)
        if mat_last.size == 0:
            mat_nse = np.asarray(mat_summary.get("nse_db_last_1s", {}).get(safe_name, []), dtype=float)
            if mat_nse.size > 0:
                mat_last = -mat_nse

        mean_py_last = float(np.mean(py_last)) if py_last.size > 0 else float("nan")
        mean_mat_last = float(np.mean(mat_last)) if mat_last.size > 0 else float("nan")

        rows.append(
            {
                "algorithm": alg,
                "runtime_py_s": py_rt,
                "runtime_mat_s": mat_rt,
                "speedup_mat_over_py": (mat_rt / py_rt) if py_rt > 0 else float("nan"),
                "curve_rmse_db": float(np.sqrt(np.mean(curve_delta**2))),
                "curve_mean_abs_db": float(np.mean(np.abs(curve_delta))),
                "mean_nr_py_last_1s_db": mean_py_last,
                "mean_nr_mat_last_1s_db": mean_mat_last,
                "abs_mean_nr_last_1s_delta_db": float(abs(mean_py_last - mean_mat_last))
                if np.isfinite(mean_py_last) and np.isfinite(mean_mat_last)
                else float("nan"),
            }
        )

    if not rows:
        raise RuntimeError("No common algorithms found between strict Python and strict MATLAB curve files.")

    # Overlay figure per algorithm.
    n_alg = len(labels)
    n_cols = 2
    n_rows = int(np.ceil(n_alg / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows), sharex=True)
    axes = np.asarray(axes).reshape(-1)

    for idx, alg in enumerate(labels):
        ax = axes[idx]
        py_i = py_algorithms.index(alg)
        mat_i = mat_idx[alg]
        ax.plot(py_time, py_nr[:, py_i], label="Python", linewidth=1.2)
        ax.plot(py_time, mat_nr[:, mat_i], label="MATLAB", linewidth=1.2)
        ax.set_title(alg)
        ax.set_ylabel("NR (dB)")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="best")

    for idx in range(n_alg, len(axes)):
        axes[idx].axis("off")

    axes[min(n_alg - 1, len(axes) - 1)].set_xlabel("Time (s)")
    fig.suptitle("Strict Equivalence: Convergence Overlay (NR)")
    fig.tight_layout()
    overlay_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(overlay_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Delta figure (Python - MATLAB).
    fig2, ax2 = plt.subplots(figsize=(12, 5))
    for alg, delta in zip(labels, deltas):
        ax2.plot(py_time, delta, linewidth=1.0, label=alg)
    ax2.axhline(0.0, color="k", linestyle="--", linewidth=1.0)
    ax2.set_title("Strict Equivalence: NR Delta (Python - MATLAB)")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Delta NR (dB)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best", ncol=2)
    fig2.tight_layout()
    delta_png.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(delta_png, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    report = {
        "rows": rows,
        "curve_points": int(n_common),
        "overlay_png": str(overlay_png),
        "delta_png": str(delta_png),
        "py_summary": str(py_summary_path),
        "mat_summary": str(mat_summary_path),
        "py_curves": str(py_curves_path),
        "mat_curves": str(mat_curves_path),
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Strict convergence report saved to: {report_path}")
    print(f"Overlay figure saved to: {overlay_png}")
    print(f"Delta figure saved to: {delta_png}")


if __name__ == "__main__":
    main()
