from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from py_anc.acoustics import RIRManager
from py_anc.algorithms import cfxlms
from py_anc.scenarios import plot_layout_with_labels
from py_anc.utils import wn_gen


SOURCE_IDS = np.array([101], dtype=int)
REF_IDS = np.array([401, 402, 403], dtype=int)
SEC_IDS = np.array([201, 202, 203], dtype=int)
ERR_IDS = np.array([301, 302, 303], dtype=int)


def _normalize_columns(x: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x / denom


def _global_nr_metrics(d: np.ndarray, e: np.ndarray, fs: int) -> dict[str, float]:
    n = int(d.shape[0])
    if n < 16:
        raise ValueError("信号长度不足，无法计算NR。")

    win = min(max(int(round(0.5 * fs)), 8), n // 2)
    eps = np.finfo(float).eps

    nr_first = 10.0 * np.log10((float(np.mean(d[:win] ** 2)) + eps) / (float(np.mean(e[:win] ** 2)) + eps))
    nr_last = 10.0 * np.log10((float(np.mean(d[-win:] ** 2)) + eps) / (float(np.mean(e[-win:] ** 2)) + eps))
    nr_gain = nr_last - nr_first

    return {
        "nr_first_db": float(nr_first),
        "nr_last_db": float(nr_last),
        "nr_gain_db": float(nr_gain),
    }


def _rolling_nr_curve(time_axis: np.ndarray, d_ch: np.ndarray, e_ch: np.ndarray, fs: int, window_s: float = 0.25) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_axis, dtype=float).reshape(-1)
    d_vec = np.asarray(d_ch, dtype=float).reshape(-1)
    e_vec = np.asarray(e_ch, dtype=float).reshape(-1)

    if not (t.size == d_vec.size == e_vec.size):
        raise ValueError("time/d/e 长度不一致。")

    win = max(8, int(round(window_s * fs)))
    if d_vec.size < win:
        eps = np.finfo(float).eps
        nr_const = 10.0 * np.log10((float(np.mean(d_vec**2)) + eps) / (float(np.mean(e_vec**2)) + eps))
        return t, np.full_like(t, nr_const, dtype=float)

    kernel = np.ones(win, dtype=float) / float(win)
    eps = np.finfo(float).eps
    d_pow = np.convolve(d_vec**2, kernel, mode="valid")
    e_pow = np.convolve(e_vec**2, kernel, mode="valid")
    nr = 10.0 * np.log10((d_pow + eps) / (e_pow + eps))
    return t[win - 1 :], nr


def _build_manager_from_sample(sample: dict[str, np.ndarray | float | int], fs: int) -> RIRManager:
    mgr = RIRManager()
    mgr.room = np.asarray(sample["room_size"], dtype=float)
    mgr.fs = int(fs)
    mgr.sound_speed = float(sample["sound_speed"])
    mgr.image_source_order = int(sample["image_order"])
    mgr.material_absorption = float(sample["material_absorption"])
    mgr.compensate_fractional_delay = True
    mgr.fractional_delay_shift = None

    mgr.add_primary_speaker(int(SOURCE_IDS[0]), np.asarray(sample["source_position"], dtype=float))

    ref_positions = np.asarray(sample["ref_positions"], dtype=float)
    sec_positions = np.asarray(sample["sec_positions"], dtype=float)
    err_positions = np.asarray(sample["err_positions"], dtype=float)

    for i in range(len(REF_IDS)):
        mgr.add_reference_microphone(int(REF_IDS[i]), ref_positions[i])
        mgr.add_secondary_speaker(int(SEC_IDS[i]), sec_positions[i])
        mgr.add_error_microphone(int(ERR_IDS[i]), err_positions[i])

    mgr.build(verbose=False)
    return mgr


def _build_weight_tensor_from_w_full(w_full: np.ndarray, filter_len: int, n_ref: int, n_sec: int) -> np.ndarray:
    """将 [sec, ref, L] 重排为 CFxLMS 使用的 [L, ref, sec]。"""
    w = np.zeros((filter_len, n_ref, n_sec), dtype=float)
    arr = np.asarray(w_full, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"w_full 期望三维 [sec,ref,L]，实际 {arr.shape}")

    keep_sec = min(n_sec, arr.shape[0])
    keep_ref = min(n_ref, arr.shape[1])
    keep_len = min(filter_len, arr.shape[2])
    for k in range(keep_sec):
        w[:keep_len, :keep_ref, k] = arr[k, :keep_ref, :keep_len].T
    return w


def _build_weight_tensor_from_diag(w_opt: np.ndarray, filter_len: int, n_ref: int, n_sec: int) -> np.ndarray:
    """兼容旧数据: 将 [node, L] 放入 [L, ref, sec] 的对角线。"""
    w = np.zeros((filter_len, n_ref, n_sec), dtype=float)
    n_nodes = min(w_opt.shape[0], n_ref, n_sec)
    keep = min(filter_len, w_opt.shape[1])
    for i in range(n_nodes):
        w[:keep, i, i] = np.asarray(w_opt[i, :keep], dtype=float)
    return w


def _build_weight_tensor_from_filter_coeffs(
    filter_coeffs: dict[int, np.ndarray],
    sec_ids: np.ndarray,
    filter_len: int,
    n_ref: int,
) -> np.ndarray:
    """将 CFxLMS 返回的 filter_coeffs 组装为 [L, ref, sec]。"""
    n_sec = int(len(sec_ids))
    w = np.zeros((filter_len, n_ref, n_sec), dtype=float)
    for k, sec_id in enumerate(sec_ids):
        w_k = np.asarray(filter_coeffs[int(sec_id)], dtype=float)
        if w_k.ndim != 2:
            raise ValueError(f"filter_coeffs[{int(sec_id)}] 期望二维，实际 {w_k.shape}")
        keep_l = min(filter_len, w_k.shape[0])
        keep_r = min(n_ref, w_k.shape[1])
        w[:keep_l, :keep_r, k] = w_k[:keep_l, :keep_r]
    return w


def _simulate_error_with_fixed_w(
    mgr: RIRManager,
    reference_signal: np.ndarray,
    desired_signal: np.ndarray,
    w: np.ndarray,
    sec_ids: np.ndarray,
    err_ids: np.ndarray,
) -> np.ndarray:
    n_samples, n_ref = reference_signal.shape
    l = int(w.shape[0])
    num_sec = int(len(sec_ids))
    num_err = int(len(err_ids))

    s_paths = [[None for _ in range(num_err)] for _ in range(num_sec)]
    s_lens = np.zeros((num_sec, num_err), dtype=np.int32)

    max_ls = 0
    for k, sec_id in enumerate(sec_ids):
        for m, err_id in enumerate(err_ids):
            s = np.asarray(mgr.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
            s_paths[k][m] = s
            s_lens[k, m] = int(s.size)
            max_ls = max(max_ls, int(s.size))

    x_taps = np.zeros((max(l, max_ls), n_ref), dtype=float)
    y_taps = []
    for k in range(num_sec):
        sec_len = int(np.max(s_lens[k]))
        y_taps.append(np.zeros(sec_len, dtype=float))

    e = np.zeros((n_samples, num_err), dtype=float)

    for n in range(n_samples):
        x_taps[1:, :] = x_taps[:-1, :]
        x_taps[0, :] = reference_signal[n, :]

        for k in range(num_sec):
            y = float(np.sum(w[:, :, k] * x_taps[:l, :]))
            y_taps[k][1:] = y_taps[k][:-1]
            y_taps[k][0] = y

        for m in range(num_err):
            y_filtered = 0.0
            for k in range(num_sec):
                s = s_paths[k][m]
                ls = int(s_lens[k, m])
                y_filtered += float(np.dot(s, y_taps[k][:ls]))
            e[n, m] = desired_signal[n, m] + y_filtered

    return e


def _plot_nr_comparison(
    room_idx: int,
    time_axis: np.ndarray,
    d: np.ndarray,
    e_adapt: np.ndarray,
    e_fixed: np.ndarray,
    recorded_nr_last_db: float,
    fs: int,
) -> None:
    n_mics = d.shape[1]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    axes = np.atleast_1d(axes).reshape(-1)

    t_ref = None
    curves_adapt = []
    curves_fixed = []

    for m in range(min(n_mics, 3)):
        t1, nr_adapt = _rolling_nr_curve(time_axis, d[:, m], e_adapt[:, m], fs=fs)
        t2, nr_fixed = _rolling_nr_curve(time_axis, d[:, m], e_fixed[:, m], fs=fs)

        curves_adapt.append(nr_adapt)
        curves_fixed.append(nr_fixed)
        t_ref = t1

        ax = axes[m]
        ax.plot(t1, nr_adapt, label="重跑CFxLMS(自适应)", linewidth=1.4)
        ax.plot(t2, nr_fixed, label="存储W_opt(固定滤波器)", linewidth=1.4)
        ax.set_title(f"Err Mic {int(ERR_IDS[m])}")
        ax.set_xlabel("时间 (s)")
        ax.set_ylabel("NR (dB)")
        ax.grid(True, alpha=0.3)
        if m == 0:
            ax.legend(loc="best")

    if t_ref is not None and curves_adapt and curves_fixed:
        mean_adapt = np.mean(np.vstack(curves_adapt), axis=0)
        mean_fixed = np.mean(np.vstack(curves_fixed), axis=0)
        ax_mean = axes[3]
        ax_mean.plot(t_ref, mean_adapt, label="三麦平均-自适应", linewidth=1.6)
        ax_mean.plot(t_ref, mean_fixed, label="三麦平均-固定W_opt", linewidth=1.6)
        ax_mean.axhline(float(recorded_nr_last_db), color="k", linestyle="--", linewidth=1.0, label="数据集记录nr_last_db")
        ax_mean.set_title("三麦平均 NR 曲线对比")
        ax_mean.set_xlabel("时间 (s)")
        ax_mean.set_ylabel("NR (dB)")
        ax_mean.grid(True, alpha=0.3)
        ax_mean.legend(loc="best")

    fig.suptitle(f"Room #{room_idx} 降噪对比")


def _load_sample(f: h5py.File, idx: int) -> dict[str, np.ndarray | float | int]:
    w_full = None
    if "raw/W_full" in f:
        w_full = np.asarray(f["raw/W_full"][idx], dtype=float)

    source_seed = None
    if "raw/qc_metrics/source_seed" in f:
        source_seed = int(f["raw/qc_metrics/source_seed"][idx])

    return {
        "room_size": np.asarray(f["raw/room_params/room_size"][idx], dtype=float),
        "source_position": np.asarray(f["raw/room_params/source_position"][idx], dtype=float),
        "ref_positions": np.asarray(f["raw/room_params/ref_positions"][idx], dtype=float),
        "sec_positions": np.asarray(f["raw/room_params/sec_positions"][idx], dtype=float),
        "err_positions": np.asarray(f["raw/room_params/err_positions"][idx], dtype=float),
        "sound_speed": float(f["raw/room_params/sound_speed"][idx]),
        "material_absorption": float(f["raw/room_params/material_absorption"][idx]),
        "image_order": int(f["raw/room_params/image_source_order"][idx]),
        "w_opt": np.asarray(f["raw/W_opt"][idx], dtype=float),
        "w_full": w_full,
        "source_seed": source_seed,
        "recorded_nr_last_db": float(f["raw/qc_metrics/nr_last_db"][idx]),
        "recorded_nr_gain_db": float(f["raw/qc_metrics/nr_gain_db"][idx]),
        "recorded_mu": float(f["raw/qc_metrics/mu_used"][idx]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="随机抽查数据集房间并复现布局与降噪曲线。")
    parser.add_argument(
        "--dataset-h5",
        type=str,
        default=str(ROOT_DIR / "python_scripts" / "cfxlms_qc_dataset_500.h5"),
        help="待抽查的 HDF5 数据集路径。",
    )
    parser.add_argument("--num-samples", type=int, default=2, help="随机抽查房间数量。")
    parser.add_argument("--indices", type=str, default="", help="可选，逗号分隔的固定房间索引列表。")
    parser.add_argument("--seed", type=int, default=20260329, help="随机抽查与噪声生成随机种子。")
    parser.add_argument("--duration-s", type=float, default=None, help="仿真时长，默认读取数据集配置。")
    parser.add_argument("--f-low", type=float, default=100.0, help="白噪声低截止频率。")
    parser.add_argument("--f-high", type=float, default=1500.0, help="白噪声高截止频率。")
    parser.add_argument("--filter-len", type=int, default=512, help="CFxLMS 滤波器阶数。")
    parser.add_argument("--mu", type=float, default=1.0e-4, help="重跑CFxLMS时的步长。")
    parser.add_argument("--no-replay-recorded-noise", action="store_true", help="不使用数据集记录的 source_seed 复现激励。")
    parser.add_argument("--no-show", action="store_true", help="仅计算不弹窗绘图。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    h5_path = Path(args.dataset_h5)
    if not h5_path.exists():
        raise FileNotFoundError(f"数据集文件不存在: {h5_path}")

    with h5py.File(str(h5_path), "r") as f:
        n_rooms = int(f["raw/room_params/room_size"].shape[0])

        cfg = {}
        if "config_json" in f.attrs:
            try:
                cfg = json.loads(f.attrs["config_json"])
            except Exception:
                cfg = {}

        fs = int(cfg.get("fs", 4000))
        duration_s = float(args.duration_s) if args.duration_s is not None else float(cfg.get("noise_duration_s", 7.0))
        f_low = float(args.f_low)
        f_high = float(args.f_high)

        if args.indices.strip():
            indices = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
        else:
            k = max(1, min(int(args.num_samples), n_rooms))
            indices = random.Random(args.seed).sample(range(n_rooms), k=k)

        for idx in indices:
            if idx < 0 or idx >= n_rooms:
                raise ValueError(f"索引越界: {idx} (有效范围: 0~{n_rooms - 1})")

        print("========== 抽查参数 ==========")
        print(f"数据集: {h5_path}")
        print(f"抽查索引: {indices}")
        print(f"fs={fs}, duration={duration_s:.2f}s, band=[{f_low:.1f},{f_high:.1f}]Hz")
        print(f"L={int(args.filter_len)}, mu={float(args.mu):.2e}")
        print(f"复现源噪声: {not args.no_replay_recorded_noise}")
        print(f"弹窗绘图: {not args.no_show}")
        print("============================")

        for idx in indices:
            sample = _load_sample(f, idx)
            mgr = _build_manager_from_sample(sample, fs=fs)

            fig_layout, _ = plot_layout_with_labels(
                mgr,
                source_ids=SOURCE_IDS,
                ref_ids=REF_IDS,
                sec_ids=SEC_IDS,
                err_ids=ERR_IDS,
                title=f"Room #{idx} Layout",
            )

            if (not args.no_replay_recorded_noise) and sample.get("source_seed", None) is not None:
                noise_rng = np.random.default_rng(int(sample["source_seed"]))
            else:
                noise_rng = np.random.default_rng(args.seed + idx)

            noise, t = wn_gen(fs=fs, duration=duration_s, f_low=f_low, f_high=f_high, rng=noise_rng)
            source_signal = _normalize_columns(noise)
            time_axis = np.asarray(t[:, 0], dtype=float)

            d = mgr.calculate_desired_signal(source_signal, len(time_axis))
            x = mgr.calculate_reference_signal(source_signal, len(time_axis))
            x = _normalize_columns(x)

            run_params = {
                "time": time_axis,
                "rir_manager": mgr,
                "L": int(args.filter_len),
                "mu": float(args.mu),
                "reference_signal": x,
                "desired_signal": d,
                "verbose": False,
                "normalized_update": False,
                "norm_epsilon": 1.0e-8,
            }
            rerun = cfxlms(run_params)
            e_adapt = np.asarray(rerun["err_hist"], dtype=float)

            w_full = sample.get("w_full", None)
            if w_full is not None:
                w_fixed = _build_weight_tensor_from_w_full(
                    np.asarray(w_full, dtype=float),
                    int(args.filter_len),
                    x.shape[1],
                    len(SEC_IDS),
                )
                fixed_desc = "dataset W_full"
            else:
                w_opt = np.asarray(sample["w_opt"], dtype=float)
                w_fixed = _build_weight_tensor_from_diag(w_opt, int(args.filter_len), x.shape[1], len(SEC_IDS))
                fixed_desc = "legacy W_opt(diagonal only)"

            e_fixed = _simulate_error_with_fixed_w(mgr, x, d, w_fixed, SEC_IDS, ERR_IDS)

            # 链路自检：将 rerun 的最终滤波器固定后再次仿真。
            w_rerun_final = _build_weight_tensor_from_filter_coeffs(
                rerun["filter_coeffs"],
                sec_ids=SEC_IDS,
                filter_len=int(args.filter_len),
                n_ref=x.shape[1],
            )
            e_fixed_rerun = _simulate_error_with_fixed_w(mgr, x, d, w_rerun_final, SEC_IDS, ERR_IDS)

            nr_adapt = _global_nr_metrics(d, e_adapt, fs)
            nr_fixed = _global_nr_metrics(d, e_fixed, fs)
            nr_fixed_rerun = _global_nr_metrics(d, e_fixed_rerun, fs)
            rec_last = float(sample["recorded_nr_last_db"])
            rec_gain = float(sample["recorded_nr_gain_db"])

            print(f"\n[Room #{idx}] 记录值 vs 复算值")
            print(f"  fixed source: {fixed_desc}")
            print(f"  recorded: nr_last={rec_last:.4f} dB, nr_gain={rec_gain:.4f} dB, mu={float(sample['recorded_mu']):.2e}")
            print(f"  fixed W : nr_last={nr_fixed['nr_last_db']:.4f} dB, nr_gain={nr_fixed['nr_gain_db']:.4f} dB")
            print(f"  rerun ANC: nr_last={nr_adapt['nr_last_db']:.4f} dB, nr_gain={nr_adapt['nr_gain_db']:.4f} dB")
            print(
                "  rerun final-W fixed: "
                f"nr_last={nr_fixed_rerun['nr_last_db']:.4f} dB, nr_gain={nr_fixed_rerun['nr_gain_db']:.4f} dB"
            )
            print(f"  delta(fixed-recorded): {nr_fixed['nr_last_db'] - rec_last:+.4f} dB")

            if args.no_show:
                plt.close(fig_layout)
            else:
                _plot_nr_comparison(
                    room_idx=idx,
                    time_axis=time_axis,
                    d=d,
                    e_adapt=e_adapt,
                    e_fixed=e_fixed,
                    recorded_nr_last_db=rec_last,
                    fs=fs,
                )
                plt.show()


if __name__ == "__main__":
    main()
