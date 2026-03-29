"""verify_w_apply_to_room.py

加载 HDF5 中存储的 W（raw/W_full / raw/W_opt），将其作为固定控制滤波器应用到对应房间的仿真中，计算并比较残差能量。

用法示例：
    python verify_w_apply_to_room.py --h5 python_impl/python_scripts/cfxlms_qc_dataset_cross_500_seeded.h5 --room 0 --show-plot False
"""
from pathlib import Path
import sys
import importlib.util
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def _normalize_columns(x_arr: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x_arr), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x_arr / denom


def _build_weight_tensor_from_w_full(w_full: np.ndarray, filter_len: int, n_ref: int, n_sec: int) -> np.ndarray:
    w = np.zeros((filter_len, n_ref, n_sec), dtype=float)
    arr = np.asarray(w_full, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f'w_full 期望三维 [sec,ref,L]，实际 {arr.shape}')
    keep_sec = min(n_sec, arr.shape[0])
    keep_ref = min(n_ref, arr.shape[1])
    keep_len = min(filter_len, arr.shape[2])
    for k in range(keep_sec):
        w[:keep_len, :keep_ref, k] = arr[k, :keep_ref, :keep_len].T
    return w


def rolling_mse_db(sig: np.ndarray, fs: int, window_samples: int = 500):
    n = sig.size
    if n < window_samples:
        eps = np.finfo(float).eps
        mse = float(np.mean(sig**2))
        return np.array([0.0]), np.array([10.0 * np.log10(mse + eps)])
    kernel = np.ones(window_samples, dtype=float) / float(window_samples)
    eps = np.finfo(float).eps
    pow_s = np.convolve(sig**2, kernel, mode='valid')
    t = (np.arange(len(pow_s)) + (window_samples - 1)) / float(fs)
    db = 10.0 * np.log10(pow_s + eps)
    return t, db


def simulate_fixed_filter(time, reference_signal, desired_signal, w_full, rir_manager, sec_ids, err_ids, L):
    x = np.asarray(reference_signal, dtype=float)
    d = np.asarray(desired_signal, dtype=float)
    n_samples = len(time)
    num_ref = x.shape[1]
    num_sec = len(sec_ids)
    num_err = len(err_ids)

    # 次级路径缓存
    s_paths = [[None for _ in range(num_err)] for _ in range(num_sec)]
    s_lens = np.zeros((num_sec, num_err), dtype=np.int32)
    max_ls_hat = 0
    for k, sec_id in enumerate(sec_ids):
        for m, err_id in enumerate(err_ids):
            s = np.asarray(rir_manager.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
            s_paths[k][m] = s
            s_lens[k, m] = int(s.size)
            if s.size > max_ls_hat:
                max_ls_hat = int(s.size)

    # 转为 FxLMS 所需的 tap 形状
    w_tensor = _build_weight_tensor_from_w_full(w_full, L, num_ref, num_sec)

    x_taps = np.zeros((max(L, max_ls_hat), num_ref), dtype=float)
    e = np.zeros((n_samples, num_err), dtype=float)
    y_taps = []
    for k in range(num_sec):
        sec_len = int(np.max(s_lens[k]))
        y_taps.append(np.zeros(sec_len, dtype=float))

    for n in range(n_samples):
        x_taps[1:, :] = x_taps[:-1, :]
        x_taps[0, :] = x[n, :]

        for k in range(num_sec):
            y = np.sum(w_tensor[:L, :, k] * x_taps[:L, :])
            y_taps[k][1:] = y_taps[k][:-1]
            y_taps[k][0] = y

        for m in range(num_err):
            yf = 0.0
            for k in range(num_sec):
                s = s_paths[k][m]
                ls = int(s_lens[k, m])
                yf += float(np.dot(s, y_taps[k][:ls]))
            e[n, m] = d[n, m] + yf

    return e


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', default=str(Path('python_impl') / 'python_scripts' / 'cfxlms_qc_dataset_cross_500_seeded.h5'))
    parser.add_argument('--room', type=int, default=0)
    parser.add_argument('--pairs', default='1,0', help='要绘制的 sec,ref 对，用逗号分隔，例如 "1,0" 表示 ref1->sec2（1-based 说明）')
    parser.add_argument('--show-plot', action='store_true')
    args = parser.parse_args()

    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f'HDF5 not found: {h5_path}')

    # 将项目 python_impl 加入 sys.path 以便导入 builder
    ROOT = Path.cwd()
    if (ROOT / 'python_impl').exists():
        sys.path.insert(0, str(ROOT / 'python_impl'))
    else:
        sys.path.insert(0, str(ROOT))

    # 加载构建脚本（与 Notebook 保持一致的配置/采样器）
    builder_path = Path('python_impl') / 'python_scripts' / 'build_cfxlms_qc_dataset.py'
    if not builder_path.exists():
        raise FileNotFoundError(f'无法找到构建脚本: {builder_path}')
    spec = importlib.util.spec_from_file_location('build_cfx', str(builder_path))
    build_mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = build_mod
    spec.loader.exec_module(build_mod)
    cfg = build_mod.DatasetBuildConfig()
    rng = np.random.default_rng(0)
    sampler = build_mod.AcousticScenarioSampler(cfg, rng)

    from py_anc.utils import wn_gen

    with h5py.File(str(h5_path), 'r') as hf:
        # 找到 W 数据集
        possible_w_keys = ['raw/W_full', 'raw/W_opt', 'raw/W']
        found_key = next((k for k in possible_w_keys if k in hf), None)
        if found_key is None:
            raise RuntimeError('HDF5 中未找到 W 数据集（raw/W_full/raw/W_opt/raw/W）')

        W_room_raw = np.asarray(hf[found_key][args.room], dtype=np.float32)

        # 读取房间参数并构建 manager
        sampled = {}
        sampled['room_size'] = np.asarray(hf['raw/room_params/room_size'][args.room], dtype=float)
        sampled['source_pos'] = np.asarray(hf['raw/room_params/source_position'][args.room], dtype=float)
        sampled['ref_positions'] = np.asarray(hf['raw/room_params/ref_positions'][args.room], dtype=float)
        sampled['sec_positions'] = np.asarray(hf['raw/room_params/sec_positions'][args.room], dtype=float)
        sampled['err_positions'] = np.asarray(hf['raw/room_params/err_positions'][args.room], dtype=float)
        sampled['sound_speed'] = float(hf['raw/room_params/sound_speed'][args.room]) if 'raw/room_params/sound_speed' in hf else float(cfg.sound_speed_min)
        sampled['absorption'] = float(hf['raw/room_params/material_absorption'][args.room]) if 'raw/room_params/material_absorption' in hf else float(cfg.absorption_min)
        sampled['image_order'] = int(hf['raw/room_params/image_source_order'][args.room]) if 'raw/room_params/image_source_order' in hf else int(cfg.image_order_choices[0])

        mgr = sampler.build_manager(sampled)
        mgr.build(verbose=False)

        # 生成与数据集一致的带限噪声（使用 dataset 里记录的 seed）
        source_seed = int(hf['raw/qc_metrics/source_seed'][args.room]) if 'raw/qc_metrics/source_seed' in hf else int(20260330 + args.room)
        noise, t = wn_gen(fs=cfg.fs, duration=cfg.noise_duration_s if hasattr(cfg, 'noise_duration_s') else 2.0, f_low=cfg.f_low, f_high=cfg.f_high, rng=np.random.default_rng(int(source_seed)))
        time_axis = t[:, 0].astype(float)
        source_signal = noise

        d = mgr.calculate_desired_signal(source_signal, len(time_axis))
        x = mgr.calculate_reference_signal(source_signal, len(time_axis))
        x = _normalize_columns(x)

        sec_ids = sampler.sec_ids
        err_ids = sampler.err_ids

        # 将 W_room_raw 解释为 (sec, ref, L)（与 dataset 一致）
        if W_room_raw.ndim == 4:
            # 可能的形状 (Nrooms, sec, ref, L) 已在外面索引，因此不应发生
            raise RuntimeError('W_room_raw 具有 4 维而不是预期的 3 维')
        if W_room_raw.ndim != 3:
            raise RuntimeError(f'W 在 HDF5 中的单房间数据不是 3 维: {W_room_raw.shape}')

        W_h5 = W_room_raw.astype(float)
        L = int(W_h5.shape[2])

        # 计算并比较：W_h5 与 零滤波下的残差
        e_h5 = simulate_fixed_filter(time_axis, x, d, W_h5, mgr, sec_ids, err_ids, L)
        e_zero = simulate_fixed_filter(time_axis, x, d, np.zeros_like(W_h5), mgr, sec_ids, err_ids, L)

        # 输出数值比较
        eps = np.finfo(float).eps
        mse_h5 = np.mean(e_h5**2, axis=0)
        mse_zero = np.mean(e_zero**2, axis=0)
        print(f'Room {args.room}: per-mic MSE (zero-init):', np.round(mse_zero, 6).tolist())
        print(f'Room {args.room}: per-mic MSE (HDF5 W):', np.round(mse_h5, 6).tolist())
        db_zero = 10.0 * np.log10(mse_zero + eps)
        db_h5 = 10.0 * np.log10(mse_h5 + eps)
        print(f'Room {args.room}: per-mic Error Power dB (zero):', np.round(db_zero, 3).tolist())
        print(f'Room {args.room}: per-mic Error Power dB (HDF5):', np.round(db_h5, 3).tolist())

        # 绘图对比（若请求）
        if args.show_plot:
            fs = int(cfg.fs)
            win = min(max(500, int(round(0.125 * cfg.fs))), int(len(time_axis) // 2))
            num_show_mics = min(3, e_h5.shape[1])
            fig, axes = plt.subplots(1, num_show_mics, figsize=(4 * num_show_mics, 3), sharex=True)
            if num_show_mics == 1:
                axes = [axes]
            for m in range(num_show_mics):
                t0, db0 = rolling_mse_db(e_zero[:, m], fs, window_samples=win)
                t1, db1 = rolling_mse_db(e_h5[:, m], fs, window_samples=win)
                axes[m].plot(t0, db0, label='Zero-init', linestyle='--', color='tab:red')
                axes[m].plot(t1, db1, label='HDF5 W', linestyle='-', color='tab:blue')
                axes[m].set_title(f'Err Mic {m} (Room {args.room})')
                axes[m].set_xlabel('Time (s)')
                axes[m].set_ylabel('Error Power (dB)')
                axes[m].legend()
                axes[m].grid(True, alpha=0.3)
            plt.suptitle(f'Room {args.room}: Zero vs HDF5 W')
            plt.show()


if __name__ == '__main__':
    main()
