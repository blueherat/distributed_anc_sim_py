#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluate_anc.py

等价于 `python_impl/notebooks/evaluate_anc.ipynb` 的可运行脚本。
用法: 在项目根运行 `python python_impl/notebooks/evaluate_anc.py`
"""

from __future__ import annotations

import sys
import os
import argparse
from pathlib import Path
import importlib.util
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy.random import default_rng


def find_repo_root(marker: str = 'python_impl/python_scripts/build_cfxlms_qc_dataset.py') -> Path:
    p = Path.cwd()
    for d in [p] + list(p.parents):
        if (d / marker).exists():
            return d
    raise FileNotFoundError(f'未在当前目录及父目录中找到 {marker}')


def _normalize_columns(x_arr: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x_arr), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x_arr / denom


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


class MIMO_Conditioned_ANCNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=5, stride=2, padding=2)
        self.ln1 = nn.LayerNorm((16, 65))
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.ln2 = nn.LayerNorm((32, 33))
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.ln3 = nn.LayerNorm((64, 17))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.s_fc1 = nn.Linear(32, 64)
        self.s_ln1 = nn.LayerNorm(64)
        self.s_fc2 = nn.Linear(64, 64)
        self.s_ln2 = nn.LayerNorm(64)
        self.f_fc1 = nn.Linear(128, 128)
        self.f_ln1 = nn.LayerNorm(128)
        self.f_drop = nn.Dropout(0.2)
        self.f_fc2 = nn.Linear(128, 64)
        self.f_fc3 = nn.Linear(64, 32)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                try:
                    nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                except Exception:
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x_p, x_s):
        z = self.conv1(x_p)
        z = F.gelu(self.ln1(z))
        z = self.conv2(z)
        z = F.gelu(self.ln2(z))
        z = self.conv3(z)
        z = F.gelu(self.ln3(z))
        z = self.pool(z)
        z_p = z.view(z.size(0), -1)
        z_s = F.gelu(self.s_ln1(self.s_fc1(x_s)))
        z_s = F.gelu(self.s_ln2(self.s_fc2(z_s)))
        zf = torch.cat([z_p, z_s], dim=1)
        zf = F.gelu(self.f_ln1(self.f_fc1(zf)))
        zf = self.f_drop(zf)
        zf = F.gelu(self.f_fc2(zf))
        c_pred = self.f_fc3(zf)
        return c_pred


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


def cfxlms_with_init(time, rir_manager, L, mu, reference_signal, desired_signal, w_init=None, normalized_update=False, norm_epsilon=1.0e-8, verbose=False):
    x = np.asarray(reference_signal, dtype=float)
    d = np.asarray(desired_signal, dtype=float)
    key_sec_spks = list(rir_manager.secondary_speakers.keys())
    key_err_mics = list(rir_manager.error_microphones.keys())
    num_ref_mics = len(rir_manager.reference_microphones)
    num_sec_spks = len(key_sec_spks)
    num_err_mics = len(key_err_mics)
    n_samples = len(time)

    s_paths = [[None for _ in range(num_err_mics)] for _ in range(num_sec_spks)]
    s_lens = np.zeros((num_sec_spks, num_err_mics), dtype=np.int32)
    max_ls_hat = 0
    for k, sec_id in enumerate(key_sec_spks):
        for m, err_id in enumerate(key_err_mics):
            s = np.asarray(rir_manager.get_secondary_rir(sec_id, err_id), dtype=float)
            s_paths[k][m] = s
            s_lens[k, m] = int(s.size)
            if s.size > max_ls_hat:
                max_ls_hat = int(s.size)

    if w_init is None:
        w = np.zeros((L, num_ref_mics, num_sec_spks), dtype=float)
    else:
        w = _build_weight_tensor_from_w_full(w_init, L, num_ref_mics, num_sec_spks)

    x_taps = np.zeros((max(L, max_ls_hat), num_ref_mics), dtype=float)
    xf_taps = np.zeros((L, num_ref_mics, num_sec_spks, num_err_mics), dtype=float)
    e = np.zeros((n_samples, num_err_mics), dtype=float)
    y_taps = []
    for k in range(num_sec_spks):
        sec_len = int(np.max(s_lens[k]))
        y_taps.append(np.zeros(sec_len, dtype=float))

    if verbose:
        print('开始集中式FxLMS仿真（支持初始 w）...')
    for n in range(n_samples):
        x_taps[1:, :] = x_taps[:-1, :]
        x_taps[0, :] = x[n, :]

        for k in range(num_sec_spks):
            y = np.sum(w[:, :, k] * x_taps[:L, :])
            y_taps[k][1:] = y_taps[k][:-1]
            y_taps[k][0] = y

        for m in range(num_err_mics):
            yf = 0.0
            for k in range(num_sec_spks):
                s = s_paths[k][m]
                ls = int(s_lens[k, m])
                yf += float(np.dot(s, y_taps[k][:ls]))
            e[n, m] = d[n, m] + yf

        xf_taps[1:, :, :, :] = xf_taps[:-1, :, :, :]
        for k in range(num_sec_spks):
            for m in range(num_err_mics):
                s = s_paths[k][m]
                ls_hat = int(s_lens[k, m])
                xf_taps[0, :, k, m] = s @ x_taps[:ls_hat, :]

        for k in range(num_sec_spks):
            grad_k = np.zeros((L, num_ref_mics), dtype=float)
            for m in range(num_err_mics):
                phi = xf_taps[:, :, k, m]
                if normalized_update:
                    denom = float(np.sum(phi * phi)) + norm_epsilon
                    grad_k += (phi * e[n, m]) / denom
                else:
                    grad_k += phi * e[n, m]
            w[:, :, k] = w[:, :, k] - mu * grad_k

    filter_coeffs = {sec_id: w[:, :, k].copy() for k, sec_id in enumerate(key_sec_spks)}
    return {'err_hist': e, 'filter_coeffs': filter_coeffs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=20260330)
    parser.add_argument('--duration', type=float, default=10.0)
    parser.add_argument('--h5', type=str, default='python_impl/python_scripts/cfxlms_qc_dataset_cross_500_seeded.h5')
    parser.add_argument('--model', type=str, default='best_mimo_anc_net.pth')
    parser.add_argument('--no-show', action='store_true', help='不要显示交互式图形，仅保存')
    parser.add_argument('--save-prefix', type=str, default=None, help='若提供则保存图像到该前缀')
    args = parser.parse_args()

    # 找到项目根并切换
    ROOT = find_repo_root()
    print('检测到项目根:', ROOT)
    sys.path.insert(0, str(ROOT / 'python_impl'))
    os.chdir(str(ROOT))
    print('已切换工作目录到:', Path.cwd())

    # 载入 dataset builder 模块（按 notebook 的做法，先在 sys.modules 注册）
    builder_path = Path('python_impl') / 'python_scripts' / 'build_cfxlms_qc_dataset.py'
    if not builder_path.exists():
        raise FileNotFoundError(f'无法找到构建脚本: {builder_path}，请确认在项目根下运行本脚本')
    spec = importlib.util.spec_from_file_location('build_cfx', str(builder_path))
    build_mod = importlib.util.module_from_spec(spec)
    import sys as _sys
    _sys.modules[spec.name] = build_mod
    spec.loader.exec_module(build_mod)

    cfg = build_mod.DatasetBuildConfig()
    print('数据集配置载入，fs=', cfg.fs, 'filter_len=', cfg.filter_len, 'gcc_len=', cfg.gcc_truncated_len)

    # 采样并构建测试房间
    rng = default_rng(args.seed)
    sampler = build_mod.AcousticScenarioSampler(cfg, rng)
    sampled = sampler.sample()
    mgr = sampler.build_manager(sampled)
    mgr.build(verbose=False)
    print('采样完成。房间大小:', sampled['room_size'])

    # 生成噪声并构建参考/期望信号
    from py_anc.utils import wn_gen
    duration_s = float(args.duration)
    noise_rng = np.random.default_rng(args.seed + 1)
    noise, t = wn_gen(fs=cfg.fs, duration=duration_s, f_low=cfg.f_low, f_high=cfg.f_high, rng=noise_rng)
    time_axis = t[:, 0].astype(float)
    source_signal = noise
    d = mgr.calculate_desired_signal(source_signal, len(time_axis))
    x = mgr.calculate_reference_signal(source_signal, len(time_axis))
    x = _normalize_columns(x)
    print('信号长度:', len(time_axis), '参考通道数:', x.shape[1], '误差通道数:', d.shape[1])

    # 提取前 50 ms 的参考
    ref_samples = int(cfg.ref_window_samples)
    X_ref_test = x[:ref_samples, : cfg.num_nodes].T.astype(np.float32)
    print('X_ref_test shape:', X_ref_test.shape)

    # 构建 S_full_test / P_full_test
    rir_len = int(cfg.rir_store_len)
    n_nodes = int(cfg.num_nodes)
    S_full_test = np.zeros((n_nodes, n_nodes, rir_len), dtype=np.float32)
    for i, sec_id in enumerate(sampler.sec_ids):
        for j, err_id in enumerate(sampler.err_ids):
            r = np.asarray(mgr.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
            keep = min(rir_len, r.size)
            S_full_test[i, j, :keep] = r[:keep].astype(np.float32)

    P_full_test = np.zeros((1, n_nodes, rir_len), dtype=np.float32)
    pri_id = int(sampler.source_id)
    for j, err_id in enumerate(sampler.err_ids):
        r = np.asarray(mgr.get_primary_rir(pri_id, int(err_id)), dtype=float)
        keep = min(rir_len, r.size)
        P_full_test[0, j, :keep] = r[:keep].astype(np.float32)

    print('S_full_test shape:', S_full_test.shape, 'P_full_test shape:', P_full_test.shape)

    # 载入全局 SVD 基底
    h5_path = Path(args.h5)
    if not h5_path.exists():
        raise FileNotFoundError(f'需要 HDF5 数据集以读取全局 SVD 基底: {h5_path}')
    with h5py.File(str(h5_path), 'r') as hf:
        s_comp = np.asarray(hf['processed/global_svd/S_components'], dtype=np.float32)
        s_mean = np.asarray(hf['processed/global_svd/S_mean'], dtype=np.float32)
        w_comp = np.asarray(hf['processed/global_svd/W_components'], dtype=np.float32)
        w_mean = np.asarray(hf['processed/global_svd/W_mean'], dtype=np.float32)
    print('Loaded S_components shape:', s_comp.shape, 'W_components shape:', w_comp.shape)

    # 特征计算
    feat_proc = build_mod.FeatureProcessor(cfg)
    gcc = feat_proc.compute_gcc_phat(X_ref_test)
    psd = feat_proc.compute_psd_features(X_ref_test[0])
    x_p = np.vstack([gcc, psd.reshape(1, -1)]).astype(np.float32)
    x_p_t = torch.from_numpy(x_p).unsqueeze(0)

    s_flat = S_full_test.reshape(-1).astype(np.float32)
    s_centered = (s_flat - s_mean).astype(np.float32)
    s_coeff = s_centered @ s_comp.T
    x_s_t = torch.from_numpy(s_coeff.astype(np.float32)).unsqueeze(0)
    print('x_p_t shape:', x_p_t.shape, 'x_s_t shape:', x_s_t.shape)

    # 加载模型
    possible_paths = [Path(args.model), Path('python_impl') / 'notebooks' / Path(args.model)]
    model_path = None
    for p in possible_paths:
        if p.exists():
            model_path = p
            break
    if model_path is None:
        raise FileNotFoundError('找不到模型文件 best_mimo_anc_net.pth，请先在训练目录保存该文件或把路径放到工作目录。')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MIMO_Conditioned_ANCNet().to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    print('已加载模型', model_path, 'device:', device)

    # 前向推理并重构 W_AI_init
    with torch.no_grad():
        xp = x_p_t.to(device=device, dtype=torch.float32)
        xs = x_s_t.to(device=device, dtype=torch.float32)
        c_pred = model(xp, xs)
        Wc = torch.from_numpy(w_comp.astype(np.float32)).to(device=device)
        Wm = torch.from_numpy(w_mean.astype(np.float32)).to(device=device)
        W_pred_flat = torch.matmul(c_pred, Wc) + Wm
        W_pred_flat_np = W_pred_flat.cpu().numpy().reshape(-1)
        W_AI_init = W_pred_flat_np.reshape(cfg.num_nodes, cfg.num_nodes, cfg.filter_len)
    print('W_AI_init shape:', W_AI_init.shape)

    # 调用 cfxlms
    from py_anc.algorithms import cfxlms as cfxlms_fn

    mu = float(cfg.mu_candidates[0] if len(cfg.mu_candidates) else 1.0e-4)
    L = int(cfg.filter_len)
    normalized_update = bool(cfg.anc_normalized_update)

    params = {
        'time': time_axis,
        'rir_manager': mgr,
        'L': L,
        'mu': mu,
        'reference_signal': x,
        'desired_signal': d,
        'verbose': False,
        'normalized_update': normalized_update,
        'norm_epsilon': float(cfg.anc_norm_epsilon),
    }

    print('运行 Baseline (zero-init) 仿真...')
    res_baseline = cfxlms_fn(params)
    e_baseline = np.asarray(res_baseline['err_hist'], dtype=float)
    print('Baseline 仿真完成，残差形状:', e_baseline.shape)

    print('运行 AI-init 仿真...')
    w_init_arr = np.asarray(W_AI_init, dtype=float)
    res_ai = cfxlms_with_init(time_axis, mgr, L, mu, x, d, w_init=w_init_arr, normalized_update=normalized_update, norm_epsilon=float(cfg.anc_norm_epsilon), verbose=False)
    e_ai = np.asarray(res_ai['err_hist'], dtype=float)
    print('AI-init 仿真完成，残差形状:', e_ai.shape)

    # 绘图比较
    window_samples = min(max(500, int(round(0.125 * cfg.fs))), int(len(time_axis) // 2))
    fs = int(cfg.fs)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
    axes = axes.flatten()
    colors = {'baseline': 'tab:red', 'ai': 'tab:blue'}
    linestyles = {'baseline': '--', 'ai': '-'}
    curves_baseline = []
    curves_ai = []
    t_ref = None
    for m in range(min(e_baseline.shape[1], 3)):
        t_b, db_b = rolling_mse_db(e_baseline[:, m], fs, window_samples=window_samples)
        t_a, db_a = rolling_mse_db(e_ai[:, m], fs, window_samples=window_samples)
        curves_baseline.append(db_b)
        curves_ai.append(db_a)
        t_ref = t_b
        ax = axes[m]
        ax.plot(t_b, db_b, label='Baseline (Zero-Init)', color=colors['baseline'], linestyle=linestyles['baseline'])
        ax.plot(t_a, db_a, label='Proposed (AI-Init)', color=colors['ai'], linestyle=linestyles['ai'])
        ax.set_title(f'Err Mic {int(sampler.err_ids[m])}')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Error Power (dB)')
        ax.grid(True, alpha=0.3)
        if m == 0:
            ax.legend(loc='best')

    if curves_baseline and curves_ai:
        mean_b = np.mean(np.vstack(curves_baseline), axis=0)
        mean_a = np.mean(np.vstack(curves_ai), axis=0)
        axm = axes[3]
        axm.plot(t_ref, mean_b, label='Mean Baseline', color='gray', linestyle='--', linewidth=1.6)
        axm.plot(t_ref, mean_a, label='Mean AI', color='tab:green', linestyle='-', linewidth=1.8)
        axm.set_title('Mean NR (3 mics)')
        axm.set_xlabel('Time (s)')
        axm.set_ylabel('Error Power (dB)')
        axm.grid(True, alpha=0.3)
        axm.legend(loc='best')

    plt.suptitle('Baseline vs AI-init: Rolling Error Power (dB)')
    if args.save_prefix:
        fig.savefig(f"{args.save_prefix}_rolling_error.png")
    if not args.no_show:
        plt.show()
    plt.close(fig)

    # 房间布局可视化
    try:
        fig3 = plt.figure(figsize=(8, 6))
        ax3 = fig3.add_subplot(111, projection='3d')
        mgr.plot_layout(ax=ax3)
        plt.title('Test Room Layout (3D)')
        if args.save_prefix:
            fig3.savefig(f"{args.save_prefix}_layout_3d.png")
        if not args.no_show:
            plt.show()
        plt.close(fig3)

        fig2 = plt.figure(figsize=(7, 6))
        ax2 = fig2.add_subplot(111)
        mgr.plot_layout_2d(ax=ax2)
        plt.title('Test Room Layout (Top View)')
        if args.save_prefix:
            fig2.savefig(f"{args.save_prefix}_layout_top.png")
        if not args.no_show:
            plt.show()
        plt.close(fig2)
    except Exception:
        print('警告：房间布局绘制失败（可能是 matplotlib/环境问题）')


if __name__ == '__main__':
    main()
