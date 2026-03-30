# %% [markdown]
# # 评估 Notebook：AI 初始化 vs 零初始化 MIMO-FxLMS 收敛对比（evaluate_anc.ipynb）
#
# 本 Notebook 在一个 **全新（未见）** 的仿真房间中，比较两种初始化下的 MIMO-FxLMS 收敛速度：
# - Baseline: 传统零初始化（或全零滤波器）
# - Proposed: 使用预训练的 `MIMO_Conditioned_ANCNet` 进行一次前向预测，得到 `W_AI_init` 作为初始化，然后运行 FxLMS
#
# 主要步骤：
# 1. 使用与数据集构建一致的采样分布（但不同随机种子）生成一个新房间并构建 RIR（确保在训练分布内但未复现训练样本）。
# 2. 生成同一段白噪声激励，提取参考麦克风前 50 ms 信号并按训练时的 FeatureProcessor 计算 `gcc_phat` 和 `psd_features`。
# 3. 使用训练时的全局 SVD 基（从 HDF5 读取 `S_components` / `S_mean` 与 `W_components` / `W_mean`）对 `S_full_test` 做投影，得到 `x_s`（32-d）。
# 4. 加载 `best_mimo_anc_net.pth`，前向获得 `c_pred`，重构 `W_AI_init`（shape [sec,ref,L]）。
# 5. 在同一条噪声 / 时间轴上分别运行： (a) 零初始化 FxLMS，(b) AI 初始化 FxLMS；记录残差波形 `e(t)`。
# 6. 计算滑动窗口 MSE -> dB（短时均方）并绘图比较收敛曲线，同时绘制房间布局。
#
# **依赖**：`pyroomacoustics`, `numpy`, `torch`, `matplotlib`, `h5py`（若缺少请先在环境中安装）。
#
# 注意：该 Notebook 严格沿用了数据集脚本 `build_cfxlms_qc_dataset.py` 中的 Feature/几何采样与 SVD 约定，保证与训练时的特征维度完全一致。

# %%
# 环境与路径准备
import sys
from pathlib import Path
import importlib.util
import numpy as np
import h5py
import matplotlib.pyplot as plt
import torch

# 确保项目根在 sys.path，以能导入本仓库内的包（py_anc 等）
ROOT = Path.cwd()
if (ROOT / 'python_impl').exists():
    sys.path.insert(0, str(ROOT / 'python_impl'))
else:
    # 尝试上一级（若 Notebook 在 python_impl 子目录打开时）
    sys.path.insert(0, str(ROOT))

print('工作目录:', ROOT)
print('Python 环境 torch:', torch.__version__)

# %%
# 自动寻找项目根并设置 sys.path 与工作目录（避免相对路径找不到脚本）
from pathlib import Path
import sys
import os

def find_repo_root(marker='python_impl/python_scripts/build_cfxlms_qc_dataset.py'):
    p = Path.cwd()
    for d in [p] + list(p.parents):
        if (d / marker).exists():
            return d
    raise FileNotFoundError(f'未在当前目录及父目录中找到 {marker}')

ROOT = find_repo_root()
print('检测到项目根:', ROOT)
# 将项目根加入 sys.path 以便相对导入；并切换当前工作目录以使相对路径有效
sys.path.insert(0, str(ROOT / 'python_impl'))
os.chdir(str(ROOT))
print('已切换工作目录到:', Path.cwd())

# %%
# 载入 dataset builder 中的配置与采样器（以保证与训练时一致的分布与特征计算器）
import importlib.util, types
builder_path = Path('python_impl') / 'python_scripts' / 'build_cfxlms_qc_dataset.py'
if not builder_path.exists():
    raise FileNotFoundError(f'无法找到构建脚本: {builder_path}，请确认在项目根下运行本 Notebook')
spec = importlib.util.spec_from_file_location('build_cfx', str(builder_path))
build_mod = importlib.util.module_from_spec(spec)
import sys
# 在执行模块前把 module 注册到 sys.modules，避免 dataclass 在执行期找不到模块而报错
sys.modules[spec.name] = build_mod
spec.loader.exec_module(build_mod)

# 现在我们可以使用 DatasetBuildConfig, AcousticScenarioSampler, FeatureProcessor 等类
cfg = build_mod.DatasetBuildConfig()
print('数据集配置载入，fs=', cfg.fs, 'filter_len=', cfg.filter_len, 'gcc_len=', cfg.gcc_truncated_len)

# %%
# 采样并构建一个新的（未见）测试房间——使用不同的随机种子以避免与训练集重复
from numpy.random import default_rng
seed_new = 20260330  # 可改为任意不同于训练的种子以保证未见性
rng = default_rng(seed_new)
sampler = build_mod.AcousticScenarioSampler(cfg, rng)
sampled = sampler.sample()  # 可能抛出 QCError，如果抛出可换 seed 重试
mgr = sampler.build_manager(sampled)
mgr.build(verbose=False)
print('采样完成。房间大小:', sampled['room_size'], '声速:', sampled['sound_speed'])
print('参考麦克风位置:', sampled['ref_positions'])
print('次级扬声器位置:', sampled['sec_positions'])
print('误差麦克风位置:', sampled['err_positions'])

# %%
# 使用与数据集一致的带限白噪声作为激励，并生成参考/期望信号
from py_anc.utils import wn_gen

duration_s = 2.0  # 仿真时长（s），可调整，但下面的特征只取前 50 ms

# %%
# 生成噪声、计算 reference (x) 与 desired (d)
from py_anc.utils import wn_gen

noise_rng = np.random.default_rng(seed_new + 1)
noise, t = wn_gen(fs=cfg.fs, duration=duration_s, f_low=cfg.f_low, f_high=cfg.f_high, rng=noise_rng)
time_axis = t[:, 0].astype(float)

# 归一化列（与数据集处理一致）
def _normalize_columns(x_arr: np.ndarray) -> np.ndarray:
    denom = np.max(np.abs(x_arr), axis=0, keepdims=True)
    denom = np.where(denom < np.finfo(float).eps, 1.0, denom)
    return x_arr / denom

# 与 spotcheck 保持一致：先归一化激励，再计算 d/x
source_signal = _normalize_columns(noise)

# 按 RIRManager API 生成 d 和 x（已在 mgr 中注册了设备）
d = mgr.calculate_desired_signal(source_signal, len(time_axis))
x = mgr.calculate_reference_signal(source_signal, len(time_axis))
# 再对参考通道做列归一化以保证特征尺度一致
x = _normalize_columns(x)
print('信号长度:', len(time_axis), '参考通道数:', x.shape[1], '误差通道数:', d.shape[1])

# %%
# 提取前 50 ms 的参考麦克风波形（与数据集一致的窗口长度）
ref_samples = int(cfg.ref_window_samples)
fs = int(cfg.fs)
# 必须偏移 0.5 秒！
start_idx = _resolve_feature_window_start(len(x), cfg)
X_ref_test = x[start_idx : start_idx + ref_samples, : cfg.num_nodes].T.astype(np.float32)
print('X_ref_test shape:', X_ref_test.shape)

# 构建 S_full_test: 次级通路完整矩阵 [sec, err, rir_len]，并按 cfg.rir_store_len 截断/补零
rir_len = int(cfg.rir_store_len)
n_nodes = int(cfg.num_nodes)
S_full_test = np.zeros((n_nodes, n_nodes, rir_len), dtype=np.float32)
for i, sec_id in enumerate(sampler.sec_ids):
    for j, err_id in enumerate(sampler.err_ids):
        r = np.asarray(mgr.get_secondary_rir(int(sec_id), int(err_id)), dtype=float)
        keep = min(rir_len, r.size)
        S_full_test[i, j, :keep] = r[:keep].astype(np.float32)

# primary/ P_full_test（primary->err）
P_full_test = np.zeros((1, n_nodes, rir_len), dtype=np.float32)
pri_id = int(sampler.source_id)
for j, err_id in enumerate(sampler.err_ids):
    r = np.asarray(mgr.get_primary_rir(pri_id, int(err_id)), dtype=float)
    keep = min(rir_len, r.size)
    P_full_test[0, j, :keep] = r[:keep].astype(np.float32)

# 计算理论参考 W_opt（通过局部解或使用构建器的快速返回，若想要严格按最优解需要求解 Wiener 方程；这里我们不用于仿真，仅保存作参考）
# 训练脚本里 W_opt 是由 ANCDatasetBuilder.evaluate_anc 计算并写入 HDF5；我们这里不重复推导，只保留占位
W_opt_test = None  # 如需，可以通过运行 ANCDatasetBuilder.evaluate_anc 获得（开销较大）
print('S_full_test shape:', S_full_test.shape, 'P_full_test shape:', P_full_test.shape)

# %%
# 从已有 HDF5 数据集中加载全局 SVD 基底（用于将 S_full_test 投影为 32-d 条件向量）
h5_path = Path('python_impl') / 'python_scripts' / 'cfxlms_qc_dataset_cross_500_seeded.h5'
if not h5_path.exists():
    raise FileNotFoundError(f'需要 HDF5 数据集以读取全局 SVD 基底: {h5_path}，请先生成或提供该文件')
with h5py.File(str(h5_path), 'r') as hf:
    s_comp = np.asarray(hf['processed/global_svd/S_components'], dtype=np.float32)
    s_mean = np.asarray(hf['processed/global_svd/S_mean'], dtype=np.float32)
    w_comp = np.asarray(hf['processed/global_svd/W_components'], dtype=np.float32)
    w_mean = np.asarray(hf['processed/global_svd/W_mean'], dtype=np.float32)
print('Loaded S_components shape:', s_comp.shape, 'W_components shape:', w_comp.shape)


def _resolve_feature_window_start(total_samples: int, cfg, preferred_idx: int | None = None) -> int:
    window = int(cfg.ref_window_samples)
    latest_start = max(total_samples - window, 0)
    start_min = int(round(float(cfg.warmup_start_s_min) * float(cfg.fs)))
    start_max = min(int(round(float(cfg.warmup_start_s_max) * float(cfg.fs))), latest_start)
    if preferred_idx is not None:
        return max(0, min(int(preferred_idx), latest_start))
    if start_max < start_min:
        return 0
    return int((start_min + start_max) // 2)


def _standardize_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean) / std).astype(np.float32)


def _destandardize_array(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (x * std + mean).astype(np.float32)


def _build_identity_norm_stats(x_p_shape: tuple[int, ...], x_s_dim: int, y_dim: int):
    return {
        'x_p': {'mean': np.zeros(x_p_shape, dtype=np.float32), 'std': np.ones(x_p_shape, dtype=np.float32)},
        'x_s': {'mean': np.zeros((x_s_dim,), dtype=np.float32), 'std': np.ones((x_s_dim,), dtype=np.float32)},
        'y_c': {'mean': np.zeros((y_dim,), dtype=np.float32), 'std': np.ones((y_dim,), dtype=np.float32)},
    }


def _room_color_series(count: int, cmap_name: str = 'tab20'):
    cmap = plt.get_cmap(cmap_name)
    if count <= 0:
        return []
    if count == 1:
        return [cmap(0.0)]
    return [cmap(i / max(count - 1, 1)) for i in range(count)]

# %%
# 使用与数据集一致的 FeatureProcessor 计算 GCC-PHAT 与 PSD（保证维度一致）
feat_proc = build_mod.FeatureProcessor(cfg)
gcc = feat_proc.compute_gcc_phat(X_ref_test)  # shape (3, 129)
psd = feat_proc.compute_psd_features(X_ref_test[0])  # shape (129,)
# 合并为训练期望的 x_p 形状 (1, 4, 129)
x_p = np.vstack([gcc, psd.reshape(1, -1)]).astype(np.float32)  # (4,129)
x_p_t = torch.from_numpy(x_p).unsqueeze(0)  # (1,4,129)

# 将 S_full_test 扁平化并投影到 S_components 得到物理条件向量 (1,32)
s_flat = S_full_test.reshape(-1).astype(np.float32)  # shape (s_feature_dim,)
s_centered = (s_flat - s_mean).astype(np.float32)
# s_comp shape: (n_components, s_feature_dim) -> proj: (s_feature_dim,) @ (s_feature_dim, n_comp) -> (n_comp,)
s_coeff = s_centered @ s_comp.T  # (32,)
x_s_t = torch.from_numpy(s_coeff.astype(np.float32)).unsqueeze(0)  # (1,32)

print('x_p_t shape:', x_p_t.shape, 'x_s_t shape:', x_s_t.shape)

# %% [markdown]
# ## 载入预训练模型并做一次前向推理，得到 AI 初始化的控制滤波器 W_AI_init
# 模型代码与训练时一致（为避免导入训练脚本带来的副作用，这里直接复刻模型定义）。

# %%
import torch.nn as nn
import torch.nn.functional as F

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

# 加载权重（查找常见路径）
possible_paths = [Path('best_mimo_anc_net_500room_200epoch.pth'), Path('python_impl') / 'notebooks' / 'best_mimo_anc_net_500room_200epoch.pth']
model_path = None
for p in possible_paths:
    if p.exists():
        model_path = p
        break
if model_path is None:
    raise FileNotFoundError('找不到模型文件 best_mimo_anc_net_500room_200epoch.pth，请先在训练目录保存该文件或把路径放到工作目录。')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MIMO_Conditioned_ANCNet().to(device)
state = torch.load(str(model_path), map_location=device)
state_dict = state['model_state_dict'] if isinstance(state, dict) and 'model_state_dict' in state else state
norm_stats = state.get('norm_stats') if isinstance(state, dict) else None
if norm_stats is None:
    norm_stats = _build_identity_norm_stats((4, int(cfg.gcc_truncated_len)), int(s_comp.shape[0]), int(w_comp.shape[0]))
model.load_state_dict(state_dict)
model.eval()
print('已加载模型', model_path, 'device:', device)

# %%
# 前向预测并物理解码得到 W_AI_init（shape: sec x ref x L）
x_p_model = _standardize_array(x_p, norm_stats['x_p']['mean'], norm_stats['x_p']['std'])
x_s_model = _standardize_array(s_coeff.astype(np.float32), norm_stats['x_s']['mean'], norm_stats['x_s']['std'])
x_p_t = torch.from_numpy(x_p_model).unsqueeze(0)
x_s_t = torch.from_numpy(x_s_model).unsqueeze(0)

with torch.no_grad():
    xp = x_p_t.to(device=device, dtype=torch.float32)
    xs = x_s_t.to(device=device, dtype=torch.float32)
    c_pred = model(xp, xs)  # (1,32)
    # 将全局 W_components 与 W_mean 转为 torch tensor（保证在 CPU 上）
    c_pred_np = c_pred.cpu().numpy().reshape(-1)
    c_pred_denorm = _destandardize_array(c_pred_np, norm_stats['y_c']['mean'], norm_stats['y_c']['std'])
    W_pred_flat_np = c_pred_denorm @ w_comp + w_mean
    W_AI_init = W_pred_flat_np.reshape(cfg.num_nodes, cfg.num_nodes, cfg.filter_len)  # (sec,ref,L)
print("诊断 c_pred (AI 预测的 32 维系数):")
print(c_pred.cpu().numpy())
print('W_AI_init shape:', W_AI_init.shape)

# %%
# 对比一下 AI 预测的 W 和真实的 W_opt 到底差多远？
print("AI 预测的 W_init 平均能量:", np.mean(W_AI_init**2))
print("AI 预测的 W_init 最大值:", np.max(np.abs(W_AI_init)))

# (如果你能从 mgr 里算出测试房间的真实 W_opt)
# print("真实的 W_opt 最大值:", np.max(np.abs(W_opt_true)))

# 画一张前 100 个采样点的对比图
plt.figure()
plt.plot(W_AI_init[0, 0, :100], label='AI Pred (Sec1-Ref1)')
plt.title("Check Filter Waveform")
plt.legend()
plt.show()

# %% [markdown]
# ## 仿真对比：实现支持初始 W 的 CFxLMS 并运行 Baseline/AI 两次仿真
# 接下来我们调用仓库内的 `py_anc.algorithms.cfxlms.cfxlms`（作为 baseline），并实现一个带初始权重的 `cfxlms_with_init` 供 AI 初始化使用。

# %%
from py_anc.algorithms import cfxlms as cfxlms_fn

# 复制仓库中辅助函数（用于将 W_full(sec,ref,L) 转为 CFxLMS 使用的形状 [L,ref,sec]）
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

# 可复用的 cfxlms_with_init（复制自 py_anc.algorithms.cfxlms 并添加可选 w_init 参数）
def cfxlms_with_init(time, rir_manager, L, mu, reference_signal, desired_signal, w_init=None, normalized_update=False, norm_epsilon=1.0e-8, verbose=False):
    # 参数与 py_anc.algorithms.cfxlms 一致，返回同样的结构{'err_hist','filter_coeffs'}
    x = np.asarray(reference_signal, dtype=float)
    d = np.asarray(desired_signal, dtype=float)
    key_sec_spks = list(rir_manager.secondary_speakers.keys())
    key_err_mics = list(rir_manager.error_microphones.keys())
    num_ref_mics = len(rir_manager.reference_microphones)
    num_sec_spks = len(key_sec_spks)
    num_err_mics = len(key_err_mics)
    n_samples = len(time)

    # 缓存次级路径
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

    # 初始化 w：若提供 w_init（sec,ref,L），则转换；否则零初始化
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

# 选取相同的 mu 与参数以保证公平比较
mu = float(cfg.mu_candidates[0] if len(cfg.mu_candidates) else 1.0e-4)
L = int(cfg.filter_len)
normalized_update = bool(cfg.anc_normalized_update)

# 运行 baseline（零初始化）
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
res_baseline = cfxlms_fn(params)
e_baseline = np.asarray(res_baseline['err_hist'], dtype=float)
print('Baseline 仿真完成，残差形状:', e_baseline.shape)

# 运行 AI 初始化（使用 W_AI_init）
w_init_arr = np.asarray(W_AI_init, dtype=float)  # shape: (sec,ref,L)
res_ai = cfxlms_with_init(time_axis, mgr, L, mu, x, d, w_init=w_init_arr, normalized_update=normalized_update, norm_epsilon=float(cfg.anc_norm_epsilon), verbose=False)
e_ai = np.asarray(res_ai['err_hist'], dtype=float)
print('AI-init 仿真完成，残差形状:', e_ai.shape)

# %%
# 计算滑动窗口的短时能量并转换为 dB（rolling MSE -> dB）
def rolling_mse_db(sig: np.ndarray, fs: int, window_samples: int = 500):
    # sig: 1D 时间序列
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

# 选择窗口大小：优先 500~1000 样本（这里以 500 为例）
window_samples = min(max(500, int(round(0.125 * cfg.fs))), int(len(time_axis) // 2))
print('window_samples=', window_samples, 'window_s=', window_samples / cfg.fs)

# 为三个误差麦克风绘制各自的 NR 曲线（Baseline vs AI）
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

# 平均曲线
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
plt.show()

# %%
# 从 HDF5 中选取训练集中的若干房间并逐一评估：
# 与 spotcheck_qc_dataset_rooms.py 保持一致的流程：
# 1) 复现噪声（并归一化 source_signal）
# 2) 若存在 raw/W_full，则直接读取该切片并返回 (sec,ref,L)
# 3) 使用与 spotcheck 相同的定常仿真函数（不进行自适应更新）来计算 e_h5
# 4) 对比 Baseline (adaptive zero-init)、HDF5 固定 W、AI-init (adaptive starting from AI W)、HDF5-init(adaptive starting from stored W)

h5_rooms = 'first:20'  # 根据需要修改
print('h5_rooms=', h5_rooms)
import h5py
from py_anc.utils import wn_gen
from py_anc.algorithms import cfxlms as cfxlms_fn


def _parse_indices(s: str, n_rooms: int) -> list:
    s = s.strip()
    if s.startswith('first:'):
        k = int(s.split(':', 1)[1])
        return list(range(min(k, n_rooms)))
    parts = []
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        if '-' in token:
            a, b = token.split('-', 1)
            parts.extend(list(range(int(a), int(b) + 1)))
        else:
            parts.append(int(token))
    seen = set(); out = []
    for v in parts:
        if v < 0 or v >= n_rooms:
            continue
        if v not in seen:
            seen.add(v); out.append(v)
    return out


def _load_w_full_from_hf(hf, idx, n_nodes, filter_len):
    """直接读取 raw/W_full 的切片并返回 (sec,ref,L)。与 spotcheck 保持一致。"""
    ds_name = 'raw/W_full'
    if ds_name not in hf:
        raise FileNotFoundError("HDF5 中未找到 'raw/W_full' 数据集。请使用包含完整 W_full 的 HDF5 文件。")
    arr = np.asarray(hf[ds_name][idx], dtype=float)
    print(f"raw/W_full[{idx}] 切片 shape = {arr.shape}, dtype={arr.dtype}")
    if arr.ndim != 3:
        raise RuntimeError(f"Room {idx}: raw/W_full 切片形状 {arr.shape} 非三维 (sec,ref,L)。请使用与构建脚本一致的 HDF5 文件。")
    return arr.copy()


def _simulate_error_with_fixed_w(
    mgr,
    reference_signal: np.ndarray,
    desired_signal: np.ndarray,
    w: np.ndarray,
    sec_ids: np.ndarray,
    err_ids: np.ndarray,
) -> np.ndarray:
    """从 spotcheck 复制的定常仿真：输入 w 为 (L, ref, sec)，返回 e(t) (n_samples, n_err)。"""
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

# 准备收集 AI-init 与 HDF5-opt（跨房间对比用）
ai_init_list = []
ai_room_idxs = []
h5_init_list = []
h5_room_idxs = []

with h5py.File(str(h5_path), 'r') as hf:
    # 推断样本数（必须以 raw/W_full 为准）
    if 'raw/W_full' in hf:
        n_rooms = int(hf['raw/W_full'].shape[0])
    else:
        raise FileNotFoundError("HDF5 中未找到 'raw/W_full' 数据集。请使用包含完整 W_full 的 HDF5 文件。")

    indices = _parse_indices(h5_rooms, n_rooms)
    if not indices:
        raise ValueError('未解析到有效的房间索引，请修改 h5_rooms 变量。')
    print('将评估以下 HDF5 房间索引:', indices)

    for idx in indices:
        print(f'--- 评估房间 idx={idx} ---')
        sampled = {}
        rp = 'raw/room_params'
        sampled['room_size'] = np.asarray(hf[f'{rp}/room_size'][idx], dtype=float)
        if f'{rp}/source_position' in hf:
            sampled['source_pos'] = np.asarray(hf[f'{rp}/source_position'][idx], dtype=float)
        elif f'{rp}/source_pos' in hf:
            sampled['source_pos'] = np.asarray(hf[f'{rp}/source_pos'][idx], dtype=float)
        sampled['ref_positions'] = np.asarray(hf[f'{rp}/ref_positions'][idx], dtype=float)
        sampled['sec_positions'] = np.asarray(hf[f'{rp}/sec_positions'][idx], dtype=float)
        sampled['err_positions'] = np.asarray(hf[f'{rp}/err_positions'][idx], dtype=float)
        sampled['sound_speed'] = float(hf[f'{rp}/sound_speed'][idx]) if f'{rp}/sound_speed' in hf else float(cfg.sound_speed_min)
        sampled['absorption'] = float(hf[f'{rp}/material_absorption'][idx]) if f'{rp}/material_absorption' in hf else float(cfg.absorption_min)
        sampled['image_order'] = int(hf[f'{rp}/image_source_order'][idx]) if f'{rp}/image_source_order' in hf else int(cfg.image_order_choices[0])

        # 使用 sampler.build_manager/sample 构建 mgr（与采样器保持一致）
        mgr_i = sampler.build_manager(sampled)
        mgr_i.build(verbose=False)

        source_seed = int(hf['raw/qc_metrics/source_seed'][idx]) if 'raw/qc_metrics/source_seed' in hf else int(20260330 + idx)
        noise, t = wn_gen(fs=cfg.fs, duration=cfg.noise_duration_s, f_low=cfg.f_low, f_high=cfg.f_high, rng=np.random.default_rng(int(source_seed)))
        time_axis_i = t[:, 0].astype(float)

        # 与 spotcheck 保持一致：先归一化激励，再计算 d/x
        source_signal = _normalize_columns(noise)

        d_i = mgr_i.calculate_desired_signal(source_signal, len(time_axis_i))
        x_i = mgr_i.calculate_reference_signal(source_signal, len(time_axis_i))
        x_i = _normalize_columns(x_i)
        warmup_idx_i = int(hf['raw/qc_metrics/warmup_start_index'][idx]) if 'raw/qc_metrics/warmup_start_index' in hf else None

        # 特征（优先使用 processed）
        if 'processed/gcc_phat' in hf and 'processed/psd_features' in hf:
            gcc_i = np.asarray(hf['processed/gcc_phat'][idx], dtype=np.float32)
            psd_i = np.asarray(hf['processed/psd_features'][idx], dtype=np.float32)
        else:
            feat_proc = build_mod.FeatureProcessor(cfg)
            ref_samples = int(cfg.ref_window_samples)
            start_idx_i = _resolve_feature_window_start(len(x_i), cfg, preferred_idx=warmup_idx_i)
            X_ref_test_i = x_i[start_idx_i : start_idx_i + ref_samples, : cfg.num_nodes].T.astype(np.float32)
            gcc_i = feat_proc.compute_gcc_phat(X_ref_test_i)
            psd_i = feat_proc.compute_psd_features(X_ref_test_i[0])
        x_p_i = np.vstack([gcc_i, psd_i.reshape(1, -1)]).astype(np.float32)
        x_p_model_i = _standardize_array(x_p_i, norm_stats['x_p']['mean'], norm_stats['x_p']['std'])
        x_p_t_i = torch.from_numpy(x_p_model_i).unsqueeze(0)

        # 条件向量
        if 'processed/S_pca_coeffs' in hf:
            s_coeff_i = np.asarray(hf['processed/S_pca_coeffs'][idx], dtype=np.float32)
        else:
            if 'raw/S_paths_full' in hf:
                s_flat_i = np.asarray(hf['raw/S_paths_full'][idx], dtype=np.float32).reshape(-1)
            else:
                s_flat_i = np.asarray(hf['raw/S_paths'][idx], dtype=np.float32).reshape(-1)
            s_centered_i = (s_flat_i - s_mean).astype(np.float32)
            s_coeff_i = s_centered_i @ s_comp.T
        x_s_model_i = _standardize_array(s_coeff_i.astype(np.float32), norm_stats['x_s']['mean'], norm_stats['x_s']['std'])
        x_s_t_i = torch.from_numpy(x_s_model_i).unsqueeze(0)

        # AI-init W
        with torch.no_grad():
            xp = x_p_t_i.to(device=device, dtype=torch.float32)
            xs = x_s_t_i.to(device=device, dtype=torch.float32)
            c_pred_i = model(xp, xs)
            c_pred_i_np = c_pred_i.cpu().numpy().reshape(-1)
            c_pred_i_denorm = _destandardize_array(c_pred_i_np, norm_stats['y_c']['mean'], norm_stats['y_c']['std'])
            W_pred_flat_np = c_pred_i_denorm @ w_comp + w_mean
            W_AI_init_i = W_pred_flat_np.reshape(cfg.num_nodes, cfg.num_nodes, cfg.filter_len)
            # 收集 AI-init（跨房间对比用）
            ai_init_list.append(W_AI_init_i.copy())
            ai_room_idxs.append(int(idx))

        # 强制从 HDF5 读取完整的 raw/W_full（不存在则抛错），并按 spotcheck 流程用定常仿真计算 e_h5
        try:
            W_h5_room = _load_w_full_from_hf(hf, idx, cfg.num_nodes, cfg.filter_len)
            print(f"Room {idx}: 成功从 HDF5 (raw/W_full) 读取 W，shape={W_h5_room.shape}")
            # 收集 HDF5-opt（跨房间对比用）
            h5_init_list.append(W_h5_room.copy())
            h5_room_idxs.append(int(idx))
        except Exception as e:
            raise RuntimeError(f"Room {idx}: 读取 raw/W_full 失败或格式不匹配: {e}")

        # 新增：绘制 AI-init 与 HDF5 最优（W_full）在选定 sec-ref 对的前 50 个抽头对比图
        try:
            sec_plot = 0
            ref_plot = 0
            sec_n, ref_n, L_n = W_h5_room.shape
            if sec_plot >= sec_n:
                sec_plot = 0
            if ref_plot >= ref_n:
                ref_plot = 0
            n_taps_plot = min(50, L_n)
            ai_plot = W_AI_init_i[sec_plot, ref_plot, :n_taps_plot]
            h5_plot = W_h5_room[sec_plot, ref_plot, :n_taps_plot]
            plt.figure(figsize=(6, 3))
            plt.plot(ai_plot, label='AI-init', color='tab:blue')
            plt.plot(h5_plot, label='HDF5-opt', color='tab:orange', linestyle='--')
            plt.title(f'Filter taps comparison sec={sec_plot} ref={ref_plot} (first {n_taps_plot})')
            plt.xlabel('Tap index')
            plt.ylabel('Amplitude')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        except Exception:
            pass

        # 运行三种初始化的仿真：Zero-init (adaptive baseline), HDF5-fixed (定常仿真), AI-init (adaptive starting from AI W)
        params_i = {
            'time': time_axis_i,
            'rir_manager': mgr_i,
            'L': int(cfg.filter_len),
            'mu': float(cfg.mu_candidates[0] if len(cfg.mu_candidates) else 1.0e-4),
            'reference_signal': x_i,
            'desired_signal': d_i,
            'verbose': False,
            'normalized_update': bool(cfg.anc_normalized_update),
            'norm_epsilon': float(cfg.anc_norm_epsilon),
        }

        # Zero-init (adaptive)
        res_zero = cfxlms_fn(params_i)
        e_zero = np.asarray(res_zero['err_hist'], dtype=float)

        # HDF5-fixed: 将 raw W_full 转为 (L,ref,sec) 并做定常仿真
        L_local = int(cfg.filter_len)
        num_ref_mics = x_i.shape[1]
        sec_ids_arr = np.array(list(mgr_i.secondary_speakers.keys()), dtype=int)
        err_ids_arr = np.array(list(mgr_i.error_microphones.keys()), dtype=int)
        w_fixed = _build_weight_tensor_from_w_full(W_h5_room, L_local, num_ref_mics, len(sec_ids_arr))
        e_h5 = _simulate_error_with_fixed_w(mgr_i, x_i, d_i, w_fixed, sec_ids_arr, err_ids_arr)

        # HDF5-init: adaptive starting from stored HDF5 optimal W
        res_h5_init = cfxlms_with_init(time_axis_i, mgr_i, params_i['L'], params_i['mu'], x_i, d_i, w_init=W_h5_room, normalized_update=params_i['normalized_update'], norm_epsilon=params_i['norm_epsilon'], verbose=False)
        e_h5_init = np.asarray(res_h5_init['err_hist'], dtype=float)

        # AI-init: adaptive starting from AI-predicted W
        res_ai = cfxlms_with_init(time_axis_i, mgr_i, params_i['L'], params_i['mu'], x_i, d_i, w_init=W_AI_init_i, normalized_update=params_i['normalized_update'], norm_epsilon=params_i['norm_epsilon'], verbose=False)
        e_ai = np.asarray(res_ai['err_hist'], dtype=float)

        # 初始 error dB（取滑动窗口的第一个点）——针对前 3 个误差麦克风
        fs_i = int(cfg.fs)
        win_local = min(max(500, int(round(0.125 * cfg.fs))), int(len(time_axis_i) // 2))
        def _initial_db_vals(e_arr, n_mics=3):
            out = []
            for m in range(min(e_arr.shape[1], n_mics)):
                t_m, db_m = rolling_mse_db(e_arr[:, m], fs_i, window_samples=win_local)
                if db_m.size > 0:
                    out.append(float(db_m[0]))
                else:
                    eps = np.finfo(float).eps
                    if e_arr.shape[0] >= win_local:
                        mse0 = float(np.mean(e_arr[:win_local, m] ** 2))
                    else:
                        mse0 = float(np.mean(e_arr[:, m] ** 2))
                    out.append(float(10.0 * np.log10(mse0 + eps)))
            while len(out) < n_mics:
                out.append(np.nan)
            return out

        init_db_zero = _initial_db_vals(e_zero, n_mics=3)
        init_db_h5 = _initial_db_vals(e_h5, n_mics=3)
        init_db_ai = _initial_db_vals(e_ai, n_mics=3)
        init_db_h5_init = _initial_db_vals(e_h5_init, n_mics=3)

        print('Room', idx)
        print('  Initial error dB HDF5-fixed :', init_db_h5)
        print('  Initial error dB Zero-init  :', init_db_zero)
        print('  Initial error dB AI-init    :', init_db_ai)
        print('  Initial error dB HDF5-init(adapt):', init_db_h5_init)

        # 在同一张图里绘制四者的初始 dB（每个误差麦克风一组条形）
        nplot = 3
        labels = [f'Mic {int(sampler.err_ids[m])}' if m < len(sampler.err_ids) else f'Mic{m}' for m in range(nplot)]
        x = np.arange(nplot)
        width = 0.2
        fig_bar, ax_bar = plt.subplots(figsize=(7, 3))
        h5_vals = np.array(init_db_h5, dtype=float)
        zero_vals = np.array(init_db_zero, dtype=float)
        ai_vals = np.array(init_db_ai, dtype=float)
        h5_init_vals = np.array(init_db_h5_init, dtype=float)
        ax_bar.bar(x - 1.5*width, h5_vals, width, label='HDF5-fixed', color='tab:orange')
        ax_bar.bar(x - 0.5*width, zero_vals, width, label='Zero-init', color='tab:red')
        ax_bar.bar(x + 0.5*width, ai_vals, width, label='AI-init', color='tab:blue')
        ax_bar.bar(x + 1.5*width, h5_init_vals, width, label='HDF5-init(adapt)', color='tab:purple')
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels)
        ax_bar.set_ylabel('Initial Error Power (dB)')
        ax_bar.set_title(f'Initial Error (first window) — Room {idx}')
        ax_bar.legend()
        for i in range(nplot):
            for j, arr in enumerate([h5_vals, zero_vals, ai_vals, h5_init_vals]):
                val = arr[i]
                if np.isfinite(val):
                    xpos = x[i] - 1.5*width + j * width
                    ax_bar.text(xpos, val, f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.show()

        # 原有的时间曲线绘图（含 HDF5-init adaptive 曲线）
        win = min(max(500, int(round(0.125 * cfg.fs))), int(len(time_axis_i) // 2))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True, sharex=True)
        axes = axes.flatten()
        colors = {'baseline': 'tab:red', 'ai': 'tab:blue', 'h5_fixed': 'tab:orange', 'h5_init': 'tab:purple'}
        for m in range(min(e_zero.shape[1], 3)):
            t_b, db_b = rolling_mse_db(e_zero[:, m], fs_i, window_samples=win)
            t_a, db_a = rolling_mse_db(e_ai[:, m], fs_i, window_samples=win)
            t_h, db_h = rolling_mse_db(e_h5[:, m], fs_i, window_samples=win)
            t_h_init, db_h_init = rolling_mse_db(e_h5_init[:, m], fs_i, window_samples=win)
            ax = axes[m]
            ax.plot(t_b, db_b, label='Zero-init', color=colors['baseline'], linestyle='--')
            ax.plot(t_a, db_a, label='AI-init', color=colors['ai'], linestyle='-')
            ax.plot(t_h, db_h, label='HDF5-fixed', color=colors['h5_fixed'], linestyle=':')
            ax.plot(t_h_init, db_h_init, label='HDF5-init(adapt)', color=colors['h5_init'], linestyle='-.')
            ax.set_title(f'Room {idx} Err Mic {int(sampler.err_ids[m])}')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Error Power (dB)')
            ax.grid(True, alpha=0.3)
            if m == 0:
                ax.legend(loc='best')

        mean_b = np.mean(np.vstack([rolling_mse_db(e_zero[:, m], fs_i, window_samples=win)[1] for m in range(min(e_zero.shape[1], 3))]), axis=0)
        mean_a = np.mean(np.vstack([rolling_mse_db(e_ai[:, m], fs_i, window_samples=win)[1] for m in range(min(e_ai.shape[1], 3))]), axis=0)
        mean_h = np.mean(np.vstack([rolling_mse_db(e_h5[:, m], fs_i, window_samples=win)[1] for m in range(min(e_h5.shape[1], 3))]), axis=0)
        mean_h_init = np.mean(np.vstack([rolling_mse_db(e_h5_init[:, m], fs_i, window_samples=win)[1] for m in range(min(e_h5_init.shape[1], 3))]), axis=0)
        axm = axes[3]
        axm.plot(t_b, mean_b, label='Mean Zero-init', color='gray', linestyle='--', linewidth=1.6)
        axm.plot(t_b, mean_a, label='Mean AI-init', color='tab:green', linestyle='-', linewidth=1.8)
        axm.plot(t_b, mean_h, label='Mean HDF5-fixed', color='tab:orange', linestyle=':')
        axm.plot(t_b, mean_h_init, label='Mean HDF5-init(adapt)', color='tab:purple', linestyle='-.')
        axm.set_title('Mean NR (3 mics)')
        axm.set_xlabel('Time (s)')
        axm.set_ylabel('Error Power (dB)')
        axm.grid(True, alpha=0.3)
        axm.legend(loc='best')

        plt.suptitle(f'Room {idx}: Zero / HDF5-fixed / AI-init / HDF5-init(adapt) Comparison')
        plt.show()

    # Cross-room AI-init taps overlay（若收集到多个房间的 AI-init，将进行跨房间比较）
    if len(ai_init_list) > 0:
        sec_plot = 0
        ref_plot = 0
        n_taps_cross = min(50, cfg.filter_len)
        plt.figure(figsize=(10, 5))
        for i_w, W in enumerate(ai_init_list):
            label = f'room {ai_room_idxs[i_w]}'
            plt.plot(W[sec_plot, ref_plot, :n_taps_cross], alpha=0.7, label=label)
        plt.title(f'Cross-room AI-init taps sec={sec_plot} ref={ref_plot} (first {n_taps_cross})')
        plt.xlabel('Tap index')
        plt.ylabel('Amplitude')
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.show()

        arr_ai = np.stack([w[sec_plot, ref_plot, :n_taps_cross] for w in ai_init_list], axis=0)
        mean_ai = arr_ai.mean(axis=0)
        std_ai = arr_ai.std(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(mean_ai, color='tab:blue', linewidth=1.6, label='AI-init mean')
        plt.fill_between(range(n_taps_cross), mean_ai - std_ai, mean_ai + std_ai, color='tab:blue', alpha=0.2, label='AI-init ±1 std')
        plt.title(f'Cross-room AI-init Mean ± Std sec={sec_plot} ref={ref_plot} (first {n_taps_cross})')
        plt.xlabel('Tap index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print('No AI-init collected across rooms to plot.')

    # Cross-room HDF5-opt taps overlay（若收集到多个房间的 HDF5-opt，将进行跨房间比较）
    if len(h5_init_list) > 0:
        sec_plot = 0
        ref_plot = 0
        n_taps_cross = min(50, cfg.filter_len)
        h5_colors = _room_color_series(len(h5_init_list), cmap_name='tab20')
        plt.figure(figsize=(10, 5))
        for i_w, W in enumerate(h5_init_list):
            label = f'room {h5_room_idxs[i_w]}'
            plt.plot(W[sec_plot, ref_plot, :n_taps_cross], alpha=0.75, label=label, color=h5_colors[i_w])
        plt.title(f'Cross-room HDF5-opt taps sec={sec_plot} ref={ref_plot} (first {n_taps_cross})')
        plt.xlabel('Tap index')
        plt.ylabel('Amplitude')
        plt.legend(ncol=2, fontsize=8)
        plt.grid(True, alpha=0.3)
        plt.show()

        arr_h5 = np.stack([w[sec_plot, ref_plot, :n_taps_cross] for w in h5_init_list], axis=0)
        mean_h5 = arr_h5.mean(axis=0)
        std_h5 = arr_h5.std(axis=0)
        plt.figure(figsize=(10, 4))
        plt.plot(mean_h5, color='tab:orange', linewidth=1.6, label='HDF5-opt mean')
        plt.fill_between(range(n_taps_cross), mean_h5 - std_h5, mean_h5 + std_h5, color='tab:orange', alpha=0.2, label='HDF5-opt ±1 std')
        plt.title(f'Cross-room HDF5-opt Mean ± Std sec={sec_plot} ref={ref_plot} (first {n_taps_cross})')
        plt.xlabel('Tap index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print('No HDF5-opt collected across rooms to plot.')

    # Combined: 同一张图展示 AI-init 与 HDF5-opt 的 Cross-room Mean ± Std
    if len(ai_init_list) > 0 and len(h5_init_list) > 0:
        sec_plot = 0
        ref_plot = 0
        n_taps_cross = min(50, cfg.filter_len)
        arr_ai = np.stack([w[sec_plot, ref_plot, :n_taps_cross] for w in ai_init_list], axis=0)
        arr_h5 = np.stack([w[sec_plot, ref_plot, :n_taps_cross] for w in h5_init_list], axis=0)
        mean_ai = arr_ai.mean(axis=0)
        std_ai = arr_ai.std(axis=0)
        mean_h5 = arr_h5.mean(axis=0)
        std_h5 = arr_h5.std(axis=0)

        plt.figure(figsize=(12, 5))
        h5_colors = _room_color_series(arr_h5.shape[0], cmap_name='tab20')
        ai_colors = _room_color_series(arr_ai.shape[0], cmap_name='tab10')
        # 绘制所有房间的曲线作为底色（可选，alpha小）
        for r in range(arr_h5.shape[0]):
            plt.plot(arr_h5[r], color=h5_colors[r], alpha=0.25)
        for r in range(arr_ai.shape[0]):
            plt.plot(arr_ai[r], color=ai_colors[r], alpha=0.20)

        # 绘制均值与 std 阴影
        plt.plot(mean_h5, color='tab:orange', linewidth=2.0, label='HDF5-opt mean')
        plt.fill_between(range(n_taps_cross), mean_h5 - std_h5, mean_h5 + std_h5, color='tab:orange', alpha=0.15)
        plt.plot(mean_ai, color='tab:blue', linewidth=2.0, label='AI-init mean')
        plt.fill_between(range(n_taps_cross), mean_ai - std_ai, mean_ai + std_ai, color='tab:blue', alpha=0.15)

        plt.title(f'Cross-room Mean ± Std Comparison (AI-init vs HDF5-opt) sec={sec_plot} ref={ref_plot} (first {n_taps_cross})')
        plt.xlabel('Tap index')
        plt.ylabel('Amplitude')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print('Not enough data to plot combined AI vs HDF5 cross-room comparison.')

# %%
# 房间布局绘制（3D）——使用 RIRManager 自带绘图方法，效果学术与清晰
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
mgr.plot_layout(ax=ax)
plt.title('Test Room Layout (3D)')
plt.show()

# 2D 顶视图（更清晰的平面分布）
fig2, ax2 = plt.subplots(figsize=(7, 6))
mgr.plot_layout_2d(ax=ax2)
plt.title('Test Room Layout (Top View)')
plt.show()

# %% [markdown]
# ---
# ### 说明与后续步骤
# - 本 Notebook 使用与数据集构建一致的特征与几何采样方法，确保 `x_p` 与 `x_s` 的维度/统计分布与训练阶段对齐。
# - 若需要在更多不同房间上做统计对比，可把上面的采样与仿真循环化（改变 `seed_new` 或多次调用 `sampler.sample()`），并收集 Mean NR / 收敛时间等指标。
# - 若你的环境缺少 `pyroomacoustics`，请安装：`pip install pyroomacoustics`。
#
# 需要我现在在当前 `myenv` 环境中运行该 Notebook 的全部 cell 吗？（注意：仿真仅2s，耗时较短，但若进行大量重复实验请确保有足够时间）

# %% [markdown]
# 两个问题：
# - 1.输入时间段任意改变，几乎不会影响输出
# - 2.在训练集中测试，ai-init和在测试集中一样，确实略高于零初始化，但差距不大。
#
# 可能的问题：
# - 1.训练时没有归一化，导致网络只会求平均值（这种情况怎么排查）理论上确实应该各不相同，但是网络输出几乎却一模一样。
# - 2.loss本身不合理，导致网络虽然loss很低，但并没有学到有用的特征（这种情况怎么排查）
# %% [markdown]
# **2026-03-30 修正说明**
# - 评估 notebook 现在会优先读取训练 checkpoint 里保存的 `norm_stats`，先对 `x_p` / `x_s` 做和训练一致的标准化，再把网络输出的 `y_c` 反标准化后重构 `W_AI_init`，避免训练和评估使用不同尺度。
# - 新房间特征窗口不再硬编码为 `0.5 s`，而是改成和数据集构建脚本一致的 warmup 合法区间；对 HDF5 房间回退重算特征时，也会优先复用保存下来的 `warmup_start_index`。
# - 训练集房间的跨房间 `HDF5-opt (W_full)` 叠加图已改成按房间分配不同颜色；合并对比图中的 HDF5 背景曲线也同步改成多色，避免所有房间都挤成同一条橙色视觉编码。
