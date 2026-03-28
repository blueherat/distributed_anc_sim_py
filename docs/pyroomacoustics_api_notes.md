# pyroomacoustics API 速查与工程映射

本文针对本仓库 ANC 迁移场景，整理 pyroomacoustics 中更成熟、常用且稳定的 API。

## 1. 房间与 RIR 建模

### 1.1 ShoeBox 房间模型

- 入口类：`pyroomacoustics.ShoeBox`
- 典型参数：
  - `p`: 房间尺寸 `[Lx, Ly, Lz]`
  - `fs`: 采样率
  - `max_order`: 镜像源阶数（等价 MATLAB `ImageSourceOrder`）
  - `absorption`: 墙面吸收系数（标量或材料对象）

示例：

```python
import pyroomacoustics as pra
room = pra.ShoeBox(
    p=[5.0, 5.0, 5.0],
    fs=4000,
    max_order=2,
    absorption=0.5,
)
```

### 1.2 添加声源与麦克风阵列

- 声源：`room.add_source(position)`
- 麦克风阵列：`pra.MicrophoneArray(mic_positions, fs)`
- 将阵列加入房间：`room.add_microphone_array(mic_array)`

注意：`mic_positions` 形状是 `(dim, n_mics)`，不是 `(n_mics, dim)`。

### 1.3 计算 RIR

- 调用：`room.compute_rir()`
- 读取：`room.rir[mic_index][source_index]`

建议：

- pyroomacoustics 返回的是“变长”RIR列表。为了匹配 MATLAB 矩阵式处理，建议统一零填充到同长度。
- 在算法迭代里频繁读取 RIR 时，务必预缓存到字典（本仓库 Python 版已经这么做）。

## 2. 声学参数控制

### 2.1 声速

可通过常量系统设置：

```python
import pyroomacoustics as pra
pra.constants.set("c", 343.0)
```

### 2.2 吸收与材料

- 快速建模：直接用 `absorption=0.5`
- 细化建模：使用 `pra.Material`，为不同频带设置吸收

对于 ANC 算法调参与速度测试，建议先用标量吸收快速迭代。

## 3. 成熟的实用函数（建议优先使用）

### 3.1 STFT 与滤波器工具

- `pyroomacoustics.transform.STFT`
- `pyroomacoustics.adaptive` 下的自适应滤波工具

适用：块处理 ANC、频域滤波器更新、实时处理原型。

### 3.2 波束形成相关

- `pyroomacoustics.beamforming`

适用：多麦前端增强 + ANC 级联实验。

### 3.3 指标评估

- `pyroomacoustics.metrics`

适用：语音场景的 STOI、PESQ 相关实验（若后续扩展到语音噪声控制）。

## 4. 在本项目中的映射关系

- MATLAB `acousticRoomResponse` -> Python `ShoeBox + compute_rir`
- MATLAB `RIRManager` 字典缓存 -> Python `RIRManager` 的 `primary_rirs/reference_rirs/secondary_rirs`
- 多参考麦克风输入：
  - MATLAB `calculateReferenceSignal` -> Python `calculate_reference_signal`
  - 两端都采用 `source -> primary->reference RIR 卷积` 生成参考通道

## 5. 白噪声与绘图建议（优先 Python 库）

### 5.1 白噪声

- 已在本项目用 `numpy` 实现频域相位随机带限白噪声（`py_anc.utils.wn_gen`）。
- 若你需要更丰富谱形控制，可用 `scipy.signal` 设计滤波器后对白噪声滤波。

### 5.2 绘图

- 时域、频域、NSE：建议 `matplotlib + scipy.signal.welch`
- 本项目已有：
  - `py_anc.viz.plot_results`
  - `py_anc.viz.plot_tap_weights`

## 6. 工程实践建议

- RIR 计算和算法迭代分离：先 `build()` 缓存后再跑算法，避免重复建模。
- 统一 ID 到索引映射：`ref_idx/sec_idx/err_idx`，可避免多节点拓扑下索引错误。
- 先用短时长做算法等价验证，再用长时长做速度基准。
- 若后续要做更大规模节点数，可优先考虑：
  - NumPy 向量化
  - Numba 加速
  - 分块频域实现（STFT）
