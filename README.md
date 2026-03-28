# Distributed ANC Simulation (MATLAB + Python)

该仓库同时包含 MATLAB 与 Python 两套分布式 ANC 仿真实现，当前已按实现语言拆分为两个主目录，便于独立开发与对照验证。

## 目录结构

```text
.
├── matlab_impl/
│   ├── +acoustics/
│   ├── +algorithms/
│   ├── +topology/
│   ├── +utils/
│   ├── +viz/
│   ├── test.m
│   ├── demo.m
│   └── run_equivalence_matlab.m
├── python_impl/
│   ├── py_anc/
│   │   ├── acoustics/
│   │   ├── algorithms/
│   │   ├── scenarios/
│   │   ├── topology/
│   │   ├── utils/
│   │   └── viz/
│   ├── python_scripts/
│   ├── test.py
│   ├── test_environment_builder.py
│   ├── verify_migration.py
│   └── requirements.txt
└── docs/
```

## 支持算法

- CFxLMS
- ADFxLMS
- ADFxLMS-BC
- Diff-FxLMS
- DCFxLMS
- CDFxLMS
- MGDFxLMS

## MATLAB 运行方式

### 1) 非 CLI 直接改参数测试（推荐）

在 MATLAB 中直接运行：

```matlab
run('matlab_impl/test.m')
```

说明：
- 在 `matlab_impl/test.m` 顶部直接修改算法列表与参数。
- 脚本已加入路径自举，可从仓库任意工作目录运行。

### 2) 全算法短时等价性批测

```matlab
run('matlab_impl/run_equivalence_matlab.m')
```

输出：
- `python_impl/python_scripts/equivalence_matlab_summary.json`

## Python 运行方式

先安装依赖：

```bash
pip install -r python_impl/requirements.txt
```

### 1) 非 CLI 直接改参数测试（推荐）

```bash
python python_impl/test.py
```

说明：
- 在 `python_impl/test.py` 顶部直接修改算法、步长、滤波器长度、房间与布局配置。
- 运行后会在 `python_impl/python_scripts/` 输出：
  - `layout_preview_test.png`（二维平面俯视图）
  - `last_run_summary_from_test.json`
  - 误差信号对比图与 tap 权重图

### 2) 场景构建接口与房间布局可视化测试

```bash
python python_impl/test_environment_builder.py
```

输出：
- `python_impl/python_scripts/layout_preview_environment_builder.png`（二维平面俯视图）
- `python_impl/python_scripts/scenario_preview_configs.json`

### 3) CLI 批量实验（可选）

```bash
python python_impl/python_scripts/run_experiment.py --algorithms "CFxLMS,ADFxLMS,ADFxLMS-BC,Diff-FxLMS,DCFxLMS,CDFxLMS,MGDFxLMS" --duration 2 --output-json "python_impl/python_scripts/equivalence_py_summary.json"
```

## Python 场景参数接口（用于训练数据构造）

位于 `python_impl/py_anc/scenarios/`：

- `RoomConfig`：房间尺寸、采样率、声速、反射阶数、吸收参数
- `NodeRadialLayout`：节点方位角、ref/sec/err 半径、z 偏移
- `ScenarioConfig`：主源位置、节点列表、设备 ID 起始值
- `build_manager_from_config`：由配置构建 RIRManager 与设备 ID
- `sample_asymmetric_scenario`：随机采样非对称布局
- `plot_layout_with_labels`：房间框架与设备标注图

布局约束：
- 已校验 `ref_radius < sec_radius < err_radius`。
- 通过半径层级保证“ref 最近、sec 次之、error 最外圈”。
- `sample_asymmetric_scenario` 默认启用节点最小间距约束（`min_inter_node_distance=0.35`，单位米），自动重采样避免节点簇拥过近。

示例（自定义最小间距）：

```python
scenario = sample_asymmetric_scenario(
  seed=2026,
  num_nodes=4,
  room=room_cfg,
  min_inter_node_distance=0.45,
)
```

关键实现说明：
- Python 布局预览默认采用二维平面图（X-Y），不再输出 3D 视图。
- Python RIR 构建默认开启分数延迟补偿（`RoomConfig.compensate_fractional_delay=True`），用于去除 pyroomacoustics 引入的固定群时延，保证 ANC 因果关系与 MATLAB 版本更一致。

## MATLAB -> Python 迁移正确性核验

### 推荐步骤

1. 运行 MATLAB 批测生成 `equivalence_matlab_summary.json`
2. 运行 Python 批测生成 `equivalence_py_summary.json`
3. 运行迁移核验脚本：

```bash
python python_impl/verify_migration.py
```

输出：
- `python_impl/python_scripts/equivalence_compare_report.json`

说明：
- 该报告比较各算法运行时间、NR（d/e）与 NSE（e/d）平均值差异。
- MATLAB 与 Python 的绝对 NSE 值可存在偏差，主要来自两端房间声学建模（RIR）实现细节差异；若算法流程与趋势一致、且两端可稳定运行，可视为迁移正确。

## 严格等价模式（共享同一份输入与次级通路）

为避免 MATLAB 与 pyroomacoustics 的 RIR 细节差异影响比较，仓库提供严格等价模式：
- 先由 Python 生成共享数据集（`strict_equiv_dataset.mat`），其中包含统一的 `time/x/d` 与次级通路 `sec_rirs`。
- MATLAB 与 Python 都加载这同一份数据集运行算法。
- 自动生成收敛叠加图与差异曲线图。

### 分步运行

1. 生成共享数据集：

```bash
python python_impl/python_scripts/generate_strict_dataset.py
```

2. MATLAB 严格等价运行：

```matlab
run('matlab_impl/run_equivalence_matlab_strict.m')
```

3. Python 严格等价运行：

```bash
python python_impl/python_scripts/run_strict_equivalence.py
```

4. 生成收敛差异报告与图：

```bash
python python_impl/python_scripts/compare_strict_convergence.py
```

### 一键运行（Windows PowerShell）

```powershell
powershell -ExecutionPolicy Bypass -File .\run_strict_equivalence.ps1
```

### 严格等价输出文件

- `python_impl/python_scripts/strict_equiv_dataset.mat`
- `python_impl/python_scripts/strict_mat_summary.json`
- `python_impl/python_scripts/strict_mat_curves.mat`
- `python_impl/python_scripts/strict_py_summary.json`
- `python_impl/python_scripts/strict_py_curves.npz`
- `python_impl/python_scripts/strict_convergence_report.json`
- `python_impl/python_scripts/strict_convergence_overlay.png`
- `python_impl/python_scripts/strict_convergence_delta.png`

## 备注

- Python 绘图若出现中文字体缺失告警，不影响算法结果，仅影响图中文字形显示。
- `run_experiment.py` 的 `--output-json` 相对路径现按当前工作目录解析，默认输出路径为 `python_impl/python_scripts/last_run_summary.json`。
