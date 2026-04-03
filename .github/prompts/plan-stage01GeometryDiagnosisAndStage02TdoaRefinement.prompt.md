## Plan: Stage01几何诊断与Stage02 TDOA提精

目标是两条并行主线：
1) 在 01 中解释“为什么有些样本在三角形外失败，但也有部分外部样本成功”，并提供可重复的 inside/outside 诊断程序（脚本 + notebook）。
2) 在 02 中把 GCC->TDOA 回归从当前约 0.0024s 量级降到目标 < 1e-4s（预测 TDOA 与真实 TDOA 的差值），通过输入/输出重构 + 架构重构 + 损失与训练策略升级实现。

**Steps**
1. Phase A: 复现实验与统一诊断口径（阻塞后续所有步骤）
2. A1. 固化当前基线口径（*阻塞步骤 B/C*）：记录 01 使用数据与结果、02 使用数据与结果，统一主指标为 `tdoa_mae_s`（秒）并保留 `tdoa_mae_m`（米）辅助解释。
3. A2. 新增统一样本级诊断表规范（*阻塞步骤 B/C*）：定义每条样本输出字段（split、sample_index、inside_triangle、triangle_area、jacobian_condition、small_angle_flag、tdoa_abs_err_s、xy_err_m、success_flag）。
4. Phase B: 01 三角形内外性能诊断（脚本与 notebook 同时交付）
5. B1. 在 01 增加可复用分析函数（*depends on A2*）：从已有模型预测与真值生成样本级 CSV，包含 inside/outside 分类与几何指标。
6. B2. 新增 01 CLI 脚本（*depends on B1*）：一键输出 inside/outside 汇总指标（count、median/p90、success rate）与分桶统计（outside 但成功、outside 且失败；按 cond/area 分层）。
7. B3. 更新 01 notebook 分析单元（*parallel with B2 after B1*）：可视化 inside/outside 的误差分布、条件数与面积交互热图，解释“外部也可能成功”的条件区间。
8. B4. 产出解释结论模板（*depends on B2+B3*）：给出失败主因排序（几何病态 > outside 标签本身），并标注可操作阈值建议。
9. Phase C: 02 输入/输出与网络架构重构（较大改造）
10. C1. I/O 对齐与目标重参数化（*depends on A2*）：
11. C1.1 对齐 pair 顺序：将 GCC 通道顺序与 pair_dist/target 顺序一致（统一为 [01,12,02]）。
12. C1.2 目标改为“归一化 lag”而非秒：`lag_samples = tdoa_seconds * fs`，再按每对物理上界 `d_ij/c*fs` 做归一化到 [-1,1]。
13. C1.3 输出参数化改为 2 自由度 + 1 约束重建：网络只预测两对时延，第三对由一致性关系重建，显式降低不一致误差。
14. C2. 编码器改造（*depends on C1*）：移除全局平均池化，改为“保留位置信息”的时序编码（多尺度 Conv1d + 可选 attention / flatten head），避免抹平峰值位置信息。
15. C3. 损失函数升级（*depends on C1+C2*）：主损失用 Huber/L1（归一化 lag 空间），叠加一致性约束损失与可选几何约束损失（`c*tdoa` 与距离差一致）。
16. C4. 训练策略升级（*depends on C2+C3*）：两阶段课程学习（先 anechoic/弱反射，再 full reflection mix），并启用学习率调度与早停。
17. C5. 可选扩展（*parallel with C4 after C2*）：多任务头（coarse lag classification + fine residual regression）以提升到 1e-4 目标所需的亚采样精度稳定性。
18. Phase D: 最小实验矩阵与收敛判定
19. D1. 运行 ablation 矩阵（*depends on C1-C4*）：
20. D1.1 Baseline（现有 notebook 模型）
21. D1.2 +I/O 对齐与目标归一化
22. D1.3 +新编码器（去全局池化）
23. D1.4 +损失升级
24. D1.5 +两阶段课程学习
25. D1.6 +多任务头（可选）
26. D2. 统一评估与过线标准（*depends on D1*）：`iid_val`、`geom_val`、`iid_test`、`geom_test` 全部报告 `tdoa_mae_s`；主验收为验证集 `tdoa_mae_s < 1e-4`，并检查 test 不回退。

**Relevant files**
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/01_source_localization_anechoic_2d/src/analysis_utils.py` — 现有 inside 判定与敏感性统计复用点。
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/01_source_localization_anechoic_2d/src/relative_distance_notebook_utils.py` — 01 模型预测与评估管线，新增样本级导出落点。
- `z:/anc/distributed_anc_sim_py/python_impl/python_scripts/hypothesis_validation_common.py` — 几何指标、inside_convex_hull、稳定几何过滤与审计复用。
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/01_source_localization_anechoic_2d/notebooks/analysis.ipynb` — 增加 inside/outside 诊断可视化单元。
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/01_source_localization_anechoic_2d/notebooks/relative_distance_train_eval.ipynb` — 对接新诊断脚本输出。
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/src/gcc_reflection_notebook_utils.py` — 02 的 bundle、模型、训练主改造文件（I/O、架构、loss）。
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/notebooks/gcc_to_tdoa_train_eval.ipynb` — 新训练参数与实验矩阵入口。
- `z:/anc/distributed_anc_sim_py/python_impl/experiments/hypothesis_validation/02_source_localization_reflections/src/build_dataset.py` — 课程学习/数据混合比例可调入口。

**Verification**
1. 01 inside/outside 程序验证：输出 CSV + JSON，必须包含 inside/outside 分组统计、outside-success 子群统计、按 `jacobian_condition`/`triangle_area` 分桶结果。
2. 01 一致性验证：脚本与 notebook 对同一 run 输出相同统计值（允许浮点微差）。
3. 02 训练过程验证：history 中应看到 `tdoa_mae_s` 进入 1e-4 量级前的单调下降趋势；若平台化需由 ablation 定位到具体瓶颈组件。
4. 02 结果验证：summary 同时输出秒与米，并确认主指标 `tdoa_mae_s` 在目标 split 过线。
5. 反演合理性验证：检查三对 TDOA 一致性残差（第三对由前两对重建后误差应显著下降）。
6. 回归风险验证：对比改造前后 `xy` 误差，确保 TDOA 提升不引入几何反演退化。

**Decisions**
- 01 诊断程序交付形式：脚本 + notebook 两者都要。
- 02 目标指标：预测 TDOA 与真实 TDOA 的差值（秒）目标 < 0.0001。
- 02 改造范围：允许较大改造（可做两阶段训练/多任务/更大模型）。
- 范围内只处理 Python 01/02 流程；不改 MATLAB 链路与 03-05 阶段实现。

**Further Considerations**
1. 目标 split 建议优先固定为 `geom_val`（更严格），同时跟踪 `iid_val` 作为收敛稳定性指标。
2. 若 1e-4 在 full reflection 混合下不可达，建议先在 anechoic 子集验证上界，再分离“任务物理极限”与“模型容量不足”。
3. 若课程学习显著有效，可将其沉淀为 02 默认训练配置，避免 notebook 手工调参漂移。
