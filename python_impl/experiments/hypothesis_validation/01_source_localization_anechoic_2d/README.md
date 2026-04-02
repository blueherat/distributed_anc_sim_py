# 01 Source Localization Anechoic 2D

假设：
- 纯直达声
- 二维平面
- 固定声速
- 白噪声源
- 已知 3 个参考麦位置和 `xref`

目标：
- 验证 `xref + ref_positions -> source_position` 是否可行。
- 同时比较解析 GCC/TDOA 基线、`GCC + ref_positions` 小模型、`xref covariance + ref_positions` 小模型。
- 必须拆分为 IID 和几何 holdout。

命令：
```powershell
python src/build_dataset.py --level L1
python src/train.py --level L1 --live-plot
python src/evaluate.py --level L1
```

也可以直接运行 `notebooks/train.ipynb`，它会调用同一套共享函数并实时刷新 loss 图。

通过标准：
- 解析 baseline test 中位误差 `< 0.05 m`
- 最好学习模型 IID test 中位误差 `< 0.05 m`
- 最好学习模型几何 holdout test 中位误差 `< 0.10 m`
