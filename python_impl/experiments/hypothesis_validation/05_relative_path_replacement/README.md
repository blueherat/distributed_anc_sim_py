# 05 Relative Path Replacement

假设：
- 保留第 03 步的最简控制条件
- 去掉绝对坐标输入
- 改用 `xref` 派生特征、`E2R_paths`、`S2R_paths`
- `R2R_paths` 只做可选扩展

目标：
- 验证相对路径是否可以替代绝对位置，而 replay 均值损失不超过 `0.5 dB`。

命令：
```powershell
python src/build_dataset.py --level L1
python src/train.py --level L1 --feature-kind relative --live-plot
python src/evaluate.py --level L1 --feature-kind relative
```

如果 `relative` 未通过 gate，再跑一次 `--feature-kind relative_plus_r2r`。也可以直接运行 `notebooks/train.ipynb`。
