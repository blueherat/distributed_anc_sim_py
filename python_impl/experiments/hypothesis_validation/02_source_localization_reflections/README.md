# 02 Source Localization Reflections

假设：
- 单次反射
- 二维平面
- 固定声速
- 白噪声源
- 已知 3 个参考麦位置和 `xref`

目标：
- 验证在一次反射下，`xref + ref_positions -> source_position` 是否仍然稳定。
- 当前默认只做 `2A` 单次反射，不直接扩到重混响。

命令：
```powershell
python src/build_dataset.py --level L1
python src/train.py --level L1 --live-plot
python src/evaluate.py --level L1
```

也可以直接运行 `notebooks/train.ipynb`。
