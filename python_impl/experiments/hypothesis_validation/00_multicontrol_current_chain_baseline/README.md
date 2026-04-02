# 00 Multi-Control Baseline

这个步骤只做当前 `3ref+3speaker+3errormic` 链路的最小基线审计，不参与后续 gate。

目标：
- 在 `200` 房间子集上跑通当前 3x3 canonical-Q 训练链路。
- 记录 validation replay、exact upper bound gap、plant residual 和典型房间的 `Q/W` 图。

命令：
```powershell
python src/build_dataset.py --num-rooms 200
python src/train.py --epochs 30
python src/evaluate.py
```

输出：
- `data/multicontrol_baseline_200.h5`
- `results/train_summary.json`
- `results/eval/summary.json`
- `results/eval/*.png`
