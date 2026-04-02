# 03 Single-Control Hyperplane Anechoic

假设：
- `3 refs + 1 speaker + 1 error mic`
- 纯直达声
- 已知 `ref/speaker/error` 绝对位置
- `xref` 负责提供源位置相关信息

目标：
- 证明超平面损失 `||P W - q*||` 比直接对标某个 `W*` 更合理。
- 比较 `hyperplane` 和 `W-MSE` 两种训练目标。

命令：
```powershell
python src/build_dataset.py --level L1
python src/train.py --level L1 --live-plot
python src/evaluate.py --level L1
```

也可以直接运行 `notebooks/train.ipynb`。
