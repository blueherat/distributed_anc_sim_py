# 04 Single-Control Hyperplane Reflections

假设：
- `3 refs + 1 speaker + 1 error mic`
- 单次反射
- 其它设置与第 03 步一致

目标：
- 保持损失函数和评估方式不变，只放宽传播条件。
- 验证一次反射下超平面损失是否仍明显优于 `W-MSE` baseline。

命令：
```powershell
python src/build_dataset.py --level L1
python src/train.py --level L1 --live-plot
python src/evaluate.py --level L1
```

也可以直接运行 `notebooks/train.ipynb`。
