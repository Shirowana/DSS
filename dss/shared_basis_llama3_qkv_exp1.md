# Llama3-8B Shared Basis 第一轮实验记录

## 实验目的

本轮实验只关注 **shared basis 阶段**，用于初步验证在 `Llama3-8B` 上对 `q_proj / k_proj / v_proj` 做共享基拟合时，是否能把权重矩阵有效压到更接近对角带的形式。

本记录**不包含**后续 DSS 正式训练结果。

## 配置摘要

- 模型：`Meta-Llama-3-8B`
- target modules：`qkv`
- shared basis 超参数：
  - `offset=0`
  - `basis_lr=0.01`
  - `basis_iters=1000`
  - `basis_device=cuda`
  - `max_layers_per_group=0`
  - `count=32`（每组使用 32 层）

## 实验结果

| Group | Shape | 最终时间 | 最终 loss | 平均平方比值 | 平均范数比值 | 带外/带内能量范数 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `q_proj` | `(4096, 4096)` | `21m 34s` | `389.5681` | `389.5681 / 32 = 12.17` | `sqrt(12.17) ≈ 3.49` | `≈ 3.49 : 1` |
| `k_proj` | `(1024, 4096)` | `3m 57s` | `687.0541` | `687.0541 / 32 = 21.47` | `sqrt(21.47) ≈ 4.63` | `≈ 4.63 : 1` |
| `v_proj` | `(1024, 4096)` | `3m 57s` | `909.5194` | `909.5194 / 32 = 28.42` | `sqrt(28.42) ≈ 5.33` | `≈ 5.33 : 1` |

说明：

- 这里的 loss 形式为：
-  
  $$
  \sum_i \frac{\|off(AW_iB)\|^2}{\|diag(AW_iB)\|^2}
  $$

- 因为每组使用 `count=32` 个矩阵，所以先除以 `32` 得到单矩阵平均平方比值，再开方得到更直观的“带外能量范数 / 带内能量范数”。
- `k_proj` 和 `v_proj` 的 shape 采用 `Llama3-8B` 的 GQA 结构理解填写，即 `(1024, 4096)`。

## 日志摘录

### q_proj

```text
[basis][q_proj] 990/1000 (99.0%) loss=389.5717 iter_time=1s avg_iter=1s elapsed=21m 21s eta=13s
[basis][q_proj] 1000/1000 (100.0%) loss=389.5681 iter_time=1s avg_iter=1s elapsed=21m 34s eta=0s
```

### k_proj

```text
[basis][k_proj] 990/1000 (99.0%) loss=687.0586 iter_time=0s avg_iter=0s elapsed=3m 55s eta=2s
[basis][k_proj] 1000/1000 (100.0%) loss=687.0541 iter_time=0s avg_iter=0s elapsed=3m 57s eta=0s
```

### v_proj

```text
[basis][v_proj] 970/1000 (97.0%) loss=909.5991 iter_time=0s avg_iter=0s elapsed=3m 50s eta=7s
[basis][v_proj] 980/1000 (98.0%) loss=909.5447 iter_time=0s avg_iter=0s elapsed=3m 52s eta=5s
[basis][v_proj] 990/1000 (99.0%) loss=909.5232 iter_time=0s avg_iter=0s elapsed=3m 54s eta=2s
[basis][v_proj] 1000/1000 (100.0%) loss=909.5194 iter_time=0s avg_iter=0s elapsed=3m 57s eta=0s
```

## 结果总结

- `q_proj` 是本轮实验里效果最好的一组，但即便如此，最终 `带外能量范数 : 带内能量范数 ≈ 3.49 : 1`，仍然不能算“近对角化很好”。
- `k_proj`、`v_proj` 也都出现了明显下降，说明 shared basis 确实在学习；但最终结果分别约为 `4.63 : 1` 和 `5.33 : 1`，带外能量仍明显高于带内能量。
- 从这轮 `q/k/v` 的结果看，当前设置下 shared basis 能带来一定改善，但整体只能算“有改善，但不够强”。

## 为什么暂时不纳入 u/d

本轮实验先只总结 `qkv`，没有把 `up_proj / down_proj` 纳入最终结论，原因是这两组的单组训练时间明显更长，等待完整结果的成本较高。

当前已有 `up_proj` 日志片段：

```text
[basis] fitting group=up_proj, count=32, shape=(14336, 4096)
[basis][up_proj] 1/1000 (0.1%) loss=458881.1250 iter_time=5s avg_iter=5s elapsed=5s eta=1h 15m 25s
[basis][up_proj] 10/1000 (1.0%) loss=56205.5156 iter_time=9s avg_iter=9s elapsed=1m 25s eta=2h 20m 25s
[basis][up_proj] 20/1000 (2.0%) loss=29128.2656 iter_time=9s avg_iter=9s elapsed=2m 55s eta=2h 22m 35s
```

这说明：

- `up_proj` 的 shape 已经达到 `(14336, 4096)`
- 单 iter 时间显著高于 `q/k/v`
- 早期日志已经显示整组 ETA 在 `2h+`

因此，本轮先完成 `qkv` 的效果验证，`ud` 留待后续单独分析。
