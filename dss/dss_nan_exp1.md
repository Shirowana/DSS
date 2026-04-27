# DSS NaN 问题定位记录

## 实验目的

本记录用于总结当前 DSS 训练中出现 NaN 的真实原因。

之前已经基本排除：

- 不是 shared basis 没保存。
- 不是模型路径错误。
- 不是导入问题。
- 不是 batch 全是 `-100`。
- 不是 `coefficient / A_inv / B_inv` 被错误放成 fp16。
- 不是 optimizer 先把正常梯度算坏。

当前已经通过两类日志确认：

1. 训练时 `y_core` 仍是正常数量级，但经过 `A_inv` 后的 `y_delta` 极大。
2. 读取 shared basis 后，发现 `A_inv / B_inv` 本身非常大，随机稀疏 `lambda` 经过逆变换后会生成远超 base weight 的 `Delta W`。

因此，当前主要问题不是训练循环或 optimizer，而是：

```text
离线 diagonalization 得到的 A / B 条件数过大，
导致 inverse path 中的 A_inv / B_inv 极度放大在线训练的 sparse lambda。
```

## 配置摘要

- 模型：`Meta-Llama-3-8B`
- target modules：`qkv`
- shared basis：`llama3_8b_dss_basis.pt`
- 诊断脚本：`diagnose_dss_basis_scale.py`
- 诊断命令核心参数：

```bash
--target_modules qkv
--max_layers_per_group 1
--probe_slots 128
--lambda_scales 1,1e-2,1e-4
```

这里的 `probe lambda` 不是训练中真实的 `lambda`，而是用于测试：

```text
Delta W = A_inv @ lambda @ B_inv
```

也就是看一个 sparse core-space 更新经过逆变换后，在原权重空间会被放大到什么程度。

## 公式对应关系

当前 DSS 前向使用：

```text
u = x (B^{-1})^T
v = u lambda^T
Delta y = v (A^{-1})^T
y = y_base + scale * Delta y
```

权重空间等价写法是：

```text
Delta W = A^{-1} lambda B^{-1}
```

代码中的对应关系：

```text
u        = x_basis
lambda   = lambda_dense
v        = y_core
Delta y  = y_delta
```

## 训练时的 NaN 现场

训练崩溃前，`q_proj` 附近出现过：

```text
y_core:   max=2.6875e+01
y_delta:  max=5.6525e+05
```

两者比例约为：

```text
5.65e5 / 2.69e1 ≈ 2.1e4
```

也就是说，core-space 输出 `y_core` 经过 `A_inv` 映回原输出空间后，被放大了大约 `2.1 万倍`。

当时即使用了：

```text
DSS_DELTA_SCALE = 1e-2
```

实际加回 attention 的扰动仍约为：

```text
1e-2 * 5.65e5 = 5.65e3
```

而 base q/k/v 输出最大值大约只有：

```text
14 ~ 33
```

所以实际 DSS 注入量约为 base output 的：

```text
5.65e3 / 33 ≈ 171 倍
5.65e3 / 14 ≈ 404 倍
```

随后 backward 中出现：

```text
grad_core: finite=0/16777216
```

并且：

```text
bad_count=6998
curr_count=6998
active_overlap=6998
```

这说明当前所有 active coefficient 的梯度都被 non-finite `grad_core` 污染了。

## Shared Basis 静态诊断结果

### 1. 原始 Llama3 权重是正常数量级

第一层 `q/k/v` 的 base weight `W0` rms 为：

| Group | `W0 rms` |
| --- | ---: |
| `q_proj` | `1.7750e-02` |
| `k_proj` | `2.5780e-02` |
| `v_proj` | `6.5104e-03` |

也就是原始 Llama3 权重整体在 `1e-2` 左右，属于正常 Transformer 权重量级。

### 2. `A W0 B` 被压得非常小

同一层经过 shared basis 变换后：

| Group | `A W0 B rms` | `core/W0 rms` |
| --- | ---: | ---: |
| `q_proj` | `5.9470e-06` | `3.3504e-04` |
| `k_proj` | `1.4449e-05` | `5.6049e-04` |
| `v_proj` | `2.9763e-06` | `4.5717e-04` |

这说明 `A W0 B` 的 rms 只有原始 `W0` 的大约：

```text
0.03% ~ 0.06%
```

换句话说，shared basis 把原始权重压小了约：

```text
q_proj: 1 / 3.3504e-04 ≈ 2985 倍
k_proj: 1 / 5.6049e-04 ≈ 1784 倍
v_proj: 1 / 4.5717e-04 ≈ 2187 倍
```

### 3. `A / B` 本身不大，但 `A_inv / B_inv` 很大

`q_proj`：

```text
||A||2≈5.2049e-02, ||A_inv||2≈2.5516e+05, cond(A)≈1.3281e+04
||B||2≈5.9613e-02, ||B_inv||2≈7.6610e+05, cond(B)≈4.5670e+04
```

`k_proj`：

```text
||A||2≈1.0151e-01, ||A_inv||2≈1.4668e+05, cond(A)≈1.4890e+04
||B||2≈5.7483e-02, ||B_inv||2≈7.6746e+05, cond(B)≈4.4115e+04
```

`v_proj`：

```text
||A||2≈6.9831e-02, ||A_inv||2≈8.6452e+05, cond(A)≈6.0370e+04
||B||2≈4.4004e-02, ||B_inv||2≈6.5044e+05, cond(B)≈2.8622e+04
```

这里最关键的是：

- `A / B` 的谱范数只有 `1e-2 ~ 1e-1`。
- 但 `A_inv / B_inv` 的谱范数达到 `1e5 ~ 1e6`。
- 条件数达到 `1e4 ~ 1e5`。

这说明矩阵不是元素值简单变大，而是变得很病态：某些方向被 `A / B` 压得很小，求逆后这些方向会被极大放大。

### 4. 稀疏 `lambda` 经过逆变换后变成巨大 `Delta W`

当随机 sparse `lambda` 有 `128` 个非零 slot，且每个 coefficient 只有：

```text
1e-4
```

得到的 `Delta W rms` 已经是：

| Group | `Delta W rms` when `coeff_abs=1e-4` | `W0 rms` | `Delta W rms / W0 rms` |
| --- | ---: | ---: | ---: |
| `q_proj` | `9.5930e+00` | `1.7750e-02` | `≈ 540x` |
| `k_proj` | `4.7763e+01` | `2.5780e-02` | `≈ 1853x` |
| `v_proj` | `1.8120e+02` | `6.5104e-03` | `≈ 27832x` |

也就是说，即使 `lambda` 的非零值只有 `1e-4`，还原成原权重空间的 `Delta W` 后，也已经比 base weight 大几百到几万倍。

当 coefficient 为：

```text
1e-2
```

`Delta W rms` 更是达到：

| Group | `Delta W rms` when `coeff_abs=1e-2` |
| --- | ---: |
| `q_proj` | `1.6286e+03` |
| `k_proj` | `3.8222e+03` |
| `v_proj` | `1.5792e+04` |

这已经完全不是 PEFT 里的“小增量”。

## 结论

当前 NaN 的真实问题已经基本查明：

```text
shared basis 阶段学到的 A / B 是病态坐标变换。
它可以把 W0 变换到很小的 core-space，
但 inverse path 中的 A_inv / B_inv 会把在线训练的 sparse lambda 极大放大。
```

因此，训练中一旦 `lambda` 稍微长起来，实际产生的：

```text
Delta W = A_inv @ lambda @ B_inv
```

或前向中的：

```text
Delta y = y_core @ A_inv.T
```

就会远超 base model 的正常尺度，最终导致 attention 反向传播数值溢出。

这和训练现场完全一致：

- `y_core` 仍是正常数量级。
- `y_delta` 被放大到 `5e5` 量级。
- 实际注入 attention 后达到 base output 的数百倍。
- `grad_core` 整张变成 non-finite。
- 所有 active coefficient 的梯度一起变坏。

## 当前风险排序

按本轮静态诊断看：

1. `v_proj` 最危险：
   - `||A_inv||2≈8.6452e+05`
   - `cond(A)≈6.0370e+04`
   - `coeff_abs=1e-4` 时 `Delta W rms / W0 rms ≈ 27832x`
2. `k_proj` 次之：
   - `coeff_abs=1e-4` 时 `Delta W rms / W0 rms ≈ 1853x`
3. `q_proj` 也不安全：
   - `coeff_abs=1e-4` 时 `Delta W rms / W0 rms ≈ 540x`

因此，继续只调小 `DSS_DELTA_SCALE` 可以暂时压住前向幅度，但会同时压弱训练信号，不是根本解决方案。

## 后续方向

后续应该优先修 shared basis 本身，而不是继续修 optimizer。

可能方向：

- diagonalization 时加入条件数约束或正交约束。
- 对过小奇异值做截断，避免 `A_inv / B_inv` 爆炸。
- 对 `A_inv / B_inv` 做归一化或重新标定。
- 用 calibration data 统计真实 `x / x_basis / y_core / y_delta` 尺度，再设计 per-group scale。
- 暂时只测单个模块，例如只测 `q_proj` 或只测 `v_proj`，验证模块敏感性。
- 降低 coefficient 学习率只能作为辅助，不应作为主要修复。
