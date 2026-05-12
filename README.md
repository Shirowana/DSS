# DSS No-Basis 主线说明

当前仓库的有效主线已经收敛为一个更纯粹的 DSS PEFT 版本：

- 不使用 shared basis
- 不使用 stage2 prune/grow
- 只研究 stage1 的 candidate 筛选与 elite 晋升
- 训练使用官方 Hugging Face `Trainer`

这份 README 以当前代码为准。旧版 shared-basis / stage2 / 自定义 trainer 方案已经归档，不再是默认训练路径。

## 1. 当前方法

对一个线性层，DSS 维护两类状态：

- 长期 elite：
  - `coefficient`
  - `coefficient_indices`
  - `curr_count`
- 临时候选：
  - `candidate_indices`
  - `grad_cache`
  - `grad_count`

其中：

- `coefficient` 是唯一可训练参数
- `coefficient_indices` 记录 elite 对应的原权重空间 flat index
- `candidate_indices` 只用于探测，不参与 optimizer 更新

### Stage1 逻辑

当前实现只保留 stage1：

1. 随机采样一批 `candidate_size` 个候选坐标
2. 在接下来 `grad_store_steps` 个 optimizer step 内，累计这些 candidate 的 signed gradient
3. 计算

```text
x_mean = abs(grad_cache / grad_count)
```

4. 根据阈值筛选 candidate
5. 按 `low / up` 约束晋升一部分到 elite pool
6. 刷新下一批 candidate，继续循环

当 `curr_count == n_frequency` 时：

- stage1 搜索结束
- 后续只训练已经选出的 elite
- 不再进行 stage2 替换

## 2. 前向与反向

当前前向已经是 no-basis fast path。

对一个 adapter：

```text
y_base = base_layer(x)
delta_w = scatter(coefficient[:curr_count], coefficient_indices[:curr_count])
y_delta = F.linear(x, delta_w)
y = y_base + y_delta
```

这里没有 `A @ Lambda @ B`，也没有 `x @ B_inv.T` 或 `... @ A_inv.T`。

candidate 的梯度收集方式是：

- elite 部分正常参与前向和反向
- candidate 不进入输出
- 但在 `delta_w` 上注册 hook
- backward 时从 `grad_delta_w` 中按 `candidate_indices` 抽取梯度
- 把它们累计进 `grad_cache`

因此：

- elite 参数仍由 PyTorch autograd 正常更新
- candidate 只贡献统计，不直接被优化器更新

## 3. 阈值模式

当前支持两种阈值模式：

- `oracle`
  - 直接对当前 `x_mean` 调用 `torch.quantile`
- `sgd`
  - 使用 pinball loss 的在线 quantile estimator

第一轮实验默认推荐：

- `threshold_mode=oracle`

## 4. Dropout

当前 DSS dropout 已接回。

它的作用位置是：

- 只对 elite `coefficient[:curr_count]` 生效
- 只在训练态生效
- 不改变 checkpoint 格式

推荐第一轮配置：

- `dropout=0.05`

## 5. PEFT 结构

当前关键代码如下：

- PEFT tuner：
  - [config.py](/root/code/quest/peft/src/peft/tuners/dss/config.py)
  - [model.py](/root/code/quest/peft/src/peft/tuners/dss/model.py)
  - [layer.py](/root/code/quest/peft/src/peft/tuners/dss/layer.py)
  - [test_dss.py](/root/code/quest/peft/src/peft/tuners/dss/test_dss.py)
- 项目侧训练入口：
  - [finetune_commonsense.py](/root/code/DSS/finetune_commonsense.py)
  - [run_dss.sh](/root/code/DSS/run_dss.sh)

### `config.py`

当前只保留 stage1 所需字段：

- `n_frequency`
- `candidate_size`
- `grad_store_steps`
- `low`
- `up`
- `ratio`
- `threshold_mode`
- `dropout`
- `quantile_lr`
- `quantile_alpha`
- `target_modules`
- `bias`

### `model.py`

`DSSModel` 负责：

- 把目标线性层替换为 `DSSLinear`
- 冻结 base model
- 只放开 DSS `coefficient` 参数

不再负责：

- 加载 shared basis
- group scale
- stage2 外部 refresh

### `layer.py`

`DSSLinear` 负责：

- 原空间 no-basis 前向
- stage1 candidate 采样
- backward hook 梯度收集
- layer 内部状态机 refresh
- sparse checkpoint 导出与恢复

## 6. 训练入口

当前训练使用官方 `Trainer`，不再使用项目侧自定义训练循环。

入口文件：

- [finetune_commonsense.py](/root/code/DSS/finetune_commonsense.py)

支持的关键训练参数：

- `--val_set_size`
- `--eval_steps`
- `--save_steps`
- `--load_best_model_at_end`
- `--resume_from_checkpoint`

### 当前设备约束

当前实现依赖 layer 内部维护 candidate/runtime 状态，因此暂时只支持单卡可见训练。

推荐做法：

```bash
CUDA_VISIBLE_DEVICES=0 bash /root/code/DSS/run_dss.sh
```

`run_dss.sh` 默认也会在未显式指定时设置：

```bash
CUDA_VISIBLE_DEVICES=0
```

### 验证集语义

当前验证集是固定切分：

- 训练开始前，从训练集一次性切出 `val_set_size`
- 这批样本之后不再参与训练
- 每次 `eval_steps` 都在同一批验证集上评估

这不是“每隔若干步临时从训练集抽一批”。

## 7. 默认实验脚本

推荐入口：

- [run_dss.sh](/root/code/DSS/run_dss.sh)

当前默认配置是第一轮 step1-only 消融：

- `target_modules=qkvud`
- `n_frequency=180000`
- `candidate_size=30000`
- `grad_store_steps=3`
- `low=500`
- `up=4000`
- `ratio=0.1`
- `threshold_mode=oracle`
- `dropout=0.05`
- `num_epochs=3`
- `batch_size=16`
- `max_length=256`
- `val_set_size=120`
- `eval_steps=10000`
- `save_steps=10000`
- `load_best_model_at_end=1`

运行方式：

```bash
bash /root/code/DSS/run_dss.sh
```

覆盖默认值示例：

```bash
UP=5000 EVAL_STEPS=5000 SAVE_STEPS=5000 bash /root/code/DSS/run_dss.sh
```

## 8. Checkpoint 与恢复训练

当前 checkpoint 优先保存 sparse adapter 状态。

单层导出格式至少包含：

- `coefficient[:curr_count]`
- `coefficient_indices[:curr_count]`

恢复时会：

- 恢复 active elite slots
- 恢复 `curr_count`
- 重建 `elite_bitset`
- 清空 candidate probe 中间缓存
- 把搜索窗口从新的 stage1 window 重新开始

也就是说，当前恢复训练不追求恢复“半轮 candidate 统计”的中间态，而是恢复稀疏 adapter 状态后继续训练。

如果运行环境里的 `torch < 2.6`，当前训练入口会自动跳过 checkpoint 中的二进制 optimizer/scheduler 恢复，只恢复：

- adapter 权重
- trainer state
- 步数推进

这是为了绕开新版 `transformers` 对 `torch.load` 安全策略的限制。

## 9. 已归档内容

下面这些内容已经不是当前主线：

- shared basis / diagonalization 训练主路径
- stage2 prune/grow
- 项目侧手写 `DSSTrainer`

归档说明：

- [统一公式.md](/root/code/DSS/统一公式.md)

如果你只是要跑当前主线，请忽略这些旧文件。

## 10. 当前完成状态

下面这些目标已经完成：

- 做一个 DSS-no-basis 分支
- 把 stage1 刷新逻辑改回 layer 内部状态机
- 重新用官方 Trainer
- 补上固定验证集、`eval_steps`、`load_best_model_at_end`
- 接上 `resume_from_checkpoint` 入口

还需要持续观察的是：

- step1-only 的最终下游指标是否能逼近或达到 baseline
- `candidate_size / low / up / grad_store_steps` 的更细致超参数打磨
- 恢复训练在真实长任务上的稳定性
