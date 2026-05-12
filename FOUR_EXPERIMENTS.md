# Four DSS Experiments

## Overview

当前主线准备对比四组实验。它们共享同一个 no-basis、stage1-only DSS 框架，区别主要在于：

- 是否使用初始化观测窗口
- 初始化结果只用来 warm-start threshold，还是直接 seed 一批 elite
- threshold 更新方式是 `oracle` 还是 `sgd`
- 是否回到最早的 DDP 激进筛选基线

## Exp 1: Oracle + Init

### Goal

验证“初始化观测窗口 + 精确 quantile warm start”是否能让 `oracle` 版本更稳、更强。

### Core Idea

- 开启初始化观测窗口
- 每个矩阵按 block 方式抽取 `5%` 的候选位置
- 不加 cap，所有矩阵都按各自参数总量的 `5%` 比例取初始化候选
- 初始化阶段只观测梯度，不更新模型
- 用前 `10` 个 step 的梯度统计计算精确 quantile
- 初始化样本严格按 8 个任务均匀抽样
- 初始化结果默认先做 `threshold_only`
- 初始化完成后，恢复正常 `oracle` 动态筛选

### Key Hyperparameters

- `threshold_mode = oracle`
- `init_enabled = 1`
- `init_steps = 10`
- `init_candidate_ratio = 0.05`
- `init_seed_mode = threshold_only`
- `candidate_size = 10000`
- `grad_store_steps = 10`
- `ratio = 0.05`
- `low = 100`
- `up = 1000`
- `dropout = 0.05`
- `lr = 8e-5`
- `num_epochs = 3`

## Exp 2: SGD + Init

### Goal

验证初始化观测窗口是否能帮助 `sgd pinball` 阈值器起步更稳，并缩小它和 `oracle` 的差距。

### Core Idea

- 初始化阶段与 Exp 1 相同
- 初始化结束后，将精确 quantile 写入 `sgd` quantile estimator 初值
- 后续继续用 pinball loss 在线更新阈值

### Key Hyperparameters

- `threshold_mode = sgd`
- `quantile_lr = 0.03`
- `quantile_alpha = 0.0`
- `init_enabled = 1`
- `init_steps = 10`
- `init_candidate_ratio = 0.05`
- `init_seed_mode = threshold_only`
- `candidate_size = 10000`
- `grad_store_steps = 10`
- `ratio = 0.05`
- `low = 100`
- `up = 1000`
- `dropout = 0.05`
- `lr = 8e-5`
- `num_epochs = 3`

## Exp 3: DDP Old Baseline

### Goal

作为最早的强基线保留，不再修改方法，只补齐评测并作为后续实验的固定对照。

### Core Idea

- 不使用初始化
- 使用最早那组激进 stage1 配置
- 双卡 DDP

### Key Hyperparameters

- `threshold_mode = oracle`
- `init_enabled = 0`
- `candidate_size = 30000`
- `grad_store_steps = 3`
- `ratio = 0.10`
- `low = 500`
- `up = 4000`
- `dropout = 0.05`
- `lr = 8e-5`
- `num_epochs = 3`
- `num_gpus = 2`

### Current Observations

- 旧配置在早期 stage1 refresh 中，不同 family 的 threshold 尺度差异明显：
  - `q_proj`: mean `0.00049175`
  - `k_proj`: mean `0.00063486`
  - `v_proj`: mean `0.00582053`
  - `up_proj`: mean `0.00111125`
  - `down_proj`: mean `0.00134237`
- `v_proj` 的 threshold 明显高于其余 `q/k/u/d`，大约高一个数量级。
- layer 间差异也很大：
  - self-attn 最低层均值约 `0.00011088`
  - self-attn 最高层均值约 `0.00682746`
  - mlp 最低层均值可到 `0.0`
  - mlp 最高层均值约 `0.00273641`
- 当前已观察到的全局 threshold 范围：
  - min `0.0`
  - mean `0.00186422`
  - max `0.01761882`
- 旧配置后期/局部模块出现过 `threshold=0` 且 `x_mean_min/mean/max=0` 的情况，例如：
  - `model.layers.10.mlp.up_proj`
  - `model.layers.10.mlp.down_proj`
  - `model.layers.12.mlp.up_proj`
  - `model.layers.12.mlp.down_proj`
- 这说明至少在某些 refresh window 中，candidate 的 `abs(mean(grad))` 会整体塌到 0；需要继续排查：
  - 是 `abs(mean)` 的符号抵消
  - 还是 hook/累计实现问题
  - 或者这些层在该阶段真的没有有效 candidate 梯度

### Current Full-Eval Results

- `boolq = 0.6905`
- `piqa = 0.8330`
- `social_i_qa = 0.7830`
- `winogrande = 0.8090`
- `ARC-Challenge = 0.6962`
- `ARC-Easy = 0.8460`
- `openbookqa = 0.8200`
- `hellaswag = 0.8908`

## Exp 4: Init Top-K To Budget Fixed Seed

### Goal

验证“只看初始化阶段的梯度统计，直接按排序填满每矩阵 18w budget”能否作为更强的 one-shot sparse baseline，同时保持与前 3 组一致的最终参数量。

### Core Idea

- 开启初始化观测窗口
- 每个矩阵按 block 方式抽取 `5%` 的候选位置
- 不加 cap，所有矩阵都按各自参数总量的 `5%` 比例取初始化候选
- 前 `10` 个 step 只观测梯度，不更新模型
- 初始化样本严格按 8 个任务均匀抽样
- 计算精确 quantile
- 对初始化候选直接按 `x_mean` 从大到小排序
- 每个矩阵直接取 `top min(n_frequency, candidate_count)` 做 `seed_elite`
- 也就是优先保证每个矩阵尽量填满 `18w` budget，而不是固定只取候选中的 `top-10%`
- 后续可以继续训练这些 elite，也可以继续动态筛选剩余 budget

### Key Hyperparameters

- `init_enabled = 1`
- `init_steps = 10`
- `init_candidate_ratio = 0.05`
- `init_seed_mode = seed_elite`
- `n_frequency = 180000`
- 其余训练超参数先默认对齐 Exp 1 或 Exp 2，再视结果细调

## Current Notes

- 训练数据预处理链路需要保留 `task_name` 字段，初始化阶段依赖它做严格 balanced-task 采样。
- 初始化阶段固定为：
  - `init_steps = 10`
  - `init_candidate_ratio = 0.05`
  - `block_size = 64`
  - exact `torch.quantile`
  - 只 forward/backward，不做 `optimizer.step()`
- 阈值调试日志已经接入训练主线：
  - 每层第一次 refresh 必打
  - 后续默认每 `1000` 个 step 打一次
- 后续需要重点比较：
  - `eval_loss_best`
  - 各任务 acc
  - 初始化 threshold
  - 初始化耗时
  - 前期 loss 曲线

## Current DDP and Init Workflow

### 1. 模型与 DDP 启动

- 两个 rank 都加载 base model 和 DSS adapter。
- 正式训练阶段使用双卡 DDP。
- 但初始化观测阶段只由 `rank0` 执行，`rank1` 不参与 init 的 forward/backward。

### 2. 初始化观测阶段

- 从带 `task_name` 的训练集里按 8 个任务均匀抽样，构造一个固定的小 init dataset。
- 当前默认是：
  - `init_steps = 10`
  - 每卡 `batch_size` 由训练配置决定
- 每一步只做：
  - forward
  - backward
- 不做：
  - `optimizer.step()`
  - scheduler step
  - 正常 stage1 promotion
- 每个 DSS 模块只在这期间累积 init candidate 的梯度统计。

### 3. 初始化收口阶段

- `rank0` 在 `finalize_dss_init()` 中逐模块计算：
  - `x_mean`
  - `torch.quantile`
- 根据实验模式产出：
  - `threshold_only`
  - 或 `seed_elite`
- 初始化阶段只更新 DSS 的初始化状态。
- **不会**在这里提前启动正常 stage1 的 candidate 刷新。

### 4. 接轨正常训练

- `sync_dss_init_state()` 把 `rank0` 生成的 init 结果同步到所有 rank。
- 所有 rank 在 sync 完成后，进入正常训练循环。
- 普通 stage1 的第一轮 `refresh_candidate_batch()`，延后到 sync 之后第一次进入常规 `maybe_refresh_stage1()` 时再触发。
- 这样可以避免：
  - `rank0` 在 init finalize 末尾提前做普通 stage1 的 DDP candidate broadcast
  - `rank1` 同时还在等待 init-state sync broadcast
  - 从而避免 DDP collective 序列错位。
