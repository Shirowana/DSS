# 新 DSS 方法与实现说明

这份文档用于让人或 AI 快速理解当前 `new/dss` 中实现的新 DSS PEFT 方法。这里的 DSS 不是旧 DSS 的复刻，也不是 SpIEL/SFT、PiCa 或 LoCA 的代码复现，而是一个新的 shared-basis + core-space sparse selective tuning 方法。

当前实现的核心目标是：

- 用离线拟合得到的 shared basis 把原始权重空间变换到 core/`Lambda` 空间。
- 在线训练时只训练 core-space 中少量稀疏 slot，也就是 `coefficient`。
- 用两阶段动态选择机制决定哪些 core-space 坐标成为长期训练的 elite。
- 用纯 PyTorch GEMM + hook 实现稳定前向和 candidate probe，不再依赖自定义 CUDA sparse kernel。

## 1. 方法总览

对一个线性层，原始权重为：

```text
W in R^{out_features x in_features}
```

DSS 不直接在 `W` 空间训练 dense 增量，而是在 shared basis 定义的 core 空间训练稀疏增量：

```text
Delta W = A_inv @ Delta Lambda @ B_inv
```

其中：

- `A_inv`、`B_inv` 是离线 shared-basis 预处理得到的冻结矩阵。
- `Delta Lambda` 是 core-space 中的稀疏更新矩阵。
- 真正训练的参数是 `coefficient`，也就是 `Delta Lambda` 上少数 elite 坐标的值。
- `coefficient_indices` 记录这些 elite 坐标在 core matrix 中的一维 flat index。

因此，一层 DSS adapter 的长期状态可以理解为：

```text
coefficient         # elite slot 的可训练数值
coefficient_indices # elite slot 在 Delta Lambda 中的位置
curr_count          # 当前已经启用的 elite 数量
```

候选位置不是长期参数。它们只是一批临时 probe 坐标：

```text
candidate_indices
```

candidate 不参与前向输出，也不会进入 optimizer。它们只在 backward 中读取梯度，用于判断哪些位置值得加入 elite pool。

## 2. 离线 shared-basis / diagonalization

DSS 的第一步是对同功能组 target modules 做 shared-basis 预处理。典型功能组包括：

```text
q_proj, k_proj, v_proj, up_proj, down_proj
```

对每个功能组，收集多个层对应的预训练权重：

```text
W_1, W_2, ..., W_L
```

然后拟合一组共享变换 `A, B`，使得：

```text
A @ W_l @ B
```

在经验上更接近对角或带状结构。当前 `shared_basis.py` 中的 `diagonalization(...)` 复用了原始 diagonalization 脚本的思路：

1. 初始化可训练矩阵 `A, B`。
2. 对每个权重矩阵归一化。
3. 构造 band mask，mask 内视为 inside-band，mask 外视为 outside-band。
4. 优化 outside-band energy / inside-band energy 的比例。
5. 优化完成后保存 `A, B`，并计算 `A_inv, B_inv`。

损失形式可以概括为：

```text
loss = sum_l ||outside_band(A W_l B)||^2 / ||inside_band(A W_l B)||^2
```

这个阶段是离线的，不属于 fine-tuning。它使用自己的 Adam optimizer 和 learning rate，与在线训练的 `DSSAdamW` 无关，也与 stage1 阈值估计器无关。

离线结果保存为 `SharedBasisPack`，训练时由 `DSSModel` 根据 `shared_basis_path` 加载。

## 3. PEFT 工程外壳

当前 DSS 按 HuggingFace PEFT tuner 风格组织。

### `config.py`

定义 `DSSConfig`，负责配置字段和合法性检查。

主要字段包括：

- `shared_basis_path`：离线 basis 文件路径。
- `basis_group_map`：功能组到 target module suffix 的映射。
- `n_frequency`：每层 elite pool 总预算。
- `candidate_size`：每轮临时候选 batch 的大小。
- `grad_store_steps`：候选梯度累计多少步后触发 stage1 refresh，也作为 stage2 尾窗统计长度。
- `low / up`：stage1 每轮最少/最多晋升数。
- `ratio`：stage1 quantile 阈值估计目标比例。
- `stage2_enabled`：是否启用第二阶段 prune/grow。
- `steady_stage_ratio`：stage1 填满后稳定训练期占总训练步数的比例。
- `update_interval`：stage2 prune/grow 周期。
- `update_counts`：stage2 初始替换预算，后续线性衰减。
- `update_margin`：未来 grow-vs-prune gate 的占位字段，当前未启用。

`config.py` 不加载 basis，不创建 runtime state，也不执行模块替换。

### `model.py`

`DSSModel` 是 PEFT tuner 壳子，负责：

1. 加载 `SharedBasisPack`。
2. 根据 `basis_group_map` 判断每个 target module 属于哪个功能组。
3. 把目标 `nn.Linear` 或兼容 `Conv1D` 替换成 `DSSLinear`。
4. 冻结 base model，只让 `coefficient` 可训练。
5. 提供外部训练循环调用的统一入口：

```python
refresh_dss_layout(global_step, total_steps, optimizer, grad_accumulation_steps)
```

这个函数在每个 optimizer step 后调用，用于消费当前 step 已经收集到的 candidate 梯度统计，并根据需要执行 stage1 refresh 或 stage2 prune/grow。

## 4. 单层 DSS 状态设计

核心实现位于 `layer.py`。

### 长期 elite 状态

```text
coefficient
coefficient_indices
elite_bitset
```

- `coefficient`：唯一真正训练的 DSS 参数，长度为 `n_frequency`。
- `coefficient_indices`：每个 elite slot 对应的 core-space flat index。
- `elite_bitset`：长度为 `out_features * in_features` 的 bool membership，用于避免 candidate 采样到已是 elite 的位置。

只有 `coefficient[:curr_count]` 和 `coefficient_indices[:curr_count]` 是当前有效 elite。

### 短期 candidate 状态

```text
candidate_indices
grad_cache
grad_count
candidate_grad_sums
candidate_grad_sq_sums
```

stage1 使用：

- `grad_cache`：candidate signed gradient 的在线累加和。
- `grad_count`：累计了多少个 optimizer step。

最终 stage1 分数是：

```text
x_mean = abs(grad_cache / grad_count)
```

注意这里是 `abs(mean(grad))`，不是 `mean(abs(grad))`。

stage2 使用：

- `candidate_grad_sums`
- `candidate_grad_sq_sums`

它们只绑定当前 stage2 update interval 尾部的临时 candidate batch。

### 运行时状态

```text
phase
curr_count
steady_phase
current_step
update_rounds
update_flag
stage2_start_step
```

- `phase`：当前是 `"stage1"` 还是 `"stage2"`。
- `curr_count`：当前 elite 数量。
- `steady_phase`：stage1 填满后还剩多少稳定训练步。
- `current_step`：stage2 当前处于 update interval 的第几步。
- `update_rounds`：已经完成多少轮 stage2 update。
- `update_flag`：当前 forward/backward 是否需要收集 stage2 candidate probe。
- `stage2_start_step`：真正允许 stage2 prune/grow 的全局步数，用于计算线性衰减分母。

## 5. 前向传播：pure PyTorch GEMM

当前 DSS 不再使用自定义 `linear_sd_dss` C++/CUDA 扩展。训练主路径是纯 PyTorch GEMM。

对一个 adapter，前向流程是：

```text
y_base = base_layer(x)
x_basis = x @ B_inv.T
lambda_dense = scatter(coefficient[:curr_count], coefficient_indices[:curr_count])
y_core = F.linear(x_basis, lambda_dense)
y_delta = y_core @ A_inv.T
y = y_base + y_delta
```

其中：

- `lambda_dense` 是 core-space dense matrix，形状为 `[out_features, in_features]`。
- `lambda_dense` 只由 elite 构造。
- candidate 不进入 `lambda_dense`，因此不影响当前 forward 输出。

构造 `lambda_dense` 时使用可微的 `scatter_add`：

```text
lambda_flat = zeros(out_features * in_features)
lambda_flat.scatter_add_(coefficient_indices, coefficient)
lambda_dense = lambda_flat.view(out_features, in_features)
```

因为这个过程是可微的，所以 elite `coefficient` 的梯度由 PyTorch autograd 自动回传，不需要手动计算 elite grad。

## 6. 反向传播与 candidate probe hook

candidate 不参与前向，但需要在 backward 中拿到 probe gradient。

当前实现是在 `lambda_dense` 上注册一个轻量 hook：

```python
lambda_dense.register_hook(self.get_core_grad_hook(...))
```

hook 收到的是：

```text
grad_core = dL / d(lambda_dense)
```

然后直接按 flat index 抽出 candidate 梯度：

```text
candidate_grad = grad_core.reshape(-1)[candidate_indices]
```

再根据 `collect_mode` 写入对应统计：

```text
stage1 -> grad_cache += candidate_grad, grad_count += 1
stage2 -> candidate_grad_sums += candidate_grad
          candidate_grad_sq_sums += candidate_grad^2
off    -> 不注册 hook
```

hook 不决定阶段，也不做筛选。阶段判断仍然在 `forward()` 中根据 runtime state 产生：

```text
"stage1"
"stage2"
"off"
```

## 7. Stage1：动态填充 elite pool

stage1 的目标是从全局 core-space 坐标中不断发现值得训练的位置，直到 elite pool 填满。

流程如下：

1. 初始化时，`curr_count = 0`。
2. 随机采样一批 candidate，大小为 `candidate_size`。
3. 采样时使用 `elite_bitset` 过滤已经是 elite 的坐标。
4. 当前 candidate batch 固定 `grad_store_steps` 个 optimizer step。
5. 每个 backward step 中，hook 收集 candidate gradient 并累加到 `grad_cache`。
6. 当 `grad_count >= grad_store_steps`，`check_reinitiate(...)` 执行 stage1 refresh。

stage1 refresh 具体做：

```text
x_mean = abs(grad_cache / grad_count)
更新 search_quantile_estimator
threshold = estimated quantile
candidate_mask = x_mean > threshold
根据 low/up/remaining_budget 选择要晋升的 candidate
写入 coefficient_indices[curr_count:curr_count+k]
coefficient 新 slot 置 0
elite_bitset[new_indices] = True
curr_count += k
```

如果 `curr_count < n_frequency`，丢弃当前 candidate batch，重新采样下一批。

如果 `curr_count == n_frequency`，stage1 结束，进入 stage2。

## 8. Stage1 阈值估计器：SGD + pinball quantile

`GPUQuantileEstimator` 负责在线估计筛选阈值。

它维护一个标量参数：

```text
quantile
```

目标分位点是：

```text
target_quantile = 1 - ratio
```

更新时使用 pinball loss：

```text
loss(x, q) = q_target * (x - quantile)      if x > quantile
             (q_target - 1) * (x - quantile) otherwise
```

实现细节：

- 第一次如果 `quantile == 0` 且 candidate 分数非零，会用 `torch.quantile(...)` 热启动。
- 后续用内部 SGD optimizer 更新这个标量阈值。
- 数据会按小 batch 分批更新，避免一次性把所有 candidate 分数塞进 pinball loss。

这个 optimizer 只训练阈值，不训练模型参数。

## 9. Stage2：可选 prune/grow

stage2 在 `stage2_enabled=True` 时启用。

stage1 填满 elite 后，先进入稳定训练期：

```text
steady_phase = ceil(steady_stage_ratio * total_steps)
```

稳定期内不做 prune/grow，只训练固定 elite。

稳定期结束后，每隔 `update_interval` 步执行一次 stage2 update。每个 interval 的最后 `grad_store_steps` 步用于 candidate probe。

stage2 candidate 和 stage1 一样是临时 batch：

```text
candidate_indices
```

它只在当前 update interval 的尾部统计，不长期驻留。

每步 backward 统计：

```text
candidate_grad_sums += candidate_grad
candidate_grad_sq_sums += candidate_grad^2
```

到 update 边界：

```text
avg_grad = candidate_grad_sums / grad_store_steps
avg_grad_sq = candidate_grad_sq_sums / grad_store_steps
```

然后执行：

```text
prune: 从 elite 中选 abs(coefficient) 最小的 k 个
grow:  从 candidate 中选 abs(avg_grad) 最大的 k 个
```

`k` 从 `update_counts` 开始，并根据真正 stage2 剩余 update rounds 线性衰减。

替换时：

```text
elite_bitset[old_indices] = False
coefficient_indices[prune_slots] = incoming_candidate_indices
coefficient[prune_slots] = 0
elite_bitset[new_indices] = True
```

当前不启用 grow-vs-prune margin gate。`update_margin` 只是未来扩展占位。

## 10. Optimizer：DSSAdamW

`optimizer.py` 实现了 `DSSAdamW`，用于替代普通 AdamW。

普通 AdamW 通常维护：

```text
step: scalar
exp_avg
exp_avg_sq
```

DSSAdamW 维护：

```text
age
exp_avg
exp_avg_sq
```

其中 `age` 是和 `coefficient` 同形状的 tensor。也就是说，每个 sparse slot 有自己的年龄。

这样做的原因是 stage2 会动态替换 slot：

- 未变化的 slot 应保留自己的动量和年龄。
- 新 grow 的 slot 应使用 candidate probe 统计进行 warm start。
- 不应因为少量 slot 替换就清空整个 `coefficient` 的 optimizer state。

stage2 替换后，`rewrite_slot_state(...)` 只局部重写 changing slots：

```text
age[prune_slots] = incoming_age
exp_avg[prune_slots] = incoming_avg_grad
exp_avg_sq[prune_slots] = incoming_avg_grad_sq
```

其中 `incoming_avg_grad` 和 `incoming_avg_grad_sq` 来自 stage2 candidate probe 窗口，并按 Adam beta 做 bias counter-correction。

外部训练时推荐：

```python
from dss import create_dss_optimizer
optimizer = create_dss_optimizer(model, lr=...)
```

## 11. 外部训练循环

DSS 不自动包装 `optimizer.step()`。训练脚本必须显式调用 layout refresh。

推荐顺序：

```python
loss.backward()
optimizer.step()
model.refresh_dss_layout(
    global_step=global_step,
    total_steps=total_train_steps,
    optimizer=optimizer,
    grad_accumulation_steps=grad_accumulation_steps,
)
optimizer.zero_grad()
```

为什么 refresh 放在 `optimizer.step()` 之后？

- backward 已经收集了 candidate probe gradient。
- optimizer step 先更新当前 elite 参数。
- 然后 DSS 才消费 candidate 统计，决定是否晋升或 prune/grow。
- 最后 zero grad，进入下一步。

如果在 optimizer step 之前改 layout，会让本 step 的梯度和参数结构不对应。

## 12. Merge / unmerge

`get_delta_weight()`、`merge()`、`unmerge()` 是 dense reference/export 路径。

`get_delta_weight()` 会把当前 elite slots 还原成 dense `Delta W`：

```text
Delta W = A_inv @ Delta Lambda @ B_inv
```

`merge()` 把 `Delta W` 加到 base layer weight。

`unmerge()` 把 `Delta W` 从 base layer weight 中减回来。

这些函数不参与训练 forward。训练 forward 直接在 core-space 使用 `lambda_dense` 和 GEMM。

## 13. 与旧 DSS / SpIEL / LoCA 的关系

### 与旧 DSS 的关系

继承的思想：

- candidate 只拿梯度，不参与 optimizer step。
- elite 真正训练并参与 forward。
- 使用 `grad_store_steps` 累计候选梯度。
- 使用 quantile / pinball loss 估计阈值。
- 通过 `low/up` 限制每轮晋升数量。

不同点：

- 不使用旧 DSS 的“左 elite + 右 candidate”同一个大数组结构。
- candidate 是临时 batch，用完就丢弃。
- 不做旧 DSS 的外循环 reset 状态机。
- 稀疏坐标位于 shared-basis core space，不是原始 `W` 空间。

### 与 SpIEL/SFT 的关系

借鉴的思想：

- fixed slot sparse values + indices。
- stage2 prune bottom-k / grow top-k。
- 对 changing slots 做 optimizer state rewrite。
- 新 grow slot 可以用候选统计 warm start optimizer state。

不同点：

- DSS 有 stage1 fill，SpIEL 通常一开始就有 active sparse pool。
- DSS 的 candidate 是临时 batch。
- DSS 现在使用 PyTorch GEMM + hook，不依赖自定义 sparse CUDA kernel。

### 与 LoCA 的关系

借鉴的是 PEFT 工程外壳：

- `config.py`
- `model.py`
- `layer.py`
- adapter injection
- merge/unmerge
- trainable parameter marking

算法本体不是 LoCA。

## 14. 当前实现边界

当前版本没有实现：

- quantization / bnb / gptq 兼容。
- 复杂 trainer 子类。
- grow-vs-prune margin gate。
- 旧 DSS reset 外循环。
- 自定义 sparse CUDA kernel。

当前版本优先保证：

- shared-basis/core-space 主线清晰。
- PEFT 注入结构规范。
- stage1/stage2 语义正确。
- forward/backward 稳定可测。
- optimizer state surgery 可控。

## 15. 远程容器路径与运行约定

后续在远程容器里调试、训练或评估 DSS 时，优先使用真实硬路径，不要默认联网下载模型或数据。

### 核心路径

```text
家目录
/data/home/7250091

DSS 项目主目录
/data/home/7250091/date/DSS

PEFT 仓库根目录
/data/home/7250091/date/quest20260313/peft

PEFT 中 DSS tuner 实现目录
/data/home/7250091/date/quest20260313/peft/src/peft/tuners/dss

conda 环境目录
/data/home/7250091/date/conda_env

conda 环境名
quest

数据集总目录
/data/home/7250091/date/datasets

本地模型总目录
/data/home/7250091/date/hf_cache_models/models
```

### 修改代码时的优先位置

如果任务是修改 DSS 的 PEFT 实现，优先进入：

```bash
cd /data/home/7250091/date/quest20260313/peft/src/peft/tuners/dss
```

这个目录下的关键文件是：

- `config.py`：DSS 配置。
- `layer.py`：DSS 核心层、stage1/stage2、forward/backward。
- `model.py`：PEFT 注入、layer 遍历、`refresh_dss_layout(...)`。
- `optimizer.py`：`DSSAdamW` 与 slot-level optimizer state rewrite。
- `shared_basis.py`：shared basis / diagonalization 相关逻辑。
- `test_dss.py`：DSS 单元测试。

如果任务是跑实验、看日志或查结果，优先进入：

```bash
cd /data/home/7250091/date/DSS
```

常用子目录：

- `scripts/`：训练、评估、批处理脚本。
- `logs_commonsense/`、`logs_gsm8k/`、`logs_math/`：实验日志。
- `results_commonsense/`、`results_gsm8k/`、`results_math/`：结果输出。
- `data_processing/`：数据预处理脚本。
- `dss/`：DSS 项目侧代码，但 PEFT tuner 实现仍以 `quest20260313/peft/src/peft/tuners/dss` 为准。

### 环境激活

脚本中建议显式激活环境：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quest
```

如果远程机器的 conda 初始化路径不同，再按实际 shell 环境微调。

### 数据与模型约定

数据优先从本地目录读取：

```bash
/data/home/7250091/date/datasets/commonsense_new
/data/home/7250091/date/datasets/math
/data/home/7250091/date/datasets/evaluate
/data/home/7250091/date/datasets/ft-training_set
```

模型优先从本地模型缓存读取：

```bash
/data/home/7250091/date/hf_cache_models/models
```

后续写训练、评估或调试脚本时，应尽量使用这些硬路径。除非明确需要重新下载，否则不要默认写成：

```python
load_dataset("xxx")
AutoModelForCausalLM.from_pretrained("meta-llama/...")
```

而应改成读取远程机器上已经准备好的数据目录和模型目录。

### 一句话记忆

远程容器里需要记住五个位置：

- 项目：`/data/home/7250091/date/DSS`
- PEFT DSS tuner：`/data/home/7250091/date/quest20260313/peft/src/peft/tuners/dss`
- 环境：`conda activate quest`
- 数据：`/data/home/7250091/date/datasets`
- 模型：`/data/home/7250091/date/hf_cache_models/models`
