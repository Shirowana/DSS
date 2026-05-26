# DSS Agent Context

## 1. Executive Summary

这个项目当前在做的是：基于 `Llama3-8B` 的 commonsense 多任务指令微调，研究一套自定义的稀疏 PEFT 方法 `DSS`，并重点比较不同 threshold 策略、不同 budget、以及不同候选筛选统计量对最终效果的影响。当前主线方法不是 LoRA，也不是带 basis 的旧 DSS，而是已经收敛成一版 **`no-basis, stage1-only DSS`**。

近期新增主线：数学推理任务。新 Agent 如果接到 Math10K、GSM8K、SVAMP、AQuA、MAWPS、`train_eval_math.sh` 或 `evaluate_math_reasoning.py` 相关请求，应优先阅读 [math_task.md](/root/code/DSS/math_task.md)。该文档是当前执行版协议，旧的 [数学推理任务.md](/root/code/DSS/数学推理任务.md) 只作为历史讨论归档。

数学任务当前入口：

- 训练 + 双卡评测：[train_eval_math.sh](/root/code/DSS/train_eval_math.sh)
- 训练-only：[run_dss_math.sh](/root/code/DSS/run_dss_math.sh)
- 数学评测：[scripts/eval_math_reasoning.sh](/root/code/DSS/scripts/eval_math_reasoning.sh)，`dataset=all` 默认双卡，指定单任务时单卡调试
- Python 评测：[evaluate_math_reasoning.py](/root/code/DSS/evaluate_math_reasoning.py)
- Math10K 预处理：[data_processing/process_math10k.py](/root/code/DSS/data_processing/process_math10k.py)

目前最重要的三个结论是：

- 当前真正训练使用的 DSS 实现位于 `quest/peft/src/peft/tuners/dss/`，不是项目根目录下那份镜像副本。
- 之前 `Llama3-8B` 某些 layer 出现 `threshold=0`、`MLP 输入为 0` 的异常，最终定位到是本地模型分片损坏，不是算法本身天然现象；现在已换成校验过的干净 HF 权重。
- 自动 resume 仍然不可信。即使环境已经升级到 `torch 2.6.0`，Hugging Face Trainer 的自动恢复在当前链路下依然没有把 scheduler/optimizer 状态无缝接回，因此恢复后的 `lr` 曲线与原始训练不一致。

如果新 Agent 只需要一句话抓重点：**先把当前 DSS 主线、正式基线结果、resume 风险和正在运行的两条单卡实验理解清楚，再决定是否继续做 threshold 对比或手动 resume 修复。**

## 2. Current Project Status

当前项目环境和模型状态如下：

- 训练环境：`quest`
- 核心版本：
  - `torch 2.6.0+cu124`
  - `transformers 4.57.1`
  - `huggingface_hub 0.36.2`
  - `accelerate 1.12.0`
  - `datasets 2.18.0`
  - `bitsandbytes 0.49.1`
- 当前可用模型路径：
  - `/root/hf_cache_models/models/Meta-Llama-3-8B`

已确认并修复的关键问题：

- 旧 `Llama3-8B` 本地模型里，`model-00002-of-00004.safetensors` 哈希不对，导致 `layer10/12.post_attention_layernorm.weight` 全 0，进而让这两层 MLP 输入塌成 0。
- 现已删除坏模型，重新下载并仅保留 HF 风格权重文件；重新 probe 后，`layer10/11/12` 的 norm weight 和前向都恢复正常。

当前仍未解决的问题：

- `checkpoint-10000` 的 resume 结果不可严格信任。
- 当前 resume 会恢复 step/epoch 和 adapter 权重，但训练轨迹并不能与原始 run 完全重合，尤其 `learning_rate` 不一致。
- DSS 自定义搜索态没有完整 checkpoint，因此即使未来手动恢复 optimizer/scheduler，也仍需额外考虑 DSS runtime 状态恢复。

当前正在运行的实验：

- `train_0`：单卡 `oracle`，大预算版本 B
- `train_1`：单卡 `sgd`，主配置不变、仅替换 threshold 更新方式

当前补充笔记：

- [FOUR_EXPERIMENTS.md](/root/code/DSS/FOUR_EXPERIMENTS.md)
- [GRADIENT_IMPORTANCE_NOTES.md](/root/code/DSS/GRADIENT_IMPORTANCE_NOTES.md)

## 3. Current DSS/PEFT Method

### 3.1 方法定义

当前这版 DSS 可以理解成“直接在原权重空间上做稀疏增量选择”的 PEFT 方法：

- **没有 basis**：不再维护额外低秩 basis 或分解结构。
- **只有 stage1**：核心工作是候选位置选择与稀疏 coefficient 训练，不走旧版多阶段训练逻辑。
- **唯一训练参数是 `coefficient`**：真正进 optimizer 的不是候选索引，而是已经被选中的稀疏位置上的可训练系数。

### 3.2 候选与梯度统计

每个模块会先维护一批 `candidate_indices`。这些位置的作用是：

- 在当前 refresh window 中接收梯度观测
- 计算候选重要性分数
- 决定是否被 promote 成正式稀疏位置

当前默认分数是：

```text
score = abs(mean(g))
```

其中 `g` 是同一个 candidate 在若干 step 中收集到的 signed gradient。实现上等价于：

```text
grad_cache = sum_t g_t
grad_count = T
x_mean = abs(grad_cache / grad_count)
```

这意味着当前分数更偏好“方向稳定”的位置，而不是单纯梯度幅值大的位置。对应的替代想法与局限已经写在 [GRADIENT_IMPORTANCE_NOTES.md](/root/code/DSS/GRADIENT_IMPORTANCE_NOTES.md)。

### 3.3 Threshold 与 promote

当前支持两种 threshold 更新方式：

- `oracle`
  - 对当前候选的 `x_mean` 直接做精确 `torch.quantile`
  - 更稳定，也更接近“真实 top-ratio”筛选
- `sgd`
  - 用 pinball loss 在线拟合目标分位数
  - 更便宜，但可能更抖、更依赖超参数

当前简化数据流是：

```text
candidate -> grad collect -> score(x_mean) -> threshold -> elite promote
```

### 3.4 关键超参数含义

| 参数 | 作用 |
| --- | --- |
| `n_frequency` | 每个模块可用的总稀疏 budget |
| `candidate_size` | 每轮 refresh 观测的候选位置数 |
| `grad_store_steps` | 每个 refresh window 累积多少 step 的梯度 |
| `ratio` | 目标 quantile/top 比例 |
| `low` | 若 threshold 过严时，保底至少选多少个位置 |
| `up` | 若 threshold 过松时，单次 refresh 最多 promote 多少个位置 |
| `threshold_mode` | `oracle` 或 `sgd` |
| `quantile_lr` | `sgd` quantile estimator 的更新步长 |
| `quantile_alpha` | 当前保留项，通常保持 `0.0` |

## 4. Code Path and Runtime Entry Points

真正使用的 DSS 代码与运行入口如下：

- DSS 配置定义：
  - `/root/code/quest/peft/src/peft/tuners/dss/config.py`
- DSS 模型包装：
  - `/root/code/quest/peft/src/peft/tuners/dss/model.py`
- DSS 核心层实现：
  - `/root/code/quest/peft/src/peft/tuners/dss/layer.py`

其中几个最重要的实现点：

- `compute_x_mean`
- `update_distribution`
- `maybe_refresh_stage1`
- `collect_candidate_grads`
- `export_sparse_checkpoint`
- `restore_sparse_checkpoint`

项目内主要训练入口：

- 单独训练：
  - [run_dss.sh](/root/code/DSS/run_dss.sh)
- 训练后自动跑 8 个 commonsense eval：
  - [train_eval.sh](/root/code/DSS/train_eval.sh)
- Python 训练主入口：
  - [finetune_commonsense.py](/root/code/DSS/finetune_commonsense.py)

数据、日志、结果与记录位置：

- 训练日志：
  - `/root/code/DSS/logs_commonsense/`
- 训练输出与 checkpoint：
  - `/root/code/DSS/output/`
- eval 输出：
  - `/root/code/DSS/results_commonsense/`
- 实验 CSV：
  - `/root/code/DSS/experiments/commonsense/runs.csv`
  - `/root/code/DSS/experiments/commonsense/index.csv`

## 5. Current Hyperparameter Baselines

### 5.1 已完成双卡 oracle 基线

当前真正可参考的历史正式基线是：

- `threshold_mode = oracle`
- `n_frequency = 180000`
- `candidate_size = 30000`
- `grad_store_steps = 3`
- `ratio = 0.1`
- `low = 500`
- `up = 4000`
- `lr = 8e-5`
- `batch_size = 16`
- `num_gpus = 2`
- `num_epochs = 3`
- `precision = bf16`

对应 run：

- `commonsense_Llama3-8B_dss_nobasis_nf180000_cand30000_gs3_20260507_161806`

### 5.2 resumecheck 结果

还有一条必须单独说明的结果：

- `commonsense_Llama3-8B_dss_nobasis_nf180000_cand30000_gs3_resumecheck_20260511`

这条结果对应的是：

- 只使用 `checkpoint-10000`
- 完成后续训练与 8 任务 eval
- 平均分更高，但**不能与正式基线严格等价比较**

原因是这条 run 的 resume 轨迹并非无缝恢复。

### 5.3 当前待比较配置

当前主线想看的两组单卡实验：

1. 单卡大预算 `oracle`（版本 B）
   - `n_frequency = 360000`
   - `candidate_size = 50000`
   - `grad_store_steps = 3`
   - `ratio = 0.1`
   - `low = 800`
   - `up = 8000`

2. 单卡 `sgd`
   - `n_frequency = 180000`
   - `candidate_size = 30000`
   - `grad_store_steps = 3`
   - `ratio = 0.1`
   - `low = 500`
   - `up = 4000`
   - `quantile_lr = 0.01`
   - `threshold_log_every_steps = 500`

如果新 Agent 需要快速复现主线，仍建议优先从双卡 `oracle 18w` 基线开始理解，再看单卡对照。

## 6. Experimental Results and Findings

### 6.1 当前正式基线结果

双卡 `oracle 18w` 旧基线的 8 任务结果为：

- `boolq = 0.6905`
- `piqa = 0.8330`
- `social_i_qa = 0.7830`
- `winogrande = 0.8090`
- `ARC-Challenge = 0.6962`
- `ARC-Easy = 0.8460`
- `openbookqa = 0.8200`
- `hellaswag = 0.8908`

### 6.2 resumecheck 结果

`checkpoint-10000` 续出来那条的 8 任务最终值是：

- `boolq = 0.7443`
- `piqa = 0.8901`
- `social_i_qa = 0.8224`
- `winogrande = 0.8824`
- `ARC-Challenge = 0.8106`
- `ARC-Easy = 0.9322`
- `openbookqa = 0.8700`
- `hellaswag = 0.9429`
- 平均分 `0.8619`

这个结果必须带着限制去看：**它不是与正式基线完全同等条件的 clean rerun**。

### 6.3 关键观察

当前最重要的实验观察有三条：

1. 某些 layer 早期曾经出现 `threshold=0` 与 `x_mean` 全 0 的情况，尤其集中在部分 `mlp.up_proj/down_proj`。
2. 这类现象有一部分并不是算法本身导致，而是旧 `Llama3-8B` 本地权重损坏造成的；当前已修复模型侧问题。
3. 即便修复了模型，`oracle` 与 `sgd` 的 threshold 稳定性、以及 budget 增大后的选点质量，仍然是接下来最值得盯的变量。

## 7. Open Problems and Risks

当前需要所有新 Agent 都注意的风险：

- 自动 resume 不可靠。当前观察到的核心问题是 `learning_rate` 曲线在恢复后与原始 run 不一致。
- DSS 自定义搜索状态不能完整 checkpoint。即使 adapter 权重恢复，`candidate_indices / grad_cache / grad_count / quantile estimator runtime state` 也不能无缝接上。
- `sgd` threshold 模式还没有完成正式结论，当前只是开始对照实验。
- budget 从 `18w` 扩到 `36w` 后，可能会带来更多位置，但也可能引入更多低质量位置；需要结合 threshold 健康日志一起看。
- 当前仓库内有一份项目内 DSS 镜像实现，但训练实际走的是 `quest/peft/src/peft/tuners/dss/`。后续改代码时不能改错位置。

## 8. Next Planned Experiments

当前明确的下一步不是再大范围改方法，而是把这几条实验链跑清楚：

1. 单卡大 budget `oracle`
   - 看 `36w + 5w candidate` 是否真正带来收益
2. 单卡 `sgd`
   - 保持主配置不变，仅替换 threshold 更新方式
   - 同时观察 threshold 健康日志
3. 继续比较 `abs(mean(g))` 与 `mean(abs(g))`
   - 至少在分析层面继续保留这个分叉
4. 若后续必须继续用 resume
   - 优先考虑手动恢复 `optimizer.pt` 与 `scheduler.pt`
   - 再决定是否进一步补 DSS runtime state 的恢复

## 9. How To Use This Document

建议新 Agent 的阅读顺序是：

1. 先读这份文档
2. 再读 [FOUR_EXPERIMENTS.md](/root/code/DSS/FOUR_EXPERIMENTS.md)
3. 再读 [GRADIENT_IMPORTANCE_NOTES.md](/root/code/DSS/GRADIENT_IMPORTANCE_NOTES.md)
4. 然后去看：
   - [train_eval.sh](/root/code/DSS/train_eval.sh)
   - [finetune_commonsense.py](/root/code/DSS/finetune_commonsense.py)
   - `/root/code/quest/peft/src/peft/tuners/dss/layer.py`

如果只需要快速确认最新实验状态，再去看：

- `/root/code/DSS/logs_commonsense/`
- `/root/code/DSS/experiments/commonsense/index.csv`

这份文档应被视为“当前总入口上下文”，其目标不是替代论文或 README，而是缩短新 Agent 的接手时间。
