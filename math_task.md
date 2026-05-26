# Math Task

本文档记录当前数学推理实验的执行版协议和后续配置决策。旧版详细讨论见 `数学推理任务.md`，后续主线以本文档为准。

## 1. 简要协议

数学推理主实验采用 Math10K 协议：

- 模型：`LLaMA3-8B`，后续补 `Qwen2.5-7B`
- 训练集：`/root/datasets/ft-training_set/math_10k.json`
- 测评集：`GSM8K`、`SVAMP`、`AQuA`、`MAWPS`
- 方法：`LoRA`、`DoRA`、`SMT`、`DSS`
- 不报告 `AddSub`、`MultiArith`、`SingleEq`，避免 Math10K/MAWPS 来源重叠争议

训练和评测统一使用 Alpaca/DIABLO 风格 prompt：

```text
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{question}

### Response: Let's think step by step.
```

训练时只对 response 部分计算 loss，prompt 部分 label mask。评测时只 decode 新生成 token，不解析 prompt。数值题抽最后数值答案；`AQuA` 抽大写选项 `A/B/C/D/E`，不接受小写字母，避免把英文冠词 `a` 误判成选项。

当前 Math10K 预处理：

- `max_length=512`
- `max_prompt_length=256`
- `validation_samples=500`
- 发现 `55/9919` 条样本总长度超过 512，其中 `19` 条在 512 截断后可能丢最终答案
- 目前先保留这些样本，不删除；后续如有必要再对比 `max_length=768`

## 2. 当前参数设置

当前 DSS 数学训练默认配置：

```text
MODEL_NAME=Llama3-8B
TARGET_MODULES=qkvud
SCORE_METHOD=snr
THRESHOLD_MODE=oracle
N_FREQUENCY=180000
CANDIDATE_SIZE=30000
GRAD_STORE_STEPS=3
LOW=500
UP=4000
RATIO=0.1
DSS_DROPOUT=0.05
LR=1e-4
NUM_EPOCHS=3
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256
BATCH_SIZE=8
GRAD_ACCUM_STEPS=2
NUM_GPUS=2
GLOBAL_BATCH_SIZE=32
EVAL_STEPS=500
SAVE_STEPS=500
PRECISION=bf16
```

训练入口：

```bash
bash train_eval_math.sh
```

当前完整训练样例：

```text
run_name=math_Llama3-8B_dss_snr_qkvud_full_20260526_055647
train_runtime=21m 4s
train_steps=885
train_loss_final=0.2995
eval_loss_best=0.3128
best_step=500
```

训练速度和显存：

- 双卡 DDP
- 约 `1.4s/it`
- 每卡显存约 `64GB`
- 对 DSS 当前 `qkvud + candidate_size=30000 + n_frequency=180000` 配置来说可接受，但比普通 LoRA/DoRA 更重

## 3. 评测任务分配

四个评测集数量：

```text
GSM8K: 1319
SVAMP: 1000
AQuA: 254
MAWPS: 238
```

后续评测采用双卡数据并行，而不是单进程单卡顺序跑：

```text
GPU0:
  GSM8K

GPU1:
  SVAMP -> AQuA -> MAWPS
```

这样样本量约为：

```text
GPU0: 1319
GPU1: 1492
```

虽然 GPU1 样本数略多，但 GSM8K 往往生成更长，实际耗时预计比较接近。

评测建议配置：

```text
EVAL_BATCH_SIZE=4
EVAL_MAX_NEW_TOKENS=512
EVAL_NUM_BEAMS=1
do_sample=False
temperature=0
```

如果 `EVAL_BATCH_SIZE=4` 显存稳定，可以再试 `8`；如果 OOM，则退回 `2`。

评测入口：

```bash
bash scripts/eval_math_reasoning.sh <adapter_dir> all <output_dir>
```

默认 `dataset=all` 时启用双卡拆分：

```text
CUDA_VISIBLE_DEVICES=0: gsm8k
CUDA_VISIBLE_DEVICES=1: svamp aqua mawps
```

`train_eval_math.sh` 默认在训练结束后调用同一个入口并启用双卡评测。如果需要临时退回旧的单卡顺序评测：

```bash
EVAL_PARALLEL_2GPU=0 bash train_eval_math.sh
```

独立单卡/单任务调试仍使用同一脚本并指定任务名：

```bash
bash scripts/eval_math_reasoning.sh <adapter_dir> gsm8k <output_dir>
```

日志口径：

- 训练 + 评测：训练阶段只写一个 train log；评测阶段不混入 train log，只在 train log 里记录两个 eval log 路径
- 双卡评测：`GPU0` 和 `GPU1` 各写一个独立 eval log，不共同写同一个文件
- 仅评测：同样生成两个独立 eval log，分别对应 `gsm8k` 和 `svamp+aqua+mawps`

`num_beams=4` 的优点是搜索更充分，模型会同时保留 4 条候选路径，可能减少贪心解码早期选错 token 的问题，因此 MISA/DIABLO/SMT 相关代码里常见 `num_beams=4`。但代价也明显：理论上每步大约要维护 4 条 beam，时间和显存开销接近放大到 `3-4x`，实际会受 batch、KV cache 和实现影响。

当前我们优先采用：

```text
num_beams=1
```

理由：

- 更快，更适合四任务全量评测和多次调参
- 更 deterministic，`do_sample=False` 下可复现
- 数学推理主要依赖训练后的步骤生成质量，beam search 不一定稳定提升最终数值正确率
- 如果后续要和使用 `num_beams=4` 的论文严格对齐，可单独补一组 beam=4 评测作为对照
