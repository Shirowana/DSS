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
GRAD_STORE_STEPS=10
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
WARMUP_RATIO=0.03
EVAL_STEPS=500
SAVE_STEPS=500
PRECISION=bf16
```

### 2.1 即将补跑的 LoRA 基线

按与当前 math 主协议一致的数据、prompt、验证集切分与评测协议，补一条 `Llama3-8B + LoRA` 训练基线。该实验只做 train，不在同一脚本里自动接四任务评测。

配置如下：

```text
MODEL_NAME=Llama3-8B
PEFT_METHOD=lora
LORA_R=32
LORA_ALPHA=32
LORA_DROPOUT=0.05
TARGET_MODULES=qkvupdown
LR=2e-4
NUM_EPOCHS=3
MAX_LENGTH=512
MAX_PROMPT_LENGTH=256
VAL_SET_SIZE=500
BATCH_SIZE=16
GRAD_ACCUM_STEPS=1
NUM_GPUS=2
GLOBAL_BATCH_SIZE=32
WARMUP_RATIO=0.0
PRECISION=bf16
```

这里 `TARGET_MODULES=qkvupdown` 的实际含义是：

```text
q_proj
k_proj
v_proj
up_proj
down_proj
```

也就是说，这条 LoRA 基线统一对 `q / k / v / up / down` 五组线性层做微调，不包含 `o_proj` 或 `gate_proj`。

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
EVAL_BATCH_SIZE=1
EVAL_MAX_NEW_TOKENS=512
EVAL_NUM_BEAMS=1
do_sample=False
temperature=0
```

当前先把 `EVAL_BATCH_SIZE=1` 作为默认主线，优先保证输出稳定、日志清晰、双卡拆分行为可控。后续如果只做速度测试，可以单独尝试 `2/4/8`。

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

## 4. 首次成功全链路结果

第一次完整跑通的主线结果如下：

```text
run_name=math_Llama3-8B_dss_snr_qkvud_lr8e5_bs1eval_20260526_101931
model=Llama3-8B
method=DSS
target_modules=qkvud
score_method=snr
threshold_mode=oracle
lr=8e-5
epochs=3
batch_size_per_gpu=8
grad_accum_steps=2
num_gpus=2
global_batch_size=32
max_length=512
max_prompt_length=256
eval_batch_size=1
eval_max_new_tokens=512
eval_num_beams=1
train_runtime=21m 2s
train_loss_final=0.3076
eval_loss_best=0.3145
best_step=500
```

四任务得分：

```text
GSM8K: 835/1319 = 63.31
SVAMP: 741/1000 = 74.10
AQuA: 116/254 = 45.67
MAWPS: 207/238 = 86.97
4-task average: 67.51
weighted average: 67.56
```

这次结果已登记到 [runs.csv](/root/code/DSS/experiments/math/runs.csv:1)。

### 4.1 初步判断

这次结果不能算理想，主要短板在 `AQuA` 和 `GSM8K`。

- `AQuA` 只有 `45.67`，明显偏低；错误样本里有一部分甚至没有抽到合法大写选项，说明“生成稳定落到选项字母”这件事还没学牢
- `GSM8K` 有不少样本虽然中间推理接近正确，但最后一句没有明确收束，导致“取最后数值”时会抽到过程中的中间数字
- `SVAMP` 和 `MAWPS` 相对正常，说明当前链路不是完全坏掉，更像是“答案格式控制 + 任务适配”还不够稳
- 当前 DSS 超参数是从已有主线经验平移过来的，不是专门为数学任务细调出来的，因此 `lr / target_modules / score_method / threshold budget` 仍有继续搜索空间

### 4.2 后续优先排查

下一步优先看这几件事：

- 先做错误样本复查，特别是 `AQuA` 是否经常输出了推理但没稳定输出选项字母
- 对比一组 `num_beams=4` 的纯评测，看 `AQuA/GSM8K` 是否能靠更强解码显著回升
- 评估是否需要重新把训练输出进一步规范化，让模型更稳定地产生“最终答案落点”
- 如主线仍偏低，再做 `lr` 和 `target_modules` 的小范围 sweep，而不是直接大改整套协议

### 4.3 当前错误模式

这次首个完整 run 里，`AQuA` 和 `GSM8K` 的错误模式已经比较清楚：

- `AQuA`：有一部分错误不是“选错了”，而是根本没有落出合法大写选项
- `GSM8K`：有一部分错误不是“完全不会做”，而是输出里有多个数字，但没有稳定最终答案锚点

`AQuA` 当前观察：

- `pred == ""` 的样本有 `16` 条
- 其中 `13` 条输出长度正好打满 `512` 个生成 token，`2` 条是 `511` token
- 这说明一部分 `AQuA` 失败确实和 generation 被截断有关：模型一直在推理，但直到停止都没有给出可解析的 `A/B/C/D/E`

`GSM8K` 当前观察：

- 有些样本输出正文里已经出现正确答案，但最后又重复了题干数字、过程数字或“节省了多少”这种次级数值
- 当前 parser 在没有 `The final answer is ...` 这类锚点时，会退化成“取最后一个数字”
- 因此会出现“过程接近正确，但抽取结果错误”的情况
- 不过这不是全部问题；当前 `GSM8K` 的大部分错题仍然是模型本身推理或收束出了问题，而不是单纯 parser 误伤

可以记住一个简化判断：

- `AQuA` 的主要风险是“没收束到选项”
- `GSM8K` 的主要风险是“收束了，但没有稳定收束到最终答案格式”

## 5. 参考上界结果

这里先把 `MISA` 论文中的数学四任务结果作为当前文档里的参考上界栏目。

说明：

- 这部分结果来自本地参考论文 [Liu 等 - MISA Memory-Efficient LLMs Optimization with Module-wise Importance Sampling.pdf](/root/code/references/Liu%20%E7%AD%89%20-%20MISA%20Memory-Efficient%20LLMs%20Optimization%20with%20Module-wise%20Importance%20Sampling.pdf)
- 按论文 Table 4，`MISA(δ = 3%)` 是两种采样比例里更好的数学主结果，因此这里优先记这一行
- 这里的“参考上界”表示我们当前最直接对齐的一篇强结果论文，不等同于全领域绝对最终 SOTA

### 5.1 MISA 数学结果

```text
LLaMA3-8B + MISA(δ = 3%)
GSM8K: 71.3
SVAMP: 78.5
AQuA: 51.2
MAWPS: 93.3
Avg: 73.6

Qwen2.5-7B + MISA(δ = 3%)
GSM8K: 81.0
SVAMP: 88.1
AQuA: 66.1
MAWPS: 92.9
Avg: 82.0
```

### 5.2 我们与 MISA 的差距

当前我们已经有可直接比较的，是：

```text
LLaMA3-8B + DSS
GSM8K: 63.31
SVAMP: 74.10
AQuA: 45.67
MAWPS: 86.97
Avg: 67.51
```

相对 `LLaMA3-8B + MISA(δ = 3%)` 的差距为：

```text
GSM8K: -7.99
SVAMP: -4.40
AQuA: -5.53
MAWPS: -6.33
Avg: -6.09
```

从这个对照看，当前 DSS 主线和 MISA 的差距不是只出在一个任务上，而是四个任务都还有明显差距，其中：

- 最大短板是 `GSM8K`
- `AQuA` 也明显偏低，说明选择题答案落点和格式稳定性仍然不够
- `SVAMP` 相对接近一些，但仍有 `4` 分以上差距
- `MAWPS` 虽然我们分数最高，但离 MISA 也还有 `6` 分左右

### 5.3 Qwen2.5-7B 口径

`Qwen2.5-7B` 的 MISA 参考结果已经登记，但我们本地还没有 DSS 的对应完整结果。

当前原因是：

- 本地 `Qwen2.5-7B` 模型权重和 tokenizer 之前并不完整
- 因此还没有生成完整的 processed 数据并完成正式训练评测

所以现阶段能下的稳妥结论是：

- `LLaMA3-8B`：我们当前比 MISA 参考上界低 `6.09` 平均分
- `Qwen2.5-7B`：先以 `MISA Avg = 82.0` 作为后续目标线，等本地 DSS 跑通后再做同口径比较
