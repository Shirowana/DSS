# Code 基线实验协议（中文版）

## 1. 适用范围与证据口径

本文用于补跑目标论文中的代码生成基线。目标论文 PDF 是主依据，EvalPlus `v0.3.1` 的对应 tag 用于解释该版本官方 `--greedy` 的实际行为。本文不包含任何方法内部的稀疏选择、替换、预算或目标层配置。

## 2. 模型与数据

### 2.1 基础模型

- `meta-llama/Meta-Llama-3-8B-Instruct`。
- Code 任务只报告该 Instruct 模型；不能换成 LLaMA3-8B base 后仍与论文表 3 横向比较。

### 2.2 微调数据

- 数据集：`ise-uiuc/Magicoder-Evol-Instruct-110K`。
- 配置和 split：`default/train`。
- 字段：`instruction`、`response`。
- 使用全部 **111,183** 条样本。
- **不划分 validation set**，也没有训练内 test split。
- 不允许从训练集留出一部分做 best-checkpoint 选择后仍声称严格采用论文协议；若基线方法强制需要验证集，应单独标记为协议偏差。

### 2.3 最终测试集

使用 EvalPlus `0.3.1` 的 canonical Plus 数据：

| EvalPlus 数据集 | 任务数 | 同一批生成报告的指标 |
|---|---:|---|
| HumanEvalPlus | 164 | HumanEval（base tests）与 HumanEval+（base + extra tests） |
| MBPPPlus | 378 | MBPP（base tests）与 MBPP+（base + extra tests） |

这里不是四套互相独立的生成任务。对 164 个 HumanEvalPlus task 各生成一次，即可分别得到 HumanEval 和 HumanEval+；对 378 个 MBPPPlus task 各生成一次，即可分别得到 MBPP 和 MBPP+。

## 3. 训练协议
后续补跑基线，我们主要微调的超参数是学习率，以及基线特有的超参数
### 3.1 任务级硬约束

| 项目 | 设置 | 证据状态 |
|---|---|---|
| Epoch | 1 | 论文明确 |
| 使用样本 | 全部 111,183 条 | 论文明确 |
| Validation split | 无 | 论文明确 |
| 全局 batch size | 128 | 论文明确 |
| Optimizer steps | 869 | 论文明确；也与 `ceil(111183 / 128)` 一致 |
| 最大序列长度 | 3,072 tokens | 论文明确 |
| 数值精度 | bf16 | 论文明确 |
| Weight decay | 0 | 论文明确 |
| Seed | 42 | 论文明确 |
| LR scheduler | cosine decay | 论文明确 |
| Warmup ratio | 0.03 | 论文明确 |
| Maximum gradient norm | 1 | 论文明确 |

论文表 4 列出的主运行学习率为 `1.5e-4`。但附录同时明确说明代码实验中的不同稀疏方法使用了不同 sparse optimizer、selection schedule 和 learning rate。因此 `1.5e-4` 不能被宣称为所有 Code 基线的统一学习率。补跑时应优先采用各基线的 native optimizer/LR；如另做统一控制实验，再把 `1.5e-4` 作为目标论文主任务配置单独报告。

### 3.2 数据格式与 loss

- 输入源字段为 `instruction`，监督目标字段为 `response`。
- 输入加目标后的最大长度为 3,072 tokens。
- 论文未披露具体 chat template、system prompt、是否只 mask instruction loss、EOS 规则、padding side、packing 和截断方向。
- 对 Instruct 模型，chat template 的差异会显著影响训练和 EvalPlus 生成，必须把 tokenizer revision、模板文本和 label mask 规则写入 run manifest。

### 3.3 论文未统一披露的训练项

- 统一适用于所有基线的 optimizer 类型、betas 和 epsilon。
- GPU 数、每卡 microbatch、gradient accumulation；只给出 global batch 128。
- Gradient checkpointing、DeepSpeed/FSDP/ZeRO 配置。
- 是否按长度分桶或 packing。

## 4. Checkpoint 协议

Code 没有 best-checkpoint 选择：

1. 完整训练全部 869 optimizer steps。
2. 不做 validation，也不按 validation loss 选模型。
3. 训练结束后直接导出 **final in-memory adapter/model state**。
4. 不得使用 HumanEval/MBPP 的结果选 checkpoint、调参或决定重跑。

对非 adapter 型基线，“final in-memory adapter”应等价实现为导出第 869 步训练结束时的最终可部署权重；不能额外从中间 checkpoint 挑测试成绩最好的版本。

## 5. 评测协议

### 5.1 EvalPlus 版本和生成设置

| 项目 | 设置 |
|---|---|
| EvalPlus | **0.3.1**，必须固定 tag/包版本 |
| 数据 | canonical 164 HumanEvalPlus + 378 MBPPPlus tasks |
| 解码 | greedy |
| `num_beams` | 1 |
| 每题 completion 数 | 1 |
| `max_new_tokens` | 1,024（生成上限） |
| `do_sample` | `false` |
| Temperature | 不启用；`do_sample=false` 时不参与生成 |
| Generation batch size | 16；论文未披露，本协议按 `hd.md` 的统一生成器固定 |

目标论文没有单独写 evaluation batch size。EvalPlus `v0.3.1` 的官方 `run_codegen(..., greedy=True)` 会强制：

- `bs = 1`
- `n_samples = 1`
- `temperature = 0.0`

本协议采用 `hd.md` 的自定义 Hugging Face 生成器，而不是 EvalPlus 的 `run_codegen` 生成入口，因此固定 batch size 为 **16**。它仍然是 `do_sample=false`、单 beam、每题一个 completion 的 greedy decoding。为排除 padding 或批处理数值差异，首次冻结实现时必须在代表性样本上与 batch size 1 做逐条输出回归测试；测试通过后所有基线统一使用同一生成器和 batch size 16。EvalPlus 0.3.1 继续负责 sanitize 和执行评分。

自定义生成器必须显式传入 `max_new_tokens=1024`，不能继承 EvalPlus 0.3.1 provider 的默认 768-token 上限。生成器实现、依赖版本和回归测试结果必须随运行 manifest 保存。

### 5.2 Prompt、停止条件和后处理

统一使用 Hugging Face backend。每个 canonical dataset prompt 作为唯一一条 Llama-3 `user` message；不添加 system message 或任务后缀；调用 tokenizer 的 chat template，并设置 `add_generation_prompt=true`。tokenize 阶段使用 `add_special_tokens=false`。

生成停止 token 为 tokenizer EOS，以及 tokenizer 中存在且有效时的 `<|eot_id|>`。只截取 padded input width 之后的新生成 token，并用 `skip_special_tokens=true` 解码。生成配置显式清除 Llama-3-Instruct 可能自带的 sampling defaults，设置 `do_sample=false`、`num_beams=1`、`temperature=None`、`top_p=None`。之后由固定为 0.3.1 的 EvalPlus 执行 sanitize、base tests 和 plus tests。

上述 backend、prompt、停止条件和 batch size 是本项目按 `hd.md` 冻结的统一补充设置，不是论文逐字披露项。必须记录 tokenizer/model revision、Transformers 版本和 attention implementation；所有基线使用同一版本和路径。

### 5.3 执行与指标

- 使用 EvalPlus 0.3.1 执行生成代码；应在 Docker/隔离沙箱内运行不受信任代码。
- HumanEval：164 个 task 的原始 base tests 上的 Pass@1。
- HumanEval+：同 164 个 task 的 base + extra tests 上的 Pass@1。
- MBPP：378 个 task 的原始 base tests 上的 Pass@1。
- MBPP+：同 378 个 task 的 base + extra tests 上的 Pass@1。
- 每题只有一个 completion，因此报告 greedy Pass@1，不做多样本估计。
- `Avg.` 为上述四个 Pass@1 百分比的非加权算术平均。

必须保存 raw completion、sanitized solution、EvalPlus 结果缓存、包版本、数据版本和执行日志。不得只保留四个汇总数字。

## 6. 基线结果来源边界

- 论文表 3 中 LoRA、DoRA 数值标记为取自 Jiang et al. (2025)，不是目标论文按本协议重新训练的结果。
- SHiRA-SNIP、SpIEL、SMT 的数值采用目标论文自己的评测协议。
- 论文明确指出不同代码稀疏方法的 optimizer、selection schedule 和 learning rate 不同，因此补跑不能假定所有方法共用一套优化器超参数。

## 7. 运行前验收清单

- 确认模型是 LLaMA3-8B-Instruct，而不是 base。
- 校验 Magicoder split 恰为 111,183 条，且没有 validation split。
- 校验 global batch 为 128、总 optimizer steps 为 869、最大长度为 3,072。
- 训练结束只导出 final state，不做 best-checkpoint 搜索。
- 固定 EvalPlus `v0.3.1`（tag commit `e5d0ed0bab96280b60b637ec7f15b5e4841b0cb2`）；记录包和数据版本。
- 使用 greedy、1 completion、1 beam、`do_sample=false`、`temperature=None`、`top_p=None`、`max_new_tokens=1024`。
- 确认没有误用 EvalPlus 0.3.1 provider 的默认 768-token 上限。
- 使用统一 Hugging Face 生成器和 batch size 16，并保留其与 batch size 1 的输出回归测试结果。
- 对 164/378 个 task 数量做启动前断言。
- 同时报告 HumanEval、HumanEval+、MBPP、MBPP+ 和四项非加权 Avg.。

## 8. 主要依据

- 目标论文：正文 §5.3、表 3；附录 B.3、表 4，PDF pp. 7-8, 24。
- EvalPlus git tag `v0.3.1`：`evalplus/codegen.py`、`evalplus/provider/hf.py` 和 CLI 文档。
- 训练数据上游：`ise-uiuc/Magicoder-Evol-Instruct-110K` 的 `default/train` split。
