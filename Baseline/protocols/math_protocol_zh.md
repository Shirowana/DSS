# Math 基线实验协议（中文版）

## 1. 适用范围与证据口径

本文用于补跑目标论文中的数学推理基线。目标论文 PDF 是主依据；论文明确引用的 GAST 四任务评测协议仅用于补充评测数据定义。本文不包含任何方法内部的稀疏选择、替换、预算或目标层配置。

严格区分三类信息：

- **论文明确**：可以直接作为复现实验的硬约束。
- **引用协议/上游明确**：目标论文没有展开，但其明确引用的评测协议或规范数据文件给出。
- **未披露**：不能靠其他旧脚本或经验默认值冒充论文设置。

## 2. 模型与数据

### 2.1 基础模型

- `meta-llama/Llama-2-7b-hf`（LLaMA2-7B base）
- `meta-llama/Meta-Llama-3-8B`（LLaMA3-8B base）
- 不使用 Instruct 版本。

### 2.2 微调数据：Math10K

- 目标论文声称来源为：从 MetaMathQA 采样得到的 Math10K。
- 总样本数为 **9,919** 条。
- 目标论文明确使用：
  - 训练集：**9,419** 条。
  - 验证集：**500** 条。
- 训练和验证合计 9,919 条，没有另外的训练内测试集。

严格复现注意事项：论文没有给出 MetaMathQA sample IDs、数据文件哈希、500 条验证样本的索引、随机抽样算法、随机数生成器版本或独立 split 文件。仅知道全局 seed 为 42，不能据此断言采样或拆分一定使用 seed 42。

另有一项必须保留的来源冲突：目标论文写“sampled from MetaMathQA”，但它所引用的 GAST 协议把 Math10K 描述为 Hu et al. (2023)/LLM-Adapters 整理的数据；公开 LLM-Adapters `math_10k.json` 也恰好有 9,919 条。条数相同不能证明内容相同。补跑前应优先取得作者实际使用的数据文件及 split manifest；若只能采用 LLM-Adapters 文件，必须标记为“替代数据版本”，不能声称完成数据级严格复现。

### 2.3 最终测试集

仅评测下列四个官方 test split：

| 数据集 | 样本数 | 答案类型 |
|---|---:|---|
| GSM8K | 1,319 | 数值答案 |
| AQuA | 254 | 五选一，A-E |
| MAWPS | 238 | 数值答案 |
| SVAMP | 1,000 | 数值答案 |
| 合计 | **2,811** | 用于 pooled/Weighted Avg. |

不得使用 MultiArith、AddSub、SingleEq 等任务替代上述四任务协议。目标论文没有给出 9,919 条 MetaMathQA 子集的 sample IDs，因此也不能仅根据另一版本 Math10K 的构成给四个测试任务擅自贴 ID/OOD 标签。

## 3. 训练协议

后续补跑基线，我们主要微调的超参数是学习率，以及基线特有的超参数

### 3.1 任务级硬约束

| 项目 | 设置 | 证据状态 |
|---|---|---|
| Epoch | 3 | 论文明确 |
| 全局 batch size | 8 | 论文明确 |
| GPU 数 | 2 | 论文明确 |
| 每卡 microbatch | 4 | 论文明确 |
| Gradient accumulation | 1 | 论文明确 |
| Optimizer steps | 3,534 | 论文明确；也与 `ceil(9419 / 8) * 3` 一致 |
| 最大序列长度 | 512 tokens | 论文明确 |
| 数值精度 | bf16 | 论文明确 |
| Weight decay | 0 | 论文明确 |
| Seed | 42 | 论文明确 |
| LR scheduler | linear | 论文明确 |
| Warmup ratio | 0.03 | 论文明确 |
| Maximum gradient norm | 1 | 论文明确 |

论文表 4 同时列出主运行学习率：LLaMA2-7B 为 `4e-4`，LLaMA3-8B 为 `5e-5`。这是目标论文主配置中的通用优化数值，但论文没有声明每一种外部基线都共享它。补跑某个基线时：若该基线在目标论文中由作者自行实现且没有另行披露学习率，可将上述值作为对齐目标论文任务配置的首选；若基线原论文有明确 native recipe，应同时保留 native 版本并清楚区分，不能把两者混报。

### 3.2 训练数据处理

- 使用 causal language modeling 方式微调数学解题样本。
- 输入截断后的总长度不得超过 512 tokens。
- 论文没有披露：具体 prompt wrapper、是否只对 response 计算 loss、EOS 添加规则、padding side、packing、长样本截断方向、数据 shuffle 实现。
- 上述未披露项必须在代码配置和运行 manifest 中显式记录；不同基线必须保持相同，除非方法本身有不可避免的要求。

### 3.3 论文未统一披露的训练项

下列值不得从旧工程默认值中补入：

- Optimizer 类型及 Adam betas/epsilon。
- Gradient checkpointing、DeepSpeed/FSDP/ZeRO 配置。
- Dropout、数据加载 worker 数和 distributed sampler 的 padding/drop-last 行为。
- 验证 batch size，以及 validation loss 的 token 加权/样本加权计算方式。

## 4. Checkpoint 与 best model 协议

这是三个任务中唯一明确给出 best-checkpoint 规则的任务。

1. 所有运行必须完整训练到 3,534 optimizer steps，不能 early stop。
2. 在训练进度的 **75%、80%、85%、90%、95%、100%** 保存并执行验证。
3. 只在这些“eligible checkpoints”中比较 500 条验证集上的 validation loss。
4. 导出 validation loss 最低的 checkpoint/adaptor 作为最终模型。
5. 测试集不得参与 checkpoint 选择、超参数选择或失败重跑决策。

实现时必须注意：论文没有规定百分比映射为整数 step 时采用 floor、ceil 还是 round，也没有规定 loss 聚合方式。实现前应固定规则并写入 manifest。不能在看到测试准确率后改变取整规则。

## 5. 评测协议

### 5.1 生成设置

| 项目 | 设置 |
|---|---|
| 解码方式 | deterministic beam search |
| `num_beams` | 4 |
| `do_sample` | `false`（由 deterministic beam search 的语义确定） |
| 每题最终答案数 | 1 个用于计分的输出 |
| Evaluation batch size | 4；论文未披露，本协议按统一 evaluator 固定 |
| `max_new_tokens` | 512；论文未披露，本协议按统一 evaluator 固定 |
| `temperature/top_p/top_k` | 不启用 |
| 精度 | bf16；评测 dtype 未单独披露，本协议按统一 evaluator 固定 |

上述 batch size 4、512-token 上限和 bf16 采用 `hd.md` 中的统一实现。它们是本项目冻结的补充设置，不是论文逐字披露的参数。不要改用旧脚本中的 `batch_size=16`、`max_new_tokens=256` 或采样参数。

### 5.2 Prompt 与答案解析

统一使用 `hd.md` 中的 Alpaca-style prompt，并在 response 起始处固定加入 `Let's think step by step.`。目标论文声明遵循 GAST 的四任务 arithmetic-reasoning protocol；该引用协议给出的任务格式为：

- GSM8K、SVAMP：`Q:` 问题，`A:` 中包含推理和最终数值。
- AQuA：问题附 `(A)` 至 `(E)` 选项，最终答案为选项字母。
- MAWPS：`Q:` 问题，`A:` 为数值答案。

目标论文和 GAST 没有给出可执行 parser，因此本项目按 `hd.md` 冻结以下统一规则：

- GSM8K、MAWPS、SVAMP：按优先级查找 `the final answer is`、`the answer is`、`####` 和普通数字；在首个有匹配的模式内取最后一个匹配。
- 数值归一化移除逗号、美元符号和百分号，去掉末尾句点及开头等号。
- 可转为十进制时使用绝对误差 `1e-6`；否则比较归一化字符串。
- AQuA 按模式优先级抽取，并在首个有匹配的模式内取最后一个 A-E 字母。
- 无法解析时返回空字符串并计错。

这些是统一的本地补充设置，不是论文披露项。必须为 parser 保留单元测试，所有基线共用，不能根据模型输出特点修改。

### 5.3 指标

- 每个任务分别报告 accuracy（百分比）。
- `Avg.`：四个任务 accuracy 的非加权算术平均：

  `Avg = (Acc_GSM8K + Acc_AQuA + Acc_MAWPS + Acc_SVAMP) / 4`

- `Weighted Avg.`：把四个测试集合并后的 pooled accuracy：

  `Weighted Avg = (四任务正确数总和) / 2811`

- 同时保存逐样本 prompt、原始生成、解析答案、标准答案和 correct 标记，以便审计。

## 6. 基线结果来源边界

- 论文表 2 中 LoRA、DoRA 数值标记为取自 Yang et al. (2024)，不是目标论文按本协议重新训练的结果。
- SHiRA-SNIP、SpIEL、SMT 的表中数值为目标论文实现所得。
- 因此不能声称所有表 2 基线共享完全相同的 optimizer、学习率或 checkpoint 选择流程。

## 7. 运行前验收清单

- 固化 9,419/500 的 split manifest 和哈希。
- 确认基础模型是 base 而非 Instruct。
- 核对 global batch 严格为 8，且 optimizer step 总数为 3,534。
- 固化六个验证时点的整数 step 映射规则。
- 验证 best checkpoint 只按 validation loss 选择。
- 评测固定 `num_beams=4`、`do_sample=false`。
- 固定 eval batch size 4、`max_new_tokens=512`、bf16、Alpaca prompt、response prefix 和 parser 版本，并在 manifest 中记录。
- 同时输出四项 accuracy、Avg. 和按 2,811 条样本计算的 Weighted Avg.。

## 8. 主要依据

- 目标论文：正文 §5.2、表 2；附录 B.3、表 4，PDF pp. 7, 24-25。
- GAST（Yao et al., 2026）：§4.1、附录 B.1、表 5 和表 9，仅用于目标论文明确引用的四任务数据与格式定义。
- AGI-Edgerunners/LLM-Adapters 的 `ft-training_set/math_10k.json` 仅作为 9,919 条公开候选版本交叉核验；目标论文未提供哈希证明它就是实际训练文件。
