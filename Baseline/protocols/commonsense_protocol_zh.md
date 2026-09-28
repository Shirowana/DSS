# Commonsense 基线实验协议（中文版）

## 1. 适用范围与证据口径

本文用于补跑目标论文中的常识推理基线。目标论文 PDF 是主依据；Commonsense170K 规范上游和论文引用的标准任务 split 仅用于展开数据构成。本文不包含任何方法内部的稀疏选择、替换、预算或目标层配置。

## 2. 模型与数据

### 2.1 基础模型

- `meta-llama/Llama-2-7b-hf`（LLaMA2-7B base）
- `meta-llama/Meta-Llama-3-8B`（LLaMA3-8B base）
- 两者都不是 chat/Instruct 版本。

### 2.2 微调数据：Commonsense170K

Commonsense170K 是八个标准任务的训练 split 直接合并形成的联合微调集，共 **170,420** 条：

| 任务 | 训练条数 |
|---|---:|
| BoolQ | 9,427 |
| PIQA | 16,113 |
| Social IQA (`social_i_qa`) | 33,410 |
| HellaSwag | 39,905 |
| WinoGrande | 63,238 |
| ARC-Challenge | 1,119 |
| ARC-Easy | 2,251 |
| OpenBookQA | 4,957 |
| 合计 | **170,420** |

数据划分规则：使用规范上游已经给定的各任务 `train.json`/`test.json`；只合并八个 `train.json` 进行联合训练，八个 `test.json` 始终分开评测。这里的 `test.json` 是 LLM-Adapters/Commonsense170K 协议使用的 evaluation split 名称；对 BoolQ、PIQA、Social IQA、HellaSwag、WinoGrande 等公开标签受限的数据集，它可能对应原始 benchmark 的公开 validation/dev split，不能仅凭文件名宣称是原始 benchmark 的隐藏 test split。不得从这些 evaluation split 抽取训练或验证数据。

目标论文没有声明从 170,420 条训练数据中另划 validation set，也没有给出 validation 数量或索引。因此不能把其他仓库曾使用的 120、2,000 等验证集大小写成该论文设置。

### 2.3 最终测试集

| 数据集 | 测试条数 | 预期答案标签 |
|---|---:|---|
| BoolQ | 3,270 | `true` / `false` |
| PIQA | 1,838 | `solution1` / `solution2` |
| Social IQA | 1,954 | `answer1` ... `answer3` |
| HellaSwag | 10,042 | `ending1` ... `ending4` |
| WinoGrande | 1,267 | `option1` / `option2` |
| ARC-Easy | 2,376 | `answer1` ... `answer5`（按样本实际选项数） |
| ARC-Challenge | 1,172 | `answer1` ... `answer4` |
| OpenBookQA | 500 | `answer1` ... `answer4` |

每个任务只在上述协议封装的完整 evaluation split 上计算 accuracy，不混合为一个 pooled accuracy。

## 3. 训练协议
后续补跑基线，我们主要微调的超参数是学习率，以及基线特有的超参数
### 3.1 任务级硬约束

| 项目 | 设置 | 证据状态 |
|---|---|---|
| Epoch | 3 | 论文明确 |
| 全局 batch size | 32 | 论文明确 |
| 最大序列长度 | 256 tokens | 论文明确 |
| 数值精度 | bf16 | 论文明确 |
| Weight decay | 0 | 论文明确 |
| Seed | 42 | 论文明确 |

论文表 4 列出的主运行学习率为：

- LLaMA2-7B：`1.5e-4`
- LLaMA3-8B：`8e-5`

这两个值属于目标论文的主任务配置，但论文表 1 的基线结果全部来自不同既有工作，论文没有证明这些外部基线都使用相同学习率。补跑时可以将其作为统一控制版本的任务学习率；若目标是复现某篇基线原始数值，还应保留该基线的 native learning rate，并把两类结果分开报告。

### 3.2 训练数据处理

- 八个训练集混合为单个 170,420 条的 SFT 数据集，而不是依次按任务训练八次。
- 最大序列长度严格为 256 tokens。
- 论文未披露：八任务是否加权采样、具体 shuffle 实现、prompt wrapper、是否仅对 response 计算 loss、EOS、padding side、packing 和截断方向。
- 默认按样本均匀 shuffle 会使大数据集贡献更多更新；若采用任务均衡采样，属于额外实验，必须另行标注。

### 3.3 论文未披露的训练项

- Optimizer 类型、betas、epsilon。
- LR scheduler 和 warmup。
- Maximum gradient norm。
- GPU 数、每卡 microbatch 和 gradient accumulation；论文只给 global batch 32。
- Gradient checkpointing、DeepSpeed/FSDP/ZeRO 配置。
- Validation batch size 和 validation loss 定义（因为 validation split 本身也未披露）。

## 4. Checkpoint 与 best model 协议

目标论文没有给出 Commonsense 的 validation split、验证频率、eligible checkpoint、best metric 或 `load_best_model_at_end` 规则。因此：

- 不能声称“按 validation loss 选 best checkpoint”是论文设置。
- 不能使用八个 test split 的 accuracy 选择 checkpoint。
- 不能从 Math 的 75/80/85/90/95/100% checkpoint 规则类推到 Commonsense。

严格复现仍缺作者信息。补跑前必须预先选择并记录本地规则。若没有补充信息，最保守且不看测试集的本地约定是使用第 3 个 epoch 结束时的 final checkpoint，但必须在报告中标注为“本地约定，非论文明确设置”。

## 5. 评测协议

### 5.1 生成设置

| 项目 | 设置 |
|---|---|
| `num_beams` | 1（论文明确） |
| Greedy / sampling | `do_sample=false`，单 beam greedy decoding；论文未明确，本协议按统一 evaluator 固定 |
| Evaluation batch size | 1；论文未披露，本协议按统一 evaluator 固定 |
| `max_new_tokens` | 32；论文未披露，本协议按统一 evaluator 固定 |
| `temperature/top_p/top_k` | 不启用；论文未披露，本协议按非采样解码固定 |
| 每题计分输出数 | 1 |
| 评测 dtype | bf16；论文未单独披露，本协议按统一 evaluator 固定 |
| Padding | left padding；没有 PAD token 时使用 EOS token 作为 PAD |

上述 `do_sample=false`、batch size 1、32-token 上限、bf16 和 padding 规则采用 `hd.md` 中的统一实现。它们是本项目冻结的补充设置，不是论文逐字披露的参数。所有基线必须共用，不能按模型单独调整。

不要从本地 S2FT 脚本抄入 4 beams、`batch_size=8`、`max_new_tokens=256`、temperature 0.1、top-p 0.75 或 top-k 40；它们与目标论文明确给出的 1 beam 至少部分冲突，且不属于目标论文披露值。

### 5.2 Prompt 与答案抽取

统一使用 Alpaca-style prompt：有 `input` 时依次组织 Instruction、Input、Response；无 `input` 时组织 Instruction、Response。具体模板以 `hd.md` 所列文本为准，生成内容从 `### Response:` 后开始。

规范数据把每道题转换成显式答案标签：

- BoolQ：匹配 `true|false`。
- PIQA：匹配 `solution1|solution2`。
- Social IQA：匹配 `answer1` 至 `answer3`。
- ARC-Challenge、OpenBookQA：匹配 `answer1` 至 `answer4`。
- ARC-Easy：按样本实际选项数匹配，最多 `answer1` 至 `answer5`。
- HellaSwag：匹配 `endingN`。
- WinoGrande：匹配 `optionN`。

答案解析忽略大小写，并取生成文本中最后一个合法标签；无法解析时返回空标签并计错。BoolQ 接受最后一个 `true|false`；其余任务同时接受规范标签（如 `answer2`）及合法范围内的裸数字。Gold 与 prediction 经同一归一化函数处理。该 prompt/parser 是本项目按 `hd.md` 冻结的统一约定，不是论文披露项；必须用单元测试覆盖无标签、多个标签、大小写变化、越界标签和标签出现在解释文本中的情况。

### 5.3 指标

- 八个任务分别报告 accuracy（百分比）。
- `Avg.` 是八个任务 accuracy 的非加权算术平均，而不是按测试集大小加权：

  `Avg = sum(八项 task accuracy) / 8`

- 论文不报告 commonsense pooled/weighted accuracy。
- 保存逐样本 prompt、原始生成、解析标签、标准标签和 correct 标记。

## 6. 基线结果来源边界

论文表 1 的基线值不是目标论文统一重跑所得：

- LoRA：取自 Xu et al. (2026)。
- DoRA、SpIEL、SMT：取自 He et al. (2025)。
- SHiRA-SNIP：取自 Bhardwaj et al. (2024)。

因此表 1 只能用于结果比较，不能反推出这些方法共享同一 optimizer、学习率、prompt、checkpoint 或解码实现。补跑结果应同时报告“统一控制协议”和“native recipe”中的哪一种。

## 7. 运行前验收清单

- 验证八个 train split 数量之和恰为 170,420。
- 验证八个 test split 数量分别为 3,270/1,838/1,954/10,042/1,267/2,376/1,172/500。
- 确认基础模型是 base 而非 Instruct。
- 核对 3 epochs、global batch 32、最大长度 256、bf16、weight decay 0、seed 42。
- 在启动前锁定 checkpoint 规则，禁止依据 test accuracy 选模型。
- 评测固定 `num_beams=1`、`do_sample=false`、batch size 1、`max_new_tokens=32` 和 bf16。
- 使用冻结的 Alpaca prompt、最后合法标签 parser、left padding 和 EOS-as-PAD 规则。
- 报告八项 accuracy 和非加权 Avg.，不额外用 pooled accuracy 替代。

## 8. 主要依据

- 目标论文：正文 §5.1、表 1；附录 B.3、表 4，PDF pp. 7, 24。
- Commonsense170K 规范上游：AGI-Edgerunners/LLM-Adapters 的 `ft-training_set/commonsense_170k.json` 及八个任务的 train/test JSON。
- GAST（Yao et al., 2026）§4.1、附录表 5 和表 8，仅用于标准 split 和答案格式的交叉核验。
