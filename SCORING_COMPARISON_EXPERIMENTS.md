# DSS Scoring 对比实验

## 适用场景

当请求涉及下面这些内容时，应优先读这份文档：

- 7 种 `score_method` 的定义与对比
- 当前 scoring 实验跑到哪一步
- 哪一条是当前 `abs_mean` 主线基线
- 队列调度器如何顺序发起 scoring 实验
- 8 个 commonsense 任务的结果占位和回填格式

这份文档专门用于记录 DSS 在不同单位置重要性指标下的对比实验。目标是统一：

- 方法定义与数学含义
- 训练/评测配置
- 8 个 commonsense 任务结果记录格式
- 后续实验结论与备注

当前这 7 种方法都建立在同一个 refresh window 内，对 candidate 位置累计得到的梯度序列上。设某个参数位置在窗口内收集到的梯度为：

- `g_{i,1}, g_{i,2}, ..., g_{i,n}`

对应的样本统计量记为：

- `mu_hat = (1/n) * sum g`
- `a_hat = (1/n) * sum |g|`
- `v_hat = (1/n) * sum g^2`
- `sigma_hat = sqrt(max(v_hat - mu_hat^2, 0))`

然后把这些统计量代入不同的 scoring 公式，得到当前窗口对该位置的 importance score。

## 七种方法简介

### 1. `mean_abs`

- 公式：`S^(1) = E[|g|]`
- 样本版：`a_hat = mean(abs(g))`
- 含义：偏好平均梯度幅值大的位置，不要求方向一致
- 作用：适合和当前 `abs_mean` 做直接对照，检验正负抵消到底影响多大

### 2. `abs_mean`

- 公式：`S^(2) = |E[g]|`
- 样本版：`|mu_hat| = abs(mean(g))`
- 含义：偏好长期存在稳定净推动方向的位置
- 作用：这是当前 DSS 主线方法，也是当前 scoring 对比的基线

### 3. `mean_square`

- 公式：`S^(3) = E[g^2]`
- 样本版：`v_hat = mean(g^2)`
- 含义：Fisher / 二阶能量风格的重要性
- 作用：通常更稳，不受符号抵消影响，适合作为稳定 baseline

### 4. `rms_over_param`

- 公式：`S^(4) = sqrt(E[g^2]) / (|theta^(0)| + eps)`
- 样本版：`sqrt(v_hat) / (|theta^(0)| + eps)`
- 含义：偏好梯度强且初始参数模长小的位置
- 作用：类似 GaLLoP 的二阶归一化版本

### 5. `abs_mean_over_param`

- 公式：`S^(5) = |E[g]| / (|theta^(0)| + eps)`
- 样本版：`|mu_hat| / (|theta^(0)| + eps)`
- 含义：把 DSS 当前的方向稳定性和 GaLLoP 的参数模长归一化结合起来
- 作用：是最自然的“当前主线 + 归一化”版本

### 6. `snr`

- 公式：`S^(6) = |E[g]| / (std(g) + eps)`
- 样本版：`|mu_hat| / (sigma_hat + eps)`
- 含义：偏好平均推动大且波动小的位置
- 作用：适合 candidate 噪声较大的情况

### 7. `newton_like`

- 公式：`S^(7) = |E[g]| / (E[g^2] + eps)`
- 样本版：`|mu_hat| / (v_hat + eps)`
- 含义：对角 Newton / inverse-Fisher 风格的一维近似
- 作用：理论味更强，但对尺度和 `eps` 可能更敏感

## 当前统一实验配置

除非特别说明，后续 scoring 对比实验都尽量保持以下设置一致，只改变 `score_method`：

- 模型：`Llama3-8B`
- 训练入口：`/root/code/DSS/train_eval.sh`
- 训练方式：双卡 DDP
- `target_modules = qkvud`
- `n_frequency = 180000`
- `candidate_size = 30000`
- `grad_store_steps = 3`，其中 `snr` 例外设为 `10`
- `ratio = 0.1`
- `low = 500`
- `up = 4000`
- `threshold_mode = oracle`
- `lr = 8e-5`
- `batch_size = 16`
- `gradient_accumulation_steps = 1`
- `num_epochs = 3`
- `max_length = 256`
- `val_set_size = 120`
- `eval_steps = 10000`
- `save_steps = 10000`
- `load_best_model_at_end = 1`
- `precision = bf16`
- `warmup_steps = 100`
- `seed = 42`
- `REPORT_TO = none`，后续默认不开 `wandb`

默认补充说明：

- threshold 主链保持不变
- `oracle` 模式下，每个 refresh window 攒满后直接对当前 score 向量做一次 `torch.quantile`
- 所有方法最终都在 8 个 commonsense 任务上做统一评测
- `snr` 是唯一显式使用 `sigma_hat = sqrt(max(v_hat - mu_hat^2, 0))` 的方差/标准差指标，因此给它更宽的 10-step refresh window；其他一阶、二阶或参数归一化指标统一保持 3-step window
- 为了避免训练结束后的长时间 `wandb` 收尾阻塞，后续 scoring 实验默认关闭在线日志上报；只有用户明确要求时才重新打开

## 结果记录模板

下面这张表专门记录 7 种 scoring 方法在统一配置下的结果。当前先放占位，后续跑完实验后逐项补齐。

| score_method | grad_store_steps | status | run_name | log_file | adapter_path | best_step | boolq | piqa | social_i_qa | hellaswag | winogrande | ARC-Challenge | ARC-Easy | openbookqa | avg |
| --- | ---: | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mean_abs` | `3` | completed | `commonsense_Llama3-8B_dss_nobasis_mean_abs_nf180000_cand30000_gs3_ddp_scoringfull_20260520_031827` | `/root/code/DSS/logs_commonsense/20260520_031827_train_eval_mean_abs.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_mean_abs_nf180000_cand30000_gs3_ddp_scoringfull_20260520_031827` | `10000` | `0.7404` | `0.8988` | `0.8163` | `0.9447` | `0.8824` | `0.8217` | `0.9314` | `0.8560` | `0.8615` |
| `abs_mean` | `3` | completed | `commonsense_Llama3-8B_dss_nobasis_nf180000_cand30000_gs3_repro8619_ddp_20260513` | `/root/code/DSS/logs_commonsense/20260513_000001_train_eval_oracle.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_nf180000_cand30000_gs3_repro8619_ddp_20260513` | `10000` | `0.7367` | `0.8906` | `0.8199` | `0.9472` | `0.8887` | `0.8242` | `0.9318` | `0.8820` | `0.8651` |
| `mean_square` | `3` | completed | `commonsense_Llama3-8B_dss_nobasis_mean_square_nf180000_cand30000_gs3_ddp_scoringfull_20260520_163158` | `/root/code/DSS/logs_commonsense/20260520_163158_train_eval_mean_square.log`; eval: `/root/code/DSS/logs_commonsense/eval_commonsense_mean_square_ckpt10000_20260521.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_mean_square_nf180000_cand30000_gs3_ddp_scoringfull_20260520_163158` | `10000` | `0.7419` | `0.8863` | `0.8229` | `0.9414` | `0.8848` | `0.8225` | `0.9293` | `0.8660` | `0.8619` |
| `rms_over_param` | `3` | completed | `commonsense_Llama3-8B_dss_nobasis_rms_over_param_nf180000_cand30000_gs3_ddp_scoringfull_20260521_080807` | `/root/code/DSS/logs_commonsense/20260521_080807_train_eval_rms_over_param.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_rms_over_param_nf180000_cand30000_gs3_ddp_scoringfull_20260521_080807` | `10000` | `0.7483` | `0.8955` | `0.8229` | `0.9442` | `0.8745` | `0.8328` | `0.9251` | `0.8880` | `0.8664` |
| `abs_mean_over_param` | `3` | completed | `commonsense_Llama3-8B_dss_nobasis_abs_mean_over_param_nf180000_cand30000_gs3_ddp_scoringfull_20260522_090700` | `/root/code/DSS/logs_commonsense/20260522_090700_train_eval_abs_mean_over_param.log`; eval: `/root/code/DSS/logs_commonsense/eval_commonsense_commonsense_Llama3-8B_dss_nobasis_abs_mean_over_param_nf180000_cand30000_gs3_ddp_scoringfull_20260522_090700_20260522_090700.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_abs_mean_over_param_nf180000_cand30000_gs3_ddp_scoringfull_20260522_090700` | `10000` | `0.7492` | `0.8993` | `0.8224` | `0.9470` | `0.8808` | `0.8285` | `0.9322` | `0.8680` | `0.8667` |
| `snr` | `10` | completed | `commonsense_Llama3-8B_dss_nobasis_snr_nf180000_cand30000_gs10_ddp_scoringfull_20260522_204322` | `/root/code/DSS/logs_commonsense/20260522_204322_train_eval_snr.log`; eval: `/root/code/DSS/logs_commonsense/eval_commonsense_commonsense_Llama3-8B_dss_nobasis_snr_nf180000_cand30000_gs10_ddp_scoringfull_20260522_204322_20260522_204322.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_snr_nf180000_cand30000_gs10_ddp_scoringfull_20260522_204322` | `10000` | `0.7609` | `0.8977` | `0.8224` | `0.9478` | `0.8800` | `0.8251` | `0.9339` | `0.8800` | `0.8685` |
| `newton_like` | `3` | completed | `commonsense_Llama3-8B_dss_nobasis_newton_like_nf180000_cand30000_gs3_ddp_scoringfull_20260523_094219` | `/root/code/DSS/logs_commonsense/20260523_094219_train_eval_newton_like.log`; eval: `/root/code/DSS/logs_commonsense/eval_commonsense_commonsense_Llama3-8B_dss_nobasis_newton_like_nf180000_cand30000_gs3_ddp_scoringfull_20260523_094219_20260523_094219.log` | `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_newton_like_nf180000_cand30000_gs3_ddp_scoringfull_20260523_094219` | `10000` | `0.7609` | `0.9010` | `0.8193` | `0.9477` | `0.8642` | `0.8294` | `0.9369` | `0.8820` | `0.8677` |

## 备注模板

后续每个方法实验完成后，建议至少补这几类备注：

- threshold 是否健康
  - 是否频繁 `threshold=0`
  - 是否频繁命中 `low`
  - 是否频繁命中 `up`
- best checkpoint 在哪一步
- 后期是否出现 overfit / eval loss 变差
- 与 `abs_mean` 基线相比，提升/下降主要来自哪些任务
- 是否出现数值问题
  - `NaN`
  - score 极端 shrink
  - `snr` 或 `newton_like` 对 `eps` 过敏

