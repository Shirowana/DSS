# Baseline experiment results

## S2FT Commonsense — Llama-2-7B — learning rate 7e-5

- **Status:** Training and full evaluation completed.
- **Run ID:** `commonsense_llama2_s2ft_20260928_lr7e5_seed42`
- **Model:** Llama-2-7B base (`Llama-2-7b-hf`)
- **Method / task:** S2FT / Commonsense
- **Overall score:** **83.0629%** — unweighted mean accuracy over the eight tasks.
- **Prepared data:** 170,420 training examples; maximum sequence length 256; concatenate prompt and response, then plain right-truncate; prompt tokens are masked from loss.

### Evaluation results

| Dataset | Correct / total | Accuracy |
|---|---:|---:|
| HellaSwag | 9,071 / 10,042 | 90.33% |
| ARC-Challenge | 888 / 1,172 | 75.77% |
| BoolQ | 2,412 / 3,270 | 73.76% |
| ARC-Easy | 2,089 / 2,376 | 87.92% |
| Social IQa | 1,578 / 1,954 | 80.76% |
| PIQA | 1,567 / 1,838 | 85.26% |
| Winogrande | 1,091 / 1,267 | 86.11% |
| OpenBookQA | 423 / 500 | 84.60% |
| **Unweighted average** | — | **83.06%** |

### Training configuration

- 2-GPU DDP; global batch size 32 (8 per GPU, gradient accumulation 2)
- 3 epochs; 15,978 optimizer steps
- Learning rate `7e-5`; linear scheduler; warmup ratio 0
- AdamW: betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay 0
- BF16; gradient checkpointing disabled; max gradient norm 1.0
- Seed 42; final-state model selection; no early stopping
- S2FT-R fixed random selection: `v_ratio=0`, `o_ratio=0.052`, `u_ratio=0`, `d_ratio=0.02`
- Selected channels: `o_proj=53`, `down_proj=7045`, `v_proj=0`, `up_proj=0`

### Artifacts

- Training log: `logs/commonsense_llama2_s2ft_20260928_lr7e5_seed42/train.log`
- Run manifest: `outputs/s2ft/commonsense/Llama-2-7b-hf/commonsense_llama2_s2ft_20260928_lr7e5_seed42/run_manifest.json`
- Evaluation summary: `results/s2ft/commonsense/Llama-2-7b-hf/commonsense_llama2_s2ft_20260928_lr7e5_seed42/summary.json`

## DSS Commonsense — Llama3-8B — default SNR, learning rate 1e-4

- **Status:** Train+eval run completed according to the supplied log.
- **Run name:** `commonsense_Llama3-8B_dss_nobasis_default_snr_nf360000_cand20000_gs10_lr1e4_20260527_1430`
- **Method / task:** DSS / Commonsense
- **Model:** Llama3-8B base
- **Total parameters:** 8,087,861,248
- **Trainable parameters:** 57,600,000 (**0.7122%**)
- **Training configuration:** 2 GPUs, BF16, 3 epochs, batch size 16, gradient accumulation 1, learning rate `1e-4`, weight decay 0, warmup steps 100, seed 42.
- **DSS configuration:** target modules `qkvud`, frequency `360000`, candidate size `20000`, gradient-store steps 10, SNR score, ratio `0.05`, threshold mode `oracle`, dropout `0.05`.
- **Evaluation note:** The supplied log records `openbookqa` as **446/500 = 89.20%**. It does not contain a complete eight-task aggregate summary, so no overall Commonsense mean is recorded here.

### Artifact

- Training/evaluation log: `logs/20260527_1430_train_eval_default_snr_nf360k_lr1e4.log`
