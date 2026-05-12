# Torch 2.6+ Upgrade Plan

## Goal

Upgrade the current `quest` environment so DSS checkpoint resume can restore optimizer and scheduler state correctly, instead of falling back to adapter-only resume.

## Current Environment

- `torch 2.5.1+cu121`
- `torchvision 0.20.1+cu121`
- `torchaudio 2.5.1+cu121`
- `transformers 4.57.1`
- `accelerate 1.12.0`
- `datasets 2.18.0`
- `bitsandbytes 0.49.1`
- `triton 3.1.0`

## Why Upgrade

Current logs show:

- `torch<2.6 blocks optimizer/scheduler torch.load in Transformers`
- resume currently restores:
  - adapter weights
  - trainer step/epoch metadata
- resume currently does **not** restore:
  - optimizer state
  - scheduler state

This breaks continuity of:

- momentum / Adam moments
- LR schedule position
- exact training trajectory

## Compatibility Risks

### Must Upgrade Together

- `torch`
- `torchvision`
- `torchaudio`

Reason:

- `torchvision 0.20.1+cu121` requires `torch==2.5.1`
- `torchaudio 2.5.1+cu121` requires `torch==2.5.1`

### Needs Verification After Upgrade

- `triton`
- `bitsandbytes`
- local compiled extensions under:
  - `/root/code/sft-llm/peft/src/linear_sd*.so`

### Likely Safe

- `transformers`
- `accelerate`
- `datasets`
- `tokenizers`
- `safetensors`
- `numpy`

## Recommended Strategy

Do **not** mutate the current `quest` environment immediately.

Preferred order:

1. Clone current env into a new env, e.g. `quest_t26`
2. Upgrade `torch`, `torchvision`, `torchaudio` together to matching versions
3. Verify CUDA compatibility
4. Rebuild local compiled extensions
5. Run resume smoke test on the saved DSS checkpoint
6. Only switch training over after resume is verified

## Suggested Validation Checklist

### Environment Validation

- `python -c "import torch; print(torch.__version__)"`
- `python -c "import torchvision, torchaudio; print(torchvision.__version__, torchaudio.__version__)"`
- `python -m pip check`

### Runtime Validation

- `python -c "import bitsandbytes as bnb; print(bnb.__version__)"`
- `python -c "import transformers, accelerate; print(transformers.__version__, accelerate.__version__)"`

### Local Extension Rebuild

- rebuild `sft-llm/peft` extension:
  - `cd /root/code/sft-llm/peft && python setup.py build_ext --inplace`

### DSS Resume Validation

Use the existing checkpoint:

- `/root/code/DSS/output/commonsense_Llama3-8B_dss_nobasis_nf180000_cand30000_gs3_rerun_20260511/checkpoint-10000`

Check that resumed logs:

- do **not** print the unsafe resume fallback warning
- restore LR near the original run's LR at step `10000`
- continue with stable speed after checkpoint evaluation

## Success Criteria

Upgrade is considered successful only if all of the following hold:

1. `pip check` is clean
2. all required imports succeed
3. local extensions rebuild successfully
4. DSS resume restores optimizer and scheduler state
5. resumed LR matches the original run near the checkpoint step

## Rollback Plan

If anything fails:

1. keep using the old `quest` env untouched
2. delete or ignore `quest_t26`
3. continue with either:
   - fresh full reruns
   - manual optimizer restore experiments in old env

