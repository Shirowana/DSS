# Three main evaluation protocols and script map

Updated: 2026-09-20

This note summarizes the evaluation code used by the paper's three task
families: Commonsense, Math10K, and code generation. It is intended for an
experimenter who needs to inspect or reuse the evaluators without inheriting
the historical Math50K/MetaMathQA/API-judge experiments. The current paper
protocols are:

- Commonsense: eight datasets, beam 1, deterministic label extraction.
- Math10K: GSM8K/AQuA/MAWPS/SVAMP, beam 4, deterministic local judging.
- Code: HumanEval(+)/MBPP(+), greedy Pass@1 through EvalPlus 0.3.1.

The current implementation uses a flat, method-independent layout directly
under `baseline/`. Models, datasets, and method implementations are not
duplicated. Shared model loading and generation live in `evaluation_core.py`;
each public task entrypoint retains its protocol-specific prompt, parser, and
metric logic.

The current bundle contains:

```text
baseline/
├── evaluation_core.py
├── eval_commonsense.py
├── eval_math.py
├── eval_code.py
├── third_party/metamath_eval/
└── tools/code_eval_sandbox/
    └── vendored bubblewrap and EvalPlus 0.3.1 runtime
```

The evaluators are independent of training orchestration.

All three evaluators consume an exported dense Hugging Face artifact. Prompt
construction, generation, parsing, and scoring do not change with the training
method.

## 1. Commonsense evaluation

### Files

- `eval_commonsense.py`: prompt construction,
  generation, answer extraction, per-example JSON, and incremental summary.
- `evaluation_core.py`: shared dense-model loading and deterministic generation.

Use `eval_commonsense.py` directly for an already exported dense artifact.

### Paper protocol

The evaluated datasets, in canonical order, are:

1. `boolq`
2. `piqa`
3. `social_i_qa`
4. `hellaswag`
5. `winogrande`
6. `ARC-Challenge`
7. `ARC-Easy`
8. `openbookqa`

Each file is read from `assets/data/evaluate/<dataset>/test.json`. Rows contain
`instruction`, optional `input`, and `answer`. The prompt is the Alpaca-style
template:

```text
Below is an instruction that describes a task[, paired with an input that
provides further context]. Write a response that appropriately completes the
request.

### Instruction:
{instruction}

[### Input:
{input}

]### Response:
```

Paper-facing decoding uses BF16, `do_sample=False`, `num_beams=1`,
`max_new_tokens=32`, and evaluation batch size 1. The tokenizer uses left
padding and EOS as PAD when no PAD token is defined.

Judging is deterministic string parsing, not likelihood ranking:

- BoolQ: the last generated `true` or `false` token.
- PIQA/Winogrande: the last valid choice 1--2.
- SocialIQA: 1--3.
- HellaSwag: 1--4.
- ARC-Easy: 1--5, subject to each sample's actual number of choices.
- ARC-Challenge/OpenBookQA: 1--4.

The evaluator writes `<dataset>.json`; every row contains the original fields
plus `output_pred`, normalized `pred`, and Boolean `flag`. It incrementally
updates `summary.json` with correct/total/accuracy per dataset and
`average_accuracy`, the unweighted mean across completed datasets. The paper's
Commonsense `Avg.` is this unweighted eight-task mean. The pipeline may also
append pooled `weighted_accuracy`, but that is not the paper table metric.

### Core code excerpt

The following is the protocol-defining subset of
`eval_commonsense.py`. Boilerplate model/path handling is omitted, but
the prompt, decoding, label normalization, and aggregation are preserved:

```python
DATASETS = [
    "boolq", "piqa", "social_i_qa", "hellaswag",
    "winogrande", "ARC-Challenge", "ARC-Easy", "openbookqa",
]

def generate_prompt(instruction: str, input_text: str | None = None) -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction}\n\n"
            f"### Input:\n{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction}\n\n"
        "### Response:\n"
    )

def normalize_label(dataset: str, text: str | None) -> str:
    if text is None:
        return ""
    raw, lowered = text.strip(), text.strip().lower()
    if dataset == "boolq":
        matches = re.findall(r"\b(true|false)\b", raw, flags=re.IGNORECASE)
        return matches[-1].lower() if matches else ""
    patterns = {
        "piqa": [r"\b(solution|choice)\s*([12])\b", r"\b([12])\b"],
        "social_i_qa": [r"\b(answer|choice)\s*([123])\b", r"\b([123])\b"],
        "hellaswag": [r"\b(ending|choice)\s*([1234])\b", r"\b([1234])\b"],
        "winogrande": [r"\b(option|choice)\s*([12])\b", r"\b([12])\b"],
        "ARC-Challenge": [r"\b(answer|choice)\s*([1234])\b", r"\b([1234])\b"],
        "ARC-Easy": [r"\b(answer|choice)\s*([12345])\b", r"\b([12345])\b"],
        "openbookqa": [r"\b(answer|choice)\s*([1234])\b", r"\b([1234])\b"],
    }
    for pattern in patterns[dataset]:
        matches = re.findall(pattern, lowered)
        if matches:
            last = matches[-1]
            choice = last[1] if isinstance(last, tuple) else last
            return f"choice{choice}"
    return ""

generation_config = GenerationConfig(
    num_beams=1,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)
generated = model.generate(
    input_ids=inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"].to(device),
    generation_config=generation_config,
    max_new_tokens=32,
)

gold = normalize_label(dataset, row.get("answer"))
pred = normalize_label(dataset, decoded_continuation)
flag = gold == pred

summary["datasets"][dataset] = {
    "correct": correct,
    "total": total,
    "accuracy": correct / max(total, 1),
}
summary["average_accuracy"] = sum(
    item["accuracy"] for item in summary["datasets"].values()
) / len(summary["datasets"])
```

The real script additionally flushes per-example JSON every 100 examples,
validates exact source counts, and can resume only from an exact source-index
prefix. Those utilities do not change the paper score.

### Standalone command

From the workspace root:

```bash
CUDA_VISIBLE_DEVICES=0 python eval_commonsense.py \
  --model_path /path/to/exported/final_model \
  --data_root /data/home/7250091/date/datasets/evaluate \
  --result_dir /path/to/result_dir
```

For the paper's LLaMA3 block, use `MODEL_NAME=Llama3-8B` and the matching
`Meta-Llama-3-8B` base model. This task does not use the Instruct checkpoint.

## 2. Math10K evaluation

### Files

- `eval_math.py`: prompt construction, generation, numeric /
  multiple-choice / symbolic answer extraction, and per-dataset scoring.
- `evaluation_core.py`: shared dense-model loading and deterministic generation.
- `third_party/metamath_eval/util.py`: boxed-expression extraction and
  symbolic equivalence for optional MATH/MATH500 evaluation. The four-task
  paper table does not invoke API judging.

### Paper protocol

The four datasets and exact counts are:

| Dataset | Count | Judge |
|---|---:|---|
| GSM8K | 1,319 | deterministic numeric equality |
| AQuA | 254 | deterministic A--E choice equality |
| MAWPS | 238 | deterministic numeric equality |
| SVAMP | 1,000 | deterministic numeric equality |

Data are read from `assets/data/evaluate`, with directory aliases `gsm8k`,
`AQuA`, `mawps`, and `SVAMP`. Rows contain `instruction`, optional `input`, and
`answer`.

The prompt uses the same Alpaca-style wrapper as Commonsense, but begins the
response with the fixed prefix:

```text
### Response: Let's think step by step.
```

Paper-facing decoding uses BF16, `do_sample=False`, `num_beams=4`, evaluation
batch size 4, and `max_new_tokens=512`. This is deterministic beam search; it
does not call the Qwen API judge.

For GSM8K/MAWPS/SVAMP, the evaluator searches the generated continuation for
the last answer marker/number, strips commas, currency and percent signs, and
compares decimals with absolute tolerance `1e-6`. AQuA extracts the last valid
choice A--E. Every dataset output is `<dataset>.json`, with `source_index`,
prompt fields, raw `output_pred`, extracted `pred`, and Boolean `flag`.

The current evaluator processes all four complete datasets and records each
row's original `source_index`. It validates canonical dataset counts before
generation and publishes `summary.json` only after all four tasks complete:

- `average_accuracy` / `4_average_accuracy`: unweighted four-task Macro.
- `weighted_accuracy`: pooled correct answers divided by all 2,811 examples.
- `judge=deterministic_numeric_or_symbolic_equivalence`.
- `api_judge=false` and `num_beams=4`.

### Core code excerpt

This shows the evaluation and aggregation logic that defines the paper score:

```python
RESPONSE_PREFIX = "Let's think step by step."

def generate_prompt(dataset: str, instruction: str, input_text: str = "") -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input "
            "that provides further context. Write a response that appropriately "
            "completes the request.\n\n"
            f"### Instruction:\n{instruction.strip()}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response: {RESPONSE_PREFIX}\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that "
        "appropriately completes the request.\n\n"
        f"### Instruction:\n{instruction.strip()}\n\n"
        f"### Response: {RESPONSE_PREFIX}\n"
    )

def normalize_numeric_text(text: str | None) -> str:
    if text is None:
        return ""
    value = text.strip().replace(",", "").replace("$", "").replace("%", "")
    value = value.rstrip(".")
    if value.startswith("="):
        value = value[1:].strip()
    return value

def numeric_equal(gold: str, pred: str) -> bool:
    gold_value = as_decimal(gold)
    pred_value = as_decimal(pred)
    if gold_value is None or pred_value is None:
        return normalize_numeric_text(gold) == normalize_numeric_text(pred)
    return abs(gold_value - pred_value) <= Decimal("1e-6")

def extract_numeric_answer(output: str) -> str:
    patterns = [
        r"the final answer is\s*:?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"the answer is\s*:?\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"\b([-+]?\d[\d,]*(?:\.\d+)?)\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output, flags=re.IGNORECASE)
        if matches:
            return normalize_numeric_text(matches[-1])
    return ""

def extract_choice_answer(output: str) -> str:
    patterns = [
        r"answer\s*:?\s*\(?([ABCDE])\)?",
        r"the answer is\s*:?\s*\(?([ABCDE])\)?",
        r"\(([ABCDE])\)", r"\b([ABCDE])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, output)
        if matches:
            return matches[-1].upper()
    return ""

generation_config = GenerationConfig(
    num_beams=4,
    do_sample=False,
    pad_token_id=tokenizer.pad_token_id,
)
generated = model.generate(
    input_ids=inputs["input_ids"].to(device),
    attention_mask=inputs["attention_mask"].to(device),
    generation_config=generation_config,
    max_new_tokens=512,
)

if dataset == "aqua":
    pred = extract_choice_answer(output)
    flag = pred == gold.upper()
else:  # gsm8k, mawps, svamp
    pred = extract_numeric_answer(output)
    flag = numeric_equal(gold, pred)
```

The final aggregation is equivalent to:

```python
stats = list(summary["datasets"].values())
summary["average_accuracy"] = sum(x["accuracy"] for x in stats) / 4
summary["4_average_accuracy"] = summary["average_accuracy"]
summary["weighted_accuracy"] = (
    sum(x["correct"] for x in stats) / sum(x["total"] for x in stats)
)
summary["num_beams"] = 4
summary["judge"] = "deterministic_numeric_or_symbolic_equivalence"
summary["api_judge"] = False
```

An important implementation detail is that decoding slices away the entire
padded input width, so only newly generated tokens are judged. No final summary
is published until all four canonical datasets are complete.

### Standalone command

```bash
CUDA_VISIBLE_DEVICES=0 python eval_math.py \
  --model_path /path/to/exported/final_model \
  --data_root /data/home/7250091/date/datasets/evaluate \
  --result_dir /path/to/result_dir
```

## 3. Code-generation evaluation

### Files

- `eval_code.py`: GPU-only deterministic generation for
  canonical HumanEvalPlus and MBPPPlus prompts. It never executes generated
  programs on the host and invokes scoring through the sandbox wrapper.
- `evaluation_core.py`: shared dense-model loading and deterministic generation.
- `tools/code_eval_sandbox/`: vendored bubblewrap/EvalPlus runtime required
  by the local backend.

### Paper protocol

- Base model: `Meta-Llama-3-8B-Instruct`.
- HumanEvalPlus: `HumanEvalPlus-v0.1.10.jsonl.gz`, exactly 164 tasks.
- MBPPPlus: `MbppPlus-v0.2.0.jsonl.gz`, exactly 378 tasks.
- Data root: `assets/data/code_generation/official/evalplus`.
- Prompt: the dataset's canonical code prompt as a single Llama-3 `user`
  message, rendered with the tokenizer chat template and
  `add_generation_prompt=True`; no system message or task-specific suffix.
- Decoding: greedy, `do_sample=False`, `num_beams=1`, one completion per task,
  `max_new_tokens=1024`, seed 42, default generation batch size 16.
- Terminators: EOS and `<|eot_id|>` when available.
- Scoring: EvalPlus 0.3.1 base and plus tests, reported as HumanEval,
  HumanEval+, MBPP, and MBPP+ Pass@1 percentages.
- Code-table `Avg.`: the unweighted mean of those four percentages.

Generation writes `humaneval_samples.jsonl` and `mbpp_samples.jsonl`, each row
having exactly `task_id` and `solution`, plus one generation-metadata JSON per
dataset. The validator requires canonical order and exact 164/378 counts.
EvalPlus sanitizes the samples, executes them only inside the sandbox, and
writes `*_samples-sanitized_eval_results.json`. A plus test counts as passed
only if both the base and plus status are `pass`.

Never run `evalplus.evaluate` directly on the host. The end-to-end entrypoint
checks the execution boundary before spending time on GPU generation and
fails closed if neither the vendored local sandbox nor Docker/Podman is usable.

### Core code excerpt

The generation contract from `eval_code.py` is:

```python
DATASET_SPECS = {
    "humaneval": DatasetSpec(
        name="humaneval",
        filename="HumanEvalPlus-v0.1.10.jsonl.gz",
        task_prefix="HumanEval/",
        expected_count=164,
    ),
    "mbpp": DatasetSpec(
        name="mbpp",
        filename="MbppPlus-v0.2.0.jsonl.gz",
        task_prefix="Mbpp/",
        expected_count=378,
    ),
}

def chat_prompt(tokenizer, problem_prompt: str) -> str:
    messages = [{"role": "user", "content": problem_prompt}]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

# Explicitly neutralize sampling defaults shipped with Llama-3-Instruct.
model.generation_config.do_sample = False
model.generation_config.temperature = None
model.generation_config.top_p = None
model.generation_config.num_beams = 1

rendered = [chat_prompt(tokenizer, problem["prompt"]) for problem in batch]
encoded = tokenizer(
    rendered, return_tensors="pt", padding=True, add_special_tokens=False
)
terminators = [tokenizer.eos_token_id]
eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in terminators:
    terminators.append(eot_id)
generated = model.generate(
    input_ids=encoded["input_ids"].to("cuda"),
    attention_mask=encoded["attention_mask"].to("cuda"),
    do_sample=False,
    num_beams=1,
    max_new_tokens=1024,
    eos_token_id=terminators,
    pad_token_id=tokenizer.pad_token_id,
    use_cache=True,
)
continuations = generated[:, encoded["input_ids"].shape[1]:]
solutions = tokenizer.batch_decode(continuations, skip_special_tokens=True)

# Exactly one JSONL row per canonical task, in canonical order.
row = {"task_id": problem["task_id"], "solution": solution}
```

Before execution, `validate_code_eval.py` checks the raw JSONL schema, task
count, task-ID prefix, uniqueness, and canonical ordering. After EvalPlus, its
score reduction is:

```python
def score_from_evalplus(path: Path):
    task_results = json.loads(path.read_text())["eval"]
    base_pass = 0
    plus_pass = 0
    for task_id, samples in task_results.items():
        assert len(samples) == 1  # exactly one completion => Pass@1
        result = samples[0]
        base_ok = result.get("base_status") == "pass"
        plus_ok = result.get("plus_status") == "pass"
        base_pass += int(base_ok)
        plus_pass += int(base_ok and plus_ok)
    count = len(task_results)
    return 100 * base_pass / count, 100 * plus_pass / count, count

summary["HumanEval"] = summary["datasets"]["humaneval"]["base_pass_at_1"]
summary["HumanEval+"] = summary["datasets"]["humaneval"]["plus_pass_at_1"]
summary["MBPP"] = summary["datasets"]["mbpp"]["base_pass_at_1"]
summary["MBPP+"] = summary["datasets"]["mbpp"]["plus_pass_at_1"]
paper_avg = (
    summary["HumanEval"] + summary["HumanEval+"]
    + summary["MBPP"] + summary["MBPP+"]
) / 4
```

The Python entrypoint performs these stages in order and stops on the first
failed invariant:

```text
check sandbox runtime
  -> validate exported model and pinned datasets
  -> GPU generation (no generated code is executed)
  -> validate exact 164/378 raw samples
  -> sanitize inside sandbox
  -> run EvalPlus base/+ tests inside sandbox
  -> validate sanitized files and exact EvalPlus result counts
  -> atomically publish summary.json
```

The local sandbox unshares user/PID/network/IPC/UTS namespaces, drops all
capabilities, uses `no_new_privs`, exposes the Python/EvalPlus runtime and
pinned datasets read-only, hides the workspace and GPUs, and applies CPU,
address-space, file-size, process-count, file-descriptor, and wall-time limits.
The container fallback uses the pinned EvalPlus 0.3.1 image digest, no network,
a read-only root, no capabilities, private IPC, and explicit resource limits.

### End-to-end command

```bash
CUDA_VISIBLE_DEVICES=0 python eval_code.py \
  --model_path /path/to/exported/final_model \
  --data_root /data/home/7250091/date/datasets/code_generation/official/evalplus \
  --result_dir /path/to/result_dir \
  --verify_batch_equivalence 16
```

If generation was interrupted, rerun with `RESUME=1`; the code verifies that
the existing JSONL is an exact canonical prefix before appending. Do not use
`OVERWRITE=1` unless replacing an intentionally disposable result.

## Required environment and common failure checks

- Python used in the original runs: `/opt/conda/envs/py312/bin/python`.
- Evaluators load the dense `final_model/` export produced by the shared trainer.
- Evaluation results must contain the complete canonical task counts; a
  partially generated JSON is not a reportable result.
- Commonsense and Math generation run on CUDA. Code generation also requires
  CUDA, while EvalPlus execution is CPU-only and sandboxed.
- Math paper results are beam 4 plus deterministic local judging. Do not mix
  them with historical beam-1 or Qwen-API-judge results.
- Code paper results are greedy/beam-1 EvalPlus Pass@1. Do not apply the math
  numeric judge or an LLM judge to code samples.
