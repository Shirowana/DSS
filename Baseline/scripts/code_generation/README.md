# Code evaluation pipeline

The Code evaluator follows this fail-closed sequence:

```text
validate model and pinned datasets
  -> generate canonical HumanEvalPlus and MBPPPlus samples on GPU
  -> validate exact task IDs, order, counts, and one completion per task
  -> sanitize inside the pinned EvalPlus sandbox
  -> execute base and plus tests inside the sandbox
  -> validate result coverage
  -> atomically publish summary.json
```

Generated programs must never be executed directly on the host.

