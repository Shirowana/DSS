from __future__ import annotations

from scripts.eval_distributed import TASKS, aggregate, balanced_partition


def test_balanced_partition_uses_every_dataset_once() -> None:
    for config in TASKS.values():
        groups = balanced_partition(config["counts"], 2)
        assigned = [name for group in groups for name in group]
        assert sorted(assigned) == sorted(config["counts"])


def test_commonsense_partition_is_balanced_by_sample_count() -> None:
    counts = TASKS["commonsense"]["counts"]
    groups = balanced_partition(counts, 2)
    loads = [sum(counts[name] for name in group) for group in groups]
    assert loads == [11214, 11205]


def test_commonsense_aggregate_is_unweighted() -> None:
    counts = TASKS["commonsense"]["counts"]
    names = list(counts)
    payloads = [
        {
            "datasets": {
                name: {"correct": counts[name], "total": counts[name], "accuracy": 1.0}
                for name in names[:4]
            }
        },
        {
            "datasets": {
                name: {"correct": 0, "total": counts[name], "accuracy": 0.0}
                for name in names[4:]
            }
        },
    ]
    summary = aggregate("commonsense", payloads)
    assert summary["average_accuracy"] == 0.5
    assert "weighted_accuracy" not in summary


def test_code_aggregate_normalizes_percent_metrics() -> None:
    summary = aggregate(
        "code",
        [
            {
                "datasets": {
                    "mbpp": {
                        "total": 378,
                        "base_pass_at_1": 50.0,
                        "plus_pass_at_1": 25.0,
                    }
                }
            },
            {
                "datasets": {
                    "humaneval": {
                        "total": 164,
                        "base_pass_at_1": 100.0,
                        "plus_pass_at_1": 75.0,
                    }
                }
            },
        ],
    )
    assert summary["HumanEval"] == 1.0
    assert summary["MBPP+"] == 0.25
    assert summary["average_accuracy"] == 0.625
