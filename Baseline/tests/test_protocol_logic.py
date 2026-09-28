import unittest

from datasets import Dataset

from data_processing.common import (
    deterministic_math_split,
    tokenization_stats,
    truncate_alpaca_tokens,
    truncate_code_tokens,
)
from eval_commonsense import normalize_label
from eval_math import extract_choice_answer, extract_numeric_answer, numeric_equal


class MathSplitTests(unittest.TestCase):
    def test_split_is_deterministic_and_complete(self):
        train_a, validation_a = deterministic_math_split(9_919, seed=42)
        train_b, validation_b = deterministic_math_split(9_919, seed=42)
        self.assertEqual((train_a, validation_a), (train_b, validation_b))
        self.assertEqual(len(train_a), 9_419)
        self.assertEqual(len(validation_a), 500)
        self.assertFalse(set(train_a).intersection(validation_a))
        self.assertEqual(sorted(train_a + validation_a), list(range(9_919)))


class TokenizationStatsTests(unittest.TestCase):
    def test_zero_supervision_rows_are_recorded(self):
        dataset = Dataset.from_dict(
            {
                "was_truncated": [False, True],
                "supervised_tokens": [3, 0],
                "length": [10, 256],
            }
        )
        with self.assertWarnsRegex(UserWarning, "1 rows contain no supervised"):
            stats = tokenization_stats(dataset)
        self.assertEqual(stats["zero_supervision_rows"], 1)


class CodeTruncationTests(unittest.TestCase):
    def test_long_prompt_preserves_both_chat_boundaries_and_full_response(self):
        source = list(range(4_000))
        target = list(range(10_000, 10_100))
        truncated_source, truncated_target = truncate_code_tokens(source, target, 3_072)
        self.assertEqual(len(truncated_source) + len(truncated_target), 3_072)
        self.assertEqual(truncated_target, target)
        self.assertEqual(truncated_source[0], source[0])
        self.assertEqual(truncated_source[-1], source[-1])

    def test_long_response_retains_terminator(self):
        source = list(range(100))
        target = list(range(10_000, 14_000)) + [128_009]
        truncated_source, truncated_target = truncate_code_tokens(source, target, 3_072)
        self.assertEqual(len(truncated_source), 64)
        self.assertEqual(len(truncated_source) + len(truncated_target), 3_072)
        self.assertEqual(truncated_target[-1], 128_009)


class AlpacaTruncationTests(unittest.TestCase):
    def test_response_is_preserved_and_prompt_is_right_truncated(self):
        source = list(range(100))
        target = [1000, 1001, 1002]
        truncated_source, truncated_target = truncate_alpaca_tokens(source, target, 10)
        self.assertEqual(truncated_source, list(range(7)))
        self.assertEqual(truncated_target, target)

    def test_long_response_keeps_eos(self):
        target = list(range(20)) + [999]
        truncated_source, truncated_target = truncate_alpaca_tokens([1, 2], target, 5)
        self.assertEqual(truncated_source, [])
        self.assertEqual(truncated_target, [0, 1, 2, 3, 999])


class CommonsenseParserTests(unittest.TestCase):
    def test_boolq_uses_last_boolean(self):
        self.assertEqual(normalize_label("boolq", "false, therefore TRUE"), "true")

    def test_choice_alias_is_canonicalized(self):
        self.assertEqual(normalize_label("piqa", "I choose choice 2"), "solution2")

    def test_arc_easy_rejects_choice_beyond_sample(self):
        row = {"instruction": "Answer1: a Answer2: b Answer3: c Answer4: d"}
        self.assertEqual(normalize_label("ARC-Easy", "answer5", row), "")

    def test_prefixed_match_has_priority_over_explanation_numbers(self):
        self.assertEqual(
            normalize_label("social_i_qa", "Answer 1 because statement 3 is wrong"),
            "answer1",
        )


class MathParserTests(unittest.TestCase):
    def test_marker_has_priority_over_other_numbers(self):
        self.assertEqual(extract_numeric_answer("First 5. The answer is 1,234.50."), "1234.50")

    def test_decimal_equivalence(self):
        self.assertTrue(numeric_equal("12.0", "12"))
        self.assertTrue(numeric_equal("1.0000001", "1"))
        self.assertFalse(numeric_equal("1.01", "1"))

    def test_aqua_choice_is_case_insensitive(self):
        self.assertEqual(extract_choice_answer("The answer is (d)."), "D")


if __name__ == "__main__":
    unittest.main()
