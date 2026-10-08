import math
import unittest
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pandas as pd

from src.asmp_13f import build_asmp_queries, build_manager_holdout, evaluate_asmp


def example_pairs() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("m1", "a1", 10.0),
            ("m1", "a2", 9.0),
            ("m1", "a3", 9.0),
            ("m2", "a1", 12.0),
            ("m2", "a3", 11.0),
        ],
        columns=["manager_cik", "typed_security_id", "position_value_usd"],
    ).assign(binary_owned=1)


def manager_split_pairs() -> pd.DataFrame:
    rows = []
    for manager_number in range(10):
        assets = ("a1", "a2", "a3") if manager_number % 2 == 0 else ("a3", "a4", "a5")
        rows.extend(
            (f"m{manager_number}", asset, value)
            for asset, value in zip(assets, (9.0, 8.0, 7.0))
        )
    return pd.DataFrame(rows, columns=example_pairs().columns[:3]).assign(binary_owned=1)


class ASMP13FTests(unittest.TestCase):
    def setUp(self) -> None:
        self.queries, self.labels = build_asmp_queries(
            example_pairs(), vocabulary=("a4", "a3", "a2", "a1")
        )

    def test_second_largest_tie_mask_preserves_original_rank(self) -> None:
        queries, labels = self.queries, self.labels
        self.assertEqual(queries.vocabulary, ("a1", "a2", "a3", "a4"))
        self.assertEqual(labels.target_columns, (1, 2))
        self.assertEqual(queries.row_indices, (0, 1))
        self.assertEqual(queries.ranked_columns[0], (0, -1, 2))
        self.assertEqual(queries.ranked_columns[1], (0, -1))
        np.testing.assert_array_equal(queries.visible.getrow(0).indices, [0, 2])
        np.testing.assert_array_equal(queries.visible.getrow(1).indices, [0])
        self.assertFalse(hasattr(queries, "target_columns"))
        self.assertNotIn(labels.target_columns[0], queries.ranked_columns[0])
        self.assertEqual(queries.visible[0, labels.target_columns[0]], 0)

    def test_uniform_baseline_is_zero_and_result_is_aggregate_only(self) -> None:
        seen_rows = []

        def score(batch):
            seen_rows.extend(batch.row_indices)
            return np.zeros((len(batch), len(batch.vocabulary)))

        result = evaluate_asmp(
            self.queries,
            self.labels,
            score,
            batch_size=1,
        )
        self.assertEqual(seen_rows, [0, 1])
        self.assertAlmostEqual(result["asmp_normalized_log_likelihood"], 0.0)
        self.assertAlmostEqual(
            result["mean_log_random_probability"],
            -(math.log(2) + math.log(3)) / 2,
        )
        self.assertAlmostEqual(result["mean_reciprocal_rank"], 0.75)
        self.assertEqual(result["coverage"], 1.0)
        self.assertIn("prefiltered_universe_not_sealed", result["benchmark"])
        self.assertNotIn("m1", str(result))
        self.assertNotIn("a1", str(result))

    def test_ratio_of_mean_log_likelihoods_and_common_candidate_set(self) -> None:
        # m1 has two candidates, m2 three. Each target has probability 1/2.
        # The observed a1 logit must not enter either denominator.
        logits = np.array(
            [[1000.0, 0.0, 1000.0, 0.0],
             [1000.0, 0.0, math.log(2.0), 0.0]]
        )
        result = evaluate_asmp(self.queries, self.labels, lambda batch: logits)
        expected = 1 - math.log(2) / ((math.log(2) + math.log(3)) / 2)
        self.assertAlmostEqual(result["asmp_normalized_log_likelihood"], expected)
        self.assertAlmostEqual(result["mean_log_target_probability"], -math.log(2))
        self.assertEqual(result["mean_reciprocal_rank"], 1.0)

    def test_explicit_out_of_vocabulary_coverage(self) -> None:
        extra = pd.DataFrame(
            [
                ("m3", "a1", 10.0, 1),
                ("m3", "a5", 9.0, 1),
            ],
            columns=example_pairs().columns,
        )
        queries, labels = build_asmp_queries(
            pd.concat([example_pairs(), extra], ignore_index=True),
            vocabulary=("a1", "a2", "a3", "a4"),
        )
        result = evaluate_asmp(
            queries,
            labels,
            lambda batch: np.zeros((len(batch), len(batch.vocabulary))),
        )
        self.assertEqual(result["input_portfolios"], 3)
        self.assertEqual(result["scored_portfolios"], 2)
        self.assertEqual(result["excluded_out_of_vocabulary"], 1)
        self.assertAlmostEqual(result["coverage"], 2 / 3)

    def test_bad_logits_or_hidden_targets_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "logit shape"):
            evaluate_asmp(self.queries, self.labels, lambda batch: np.zeros((2, 3)))
        with self.assertRaisesRegex(ValueError, "finite"):
            evaluate_asmp(
                self.queries,
                self.labels,
                lambda batch: np.full((len(batch), len(batch.vocabulary)), np.nan),
            )
        visible_target = replace(self.labels, target_columns=(0, 2))
        with self.assertRaisesRegex(ValueError, "appears in model-visible"):
            evaluate_asmp(
                self.queries,
                visible_target,
                lambda batch: np.zeros((len(batch), len(batch.vocabulary))),
            )
        invalid_target = replace(self.labels, target_columns=(99, 2))
        with self.assertRaisesRegex(ValueError, "valid vocabulary indexes"):
            evaluate_asmp(
                self.queries,
                invalid_target,
                lambda batch: np.zeros((len(batch), len(batch.vocabulary))),
            )

    def test_invalid_pairs_and_vocabulary_are_rejected(self) -> None:
        pairs = example_pairs()
        with self.assertRaisesRegex(ValueError, "unique"):
            build_asmp_queries(pd.concat([pairs, pairs.iloc[[0]]], ignore_index=True))
        with self.assertRaisesRegex(ValueError, "positive"):
            build_asmp_queries(pairs.assign(position_value_usd=0.0))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            build_asmp_queries(pairs, vocabulary=("a1", "a1", "a2", "a3"))

    def test_manager_holdout_is_deterministic_and_training_rank_two_is_visible(self) -> None:
        pairs = manager_split_pairs()
        first = build_manager_holdout(pairs, seed=17)
        shuffled_rows = pairs.sample(frac=1, random_state=31).reset_index(drop=True)
        second = build_manager_holdout(shuffled_rows, seed=17)
        self.assertEqual(first.train_manager_count, 8)
        self.assertEqual(first.test_manager_count, 2)
        self.assertEqual(first.vocabulary, ("a1", "a2", "a3", "a4", "a5"))
        self.assertEqual(first.train_ranked_columns, second.train_ranked_columns)
        np.testing.assert_array_equal(first.train_visible.toarray(), second.train_visible.toarray())
        self.assertEqual(first.test_queries.ranked_columns, second.test_queries.ranked_columns)
        self.assertEqual(first.test_labels.target_columns, second.test_labels.target_columns)
        self.assertEqual(first.test_labels.input_portfolios, 2)
        self.assertEqual(len(first.test_queries), 2)
        self.assertEqual(first.train_visible.shape, (8, 5))
        self.assertEqual(first.train_visible.nnz, 24)
        self.assertTrue(all(sequence[1] >= 0 for sequence in first.train_ranked_columns))
        self.assertTrue(all(sequence[1] == -1 for sequence in first.test_queries.ranked_columns))
        self.assertEqual(first.test_queries.vocabulary, first.vocabulary)

    def test_manager_holdout_vocab_is_training_only_and_reports_test_oov(self) -> None:
        pairs = manager_split_pairs()
        manager_ids = sorted(pairs["manager_cik"].unique())
        held_out = sorted(
            manager_ids,
            key=lambda manager: (
                sha256(f"17\0{manager}".encode("utf-8")).digest(), manager
            ),
        )[:2]
        extra = pd.DataFrame(
            # This becomes rank two for one held-out manager, but never
            # enters the training-fitted vocabulary.
            [(held_out[0], "z_test_only", 8.5, 1)], columns=pairs.columns
        )
        split = build_manager_holdout(pd.concat([pairs, extra], ignore_index=True))
        self.assertNotIn("z_test_only", split.vocabulary)
        self.assertEqual(split.test_manager_count, 2)
        self.assertEqual(split.test_labels.input_portfolios, 2)
        self.assertEqual(split.test_labels.excluded_out_of_vocabulary, 1)
        self.assertEqual(len(split.test_queries), 1)
        result = evaluate_asmp(
            split.test_queries,
            split.test_labels,
            lambda batch: np.zeros((len(batch), len(batch.vocabulary))),
        )
        self.assertEqual(result["coverage"], 0.5)

    def test_held_out_target_change_cannot_change_training_inputs(self) -> None:
        pairs = manager_split_pairs()
        original = build_manager_holdout(pairs, seed=17)
        held_out_manager = min(
            pairs["manager_cik"].unique(),
            key=lambda manager: (
                sha256(f"17\0{manager}".encode("utf-8")).digest(), manager
            ),
        )
        portfolio_assets = set(
            pairs.loc[pairs["manager_cik"].eq(held_out_manager), "typed_security_id"]
        )
        replacement = next(
            asset for asset in original.vocabulary if asset not in portfolio_assets
        )
        target_row = pairs["manager_cik"].eq(held_out_manager) & pairs[
            "position_value_usd"
        ].eq(8.0)
        self.assertEqual(int(target_row.sum()), 1)
        mutated_pairs = pairs.copy()
        mutated_pairs.loc[target_row, "typed_security_id"] = replacement
        changed = build_manager_holdout(mutated_pairs, seed=17)

        self.assertEqual(original.vocabulary, changed.vocabulary)
        self.assertEqual(original.train_manager_count, changed.train_manager_count)
        self.assertEqual(original.test_manager_count, changed.test_manager_count)
        self.assertEqual(original.train_ranked_columns, changed.train_ranked_columns)
        np.testing.assert_array_equal(
            original.train_visible.toarray(), changed.train_visible.toarray()
        )
        self.assertEqual(
            original.test_queries.ranked_columns, changed.test_queries.ranked_columns
        )
        np.testing.assert_array_equal(
            original.test_queries.visible.toarray(),
            changed.test_queries.visible.toarray(),
        )
        self.assertEqual(original.test_labels.input_portfolios, changed.test_labels.input_portfolios)
        self.assertEqual(
            sum(
                before != after
                for before, after in zip(
                    original.test_labels.target_columns,
                    changed.test_labels.target_columns,
                )
            ),
            1,
        )

    def test_manager_holdout_requires_valid_split_settings(self) -> None:
        pairs = manager_split_pairs()
        for fraction in (0, 1, -0.1, float("nan"), True, "0.2"):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, "test_fraction"):
                build_manager_holdout(pairs, test_fraction=fraction)
        with self.assertRaisesRegex(ValueError, "seed"):
            build_manager_holdout(pairs, seed=True)
        with self.assertRaisesRegex(ValueError, "at least two"):
            build_manager_holdout(example_pairs().query("manager_cik == 'm1'"))


if __name__ == "__main__":
    unittest.main()
