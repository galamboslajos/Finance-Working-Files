"""Focused integration checks for the provisional paper-aligned ASMP runner."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from scripts.run_13f_asmp_pilot import (
    ranked_matrix,
    ranked_training_matrix,
    run_assetbert_asmp,
    run_pca_asmp,
    run_word2vec_asmp,
)
from src.asmp_13f import build_asmp_queries, build_manager_holdout


class AsmpPilotTests(unittest.TestCase):
    @staticmethod
    def tiny_split():
        assets = ("a", "b", "c", "d", "e", "f")
        rows = [
            (f"m{manager}", assets[(manager + rank) % len(assets)], float(10 - rank))
            for manager in range(10)
            for rank in range(4)
        ]
        pairs = pd.DataFrame(
            rows,
            columns=["manager_cik", "typed_security_id", "position_value_usd"],
        )
        return build_manager_holdout(pairs)

    def test_ranked_matrix_preserves_original_gap_and_denominator(self) -> None:
        pairs = pd.DataFrame(
            {
                "manager_cik": ["m1"] * 4,
                "typed_security_id": ["a", "b", "c", "d"],
                "position_value_usd": [40.0, 30.0, 20.0, 10.0],
            }
        )
        queries, labels = build_asmp_queries(
            pairs, vocabulary=("a", "b", "c", "d", "e")
        )
        self.assertEqual(queries.ranked_columns, ((0, -1, 2, 3),))
        self.assertEqual(labels.target_columns, (1,))
        np.testing.assert_allclose(
            ranked_matrix(queries).toarray(), [[1.0, 0.0, 0.5, 0.25, 0.0]]
        )
        np.testing.assert_allclose(
            ranked_training_matrix(((0, 1, 2, 3),), 5).toarray(),
            [[1.0, 0.75, 0.5, 0.25, 0.0]],
        )

    def test_all_simple_models_share_an_identical_test(self) -> None:
        split = self.tiny_split()
        results = run_pca_asmp(
            split.train_visible,
            split.train_ranked_columns,
            split.test_queries,
            split.test_labels,
            dimensions=(1,),
            seed=17,
        )
        for metrics in (
            results["uniform"],
            results["popularity_log_count_plus_one"],
            results["rs_binary"]["1"],
            results["rs_ranks"]["1"],
        ):
            self.assertEqual(metrics["scored_portfolios"], len(split.test_queries))
            self.assertEqual(metrics["candidate_policy"], "frozen_vocabulary_minus_visible_holdings")
            self.assertTrue(np.isfinite(metrics["asmp_normalized_log_likelihood"]))
        self.assertAlmostEqual(results["uniform"]["asmp_normalized_log_likelihood"], 0)

    def test_word2vec_runner_uses_shared_queries(self) -> None:
        split = self.tiny_split()
        metrics = run_word2vec_asmp(
            split.train_ranked_columns,
            split.test_queries,
            split.test_labels,
            dimensions=4,
            radius=2,
            epochs=1,
            batch_size=2,
            learning_rate=0.01,
            seed=17,
        )
        self.assertEqual(metrics["scored_portfolios"], len(split.test_queries))
        self.assertEqual(metrics["fit"]["training_managers"], split.train_manager_count)
        self.assertGreater(metrics["fit"]["training_examples"], 0)
        self.assertTrue(np.isfinite(metrics["asmp_normalized_log_likelihood"]))

    def test_assetbert_runner_uses_shared_queries(self) -> None:
        try:
            import torch  # noqa: F401
        except ImportError:
            self.skipTest("Optional PyTorch is unavailable")
        split = self.tiny_split()
        metrics = run_assetbert_asmp(
            split.train_ranked_columns,
            split.test_queries,
            split.test_labels,
            dimensions=4,
            epochs=5,
            batch_size=2,
            learning_rate=0.001,
            seed=17,
            requested_device="cpu",
        )
        self.assertEqual(metrics["scored_portfolios"], len(split.test_queries))
        self.assertGreater(metrics["fit"]["selected_training_tokens_per_epoch"][0], 0)
        self.assertGreater(
            sum(metrics["fit"]["supervised_training_rank_two_tokens_per_epoch"]), 0
        )
        self.assertEqual(metrics["fit"]["training_managers"], split.train_manager_count)
        self.assertTrue(np.isfinite(metrics["asmp_normalized_log_likelihood"]))


if __name__ == "__main__":
    unittest.main()
