import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from src.representation_13f import (
    _masked_ranks,
    centered_pca,
    holdings_matrix,
    mask_second_largest,
    run_smoke_test,
)
from scripts.run_13f_pca_smoke import validate_matrix_manifest


def example_pairs() -> pd.DataFrame:
    rows = [
        ("m1", "a1", 10.0), ("m1", "a2", 9.0), ("m1", "a3", 8.0),
        ("m2", "a2", 10.0), ("m2", "a3", 9.0), ("m2", "a4", 8.0),
        ("m3", "a3", 10.0), ("m3", "a4", 9.0), ("m3", "a1", 8.0),
        ("m4", "a4", 10.0), ("m4", "a1", 9.0), ("m4", "a2", 8.0),
    ]
    return pd.DataFrame(
        rows, columns=["manager_cik", "typed_security_id", "position_value_usd"]
    ).assign(binary_owned=1)


class Representation13FTests(unittest.TestCase):
    def test_masked_ranking_excludes_observed_and_breaks_score_ties(self) -> None:
        matrix = sparse.csr_matrix([[1, 0, 0, 0], [1, 0, 0, 0]])
        ranks, choices = _masked_ranks(
            matrix,
            np.array([1, 2]),
            popularity=np.array([5.0, 2.0, 2.0, 1.0]),
        )
        np.testing.assert_array_equal(ranks, [1, 2])
        np.testing.assert_array_equal(choices, [3, 3])

    def test_runner_requires_registered_matching_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "contract_version": "13f-pit-matrix-v1",
                "materialized_quarters": [
                    {
                        "quarter_directory": "report_quarter=2026Q1",
                        "pair_file_sha256": "expected-hash",
                    }
                ],
            }
            (root / "run_manifest.json").write_text(json.dumps(manifest))
            validate_matrix_manifest(root, "2026Q1", "expected-hash")
            with self.assertRaisesRegex(ValueError, "file hash"):
                validate_matrix_manifest(root, "2026Q1", "wrong-hash")
            with self.assertRaisesRegex(ValueError, "uniquely registered"):
                validate_matrix_manifest(root, "2025Q4", "expected-hash")

    def test_mask_happens_before_matrix_construction(self) -> None:
        split = mask_second_largest(example_pairs())
        self.assertEqual(len(split.test), 4)
        self.assertEqual(len(split.train), 8)
        self.assertEqual(
            split.test.set_index("manager_cik")["typed_security_id"].to_dict(),
            {"m1": "a2", "m2": "a3", "m3": "a4", "m4": "a1"},
        )
        matrix = holdings_matrix(split, "binary")
        self.assertEqual(matrix.shape, (4, 4))
        self.assertEqual(matrix.nnz, 8)
        self.assertEqual(matrix[0, 1], 0)

    def test_rank_endpoints_and_deterministic_ties(self) -> None:
        pairs = pd.DataFrame(
            [
                ("m1", "a1", 50.0),
                ("m1", "a2", 50.0),
                ("m1", "a3", 10.0),
                ("m1", "a4", 5.0),
                ("m2", "a1", 6.0),
                ("m2", "a3", 5.0),
            ],
            columns=["manager_cik", "typed_security_id", "position_value_usd"],
        ).assign(binary_owned=1)
        split = mask_second_largest(pairs)
        matrix = holdings_matrix(split, "ranks")
        # a2 is masked; remaining m1 ranks are a4=1/3, a3=2/3, a1=1.
        self.assertAlmostEqual(matrix[0, split.assets.get_loc("a4")], 1 / 3)
        self.assertAlmostEqual(matrix[0, split.assets.get_loc("a3")], 2 / 3)
        self.assertAlmostEqual(matrix[0, split.assets.get_loc("a1")], 1.0)

    def test_sparse_centered_pca_matches_dense_singular_values(self) -> None:
        dense = np.array(
            [[1, 0, 1, 0, 0], [0, 1, 1, 0, 0],
             [1, 0, 0, 1, 0], [0, 1, 0, 0, 1]],
            dtype=np.float64,
        )
        _, singular, _, means, total = centered_pca(
            sparse.csr_matrix(dense), dimensions=2, seed=17
        )
        expected = np.linalg.svd(dense - dense.mean(axis=0), compute_uv=False)
        np.testing.assert_allclose(singular, expected[:2], atol=1e-8)
        np.testing.assert_allclose(means, dense.mean(axis=0))
        self.assertAlmostEqual(total, np.square(dense - means).sum())

    def test_end_to_end_result_is_aggregate_only(self) -> None:
        result = run_smoke_test(example_pairs(), dimensions=(1, 2), seed=17)
        self.assertEqual(result["masked_pairs"], 4)
        self.assertEqual(result["train_pairs"], 8)
        self.assertEqual(result["minimum_train_assets_per_manager"], 2)
        self.assertEqual(result["minimum_train_managers_per_asset"], 2)
        self.assertEqual(set(result["models"]), {"binary", "ranks"})
        self.assertEqual(set(result["models"]["binary"]), {"1", "2"})
        self.assertNotIn("m1", str(result))
        self.assertNotIn("a1", str(result))


if __name__ == "__main__":
    unittest.main()
