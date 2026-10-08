"""Synthetic checks for the manager-only, point-in-time ASMP universe."""

import unittest

import pandas as pd

from scripts.run_13f_manager_only_asmp import _assert_point_in_time
from src.asmp_13f import (
    MASKED_ASSET,
    build_asmp_queries,
    build_partitioned_manager_holdout,
    split_manager_ids,
)
from src.manager_only_asmp_13f import build_manager_only_quarter
from src.matrix_13f import MatrixParameters


ASSETS = tuple(f"cash_share_candidate:00000000{index}" for index in range(8))


def _row(manager: str, asset: str, value: float) -> dict[str, object]:
    return {
        "manager_cik": manager,
        "period_of_report": "2024-03-31",
        "accession_no": f"synthetic-{manager}",
        "typed_security_id": asset,
        "matrix_eligible": True,
        "value_usd_numeric": value,
    }


def _pairs(manager: str, positions: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "manager_cik": manager,
            "typed_security_id": asset,
            "position_value_usd": value,
        }
        for asset, value in positions
    )


class ManagerOnlyASMP13FTests(unittest.TestCase):
    def setUp(self) -> None:
        self.managers = tuple(f"m{index:02d}" for index in range(8))
        self.fit_ids, self.test_ids = split_manager_ids(
            self.managers, seed=17, test_fraction=0.25
        )
        self.parameters = MatrixParameters(
            maximum_position_weight=0.75,
            minimum_assets_per_manager=3,
            minimum_managers_per_asset=2,
        )

    def _classified(self) -> pd.DataFrame:
        rows = [
            _row(manager, asset, float(100 - 10 * rank))
            for manager in self.fit_ids
            for rank, asset in enumerate(ASSETS[:6])
        ]
        rows.extend(
            _row(manager, asset, float(100 - 10 * rank))
            for manager in self.test_ids
            for rank, asset in enumerate(ASSETS[:3])
        )
        # This asset has only one fit-side owner. Test-side ownership must not
        # make it pass the fit-side minimum-manager degree filter.
        rows.append(_row(self.fit_ids[0], ASSETS[6], 15.0))
        rows.extend(_row(manager, ASSETS[6], 15.0) for manager in self.test_ids)
        return pd.DataFrame(rows)

    def test_test_manager_holdings_cannot_change_fit_universe(self) -> None:
        classified = self._classified()
        original = build_manager_only_quarter(
            classified,
            eligible_manager_ids=self.managers,
            parameters=self.parameters,
            seed=17,
            test_fraction=0.25,
        )
        self.assertEqual(set(original.split.vocabulary), set(ASSETS[:6]))
        self.assertNotIn(ASSETS[6], original.split.vocabulary)
        self.assertEqual(original.fit_pairs_after_filters, len(self.fit_ids) * 6)
        self.assertTrue(original.fit_filter_checks["violations"].eq(0).all())
        self.assertEqual(original.assigned_fit_managers, len(self.fit_ids))
        self.assertEqual(original.assigned_test_managers, len(self.test_ids))
        self.assertEqual(original.split.train_manager_count, len(self.fit_ids))
        self.assertEqual(original.split.test_manager_count, len(self.test_ids))

        changed = pd.concat(
            [
                classified,
                pd.DataFrame(
                    [_row(self.test_ids[0], ASSETS[7], 1_000.0)]
                ),
            ],
            ignore_index=True,
        )
        perturbed = build_manager_only_quarter(
            changed,
            eligible_manager_ids=self.managers,
            parameters=self.parameters,
            seed=17,
            test_fraction=0.25,
        )
        self.assertEqual(perturbed.split.vocabulary, original.split.vocabulary)
        self.assertEqual(
            perturbed.split.train_ranked_columns,
            original.split.train_ranked_columns,
        )
        self.assertEqual(
            (perturbed.split.train_visible != original.split.train_visible).nnz,
            0,
        )
        self.assertEqual(
            perturbed.fit_pairs_after_filters,
            original.fit_pairs_after_filters,
        )
        self.assertEqual(
            perturbed.fit_managers_after_filters,
            original.fit_managers_after_filters,
        )
        self.assertEqual(
            perturbed.fit_pair_content_sha256,
            original.fit_pair_content_sha256,
        )
        self.assertNotEqual(
            perturbed.test_pair_content_sha256,
            original.test_pair_content_sha256,
        )

    def test_original_rank_two_survives_later_visible_oov_drop(self) -> None:
        test = _pairs(
            "test",
            [
                (ASSETS[0], 100.0),
                (ASSETS[1], 90.0),
                (ASSETS[5], 80.0),
                (ASSETS[2], 70.0),
            ],
        )
        queries, labels = build_asmp_queries(
            test,
            vocabulary=ASSETS[:5],
            out_of_vocabulary_visible="drop_after_rank_two",
        )
        self.assertEqual(labels.target_columns, (1,))
        self.assertEqual(queries.ranked_columns, ((0, MASKED_ASSET, 2),))
        self.assertEqual(queries.visible.indices.tolist(), [0, 2])
        self.assertEqual(labels.dropped_out_of_vocabulary_visible_positions, 1)
        self.assertEqual(labels.excluded_out_of_vocabulary, 0)

    def test_rank_one_and_rank_two_oov_portfolios_are_excluded(self) -> None:
        portfolios = pd.concat(
            [
                _pairs("valid", [(ASSETS[0], 100), (ASSETS[1], 90)]),
                _pairs("first_oov", [(ASSETS[5], 100), (ASSETS[1], 90)]),
                _pairs("target_oov", [(ASSETS[0], 100), (ASSETS[5], 90)]),
            ],
            ignore_index=True,
        )
        queries, labels = build_asmp_queries(
            portfolios,
            vocabulary=ASSETS[:5],
            out_of_vocabulary_visible="drop_after_rank_two",
        )
        self.assertEqual(len(queries), 1)
        self.assertEqual(labels.input_portfolios, 3)
        self.assertEqual(labels.excluded_out_of_vocabulary, 2)
        self.assertEqual(labels.target_columns, (1,))

    def test_manager_partitions_are_disjoint_and_complete(self) -> None:
        reversed_fit, reversed_test = split_manager_ids(
            tuple(reversed(self.managers)), seed=17, test_fraction=0.25
        )
        self.assertEqual((self.fit_ids, self.test_ids), (reversed_fit, reversed_test))
        self.assertFalse(set(self.fit_ids) & set(self.test_ids))
        self.assertEqual(set(self.fit_ids) | set(self.test_ids), set(self.managers))

        train = _pairs("shared", [(ASSETS[0], 100), (ASSETS[1], 90)])
        test = _pairs("shared", [(ASSETS[0], 80), (ASSETS[1], 70)])
        with self.assertRaisesRegex(ValueError, "disjoint"):
            build_partitioned_manager_holdout(train, test)

    def test_point_in_time_guard_accepts_available_snapshot(self) -> None:
        decision_at = pd.Timestamp("2024-05-15T23:59:59Z")
        classified = pd.DataFrame.from_records(
            [
                {
                    "matrix_eligible": True,
                    "source_event_available_at_utc": "2024-05-10T12:00:00Z",
                    "snapshot_decision_at_utc": decision_at,
                    "state_last_event_at_utc": "2024-05-10T12:00:00Z",
                    "period_of_report": "2024-03-31",
                }
            ]
        )
        self.assertIsNone(_assert_point_in_time(
            classified, decision_at, report_date=pd.Timestamp("2024-03-31")
        ))

    def test_point_in_time_guard_rejects_future_source_event(self) -> None:
        decision_at = pd.Timestamp("2024-05-15T23:59:59Z")
        classified = pd.DataFrame.from_records(
            [
                {
                    "matrix_eligible": True,
                    "source_event_available_at_utc": "2024-05-16T00:00:00Z",
                    "snapshot_decision_at_utc": decision_at,
                    "state_last_event_at_utc": "2024-05-10T12:00:00Z",
                    "period_of_report": "2024-03-31",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "unavailable"):
            _assert_point_in_time(
                classified, decision_at, report_date=pd.Timestamp("2024-03-31")
            )

    def test_point_in_time_guard_rejects_mismatched_snapshot(self) -> None:
        decision_at = pd.Timestamp("2024-05-15T23:59:59Z")
        classified = pd.DataFrame.from_records(
            [
                {
                    "matrix_eligible": True,
                    "source_event_available_at_utc": "2024-05-10T12:00:00Z",
                    "snapshot_decision_at_utc": "2024-05-14T23:59:59Z",
                    "state_last_event_at_utc": "2024-05-10T12:00:00Z",
                    "period_of_report": "2024-03-31",
                }
            ]
        )
        with self.assertRaisesRegex(ValueError, "different PIT snapshot"):
            _assert_point_in_time(
                classified, decision_at, report_date=pd.Timestamp("2024-03-31")
            )

    def test_point_in_time_guard_rejects_future_state_and_wrong_report(self) -> None:
        decision_at = pd.Timestamp("2024-05-15T23:59:59Z")
        classified = pd.DataFrame.from_records(
            [{
                "matrix_eligible": True,
                "source_event_available_at_utc": "2024-05-10T12:00:00Z",
                "snapshot_decision_at_utc": decision_at,
                "state_last_event_at_utc": "2024-05-16T00:00:00Z",
                "period_of_report": "2024-03-31",
            }]
        )
        with self.assertRaisesRegex(ValueError, "after the decision cutoff"):
            _assert_point_in_time(
                classified, decision_at, report_date=pd.Timestamp("2024-03-31")
            )
        classified["state_last_event_at_utc"] = "2024-05-10T12:00:00Z"
        classified["period_of_report"] = "2023-12-31"
        with self.assertRaisesRegex(ValueError, "different report period"):
            _assert_point_in_time(
                classified, decision_at, report_date=pd.Timestamp("2024-03-31")
            )


if __name__ == "__main__":
    unittest.main()
