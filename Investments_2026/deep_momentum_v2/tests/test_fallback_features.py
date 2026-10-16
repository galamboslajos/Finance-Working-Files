"""Worker composition and monthly-label quarantine on synthetic cache inputs."""
from dataclasses import replace
from types import SimpleNamespace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pyarrow as pa

from dmv2.fallback_features import build_fallback_year
from dmv2.fallback_loading import load_fallback_year
from dmv2.paper_features import PAPER_FEATURE_NAMES
from dmv2.universe import DataContractError
from test_fallback_loading import FallbackCache


class FallbackFeatureCompositionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cache = FallbackCache(Path(self.temp.name))
        self.loaded = load_fallback_year(self.cache.root, plan=self.cache.plan())

    def features(self, prices, calendar, snapshots, **kwargs):
        self.assertIs(prices, self.loaded.feature_prices)
        self.assertIs(calendar, self.loaded.plan.calendar)
        self.assertIs(kwargs["daily_cohorts"], self.loaded.plan.daily_cohorts)
        self.assertEqual(kwargs["context_policy"], "available_cohort_80pct")
        self.assertEqual(kwargs["requested_security_ids"], self.loaded.plan.requested_ids)
        rows = [{**row, "requested": row["identity_eligible"],
            "paper_prediction_eligible": row["identity_eligible"]}
            for row in self.loaded.plan.original_cohort.to_pylist()]
        return SimpleNamespace(table=pa.Table.from_pylist([{"source_security_id": r["source_security_id"],
            "signal_date": r["signal_date"], "synthetic_feature": 1.0} for r in rows]),
            coverage=pa.Table.from_pylist(rows))

    def test_frozen_feature_worker_and_new_monthly_targets_share_amended_denominators(self):
        with patch("dmv2.fallback_features.build_full_feature_panel", side_effect=self.features):
            result = build_fallback_year(self.loaded)
        self.assertEqual(result.targets.table.num_rows, self.cache.original.num_rows)
        bbby = [r for r in result.targets.table.to_pylist() if r["source_security_id"] == self.cache.bbby]
        self.assertEqual(len(bbby), 12)
        self.assertTrue(all(not r["target_available"] and not r["prediction_eligible"] for r in bbby))
        self.assertTrue(all(r["return_status"] == "identity_unresolved"
            for r in result.monthly.table.to_pylist() if r["source_security_id"] == self.cache.bbby))
        self.assertEqual(sum(result.targets.table["target_available"].to_pylist()), 11*12)
        self.assertTrue(result.provenance["paper15_common_population_recomputed"])
        self.assertFalse(result.provenance["old_features_targets_or_model_predictions_reused"])

    def test_outcome_table_never_enters_feature_worker(self):
        rows = self.loaded.outcome_prices.to_pylist()
        for row in rows: row["adj_close"] *= 100
        changed = replace(self.loaded, outcome_prices=pa.Table.from_pylist(rows))
        calls = []
        def frozen(prices, *args, **kwargs):
            calls.append(prices)
            return self.features(prices, *args, **kwargs)
        with patch("dmv2.fallback_features.build_full_feature_panel", side_effect=frozen):
            result = build_fallback_year(changed)
        self.assertIs(calls[0], self.loaded.feature_prices)
        self.assertTrue(result.provenance["feature_and_outcome_views_separate"])

    def test_changed_key_denominators_or_false_complete_repair_claim_refused(self):
        with patch("dmv2.fallback_features.build_full_feature_panel", side_effect=self.features):
            self.loaded.provenance["fully_identity_repaired"] = True
            with self.assertRaisesRegex(DataContractError, "complete identity repair"):
                build_fallback_year(self.loaded)

    def test_real_frozen_numerical_workers_accept_missing_native_without_label_fill(self):
        result = build_fallback_year(self.loaded)
        self.assertEqual(result.features.summary["registered_features"], 728)
        self.assertEqual(result.features.summary["candidate_model_columns"], 715)
        frame = result.features.table.to_pandas()
        bbby = frame.loc[frame.source_security_id.eq(self.cache.bbby), list(PAPER_FEATURE_NAMES)]
        self.assertEqual(len(bbby), 12)
        self.assertTrue(bbby.isna().all().all())
        self.assertEqual(sum(result.targets.table["target_available"].to_pylist()), 11*12)
        self.assertTrue(result.provenance["daily_historical_benchmarks_recomputed"])


if __name__ == "__main__": unittest.main()
