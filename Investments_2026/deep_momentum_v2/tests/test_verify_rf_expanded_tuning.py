"""Synthetic tests for the independent zero-fit RF tuning verifier."""
import calendar
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import sys

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_rf_expanded_tuning.py"
SPEC = importlib.util.spec_from_file_location("independent_rf_tuning_verifier", SCRIPT)
verify = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verify)
sys.path.insert(0, str(SCRIPT.parents[1] / "src"))


def metadata(year, *, rows_per_month=1):
    records = []
    for month in range(1, 13):
        previous_year, previous_month = (year - 1, 12) if month == 1 else (year, month - 1)
        signal = f"{previous_year}-{previous_month:02d}-{calendar.monthrange(previous_year, previous_month)[1]}"
        for i in range(rows_per_month):
            records.append({"source_security_id": f"security-{i}", "signal_date": signal,
                            "outcome_month": f"{year}-{month:02d}", "outcome_start": f"{year}-{month:02d}-01",
                            "outcome_end": f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]}"})
    return pd.DataFrame(records)


def grid():
    return [{"candidate_id": f"d{depth}l{leaf}f{feature}", "max_depth": depth,
             "min_samples_leaf": leaf, "max_features": feature,
             "reference_statistical_parameters": depth == 12 and leaf == 20 and feature == "sqrt"}
            for depth in (8, 12) for leaf in (20, 100) for feature in ("sqrt", 0.05)]


def score_rows(loss=2.3):
    return [{"outcome_month": f"{year}-{month:02d}", "log_loss": loss,
             "score_keys_sha256": f"keys-{year}-{month}"} for year in (2017, 2018, 2019) for month in range(1, 13)]


class ScoreTests(unittest.TestCase):
    def test_uniform_ties_lowest_class_and_logloss(self):
        probabilities = np.full((12, 10), 0.1, dtype=np.float64)
        result = verify.independent_scores(probabilities, np.ones(12), metadata(2020), expected_year=2020)
        self.assertAlmostEqual(result["summary"]["mean_monthly_log_loss"], np.log(10))
        np.testing.assert_array_equal(result["predicted_class"], np.ones(12))
        self.assertAlmostEqual(result["summary"]["mean_monthly_brier_sum10"], 0.9)

    def test_native_zero_floor_does_not_modify_probability(self):
        probabilities = np.zeros((12, 10), dtype=np.float64)
        probabilities[:, 1] = 1
        original = probabilities.copy()
        result = verify.independent_scores(probabilities, np.ones(12), metadata(2020), expected_year=2020)
        self.assertEqual(result["summary"]["floored_true_class_probability_rows"], 12)
        self.assertAlmostEqual(result["summary"]["mean_monthly_log_loss"], -np.log(1e-15))
        np.testing.assert_array_equal(original, probabilities)

    def test_equal_month_weighting_differs_from_pooled(self):
        meta = metadata(2020)
        extra = pd.concat([meta.iloc[[0]].assign(source_security_id=f"extra-{i}") for i in range(9)], ignore_index=True)
        meta = pd.concat([meta, extra], ignore_index=True)
        p = np.full((len(meta), 10), 0.1, dtype=np.float64)
        january = meta.outcome_month == "2020-01"
        p[january] = 0.01
        p[january, 0] = 0.91
        result = verify.independent_scores(p, np.ones(len(meta)), meta, expected_year=2020)
        self.assertAlmostEqual(result["summary"]["mean_monthly_log_loss"], (-np.log(.91) + 11 * np.log(10)) / 12)
        self.assertNotAlmostEqual(result["summary"]["mean_monthly_log_loss"], result["summary"]["pooled_log_loss"])

    def test_missing_label_month_fails_closed(self):
        labels = np.ones(12)
        labels[3] = np.nan
        with self.assertRaisesRegex(AssertionError, "no observed label"):
            verify.independent_scores(np.full((12, 10), .1), labels, metadata(2020), expected_year=2020)

    def test_reject_future_before_probability_inspection(self):
        with self.assertRaisesRegex(AssertionError, "2010–2023"):
            verify.independent_scores(object(), object(), metadata(2024), expected_year=2024)

    def test_bad_probability_or_fractional_class_fails(self):
        p = np.full((12, 10), .1)
        p[0, 0] = -.1
        with self.assertRaisesRegex(AssertionError, "Invalid probability"):
            verify.independent_scores(p, np.ones(12), metadata(2020), expected_year=2020)
        with self.assertRaisesRegex(AssertionError, "Noncanonical"):
            verify.independent_scores(np.full((12, 10), .1), np.full(12, 1.5), metadata(2020), expected_year=2020)


class SelectionTests(unittest.TestCase):
    def test_reference_wins_numerical_tie(self):
        candidates = grid()
        rows = {item["candidate_id"]: score_rows() for item in candidates}
        ref = next(item["candidate_id"] for item in candidates if item["reference_statistical_parameters"])
        rows[ref] = score_rows(2.3 + 0.5e-8)
        self.assertEqual(verify.select_winner(candidates, rows, [1430] * 3)["candidate_id"], ref)

    def test_outside_tolerance_loss_selects_without_reference_bias(self):
        candidates = grid()
        rows = {item["candidate_id"]: score_rows() for item in candidates}
        winner = candidates[0]["candidate_id"]
        rows[winner] = score_rows(2.3 - 2e-8)
        self.assertEqual(verify.select_winner(candidates, rows, [1430] * 3)["candidate_id"], winner)

    def test_simplicity_tie_after_reference_excluded(self):
        candidates = grid()
        rows = {item["candidate_id"]: score_rows() for item in candidates}
        ref = next(item["candidate_id"] for item in candidates if item["reference_statistical_parameters"])
        rows[ref] = score_rows(2.4)
        selected = verify.select_winner(candidates, rows, [1430] * 3)["candidate_id"]
        self.assertEqual(selected, "d8l100fsqrt")

    def test_incomplete_or_different_mask_cannot_select(self):
        candidates = grid()
        rows = {item["candidate_id"]: score_rows() for item in candidates}
        rows[candidates[0]["candidate_id"]][0]["score_keys_sha256"] = "other"
        with self.assertRaisesRegex(AssertionError, "different scoring"):
            verify.select_winner(candidates, rows, [1430] * 3)
        rows = {item["candidate_id"]: score_rows() for item in candidates[:-1]}
        with self.assertRaisesRegex(AssertionError, "Incomplete"):
            verify.select_winner(candidates, rows, [1430] * 3)


class ChronologyAndPreprocessingTests(unittest.TestCase):
    def test_december_completed_target_may_share_january_signal(self):
        train, valid = metadata(2019), metadata(2020)
        result = verify.validate_chronology(train, valid, np.ones(12, bool), train_start_year=2019, train_end_year=2019, validation_year=2020)
        self.assertEqual(result["maximum_train_outcome_end"], result["minimum_validation_signal_date"])

    def test_future_train_target_rejected(self):
        train, valid = metadata(2019), metadata(2020)
        valid.loc[0, "signal_date"] = "2019-12-29"
        with self.assertRaisesRegex(AssertionError, "beyond validation signal"):
            verify.validate_chronology(train, valid, np.ones(12, bool), train_start_year=2019, train_end_year=2019, validation_year=2020)

    def prep(self):
        names = ["one", "two", *[f"drop-{i}" for i in range(713)]]
        manifest = {"input_columns": names, "retained_columns": names[:2], "output_columns": ["one", "one__missing", "two", "two__missing"],
                    "medians": {"one": 2., "two": 5.}, "fit_rows": 3,
                    "dropped_columns": {name: "all_missing_in_training_partition" for name in names[2:]}}
        matrix = np.asarray([[1, 0, 4, 0], [2, 1, 5, 0], [3, 0, 6, 0]], dtype=np.float32)
        return manifest, matrix

    def test_prepared_medians_and_flags(self):
        manifest, matrix = self.prep()
        self.assertEqual(verify.verify_prepared_matrix(matrix, manifest)["retained_predictors"], 2)

    def test_validation_learned_median_or_imputation_detected(self):
        manifest, matrix = self.prep()
        manifest["medians"]["two"] = 99.
        with self.assertRaisesRegex(AssertionError, "median mismatch"):
            verify.verify_prepared_matrix(matrix, manifest)
        manifest, matrix = self.prep()
        matrix[1, 0] = 99
        with self.assertRaisesRegex(AssertionError, "Imputed values"):
            verify.verify_prepared_matrix(matrix, manifest)

    def test_pin_checks_bytes_and_directory_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "state.json"
            path.write_text(json.dumps({"safe": True}))
            pin = {"path": str(path), "sha256": verify.digest(path), "size_bytes": path.stat().st_size}
            self.assertEqual(verify.verify_pin(pin, roots=[temporary]), path.resolve())
            path.write_text(json.dumps({"safe": False}))
            with self.assertRaisesRegex(AssertionError, "changed"):
                verify.verify_pin(pin, roots=[temporary])


class SavedDiagnosticTests(unittest.TestCase):
    """Cross-check production-generated synthetic reports with independent math."""
    def test_all_saved_diagnostics_and_missing_label_support(self):
        from dmv2.rf_tuning_fit import _save_prediction_artifacts
        meta = metadata(2020, rows_per_month=3)
        meta["prediction_eligible"] = np.tile([True, True, False], 12)
        meta["target_class"] = pd.array(np.tile([1, 2, None], 12), dtype="Int64")
        meta.loc[1, "target_class"] = pd.NA
        meta["target_available"] = meta.target_class.notna()
        ledger = meta.set_index(list(verify.KEYS))
        probabilities = np.full((24, 10), .1, dtype=np.float64)
        probabilities[0] = np.asarray([.3, .1, .1, .1, .1, .1, .05, .05, .05, .05])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifacts, evaluation = _save_prediction_artifacts(root, ledger, probabilities, year=2020)
            source = root / "prediction_ledger.parquet"
            meta.to_parquet(source, index=False)
            prepared = {"stage": "inner", "artifacts": {"prediction_ledger": {"path": str(source), "sha256": verify.digest(source), "size_bytes": source.stat().st_size}},
                        "prediction_original_keys_sha256": verify.keys_hash(meta),
                        "prediction_keys_sha256": verify.keys_hash(meta, meta.prediction_eligible)}
            manifest = {"artifacts": artifacts, "summary": evaluation["summary"], "monthly_log_losses": evaluation["monthly_log_losses"]}
            result = verify.verify_predictions(manifest, prepared=prepared, run_root=root, expected_year=2020, budget=verify.VerificationBudget())
            self.assertEqual(result["summary"]["missing_label_rows"], 1)
            table_path = Path(artifacts["confusion"]["path"])
            table = pd.read_parquet(table_path)
            table.loc[0, "count"] += 1
            table.to_parquet(table_path, index=False)
            with self.assertRaisesRegex(AssertionError, "confusion count"):
                verify.verify_predictions(manifest, prepared=prepared, run_root=root, expected_year=2020, budget=verify.VerificationBudget())


class EventTests(unittest.TestCase):
    def test_independent_chain_rejects_counter_and_snapshot_tamper(self):
        from dmv2 import rf_tuning_control as control
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            state = {"tasks": [], "active_seconds": 0., "initiated_estimator_fits": 0, "initiated_preprocessor_fits": 0}
            control.commit_state(root, state, "PREPARED")
            result = verify.verify_event_chain(root, state)
            self.assertEqual(result["event_count"], 1)
            event = verify.read_json(root / "events" / "000001.json")
            event["ledger_snapshot"]["initiated_estimator_fits"] = 1
            (root / "events" / "000001.json").write_text(json.dumps(event))
            with self.assertRaisesRegex(AssertionError, "snapshot hash"):
                verify.verify_event_chain(root, state)

    def test_prepared_only_attempt_is_partial_and_cannot_be_full_pass(self):
        from dmv2 import rf_tuning_control as control
        frozen = SCRIPT.parents[1] / "configs" / "rf_expanded_tuning_20261020.json"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            protocol = json.loads(frozen.read_text())
            protocol["reference"] = {"pins": []}
            protocol["runtime_snapshot_for_future_matching"] = {}
            protocol_path = root / "protocol.json"
            control.atomic_json(protocol_path, protocol)
            authority = {name: True for name in ("fitting_authorized", "preprocessing_fitting_authorized", "new_financial_reads_authorized", "prediction_authorized")}
            authority.update(backtest_authorized=False, financial_2024_plus_authorized=False)
            amendment_path = root / "amendment.json"
            control.atomic_json(amendment_path, {"protocol_id": protocol["protocol_id"],
                "protocol_sha256": control.file_pin(protocol_path)["sha256"], "execution_authorized": True,
                "statistical_protocol_changed": False, "authority": authority})
            code = root / "synthetic_metadata_only.py"
            code.write_text("# No compute\n")
            run = control.prepare_run(root, protocol_path, amendment_path, code_paths=[code], runtime={"python": "synthetic"}, attempt_suffix="synthetic")
            partial = verify.verify_run(root, run, allow_partial=True, check_runtime=False)
            self.assertEqual(partial["status"], "PASS_COMPLETED_CHECKPOINTS_ONLY")
            self.assertFalse(partial["complete_search_verified"])
            self.assertEqual(partial["verified_fit_manifests"], 0)
            with self.assertRaisesRegex(AssertionError, "all 100 fits"):
                verify.verify_run(root, run, check_runtime=False)


if __name__ == "__main__":
    unittest.main()
