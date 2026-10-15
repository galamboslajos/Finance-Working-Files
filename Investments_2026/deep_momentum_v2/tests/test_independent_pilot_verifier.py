"""Known-answer checks for the independent saved-result verifier."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SOURCE = Path(__file__).resolve().parents[1] / "scripts/verify_classifier_pilot.py"
SPEC = importlib.util.spec_from_file_location("independent_pilot_verifier", SOURCE)
VERIFIER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VERIFIER)


class IndependentProbabilityChecks(unittest.TestCase):
    def frame(self, values, labels, months):
        frame = pd.DataFrame(values, columns=VERIFIER.PROBS)
        frame["source_security_id"] = [f"id{i}" for i in range(len(frame))]
        frame["signal_date"] = [(pd.Period(month, "M") - 1).end_time.date().isoformat() for month in months]
        frame["outcome_month"] = months
        frame["target_class"] = labels
        frame["target_available"] = pd.notna(labels)
        return frame

    def test_uniform_known_answer_and_missing_label(self):
        frame = self.frame(np.full((3, 10), .1), [1, 10, np.nan], ["2018-01"] * 3)
        summary, monthly, scores = VERIFIER.independent_scores(frame)
        self.assertAlmostEqual(summary["mean_monthly_log_loss"], np.log(10))
        self.assertAlmostEqual(summary["mean_monthly_brier_sum10"], .9)
        self.assertEqual(summary["mean_monthly_accuracy"], .5)
        self.assertEqual(summary["prediction_rows"], 3)
        self.assertEqual(summary["missing_label_rows"], 1)
        self.assertTrue(pd.isna(scores.iloc[2].log_loss))

    def test_perfect_probabilities_and_wrong_zero_floor(self):
        values = np.eye(10)[[0, 1, 2]]
        frame = self.frame(values, [1, 2, 1], ["2018-01"] * 3)
        summary, _, _ = VERIFIER.independent_scores(frame)
        self.assertAlmostEqual(summary["pooled_log_loss"], -np.log(1e-15) / 3)
        self.assertAlmostEqual(summary["pooled_brier_sum10"], 2 / 3)
        self.assertEqual(summary["zero_true_class_probability_rows"], 1)
        np.testing.assert_array_equal(frame[VERIFIER.PROBS], values)

    def test_months_are_equally_weighted(self):
        frame = self.frame(np.eye(10)[[0, 1, 1]], [1, 1, 1], ["2018-01", "2018-02", "2018-02"])
        summary, _, _ = VERIFIER.independent_scores(frame)
        self.assertEqual(summary["mean_monthly_accuracy"], .5)
        self.assertAlmostEqual(summary["pooled_accuracy"], 1 / 3)
        self.assertEqual(summary["mean_monthly_brier_sum10"], 1)

    def test_invalid_probability_is_rejected(self):
        frame = self.frame(np.full((1, 10), .11), [1], ["2018-01"])
        with self.assertRaisesRegex(AssertionError, "sum to one"):
            VERIFIER.independent_scores(frame)

    def test_future_metadata_rejected_before_financial_decode(self):
        table = pa.table({"source_security_id": ["one"], "signal_date": ["2019-12-31"],
            "outcome_month": ["2020-01"], "outcome_start": ["2020-01-02"],
            "outcome_end": ["2020-01-31"], "probability_1": ["do not decode financial field"]})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "future.parquet"
            pq.write_table(table, path)
            with self.assertRaisesRegex(AssertionError, "Out-of-pilot"):
                VERIFIER.read_pilot_table(path, first="2018-01", last="2019-12")

    def test_preprocessor_accepts_actual_completed_session_cutoff(self):
        ledger = pd.DataFrame({"source_security_id": ["one"], "signal_date": ["2017-11-30"],
            "fit_mask": [True], "outcome_end": ["2017-12-29"]})
        manifest = {"train_start": "2010-01-01", "train_end": "2017-12-31",
            "label_observed_through": "2017-12-29", "fit_rows": 1,
            "fit_keys_sha256": VERIFIER.canonical([["one", "2017-11-30"]]),
            "input_columns": ["feature"], "retained_columns": ["feature"], "dropped_columns": {},
            "output_columns": ["feature", "feature__missing"], "scaling": None, "clipping": None, "winsorization": None}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps({"manifest": manifest, "fingerprint": VERIFIER.canonical(manifest)}))
            self.assertEqual(VERIFIER.verify_preprocessor(path, ledger)["fit_rows"], 1)
            manifest["label_observed_through"] = "2017-12-28"
            path.write_text(json.dumps({"manifest": manifest, "fingerprint": VERIFIER.canonical(manifest)}))
            with self.assertRaisesRegex(AssertionError, "uncompleted"):
                VERIFIER.verify_preprocessor(path, ledger)

    def test_receipt_binds_exact_run_and_rejects_changed_root_audit(self):
        audit = {"status": "COMPLETE", "build_identity_sha256": "sha256:" + "a" * 64}
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory)
            path = run / "audit.json"
            path.write_text(json.dumps(audit))
            binding = VERIFIER.receipt_binding(run, audit)
            self.assertEqual(binding["run_directory"], str(run.resolve()))
            self.assertEqual(binding["build_identity_sha256"], audit["build_identity_sha256"])
            self.assertEqual(binding["root_audit_sha256"], VERIFIER.digest(path))
            self.assertEqual(binding["verifier_sha256"], VERIFIER.digest(SOURCE))
            path.write_text(json.dumps({**audit, "status": "INCOMPLETE"}))
            with self.assertRaisesRegex(AssertionError, "Root audit changed"):
                VERIFIER.receipt_binding(run, audit)


if __name__ == "__main__":
    unittest.main()
