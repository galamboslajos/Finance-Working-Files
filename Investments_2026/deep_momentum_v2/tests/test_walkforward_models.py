"""Synthetic annual cohorts, fit checkpoints and16-fit outer orchestration."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _json
from dmv2.monthly import MonthWindow
from dmv2.targets import build_monthly_targets, COHORT_FIELDS
from dmv2.universe import DataContractError
import dmv2.walkforward_models as worker
import dmv2.walkforward_loading as loader
from test_pilot_loading import fixture
from test_ret import target_fixture

PROJECT = Path(__file__).resolve().parents[1]


def annual_target(**kwargs):
    kwargs["variable_members"] = True
    panel = target_fixture(**kwargs)
    year = int(kwargs["first"][:4])
    if year != 2022:
        return panel
    windows, outcomes = [], []
    for row in panel.provenance["month_windows"]:
        end = "2022-12-30" if row["outcome_month"] == "2022-12" else row["outcome_end"]
        windows.append(MonthWindow(row["outcome_month"], row["signal_date"], row["outcome_start"], end,
            (row["outcome_start"], end)))
    for row in panel.table.to_pylist():
        outcomes.append({**{name: row[name] for name in (*worker.KEYS, "outcome_month", "outcome_start", "outcome_end")},
            "outcome_end": "2022-12-30" if row["outcome_month"] == "2022-12" else row["outcome_end"],
            "requested": row["outcome_requested"], "provider_adjusted_return": row["label_return"],
            "return_status": row["source_return_status"]})
    return build_monthly_targets(panel.table.select(list(COHORT_FIELDS)), pa.Table.from_pylist(outcomes),
        label_observed_through="2022-12-30", month_windows=windows)


def setup(root):
    with patch("dmv2.pilot_loading.YEARS", tuple(range(2010, 2024))), \
            patch("test_pilot_loading.target_fixture", side_effect=annual_target):
        roles, targets = fixture(root)
    run = root / "outputs/walkforward/synthetic"
    identity = {"phase": "synthetic_root", "metadata_only_fixture": True}
    pin = canonical_sha256(identity)
    run.mkdir(parents=True)
    _json(run / "build_identity.json", identity)
    references = {}
    for year in range(2010, 2024):
        directory = run / "years" / str(year)
        directory.mkdir(parents=True)
        old_f = roles["primary"]["directory"] / "years" / str(year)
        old_t = targets["directory"] / "years" / str(year)
        fa, ta = json.loads((old_f / "audit.json").read_text()), json.loads((old_t / "audit.json").read_text())
        artifacts = {}
        for source, audit in ((old_f, fa), (old_t, ta)):
            for name, item in audit["artifacts"].items():
                path = source / item["path"]
                shutil.copyfile(path, directory / path.name)
                artifacts[name] = _artifact(directory, directory / path.name)
        audit = {"status": "COMPLETE", "year": year, "build_identity_sha256": pin,
            "feature_provenance": fa["feature_provenance"], "target_artifact_sha256": ta["target_artifact_sha256"],
            "artifacts": artifacts}
        _json(directory / "audit.json", audit)
        references[year] = {"directory": directory, "audit": audit,
            "build_identity_sha256": pin, "audit_pin": _artifact(root, directory / "audit.json")}
    (root / "configs").mkdir()
    shutil.copyfile(PROJECT / "configs/classifier_pilot.json", root / "configs/classifier_pilot.json")
    cfg = json.loads((PROJECT / "configs/walkforward_2020_2023.json").read_text())
    return run, cfg, pin, references


class FakeEstimator:
    fit_calls = 0
    def __init__(self, **parameters):
        self.parameters = parameters
    def fit(self, X, y):
        FakeEstimator.fit_calls += 1
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self
    def predict_proba(self, X):
        return np.full((len(X), 10), .1)
    def predict(self, X, *, output_margin):
        assert output_margin
        return np.zeros((len(X), 10), dtype=np.float32)
    def get_params(self, deep=False):
        return self.parameters
    def save_model(self, path):
        Path(path).write_text(json.dumps({"parameters": self.parameters,
            "classes": self.classes_.tolist(), "width": self.n_features_in_}))
    def load_model(self, path):
        saved = json.loads(Path(path).read_text())
        self.parameters, self.classes_, self.n_features_in_ = saved["parameters"], np.array(saved["classes"]), saved["width"]


class WalkforwardModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name).resolve()
        cls.run_directory, cls.cfg, cls.identity, cls.references = setup(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def resolver(self, root, run, identity, year):
        self.assertIsInstance(identity, dict)
        return self.references[year]

    def test_configuration_keeps_exact16_fits_and_rolling120_completed_months(self):
        worker._configuration(self.root, self.cfg)
        for kind in ("fits", "years", "train_start", "params", "future"):
            cfg = deepcopy(self.cfg)
            if kind == "fits": cfg["execution"]["maximum_new_model_fits"] = 20
            elif kind == "years": cfg["test_years"].append(2024)
            elif kind == "train_start": cfg["folds"][1]["train_outcome_start"] = "2010-01-01"
            elif kind == "params": cfg["model_recipe"]["estimators"]["rf"]["n_estimators"] = 10
            else: cfg["execution"]["financial_2024_plus_analysis"] = True
            with self.subTest(kind=kind), self.assertRaises(DataContractError):
                worker._configuration(self.root, cfg)

    def test_planning_reads_metadata_only_and_preserves_ten_year_window(self):
        original, columns_read = pq.ParquetFile.read, []
        def guarded(parquet, columns=None, *args, **kwargs):
            self.assertIsNotNone(columns)
            self.assertNotIn("target_class", columns)
            self.assertNotIn("label_return", columns)
            columns_read.append(columns)
            return original(parquet, columns=columns, *args, **kwargs)
        with patch.object(loader, "resolve_walkforward_year", side_effect=self.resolver), \
                patch.object(pq.ParquetFile, "read", guarded):
            plan = worker.plan_walkforward_fold(self.root, self.run_directory, self.cfg, self.identity, 2023)
        self.assertEqual(tuple(plan["years"]), tuple(range(2013, 2024)))
        self.assertEqual(plan["provenance"]["label_observed_through"], "2022-12-30")
        self.assertTrue(columns_read)

    def test_fold_materializer_has_no_overlapping_outcomes_and_retains_missing_labels(self):
        with patch.object(loader, "resolve_walkforward_year", side_effect=self.resolver):
            plan = worker.plan_walkforward_fold(self.root, self.run_directory, self.cfg, self.identity, 2020)
        train, test = worker._load_fold(plan)
        self.assertEqual(len(train.X.columns), 715)
        self.assertLess(train.ledger.outcome_end.max(), test.ledger.outcome_start.min())
        self.assertTrue(train.ledger.outcome_start.ge("2010-01-01").all())
        self.assertTrue(train.ledger.outcome_end.le("2019-12-31").all())
        self.assertEqual(test.ledger.outcome_month.nunique(), 12)
        self.assertGreater(train.y.isna().sum(), 0)

    def test_future_annual_key_is_rejected_before_financial_decode(self):
        reference = deepcopy(self.references[2023])
        directory = reference["directory"]
        path = directory / "features.parquet"
        original = path.read_bytes()
        self.addCleanup(lambda: path.write_bytes(original))
        table = pq.read_table(path)
        dates = table["signal_date"].to_pylist(); dates[0] = "2023-12-29"
        table = table.set_column(table.schema.get_field_index("signal_date"), "signal_date", pa.array(dates))
        pq.write_table(table, path)
        reference["audit"]["artifacts"]["features"] = _artifact(directory, path)
        audit_path = directory / "audit.json"; original_audit = audit_path.read_bytes()
        self.addCleanup(lambda: audit_path.write_bytes(original_audit))
        _json(audit_path, reference["audit"])
        with self.assertRaises(DataContractError):
            worker._annual_inputs(self.root, reference, 2023)

    def test_full16_synthetic_fits_then_stitch_and_reload_receipts(self):
        FakeEstimator.fit_calls = 0
        class FakeXgb(FakeEstimator):
            pass
        def load_model(kind, path):
            if kind == "rf":
                import joblib
                return joblib.load(path)
            result = FakeXgb(); result.load_model(path); return result
        real_evaluate = worker._save_evaluation
        failure = {"pending": True}
        def fail_once_after_fit(*args, **kwargs):
            if failure["pending"]:
                failure["pending"] = False
                self.assertTrue((args[0] / "FIT_COMPLETE.json").is_file())
                raise RuntimeError("Synthetic interrupted metrics after the model was saved")
            return real_evaluate(*args, **kwargs)
        with patch.object(loader, "resolve_walkforward_year", side_effect=self.resolver), \
                patch.object(worker, "_constructor", return_value=FakeEstimator), \
                patch.object(worker, "_load_estimator", side_effect=load_model), \
                patch.object(worker, "_save_evaluation", side_effect=fail_once_after_fit):
            with self.assertRaisesRegex(RuntimeError, "interrupted metrics"):
                worker.run_walkforward_fold(self.root, self.run_directory, self.cfg, self.identity, 2020)
            self.assertEqual(FakeEstimator.fit_calls, 1)
            for year in worker.YEARS:
                worker.run_walkforward_fold(self.root, self.run_directory, self.cfg, self.identity, year)
            destination = worker.finalize_walkforward_models(self.root, self.run_directory, self.cfg, self.identity)
            # Completed reruns authenticate/reuse and must not call another fit.
            worker.run_walkforward_fold(self.root, self.run_directory, self.cfg, self.identity, 2020)
            worker.finalize_walkforward_models(self.root, self.run_directory, self.cfg, self.identity)
        self.assertEqual(FakeEstimator.fit_calls, 16)
        saved = worker.read_walkforward_models(destination, root=self.root, identity_sha256=self.identity)
        self.assertEqual(saved["audit"]["summary"]["fitted_arms"], 16)
        self.assertEqual(saved["audit"]["summary"]["preprocessor_fits"], 8)
        for arm in worker.ARMS:
            frame = pq.read_table(destination / "arms" / arm["arm_id"] / "predictions.parquet").to_pandas()
            self.assertEqual(frame.outcome_month.nunique(), 48)
            self.assertEqual(set(frame.fit_year), {2020, 2021, 2022, 2023})

    def test_started_fit_without_complete_checkpoint_cannot_automatically_repeat(self):
        with tempfile.TemporaryDirectory() as temp:
            models = Path(temp)
            directory = models / "folds/2020/arms/rf_paper15"
            directory.mkdir(parents=True)
            (directory / "FIT_ATTEMPT.json").write_text("{}")
            with patch.object(worker, "_constructor", side_effect=AssertionError("No duplicate fit")), \
                    self.assertRaisesRegex(DataContractError, "automatic duplicate"):
                worker._fit_checkpoint(directory, kind="rf", train_X=np.ones((10, 2), dtype=np.float32),
                    train_y=np.arange(1, 11), parameters={}, package_versions={}, identity="a", phase_pin="b",
                    expected_metadata={}, maximum_total_fits=16, models_root=models)


if __name__ == "__main__":
    unittest.main()
