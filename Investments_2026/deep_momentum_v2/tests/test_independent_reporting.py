"""Small real saved-receipt fixtures for passive independent-run reporting."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.catalog import canonical_sha256
from dmv2.full_panel import _artifact, _hash, _json
import dmv2.fallback_reporting as passive
import dmv2.independent_reporting as reporting
from dmv2.universe import DataContractError


def fixture(root, *, identity_updates=None):
    parent_identity = {"phase": "data_limited_feature_target_rebuild_v1", "synthetic": True}
    parent_pin = canonical_sha256(parent_identity)
    parent = root / "outputs/data_limited_backtest" / ("2010-2019-" + parent_pin[7:23])
    parent_models = parent / "models"
    parent_models.mkdir(parents=True)
    _json(parent / "build_identity.json", parent_identity)
    _json(parent / "audit.json", {"status": "COMPLETE", "build_identity_sha256": parent_pin})
    parent_phase = {"phase": "data_limited_four_arm_models_v1", "parent_identity_sha256": parent_pin}
    parent_model_pin = canonical_sha256(parent_phase)
    _json(parent_models / "build_identity.json", parent_phase)
    _json(parent_models / "audit.json", {"status": "COMPLETE", "build_identity_sha256": parent_pin,
        "model_identity_sha256": parent_model_pin, "summary": {"fitted_arms": 4}})
    parent_pins = {str(p.relative_to(root)): _artifact(root, p) for p in (
        parent / "build_identity.json", parent / "audit.json",
        parent_models / "build_identity.json", parent_models / "audit.json")}
    identity = {"phase": "independent_validation_feature_repair_v1",
        "configuration": {"execution": {"maximum_new_model_fits": 0, "maximum_new_preprocessor_fits": 0},
            "validation_reprediction": {"fitted_parent_directory": str(parent.relative_to(root)),
                "fitted_parent_build_identity_sha256": parent_pin}},
        "years": {"2018": {}, "2019": {}}, "workers": {},
        "parent_identity_pin": _artifact(root, parent / "build_identity.json"),
        "fitted_parent_directory": str(parent.relative_to(root)),
        "fitted_parent_build_identity_sha256": parent_pin}
    for field, value in (identity_updates or {}).items():
        identity[field] = value
    identity["configuration"].setdefault("backtest", {"events": "original_eight_plus_SNI", "top_k_per_side": 10})
    identity["configuration_sha256"] = canonical_sha256(identity["configuration"])
    pin = canonical_sha256(identity)
    run = root / "outputs/independent_backtest" / ("2018-2019-" + pin[7:23])
    run.mkdir(parents=True)
    _json(run / "build_identity.json", identity)
    annuals = {}
    for year in (2018, 2019):
        directory = run / "years" / str(year)
        directory.mkdir(parents=True)
        # These are intentionally not valid Parquet. A passive read must never
        # decode them, nor serialized matrices/estimators below.
        (directory / "features.parquet").write_bytes(b"DO NOT DECODE FEATURE PANEL")
        _json(directory / "audit.json", {"status": "COMPLETE", "year": year,
            "build_identity_sha256": pin, "summary": {"original_rows": 120,
                "prediction_eligible_rows": 119, "labelled_rows": 118},
            "artifacts": {"features": _artifact(directory, directory / "features.parquet")}})
        annuals[str(year)] = _artifact(run, directory / "audit.json")
    _json(run / "audit.json", {"status": "COMPLETE", "build_identity_sha256": pin,
        "years": annuals, "research_status": "DATA_LIMITED_RESTRICTED_UNIVERSE"})
    models = run / "models"
    models.mkdir()
    phase = {"phase": "validation_reprediction_v1", "parent_identity_sha256": pin,
        "new_estimator_fits": 0, "preprocessing_fits": 0,
        "data_plan": {"fitted_parent_run": str(parent.relative_to(root)),
            "fitted_parent_build_identity_sha256": parent_pin,
            "fitted_parent_model_identity_sha256": parent_model_pin,
            "new_estimator_fits": 0, "preprocessing_fits": 0,
            "years": {"2018": {}, "2019": {}}, "source_pins": parent_pins}}
    _json(models / "build_identity.json", phase)
    model_pin = canonical_sha256(phase)
    arms = {}
    for arm in sorted(reporting.ARMS | {"uniform", "train_prior"}):
        directory = models / ("arms" if arm in reporting.ARMS else "refs") / arm
        directory.mkdir(parents=True)
        (directory / "model.joblib").write_bytes(b"DO NOT DESERIALIZE MODEL")
        _json(directory / "audit.json", {"status": "COMPLETE", "phase": "validation_reprediction_v1",
            "build_identity_sha256": pin, "model_identity_sha256": model_pin,
            "metadata": {"new_fit_count": 0, "reused_fit_count": 1,
                "preprocessing_fit_count": 0, "parent_model_identity_sha256": parent_model_pin},
            "summary": {"outcome_months": 24, "mean_monthly_log_loss": 2.3}})
        arms[arm] = _artifact(models, directory / "audit.json")
    _json(models / "audit.json", {"status": "COMPLETE", "phase": "validation_reprediction_v1",
        "build_identity_sha256": pin, "model_identity_sha256": model_pin,
        "summary": {"newly_fitted_arms": 0, "reused_fitted_arms": 4, "preprocessing_fits": 0,
            "fitted_parent_model_identity_sha256": parent_model_pin},
        "arms": arms, "artifacts": {"build_identity.json": _artifact(models, models / "build_identity.json")}})
    backtest = run / "backtest"
    backtest.mkdir()
    backtest_phase = {"phase": "independent_quote_backtest_v1", "parent_identity_sha256": pin}
    _json(backtest / "build_identity.json", backtest_phase)
    comparison = pd.DataFrame([{"arm": arm, "basket": basket, "variant": variant,
        "cost_bps": cost, "complete_proxy_path": True} for arm, basket, variant, cost in
        itertools.product(sorted(reporting.ARMS), ("class_1_10", "class_2_9"),
            ("unstopped", "asset_stop_10pct"), (0, 5, 10, 25))])
    comparison.to_parquet(backtest / "comparison.parquet", index=False)
    _json(backtest / "audit.json", {"status": "COMPLETE", "phase": "independent_quote_backtest_v1",
        "build_identity_sha256": pin, "phase_identity_sha256": canonical_sha256(backtest_phase),
        "research_status": "DATA_LIMITED_RESTRICTED_UNIVERSE", "scenarios": 64,
        "complete_proxy_paths": 64, "failed_closed_paths": 0, "execution_seconds": .01,
        "artifacts": {name: _artifact(backtest, backtest / name)
            for name in ("build_identity.json", "comparison.parquet")}})
    return run, pin


def receipts(run):
    return {label: "sha256:" + _hash(run / path) for label, path in (
        ("features", "audit.json"), ("models", "models/audit.json"), ("backtest", "backtest/audit.json"))}


def repin_models(run, phase):
    """Keep synthetic receipts self-consistent while testing semantic guards."""
    models = run / "models"
    _json(models / "build_identity.json", phase)
    model_pin = canonical_sha256(phase)
    audit = json.loads((models / "audit.json").read_text())
    for name, item in audit["arms"].items():
        path = models / item["path"]
        arm = json.loads(path.read_text())
        arm["model_identity_sha256"] = model_pin
        _json(path, arm)
        audit["arms"][name] = _artifact(models, path)
    audit["model_identity_sha256"] = model_pin
    audit["artifacts"]["build_identity.json"] = _artifact(models, models / "build_identity.json")
    _json(models / "audit.json", audit)


class IndependentReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run, self.pin = fixture(self.root)

    def read(self, **updates):
        args = {"identity_sha256": self.pin, "expected_receipts": receipts(self.run),
            "verify_sources": False}
        args.update(updates)
        return reporting.read_independent_run(self.root, self.run, **args)

    def test_passive_reads_real_small_receipts_and_only_sixtyfour_row_report(self):
        original = passive.pq.ParquetFile
        def small_report_only(path, *args, **kwargs):
            self.assertEqual(Path(path), self.run / "backtest/comparison.parquet")
            return original(path, *args, **kwargs)
        with patch.object(passive.pq, "ParquetFile", side_effect=small_report_only), \
             patch("pickle.load", side_effect=AssertionError("Model deserialization")), \
             patch.object(reporting, "_verify_artifacts", side_effect=AssertionError("Deep source verification")):
            saved = self.read()
        self.assertEqual(saved["summary"]["outcome_years"], [2018, 2019])
        self.assertEqual(len(saved["model_comparison"]), 6)
        self.assertEqual(len(saved["backtest_comparison"]), 64)
        self.assertFalse(saved["summary"]["models_deserialized"])
        self.assertFalse(saved["summary"]["quotes_compiled"])
        self.assertFalse(saved["summary"]["numerical_work_started"])
        self.assertFalse(saved["summary"]["full_source_bytes_verified"])

    def test_any_changed_final_audit_rejected_against_original_explicit_receipt(self):
        for relative in ("audit.json", "models/audit.json", "backtest/audit.json"):
            with self.subTest(relative=relative):
                expected = receipts(self.run)
                path = self.run / relative
                original = path.read_bytes()
                path.write_bytes(original + b"\n")
                with self.assertRaises(DataContractError):
                    self.read(expected_receipts=expected)
                path.write_bytes(original)

    def test_stale_identity_and_outside_output_directory_rejected(self):
        with self.assertRaises(DataContractError):
            self.read(identity_sha256="sha256:" + "a" * 64)
        with self.assertRaises(DataContractError):
            reporting.read_independent_run(self.root, self.root,
                identity_sha256=self.pin, expected_receipts=receipts(self.run))

    def test_all_final_receipts_must_be_nonnull_complete_sha256_values(self):
        for label in ("features", "models", "backtest"):
            for invalid in (None, "", "sha256:abc"):
                with self.subTest(label=label, invalid=invalid):
                    expected = receipts(self.run)
                    expected[label] = invalid
                    with self.assertRaises(DataContractError):
                        self.read(expected_receipts=expected)

    def test_wrong_year_in_annual_receipt_rejected_even_with_new_final_receipt(self):
        path = self.run / "years/2018/audit.json"
        annual = json.loads(path.read_text())
        annual["year"] = 2020
        _json(path, annual)
        audit = json.loads((self.run / "audit.json").read_text())
        audit["years"]["2018"] = _artifact(self.run, path)
        _json(self.run / "audit.json", audit)
        with self.assertRaises(DataContractError):
            self.read()

    def test_identity_cannot_authorize_new_model_fits_or_later_outcome_year(self):
        for alteration in ("fits", "years"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary).resolve()
                updates = {"years": {"2018": {}, "2020": {}}} if alteration == "years" else {
                    "configuration": {"execution": {"maximum_new_model_fits": 1}}}
                run, pin = fixture(root, identity_updates=updates)
                with self.assertRaises(DataContractError):
                    reporting.read_independent_run(root, run, identity_sha256=pin,
                        expected_receipts=receipts(run))

    def test_reprediction_fit_or_preprocessing_count_changed_semantically_rejected(self):
        phase_path = self.run / "models/build_identity.json"
        original = json.loads(phase_path.read_text())
        for field in ("new_estimator_fits", "preprocessing_fits"):
            with self.subTest(field=field):
                phase = deepcopy(original)
                phase[field] = 1
                repin_models(self.run, phase)
                with self.assertRaises(DataContractError):
                    self.read()
                repin_models(self.run, original)

    def test_missing_parent_receipt_is_not_silently_accepted_by_light_reader(self):
        phase = json.loads((self.run / "models/build_identity.json").read_text())
        missing = next(key for key in phase["data_plan"]["source_pins"] if key.endswith("models/audit.json"))
        phase["data_plan"]["source_pins"].pop(missing)
        repin_models(self.run, phase)
        with self.assertRaises(DataContractError):
            self.read()

    def test_changed_parent_identity_reference_rejected_despite_valid_new_phase_hash(self):
        phase = json.loads((self.run / "models/build_identity.json").read_text())
        phase["data_plan"]["fitted_parent_build_identity_sha256"] = "sha256:" + "b" * 64
        repin_models(self.run, phase)
        with self.assertRaises(DataContractError):
            self.read()

    def test_oversized_passive_json_refused_before_read_text(self):
        path = self.run / "build_identity.json"
        with path.open("wb") as stream:
            stream.truncate(passive.MAX_JSON_BYTES + 1)
        with self.assertRaises(DataContractError):
            self.read()

    def test_comparison_redirect_to_panel_is_rejected_before_decode(self):
        path = self.run / "backtest/audit.json"
        audit = json.loads(path.read_text())
        audit["artifacts"]["comparison.parquet"] = {
            "path": "../years/2018/features.parquet", "size_bytes": 10, "sha256": "a" * 64}
        _json(path, audit)
        with patch.object(passive.pq, "ParquetFile", side_effect=AssertionError("Panel decode attempted")), \
             self.assertRaises(DataContractError):
            self.read()


if __name__ == "__main__":
    unittest.main()
