"""Interactive rerun state checks; notebook cells execute with mocked workers."""
import ast
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch


def cell_containing(needle):
    path = Path(__file__).resolve().parents[1] / "notebooks/00_research_master.ipynb"
    notebook = json.loads(path.read_text())
    matches = ["".join(cell["source"]) for cell in notebook["cells"]
               if cell["cell_type"] == "code" and needle in "".join(cell["source"])]
    if len(matches) != 1:
        raise AssertionError(f"Expected one notebook cell containing {needle!r}")
    return matches[0]


class InteractiveRerunTests(unittest.TestCase):
    def namespace(self):
        return {"catalog": Mock(), "settings": {}, "display": Mock(), "Markdown": str}

    def test_failed_monthly_rerun_clears_previous_results_before_worker_call(self):
        state = self.namespace()
        names = ("monthly_run", "daily_df", "monthly_df", "numerical_preview", "monthly_artifact_path")
        state.update({name: object() for name in names})
        state["run_monthly_sample"] = Mock(side_effect=ValueError("bounded read failed"))
        with self.assertRaisesRegex(ValueError, "bounded read failed"):
            exec(cell_containing("run_monthly_sample(catalog, settings)"), state)
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(state[name])

    def test_disabled_monthly_rerun_clears_previous_results(self):
        state = self.namespace()
        names = ("monthly_run", "daily_df", "monthly_df", "numerical_preview", "monthly_artifact_path")
        state.update({name: object() for name in names})
        state["run_monthly_sample"] = Mock(return_value=None)
        exec(cell_containing("run_monthly_sample(catalog, settings)"), state)
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(state[name])

    def test_failed_inspection_rerun_clears_previous_snapshot(self):
        state = self.namespace()
        state["inspection"] = object()
        state["run_inspection"] = Mock(side_effect=ValueError("membership read failed"))
        with self.assertRaisesRegex(ValueError, "membership read failed"):
            exec(cell_containing("run_inspection(catalog, settings)"), state)
        self.assertIsNone(state["inspection"])

    def test_preview_cell_clears_old_preview_when_monthly_frame_is_unavailable(self):
        state = {"monthly_df": None, "numerical_preview": object()}
        exec(cell_containing("numerical_preview = monthly_df.loc"), state)
        self.assertIsNone(state["numerical_preview"])


    def test_failed_model_plan_rerun_clears_previous_planning_state(self):
        names = ("model_plan", "model_arms_df", "model_schedule", "model_schedule_df", "model_readiness")
        state = {"ROOT": Path(__file__).resolve().parents[1], "display": Mock(),
                 **{name: object() for name in names}}
        with patch("dmv2.model_design.load_model_plan", side_effect=ValueError("invalid draft")):
            with self.assertRaisesRegex(ValueError, "invalid draft"):
                exec(cell_containing("model_plan = load_model_plan("), state)
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(state[name])

    def test_model_planning_cell_is_metadata_only_and_keeps_completed_label_cutoffs(self):
        state = {"ROOT": Path(__file__).resolve().parents[1], "display": Mock()}
        exec(cell_containing("model_plan = load_model_plan("), state)
        self.assertEqual(len(state["model_arms_df"]), 4)
        self.assertEqual(len(state["model_schedule_df"]), 48)
        first = state["model_schedule_df"].iloc[0]
        self.assertEqual(first.prediction_outcome_month, "2020-01")
        self.assertEqual(first.feature_month, "2019-12")
        self.assertEqual(first.train_last_feature_month, "2019-11")
        self.assertFalse(state["model_readiness"]["market_data_read"])
        self.assertFalse(state["model_readiness"]["model_fitted"])
        self.assertFalse(state["model_readiness"]["execution_authorized"])


    def test_failed_shared_target_rerun_clears_previous_labels_and_payoffs(self):
        names = ("target_ret_protocol", "shared_target_run", "target_classes_df",
                 "target_diagnostics_df", "ret_sample_payoffs")
        state = {"ROOT": Path(__file__).resolve().parents[1], "display": Mock(),
                 "feature_run": None, "monthly_run": None,
                 **{name: object() for name in names}}
        with patch("dmv2.target_workflow.load_target_ret_protocol", side_effect=ValueError("invalid target policy")):
            with self.assertRaisesRegex(ValueError, "invalid target policy"):
                exec(cell_containing("target_ret_protocol = load_target_ret_protocol(ROOT)"), state)
        for name in names:
            with self.subTest(name=name):
                self.assertIsNone(state[name])

    def test_missing_existing_sample_does_not_launch_target_or_return_reads(self):
        state = {"ROOT": Path(__file__).resolve().parents[1], "display": Mock(),
                 "feature_run": None, "monthly_run": None}
        with patch("dmv2.target_workflow.run_shared_target_sample") as worker:
            exec(cell_containing("target_ret_protocol = load_target_ret_protocol(ROOT)"), state)
            worker.assert_not_called()
        self.assertIsNone(state["shared_target_run"])
        self.assertFalse(state["target_ret_protocol"]["execution"]["full_history_build_enabled"])

    def test_failed_full_target_verification_clears_saved_target_state(self):
        names = ("full_target_run", "full_target_dataset", "full_monthly_outcome_dataset",
                 "full_target_coverage_df", "full_target_class_coverage_df", "full_ret_coverage_df")
        state = {"catalog": Mock(), **{name: object() for name in names}}
        with patch("dmv2.full_target_panel.read_full_target_run", side_effect=ValueError("stale target run")):
            with self.assertRaisesRegex(ValueError, "stale target run"):
                exec(cell_containing("full_target_run = read_full_target_run(catalog)"), state)
        for name in names:
            self.assertIsNone(state[name])

    def test_missing_full_target_run_does_not_build_or_fit(self):
        state = {"catalog": Mock()}
        with patch("dmv2.full_target_panel.read_full_target_run", side_effect=FileNotFoundError), \
             patch("dmv2.full_target_panel.build_full_targets") as build, \
             patch("dmv2.full_target_panel.read_full_targets") as labels:
            exec(cell_containing("full_target_run = read_full_target_run(catalog)"), state)
        build.assert_not_called()
        labels.assert_not_called()
        self.assertIsNone(state["full_target_run"])

    def test_failed_classifier_pilot_verification_clears_old_results(self):
        state = {"catalog": Mock(), "classifier_pilot_run": object(),
                 "classifier_pilot_comparison_df": object()}
        with patch("dmv2.classifier_pilot.read_classifier_pilot_run", side_effect=ValueError("stale pilot")):
            with self.assertRaisesRegex(ValueError, "stale pilot"):
                exec(cell_containing("classifier_pilot_run = read_classifier_pilot_run(catalog)"), state)
        self.assertIsNone(state["classifier_pilot_run"])
        self.assertIsNone(state["classifier_pilot_comparison_df"])

    def test_missing_classifier_pilot_never_prepares_inputs_or_fits(self):
        state = {"catalog": Mock()}
        with patch("dmv2.classifier_pilot.read_classifier_pilot_run", side_effect=FileNotFoundError), \
             patch("dmv2.classifier_pilot.run_classifier_pilot") as run, \
             patch("dmv2.classifier_pilot.load_pilot_data") as load, \
             patch("dmv2.classifier_pilot.fit_predict_pilot_arm") as fit:
            exec(cell_containing("classifier_pilot_run = read_classifier_pilot_run(catalog)"), state)
        run.assert_not_called()
        load.assert_not_called()
        fit.assert_not_called()
        self.assertIsNone(state["classifier_pilot_run"])

    def test_failed_bbby_review_clears_old_status_and_propagates_invalid_evidence(self):
        state = {"ROOT": Path(__file__).resolve().parents[1],
                 "bbby_identity_review": object(), "bbby_identity_summary": object()}
        with patch("dmv2.identity_amendment.read_saved_identity_review",
                   side_effect=ValueError("changed evidence hash")):
            with self.assertRaisesRegex(ValueError, "changed evidence hash"):
                exec(cell_containing("bbby_identity_review = read_saved_identity_review("), state)
        self.assertIsNone(state["bbby_identity_review"])
        self.assertIsNone(state["bbby_identity_summary"])

    def test_missing_bbby_review_is_passive_and_clears_old_status(self):
        state = {"ROOT": Path(__file__).resolve().parents[1],
                 "bbby_identity_review": object(), "bbby_identity_summary": object()}
        with patch("dmv2.identity_amendment.read_saved_identity_review",
                   side_effect=FileNotFoundError) as reader, patch("builtins.print"):
            exec(cell_containing("bbby_identity_review = read_saved_identity_review("), state)
        reader.assert_called_once_with(
            state["ROOT"] / "data/derived/identity_amendments/old-bbby-native-conflict-20261016-v1")
        self.assertIsNone(state["bbby_identity_review"])
        self.assertIsNone(state["bbby_identity_summary"])

    def test_verified_bbby_review_preserves_active_data_and_runs(self):
        preserved = {name: object() for name in (
            "catalog", "daily_df", "monthly_df", "features_df", "full_feature_run",
            "full_target_run", "classifier_pilot_run", "daily_backtest_run", "action_backtest_run")}
        summary = {"replacement_native_id_resolved": False, "corrected_features_rebuilt": False,
                   "models_refitted": False, "affected_training_examples": 96}
        saved = {"summary": summary, "path": Path("/private/saved-review")}
        state = {"ROOT": Path(__file__).resolve().parents[1], "json": json, **preserved}
        with patch("dmv2.identity_amendment.read_saved_identity_review", return_value=saved) as reader, \
             patch("builtins.print") as output:
            exec(cell_containing("bbby_identity_review = read_saved_identity_review("), state)
        reader.assert_called_once()
        self.assertIs(state["bbby_identity_review"], saved)
        self.assertIs(state["bbby_identity_summary"], summary)
        self.assertIn("still unresolved", " ".join(str(c) for c in output.call_args_list))
        for name, original in preserved.items():
            with self.subTest(name=name):
                self.assertIs(state[name], original)

    def test_bbby_review_cell_has_only_passive_reader_and_json_display_calls(self):
        tree = ast.parse(cell_containing("bbby_identity_review = read_saved_identity_review("))
        imports = [(node.module, [alias.name for alias in node.names])
                   for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        self.assertEqual(imports, [("dmv2.identity_amendment", ["read_saved_identity_review"])])
        self.assertFalse(any(isinstance(node, ast.Import) for node in ast.walk(tree)))
        calls = {ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)}
        self.assertEqual(calls, {"read_saved_identity_review", "json.dumps", "print"})
        handlers = [ast.unparse(node.type) for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)]
        self.assertEqual(handlers, ["FileNotFoundError"])

    def test_bootstrap_clears_bbby_review_before_loading_settings(self):
        bootstrap = cell_containing("cwd = Path.cwd().resolve()")
        self.assertIn("bbby_identity_review = bbby_identity_summary = None", bootstrap)
        self.assertLess(bootstrap.index("bbby_identity_review ="), bootstrap.index("cwd = Path.cwd()"))


if __name__ == "__main__":
    unittest.main()
