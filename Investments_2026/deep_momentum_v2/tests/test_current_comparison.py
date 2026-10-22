"""Public aggregate/readiness contracts only; no research artifact reads."""
from copy import deepcopy
import ast
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from dmv2 import current_comparison as m

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    return {"schema": "dmv2_path_target_protocol_v1", "protocol_id": "path-target-engineering-20261022-v1",
            "status": "ENGINEERING_ONLY_NOT_FIT_READY",
            "execution": {"model_fitting_enabled": False, "full_history_build_enabled": False,
                          "notebook_compute_enabled": False, "max_estimator_fits": 0},
            "chronology": {"research_start": "2010-01-01", "last_outcome_year": 2023,
                           "rolling_train_years": 10, "protected_from": "2024-01-01"},
            "target_families": ["original_endpoint_control", "executable_return", "return_minus_drawdown"],
            "candidate_utility": {"formula": "cash_return - lambda * maximum_drawdown", "lambda": 1.0,
                                  "selection_status": "engineering_candidate_not_empirically_selected"}}


class PublicComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for rel in (m.REPORT_PATH, m.FIXTURE_PATH):
            target = self.root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((ROOT / rel).read_bytes())
        self.fixture = json.loads((self.root / m.FIXTURE_PATH).read_text())

    def save(self, data):
        (self.root / m.FIXTURE_PATH).write_text(json.dumps(data))

    def test_latest_version_counts_and_ret_focus(self):
        result = m.read_current_comparison(self.root)
        self.assertEqual(result["input_version"], m.INPUT_VERSION)
        self.assertEqual(result["counts"], m.COUNTS)
        self.assertEqual(len(result["rows"]), 30)
        self.assertEqual(len(result["ret_with_stops"]), 5)
        rows = {row["model"]: row for row in result["ret_with_stops"]}
        self.assertEqual(rows["RF · expanded baseline"]["net_total_return"], -.0835)
        self.assertEqual(rows["RF · expanded, prior CV settings"]["net_total_return"], .1587)
        self.assertIsNone(rows["RF · Paper15"]["net_total_return"])

    def test_all_withheld_performance_is_null_including_annuals(self):
        result = m.read_current_comparison(self.root)
        for row in result["rows"]:
            if row["status"] == "withheld":
                self.assertTrue(all(row[key] is None for key in m.METRICS))
        for row in result["annual_returns"]:
            if row["status"] == "withheld":
                self.assertTrue(all(value is None for value in row["returns"].values()))

    def test_only_two_bounded_public_files_are_read(self):
        real = Path.read_bytes
        allowed = {(self.root / m.REPORT_PATH).resolve(), (self.root / m.FIXTURE_PATH).resolve()}
        def read(path):
            self.assertIn(path, allowed)
            return real(path)
        with patch.object(Path, "read_bytes", read):
            result = m.read_current_comparison(self.root)
        self.assertFalse(result["verification_scope"]["private_independent_receipts_checked"])
        self.assertIn("not opened or verified", result["private_verification"])

    def test_report_byte_change_fails_before_fixture_decode(self):
        path = self.root / m.REPORT_PATH
        path.write_text(path.read_text() + "\n")
        with patch.object(m, "_load_json", side_effect=AssertionError("should not decode fixture")):
            with self.assertRaisesRegex(m.ComparisonContractError, "hash changed"):
                m.read_current_comparison(self.root)

    def test_public_path_cannot_redirect_to_private_artifact(self):
        target = self.root / "outputs/private.json"
        target.parent.mkdir(); target.write_text('{"invented_private":true}')
        path = self.root / m.FIXTURE_PATH
        path.unlink(); path.symlink_to(target)
        with patch.object(Path, "read_bytes", side_effect=AssertionError("must not follow private link")):
            with self.assertRaisesRegex(m.ComparisonContractError, "cannot redirect"):
                m.read_current_comparison(self.root)

    def test_older_input_version_or_source_pin_is_refused(self):
        for field, value in (("input_version", "walkforward_source_corrected/2020-2023-20261016-v1"),
                             ("source_report", {"path": "outputs/private.json", "sha256": m.REPORT_SHA256})):
            changed = deepcopy(self.fixture); changed[field] = value; self.save(changed)
            with self.subTest(field=field), self.assertRaises(m.ComparisonContractError):
                m.read_current_comparison(self.root)

    def test_period_cost_stop_borrow_and_counts_cannot_change(self):
        cases = [("period", "trading_sessions", 503), ("period", "outcome_months", 36),
                 ("period", "continuous_nav", False), ("policy", "net_one_way_cost_bps", 5),
                 ("policy", "asset_stop_fraction", .05), ("policy", "borrow_cost", 1),
                 ("policy", "funding_cost", False), ("counts", "withheld_paths", 0)]
        for group, key, value in cases:
            changed = deepcopy(self.fixture); changed[group][key] = value; self.save(changed)
            with self.subTest(group=group, key=key), self.assertRaises(m.ComparisonContractError):
                m.read_current_comparison(self.root)

    def test_metric_value_or_missing_status_cannot_be_reinterpreted(self):
        for field, value in (("net_total_return", 0), ("status", "complete"), ("net_sharpe", False)):
            changed = deepcopy(self.fixture); changed["rows"][0][field] = value; self.save(changed)
            with self.subTest(field=field), self.assertRaises(m.ComparisonContractError):
                m.read_current_comparison(self.root)

    def test_all_model_basket_variants_remain(self):
        for action in (lambda rows: rows.pop(), lambda rows: rows.reverse(), lambda rows: rows.append(rows[0])):
            changed = deepcopy(self.fixture); action(changed["rows"]); self.save(changed)
            with self.assertRaises(m.ComparisonContractError): m.read_current_comparison(self.root)

    def test_cost_and_annual_cells_are_bound_to_report(self):
        for collection in ("cost_sensitivity", "annual_returns"):
            changed = deepcopy(self.fixture)
            if collection == "cost_sensitivity": changed[collection][0]["returns"][0]["total_return"] = 0
            else: changed[collection][0]["returns"]["2020"] = 0
            self.save(changed)
            with self.subTest(collection=collection), self.assertRaises(m.ComparisonContractError):
                m.read_current_comparison(self.root)

    def test_private_verification_claim_and_extra_path_fields_are_refused(self):
        changed = deepcopy(self.fixture)
        changed["verification_scope"]["private_independent_receipts_checked"] = True
        self.save(changed)
        with self.assertRaises(m.ComparisonContractError): m.read_current_comparison(self.root)
        changed = deepcopy(self.fixture); changed["private_receipt_path"] = "outputs/run/receipt.json"
        self.save(changed)
        with self.assertRaises(m.ComparisonContractError): m.read_current_comparison(self.root)

    def test_duplicate_keys_nonfinite_and_byte_bounds_fail(self):
        path = self.root / m.FIXTURE_PATH
        for raw in ('{"schema":1,"schema":2}', '{"x": NaN}', 'x' * (128 * 1024 + 1)):
            path.write_text(raw)
            with self.assertRaises(m.ComparisonContractError): m.read_current_comparison(self.root)

    def test_report_parser_refuses_null_fabrication_inconsistent_costs_or_years(self):
        text = (self.root / m.REPORT_PATH).read_text()
        for old, new in (("| — | — | — | — | — | — | withheld / withheld |", "| +0.00% | — | — | — | — | — | withheld / withheld |"),
                         ("+27.16% (complete)", "+28.16% (complete)"),
                         ("| +48.51% | -6.39% | -8.92% | +0.43% |", "| +98.51% | -6.39% | -8.92% | +0.43% |")):
            self.assertIn(old, text)
            with self.subTest(old=old), self.assertRaises(m.ComparisonContractError):
                m.published_report_tables(text.replace(old, new, 1))


class TargetReadinessTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "configs/path_target_protocol_20261022.json"
        self.path.parent.mkdir()

    def save(self, data): self.path.write_text(json.dumps(data))

    def test_absent_protocol_stays_unconfigured_without_build(self):
        result = m.read_path_target_readiness(self.root)
        self.assertFalse(result["configured"])
        self.assertFalse(result["target_build_verified"])
        self.assertFalse(result["computation_started"])

    def test_configured_protocol_is_metadata_only_not_a_build_receipt(self):
        self.save(protocol())
        result = m.read_path_target_readiness(self.root)
        self.assertTrue(result["configured"])
        self.assertFalse(result["target_build_verified"])
        self.assertFalse(result["computation_started"])
        self.assertEqual(result["candidate_utility"]["lambda"], 1.0)

    def test_scope_or_protected_year_expansion_is_refused(self):
        cases = [("execution", "model_fitting_enabled", True), ("execution", "full_history_build_enabled", True),
                 ("execution", "notebook_compute_enabled", True), ("execution", "max_estimator_fits", 1),
                 ("execution", "max_estimator_fits", False), ("chronology", "last_outcome_year", 2024),
                 ("chronology", "protected_from", "2025-01-01"),
                 ("candidate_utility", "selection_status", "empirically_selected")]
        for group, key, value in cases:
            changed = protocol(); changed[group][key] = value; self.save(changed)
            with self.subTest(key=key), self.assertRaises(m.ComparisonContractError):
                m.read_path_target_readiness(self.root)


class NotebookSafetyTests(unittest.TestCase):
    def test_new_cells_are_unexecuted_and_parse_without_compute_entrypoints(self):
        notebook = json.loads((ROOT / "notebooks/00_research_master.ipynb").read_text())
        identities = {"latest-published-comparison-20261022", "read-latest-published-comparison-20261022",
                      "path-target-readiness-20261022", "read-path-target-readiness-20261022"}
        new = [cell for cell in notebook["cells"] if cell.get("id") in identities]
        self.assertEqual(len(new), 4)
        self.assertIn("25.", "".join(new[0]["source"]))
        self.assertIn("26.", "".join(new[2]["source"]))
        for cell in new:
            if cell["cell_type"] == "code":
                self.assertIsNone(cell["execution_count"])
                self.assertEqual(cell["outputs"], [])
                tree = ast.parse("".join(cell["source"]))
                calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
                self.assertFalse(any(any(word in call.lower() for word in (".fit", "subprocess", "read_parquet", "joblib", "pickle", "backtest")) for call in calls))

    def test_new_comparison_cell_executes_from_published_files_only(self):
        notebook = json.loads((ROOT / "notebooks/00_research_master.ipynb").read_text())
        shown = []
        namespace = {"ROOT": ROOT, "display": shown.append}
        cell = next(cell for cell in notebook["cells"] if cell.get("id") == "read-latest-published-comparison-20261022")
        exec(compile("".join(cell["source"]), "latest_comparison_cell", "exec"), namespace)
        self.assertEqual(namespace["latest_ret_sl_df"].shape[0], 5)
        self.assertEqual(namespace["latest_comparison_df"].shape[0], 30)
        self.assertEqual(len(shown), 2)
        self.assertEqual(shown[0].iloc[0].net_total_return, "withheld")


if __name__ == "__main__":
    unittest.main()
