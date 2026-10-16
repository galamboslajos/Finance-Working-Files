"""Small synthetic orchestration guards; no research return calculations."""
from copy import deepcopy
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.backtest_loading import _pin
from dmv2.catalog import canonical_sha256
from dmv2.daily_backtest import _save_table, _write_json
import dmv2.completed_backtest as worker
from dmv2.universe import DataContractError

PROJECT = Path(__file__).resolve().parents[1]
PARENT = "sha256:" + "a" * 64


def configuration():
    return json.loads((PROJECT / "configs/independent_backtest_completion.json").read_text())


def saved_fixture(root):
    directory = root / "outputs/independent/backtest_completed"
    directory.mkdir(parents=True)
    phase = {"phase": worker.PHASE, "parent_identity_sha256": PARENT,
        "models_fitted": False, "events": [{"event_id": str(i)} for i in range(11)],
        "source_pins": []}
    phase_pin = canonical_sha256(phase)
    artifacts = {}
    _write_json(directory / "build_identity.json", phase)
    artifacts["build_identity.json"] = _pin(directory, directory / "build_identity.json")
    comparison = pd.DataFrame([dict(arm=arm, basket=basket, variant=variant, cost_bps=cost)
        for arm, basket, variant, cost in itertools.product(worker.ARMS,
            ("class_1_10", "class_2_9"), ("unstopped", "asset_stop_10pct"), (0, 5, 10, 25))])
    reports = {name: pd.DataFrame({"synthetic": [True]}) for name in worker.REPORTS}
    reports["comparison"] = comparison
    reports["selection_validation"] = pd.DataFrame({"basket_verified": [True] * 8})
    for name, frame in reports.items():
        _save_table(directory, name + ".parquet", frame, artifacts)
    audit = {"status": "COMPLETE", "phase": worker.PHASE,
        "research_status": "DATA_LIMITED_RESTRICTED_UNIVERSE",
        "build_identity_sha256": PARENT, "phase_identity_sha256": phase_pin,
        "models_fitted": False, "financial_2020_plus_read": False,
        "scenarios": 64, "artifacts": artifacts}
    _write_json(directory / "audit.json", audit)
    return directory, audit


class ConfigurationTests(unittest.TestCase):
    def test_accepts_frozen_new_configuration_and_balanced_breadth(self):
        cfg = configuration()
        self.assertEqual(worker._configuration(cfg), cfg)

    def test_refuses_period_refits_new_exclusions_and_changed_grid(self):
        for alteration in ("period", "fit", "extra_id", "cost", "stop", "unequal", "cash", "future", "budget"):
            cfg = configuration()
            if alteration == "period": cfg["backtest"]["end"] = "2020-12-31"
            elif alteration == "fit": cfg["execution"]["maximum_new_model_fits"] = 1
            elif alteration == "extra_id": cfg["execution_restrictions"]["native_ids"]["another"] = "posthoc"
            elif alteration == "cost": cfg["backtest"]["cost_bps"] = [0, 10]
            elif alteration == "stop": cfg["backtest"]["variants"]["asset_stop_10pct"] = .2
            elif alteration == "unequal": cfg["backtest"]["basket_breadth"]["equal_side_name_counts"] = False
            elif alteration == "cash": cfg["backtest"]["basket_breadth"]["zero_candidate_side"] = "pretend_cash"
            elif alteration == "future": cfg["backtest"]["basket_breadth"]["candidate_information"] = "future_quote_availability"
            else: cfg["execution"]["backtest_maximum_wall_seconds"] = 901
            with self.subTest(alteration=alteration), self.assertRaises(DataContractError):
                worker._configuration(cfg)

    def test_completion_changes_only_action_terms_and_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "configs").mkdir()
            base = json.loads((PROJECT / "configs/independent_backtest.json").read_text())
            (root / "configs/independent_backtest.json").write_text(json.dumps(base))
            identity = {"configuration": base, "configuration_sha256": canonical_sha256(base)}
            cfg = configuration()
            self.assertEqual(worker._completion_parent_configuration(root, cfg, identity)[0], base)
            for alteration in ("median_recipe", "quote_recipe", "basket", "parent"):
                changed = deepcopy(cfg)
                if alteration == "median_recipe": changed["model_recipe"]["estimators"]["rf"]["n_estimators"] += 1
                elif alteration == "quote_recipe": changed["quote_amendments"] = "new data"
                elif alteration == "basket": changed["backtest"]["basket_breadth"]["zero_candidate_side"] = "cash"
                else: changed["parent_configuration_sha256"] = "sha256:" + "d" * 64
                with self.subTest(alteration=alteration), self.assertRaises(DataContractError):
                    worker._completion_parent_configuration(root, changed, identity)

    def test_prior56_8_receipts_are_hashed_without_performance_decoding(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            completed, audit = saved_fixture(root)
            old = root / "outputs/independent/backtest"
            completed.rename(old)
            phase = json.loads((old / "build_identity.json").read_text())
            phase["phase"] = "independent_quote_backtest_v1"
            phase["events"] = phase["events"][:9]
            _write_json(old / "build_identity.json", phase)
            audit.update(phase="independent_quote_backtest_v1", phase_identity_sha256=canonical_sha256(phase),
                complete_proxy_paths=56, failed_closed_paths=8)
            audit["artifacts"]["build_identity.json"] = _pin(old, old / "build_identity.json")
            _write_json(old / "audit.json", audit)
            with patch.object(pq.ParquetFile, "read", side_effect=AssertionError("No old performance decode")):
                pins = worker._prior_backtest_pins(root, old.parent, PARENT)
            self.assertGreater(len(pins), 2)
            source = old / "comparison.parquet"
            source.write_bytes(source.read_bytes() + b"changed")
            with self.assertRaises(DataContractError):
                worker._prior_backtest_pins(root, old.parent, PARENT)


class SavedReaderTests(unittest.TestCase):
    def test_reader_checks_grid_and_does_not_implicitly_run_engine(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            directory, _ = saved_fixture(root)
            with patch.object(worker, "run_action_accounting", side_effect=AssertionError("No implicit PnL")):
                saved = worker.read_completed_backtest(directory, root=root, identity_sha256=PARENT)
            self.assertEqual(len(saved.comparison), 64)
            self.assertEqual(len(saved.selection_validation), 8)

    def test_changed_report_bytes_fail_integrity(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, _ = saved_fixture(Path(temp))
            path = directory / "selection_coverage.parquet"
            path.write_bytes(path.read_bytes() + b"changed")
            with self.assertRaises(DataContractError):
                worker.read_completed_backtest(directory)

    def test_incomplete_scenario_grid_refused_even_if_report_repin_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, audit = saved_fixture(Path(temp))
            path = directory / "comparison.parquet"
            table = pq.read_table(path).slice(1)
            pq.write_table(table, path)
            audit["artifacts"]["comparison.parquet"] = _pin(directory, path)
            _write_json(directory / "audit.json", audit)
            with self.assertRaisesRegex(DataContractError, "complete grid"):
                worker.read_completed_backtest(directory)

    def test_missing_selection_validator_receipt_refused(self):
        with tempfile.TemporaryDirectory() as temp:
            directory, audit = saved_fixture(Path(temp))
            path = directory / "selection_validation.parquet"
            pq.write_table(pa.table({"basket_verified": [True] * 7}), path)
            audit["artifacts"]["selection_validation.parquet"] = _pin(directory, path)
            _write_json(directory / "audit.json", audit)
            with self.assertRaises(DataContractError):
                worker.read_completed_backtest(directory)

    def test_phase_or_fit_claim_change_refused(self):
        for alteration in ("new_fit", "future", "parent", "phase"):
            with self.subTest(alteration=alteration), tempfile.TemporaryDirectory() as temp:
                directory, audit = saved_fixture(Path(temp))
                if alteration == "new_fit": audit["models_fitted"] = True
                elif alteration == "future": audit["financial_2020_plus_read"] = True
                elif alteration == "parent": audit["build_identity_sha256"] = "sha256:" + "b" * 64
                else: audit["phase"] = "old_backtest"
                _write_json(directory / "audit.json", audit)
                with self.assertRaises(DataContractError):
                    worker.read_completed_backtest(directory, identity_sha256=PARENT)

    def test_root_source_pin_guard_catches_changed_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            directory, audit = saved_fixture(root)
            source = root / "source.json"
            source.write_text("{}")
            phase = json.loads((directory / "build_identity.json").read_text())
            phase["source_pins"] = [_pin(root, source)]
            _write_json(directory / "build_identity.json", phase)
            audit["phase_identity_sha256"] = canonical_sha256(phase)
            audit["artifacts"]["build_identity.json"] = _pin(directory, directory / "build_identity.json")
            _write_json(directory / "audit.json", audit)
            source.write_text("{\"changed\":true}")
            with self.assertRaises(DataContractError):
                worker.read_completed_backtest(directory, root=root)


if __name__ == "__main__":
    unittest.main()
