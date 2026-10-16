"""Signed default SNI consideration and additive-evidence boundary tests."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dmv2.action_accounting import run_action_accounting
from dmv2.action_loading import EVENT_TERMS
from dmv2.backtest_accounting import DailyAccountingPolicy
from dmv2.universe import DataContractError

try:
    from dmv2 import sni_action as worker
except ImportError:
    spec = importlib.util.spec_from_file_location("dmv2.sni_action", Path(__file__).with_name("sni_action.py"))
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)


def prices():
    rows = []
    for day, native, value in [
        ("2018-03-01", worker.SNI_PREDECESSOR_ID, 90.),
        ("2018-03-06", worker.SNI_PREDECESSOR_ID, 90.),
        ("2018-03-07", worker.SNI_SUCCESSOR_ID, 25.),
        ("2018-03-08", worker.SNI_SUCCESSOR_ID, 26.),
        *[(day, "flat", 100.) for day in ("2018-03-01", "2018-03-06", "2018-03-07", "2018-03-08")],
    ]:
        rows.append({"trading_date": day, "intrinio_security_id": native,
            "open": value, "close": value, "adj_open": value, "adj_close": value,
            "adj_high": value, "adj_low": value, "frequency": "daily", "intraperiod": False,
            "price_quality_status": "accepted"})
    return pd.DataFrame(rows)


def old_events():
    return [{"event_id": ticker + "-" + values[1], "predecessor_id": values[0],
        "effective_date": values[2], "cash_per_raw_share": values[3], "stock_ratio": values[4],
        "verified": True, "arbitrary_original_audit_field": {"keep": [ticker]}}
        for ticker, values in EVENT_TERMS.items()]


def evidence(directory):
    identity = {"master": {"intrinio_security_id": worker.SNI_SUCCESSOR_ID,
        "ticker": "DISCK", "type": "Ordinary Shares - Series C"},
        "links": [{"intrinio_security_id": worker.SNI_SUCCESSOR_ID,
            "query_identifier_type": "isin", "query_identifier": "US25470F3029",
            "resolution_status": "resolved"}]}
    (directory / "successor_identity.json").write_text(json.dumps(identity))
    prices().to_parquet(directory / "predecessor_successor_controls.parquet", index=False)
    artifacts = {}
    for name in ("successor_identity.json", "predecessor_successor_controls.parquet"):
        path = directory / name
        artifacts[name] = {"path": name, "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
    receipt = {"schema": "dmv2-sni-action-evidence/v1", "event": deepcopy(worker.SNI_EVENT),
        "no_new_quote_created": True, "successor_class_verified": True,
        "settlement_or_fractional_cash_invented": False, "artifacts": artifacts}
    (directory / "sni_action_evidence.json").write_text(json.dumps(receipt))
    return receipt


class SniActionTests(unittest.TestCase):
    def account(self, side="long", quotes=None, policy=None):
        selection = pd.DataFrame([{"signal_date": "2018-02-28", "execution_date": "2018-03-01",
            "source_security_id": "sni", "intrinio_security_id": worker.SNI_PREDECESSOR_ID,
            "side": side}, {"signal_date": "2018-02-28", "execution_date": "2018-03-01",
            "source_security_id": "other", "intrinio_security_id": "flat",
            "side": "short" if side == "long" else "long"}])
        return run_action_accounting(selection, prices() if quotes is None else quotes,
            ["2018-03-01", "2018-03-06", "2018-03-07", "2018-03-08"],
            rebalances=pd.DataFrame([{"signal_date": "2018-02-28", "execution_date": "2018-03-01"}]),
            terminal_date="2018-03-08", events=[worker.SNI_EVENT], policy=policy)

    def test_long_receives_cash_plus_signed_class_c_units_and_no_sni_gap(self):
        result = self.account()
        self.assertEqual(result.status, "complete_proxy")
        action = result.action_events.iloc[0]
        raw_quantity = .5 / 90.
        self.assertAlmostEqual(action.signed_cash, raw_quantity * 65.82)
        self.assertAlmostEqual(action.new_adjusted_units, raw_quantity * 1.0584)
        self.assertAlmostEqual(result.summary["final_nav"], 1. + raw_quantity * (65.82 + 1.0584 * 26. - 90.))
        self.assertEqual(str(action.date.date()), "2018-03-07")
        self.assertEqual(action.successor_id, worker.SNI_SUCCESSOR_ID)
        self.assertFalse(result.trades.date.eq(pd.Timestamp("2018-03-07")).any())

    def test_short_owes_cash_and_delivered_shares(self):
        result = self.account(side="short")
        action = result.action_events.iloc[0]
        raw_quantity = -.5 / 90.
        self.assertLess(action.signed_cash, 0.)
        self.assertLess(action.new_adjusted_units, 0.)
        self.assertAlmostEqual(action.signed_cash, raw_quantity * 65.82)
        self.assertAlmostEqual(result.summary["final_nav"], 1. + raw_quantity * (65.82 + 1.0584 * 26. - 90.))

    def test_cash_is_not_added_to_adjustment_a_second_time(self):
        p = prices()
        predecessor = p.intrinio_security_id.eq(worker.SNI_PREDECESSOR_ID)
        successor = p.intrinio_security_id.eq(worker.SNI_SUCCESSOR_ID)
        for field in ("adj_open", "adj_close", "adj_high", "adj_low"):
            p.loc[predecessor, field] *= .5
            p.loc[successor, field] *= .25
        result = self.account(quotes=p)
        action = result.action_events.iloc[0]
        self.assertAlmostEqual(action.predecessor_factor, .5)
        self.assertAlmostEqual(action.successor_factor, .25)
        self.assertAlmostEqual(action.signed_raw_entitlement, .5 / 90.)
        self.assertAlmostEqual(action.new_adjusted_units, (.5 / 90.) * 1.0584 / .25)
        self.assertAlmostEqual(result.summary["final_nav"], 1. + (.5 / 90.) * (65.82 + 1.0584 * 26. - 90.))

    def test_stop_threshold_carries_original_mixed_package_basis(self):
        result = self.account(policy=DailyAccountingPolicy(asset_stop_fraction=.1))
        action = result.action_events.iloc[0]
        self.assertAlmostEqual(action.new_stop_price, (81. - 65.82) / 1.0584)
        self.assertNotAlmostEqual(action.new_stop_price, 25. * .9)
        self.assertTrue(result.stops.empty)

    def test_absent_successor_quote_withholds_without_invented_payout(self):
        p = prices()
        p = p.loc[~(p.intrinio_security_id.eq(worker.SNI_SUCCESSOR_ID)
            & p.trading_date.eq("2018-03-07"))]
        result = self.account(quotes=p)
        self.assertNotEqual(result.status, "complete_proxy")
        self.assertTrue(result.action_events.empty)
        self.assertFalse(result.failures.empty)

    def test_addition_preserves_all_eight_original_event_mappings(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            evidence(directory)
            originals = old_events()
            before = deepcopy(originals)
            result = worker.append_sni_action(originals, directory, prices=prices())
            self.assertEqual(originals, before)
            self.assertEqual(result[:-1], before)
            self.assertEqual(len(result), 9)
            result[0]["arbitrary_original_audit_field"]["keep"].append("changed")
            self.assertEqual(originals, before)

    def test_tampered_price_evidence_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            evidence(directory)
            with (directory / "predecessor_successor_controls.parquet").open("ab") as stream:
                stream.write(b"drift")
            with self.assertRaises(DataContractError):
                worker.load_sni_action(directory)

    def test_wrong_series_a_rejected_even_when_manifest_is_repinned(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            receipt = evidence(directory)
            path = directory / "successor_identity.json"
            item = json.loads(path.read_text())
            item["master"]["ticker"] = "DISCA"
            item["master"]["type"] = "Ordinary Shares - Series A"
            path.write_text(json.dumps(item))
            receipt["artifacts"][path.name] = {"path": path.name, "size_bytes": path.stat().st_size,
                "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}
            (directory / "sni_action_evidence.json").write_text(json.dumps(receipt))
            with self.assertRaises(DataContractError):
                worker.load_sni_action(directory)

    def test_changed_economic_terms_or_missing_old_event_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            receipt = evidence(directory)
            receipt["event"]["cash_per_raw_share"] = 90.
            (directory / "sni_action_evidence.json").write_text(json.dumps(receipt))
            with self.assertRaises(DataContractError):
                worker.append_sni_action(old_events(), directory)
            with self.assertRaises(DataContractError):
                worker.append_sni_action(old_events()[:-1], directory)

    def test_post_suspension_sni_bar_is_not_a_repair(self):
        p = prices()
        row = p.iloc[0].copy()
        row.trading_date = "2018-03-07"
        p = pd.concat([p, row.to_frame().T], ignore_index=True)
        with self.assertRaises(DataContractError):
            worker.validate_sni_action_prices(p)

    def test_future_financial_dates_rejected_before_quote_access(self):
        p = prices()
        p.loc[0, "trading_date"] = "2020-01-02"
        with self.assertRaises(DataContractError):
            worker.validate_sni_action_prices(p)


if __name__ == "__main__":
    unittest.main()
