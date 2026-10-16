"""Restricted selection is explicit, pre-ranking and independent of outcomes."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.backtest_selection import select_monthly_basket
from dmv2.catalog import canonical_sha256
from dmv2.fallback_backtest import (_annual_prices,_configuration,
    _reviewed_events,read_fallback_backtest,select_restricted_monthly_basket)
from dmv2.universe import DataContractError

PROJECT=Path(__file__).resolve().parents[1]
SESSIONS=("2017-12-29","2018-01-02","2018-01-31")


def configuration():
    return json.loads((PROJECT/"configs/data_limited_fallback.json").read_text())


def predictions(classes=(1,10)):
    rows=[]
    for side,mode in enumerate(classes):
        for i in range(13):
            native=("sec_gjP4xM" if side==0 else "sec_zmYNnz") if i==0 else f"native-{side}-{i:02d}"
            probability=.8-i/100
            values={f"probability_{k}":probability if k==mode else (1-probability)/9 for k in range(1,11)}
            rows.append(dict(source_security_id=f"source-{side}-{i:02d}",signal_date="2017-12-29",
                intrinio_security_id=native,prediction_eligible=True,paper_prediction_eligible=True,
                outcome_month="2018-01",outcome_start="2018-01-02",outcome_end="2018-01-31",
                prediction_status="predicted",**values))
    return pd.DataFrame(rows)


class RestrictedSelectionTests(unittest.TestCase):
    def select(self,**updates):
        args=dict(predictions=predictions(),basket="class_1_10",sessions=SESSIONS,configuration=configuration())
        args.update(updates)
        return select_restricted_monthly_basket(**args)

    def test_fixed_restriction_precedes_top_ten_and_replaces_no_failed_holding(self):
        source=predictions()
        old=select_monthly_basket(source,basket="class_1_10",sessions=SESSIONS)
        self.assertTrue(old.selections.intrinio_security_id.isin(["sec_gjP4xM","sec_zmYNnz"]).any())
        result=self.select()
        self.assertEqual(len(result.selections),20)
        self.assertFalse(result.selections.execution_restricted.any())
        self.assertFalse(result.selections.intrinio_security_id.isin(["sec_gjP4xM","sec_zmYNnz"]).any())
        self.assertEqual(set(result.selections.query("side=='long'").source_security_id),
                         {f"source-0-{i:02d}" for i in range(1,11)})

    def test_all_prediction_keys_flags_and_modes_remain_visible(self):
        source=predictions()
        before=source.copy(deep=True)
        result=self.select(predictions=source)
        self.assertEqual(result.ledger.source_security_id.tolist(),source.source_security_id.tolist())
        self.assertTrue(result.ledger.prediction_eligible.all())
        self.assertTrue(result.ledger.paper_prediction_eligible.all())
        self.assertEqual(int(result.ledger.trade_eligible.sum()),24)
        restricted=result.ledger[result.ledger.execution_restricted]
        self.assertEqual(set(restricted.predicted_class),{1,10})
        self.assertTrue(restricted.selection_status.eq("execution_excluded_retrospective_data_quality").all())
        pd.testing.assert_frame_equal(source,before)

    def test_restriction_and_prediction_denominators_are_reported_separately(self):
        row=self.select().monthly.iloc[0]
        self.assertEqual(row.original_rows,26)
        self.assertEqual(row.prediction_eligible,26)
        self.assertEqual(row.trade_eligible,24)
        self.assertEqual(row.execution_restricted_prediction_rows,2)
        self.assertTrue(row.retrospective_execution_restriction)
        self.assertFalse(row.historical_universe_performance_claim)

    def test_second_pair_is_separate_and_does_not_pool_extremes(self):
        result=self.select(predictions=predictions((2,9)),basket="class_2_9")
        self.assertEqual(set(result.selections.predicted_class),{2,9})
        self.assertEqual(len(result.selections.query("side=='long'")),10)
        self.assertEqual(len(result.selections.query("side=='short'")),10)

    def test_outcomes_and_price_coverage_are_irrelevant_to_selection(self):
        source=predictions()
        expected=self.select(predictions=source)
        source["target_class"]=10
        source["target_return"]=np.nan
        source["daily_quote_path_complete"]=False
        source["future_membership"]=False
        actual=self.select(predictions=source)
        pd.testing.assert_frame_equal(expected.selections,actual.selections)

    def test_unchanged_ret_gate_withholds_instead_of_using_class_numbers(self):
        result=self.select(basket="ret")
        self.assertTrue(result.selections.empty)
        self.assertTrue(result.monthly.status.eq("withheld_incomplete_120_month_class_grid").all())

    def test_trade_shortage_is_explicit_whole_rebalance_withheld(self):
        source=predictions().drop(index=[11,12]).reset_index(drop=True)
        result=self.select(predictions=source)
        self.assertEqual(len(result.selections),20)
        source=source.drop(index=10).reset_index(drop=True)
        result=self.select(predictions=source)
        self.assertTrue(result.selections.empty)
        self.assertEqual(result.rebalances.iloc[0].selection_status,"withheld_insufficient_mode_candidates")
        self.assertEqual(result.monthly.iloc[0].long_candidates,9)

    def test_already_ineligible_original_is_not_promoted_by_execution_filter(self):
        source=predictions()
        source.loc[1,"prediction_eligible"]=False
        source.loc[1,"paper_prediction_eligible"]=False
        source.loc[1,[f"probability_{k}" for k in range(1,11)]]=np.nan
        result=self.select(predictions=source)
        self.assertFalse(result.ledger.loc[1,"trade_eligible"])
        self.assertFalse(result.ledger.loc[1,"selected"])
        self.assertFalse(result.ledger.loc[1,"prediction_eligible"])

    def test_excluded_probability_vectors_are_still_validated(self):
        source=predictions()
        source.loc[0,"probability_1"]=np.nan
        with self.assertRaises(DataContractError):
            self.select(predictions=source)

    def test_frozen_recipe_refuses_extra_hindsight_exclusions_or_other_edits(self):
        for kind in ("extra_id","topk","later_period"):
            cfg=configuration()
            if kind=="extra_id":
                cfg["execution_restrictions"]["native_ids"]["new_bad_stock"]="unapproved"
            elif kind=="topk":
                cfg["backtest"]["top_k_per_side"]=9
            else:
                cfg["backtest"]["end"]="2020-12-31"
            with self.subTest(kind=kind),self.assertRaises(DataContractError):
                self.select(configuration=cfg)

    def test_future_metadata_refused_before_forecast_value_decoding(self):
        source=predictions()
        source.loc[0,"signal_date"]="2019-12-31"
        with self.assertRaises(DataContractError):
            self.select(predictions=source)


class RunnerGuardTests(unittest.TestCase):
    def event_manifest(self):
        from dmv2.action_loading import EVENT_TERMS
        return {"events":[dict(old_ticker=ticker,old_provider_id=terms[0],effective_date=terms[1],
            application_session=terms[2],cash_per_old_raw_share=terms[3],successor_raw_shares=terms[4],
            source_url="https://www.sec.gov/synthetic-primary-evidence") for ticker,terms in EVENT_TERMS.items()]}

    def test_event_reader_keeps_same_eight_legal_entitlements_without_old_pointers(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"data/derived/daily_action_inputs/action_evidence.json"
            path.parent.mkdir(parents=True)
            source=self.event_manifest();path.write_text(json.dumps(source))
            events,actual_path=_reviewed_events(root,[item["application_session"] for item in source["events"]])
            self.assertEqual(actual_path,path)
            self.assertEqual(len(events),8)
            self.assertEqual(next(event for event in events if event["event_id"].startswith("RDC"))["successor_id"],
                             "unquoted_successor:CUSIP:G3166L100")
            self.assertTrue(all(event["verified"] is True and event["knowledge_time_verified"] is False for event in events))

    def test_event_reader_refuses_new_terms_extra_event_or_missing_session(self):
        for alteration in ("cash","extra","calendar"):
            with self.subTest(alteration=alteration),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);path=root/"data/derived/daily_action_inputs/action_evidence.json"
                path.parent.mkdir(parents=True)
                source=self.event_manifest();sessions=[item["application_session"] for item in source["events"]]
                if alteration=="cash":source["events"][0]["cash_per_old_raw_share"]+=1
                if alteration=="extra":source["events"].append(source["events"][0])
                if alteration=="calendar":sessions=sessions[1:]
                path.write_text(json.dumps(source))
                with self.assertRaises(DataContractError):_reviewed_events(root,sessions)

    def test_event_reader_refuses_unreviewed_evidence_domain(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);path=root/"data/derived/daily_action_inputs/action_evidence.json"
            path.parent.mkdir(parents=True)
            source=self.event_manifest();source["events"][0]["source_url"]="https://random.example.com/claim"
            path.write_text(json.dumps(source))
            with self.assertRaises(DataContractError):
                _reviewed_events(root,[item["application_session"] for item in source["events"]])

    def test_configuration_pin_matches_current_frozen_public_recipe(self):
        self.assertEqual(_configuration(configuration()),configuration())

    def test_reader_does_not_implicitly_run_engine(self):
        with tempfile.TemporaryDirectory() as temp,patch("dmv2.fallback_backtest.run_action_accounting",side_effect=AssertionError("No implicit PnL")):
            path=Path(temp)
            (path/"audit.json").write_text(json.dumps({"status":"STARTED"}))
            with self.assertRaises(DataContractError):
                read_fallback_backtest(path)

    def test_annual_scope_checked_before_ohlc_projection(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);run=root/"new"
            year=run/"years/2018";year.mkdir(parents=True)
            path=year/"daily_prices_raw.parquet"
            pq.write_table(pa.table({"trading_date":["2020-01-02"],"intrinio_security_id":["A"]}),path)
            from dmv2.backtest_loading import _pin
            audit={"status":"COMPLETE","year":2018,"build_identity_sha256":"parent","quote_amendments_fingerprint":"quotes",
                   "artifacts":{"daily_prices_raw":_pin(year,path)}}
            (year/"audit.json").write_text(json.dumps(audit))
            with patch("dmv2.fallback_backtest._projected_bytes",side_effect=AssertionError("No OHLC projection before date guard")):
                with self.assertRaisesRegex(DataContractError,"out-of-year"):
                    _annual_prices(root,run,"parent","quotes",[])


if __name__=="__main__":
    unittest.main()
