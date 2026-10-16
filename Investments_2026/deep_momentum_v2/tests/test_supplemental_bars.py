"""Synthetic unit-transport checks; fixture prices are never research data."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from dmv2.supplemental_bars import admit_reviewed_missing_bars, MISSING_SPECS


class SupplementalBarTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.table_uri = "gs://synthetic-only/published/daily_bars.parquet"
        self.path = self.root / "synthetic_selected.parquet"
        self.receipt = self.root / "synthetic_receipt.json"
        self.receipt.write_text(json.dumps({"synthetic_only": True}))
        original, published, seen = [], [], set()
        self.mapping = {}
        for source_id, sid, day, left, right in MISSING_SPECS:
            self.mapping[source_id] = sid
            multiplier = .7 if sid == "sec_XLvxMP" else 1.
            for control in (left, right):
                if (sid, control) in seen:
                    continue
                seen.add((sid, control))
                original.append({"intrinio_security_id": sid, "trading_date": control,
                    "open": 100., "high": 110., "low": 90., "close": 105.,
                    "adj_open": 100. * multiplier, "adj_high": 110. * multiplier,
                    "adj_low": 90. * multiplier, "adj_close": 105. * multiplier,
                    "factor": 1., "split_ratio": 1., "dividend": 0., "frequency": "daily",
                    "intraperiod": False, "price_quality_status": "accepted",
                    "raw_ohlc_bounds_valid": True, "adjusted_ohlc_bounds_valid": True})
                published.append(self.published(source_id, control, open=100., high=110., low=90., close=105.))
            # Deliberately different actual-source values: never infer this bar
            # from the identical neighboring control prices.
            published.append(self.published(source_id, day, open=120., high=130., low=110., close=125.))
        self.prices = pd.DataFrame(original)
        self.rows = published
        self.provenance = {"publication_manifest_uri": "gs://synthetic-only/MANIFEST.json",
            "publication_manifest_generation": "123", "publication_manifest_canonical_sha256": "sha256:" + "2" * 64,
            "publication_manifest_canonical_pin_verified": True, "source_table_uri": self.table_uri,
            "source_table_generation": "456", "source_table_size_bytes": 10000000,
            "source_table_sha256_declared": "sha256:" + "3" * 64, "whole_source_table_sha256_reread": False,
            "raw_page_hash_verified": False, "range_receipt": self.pin(self.receipt),
            "projection_policy": "date_and_id_filter_in_Arrow_before_Python_conversion",
            "mixed_row_group_numeric_decoding_possible": True, "retained_manifest_pin_verified": False,
            "projected_compressed_bytes": 40000, "maximum_projected_compressed_bytes": 64 * 1024 * 1024}

    def published(self, source_id, day, **values):
        return {"security_id": source_id, "trading_date": day, "query_symbol": "SYNTHETIC-" + source_id,
            "adjustment": "raw", "feed": "sip", "timeframe": "1Day", "source_task_id": "synthetic-task",
            "source_request_id": "synthetic-request", "source_page_hash": "1" * 64, **values}

    def pin(self, path):
        return {"path": path.name, "size_bytes": path.stat().st_size,
            "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(), "source_uri": self.table_uri}

    def run_case(self, **changes):
        pq.write_table(pa.Table.from_pylist(self.rows), self.path)
        args = dict(prices=self.prices, evidence_root=self.root, projection_evidence=self.pin(self.path),
                    source_provenance=self.provenance, native_mapping=self.mapping,
                    review_notes="Synthetic-only reviewed source identity and provisional no-action unit transport")
        args.update(changes)
        return admit_reviewed_missing_bars(**args)

    def test_actual_source_bar_is_appended_instead_of_price_interpolation(self):
        before = self.prices.copy(deep=True)
        result = self.run_case()
        self.assertEqual(len(result.prices), len(before) + 4)
        self.assertTrue(result.ledger.status.eq("APPENDED_ACTUAL_RAW_BAR_PROVISIONAL_UNITS").all())
        pd.testing.assert_frame_equal(self.prices, before)
        pd.testing.assert_frame_equal(result.prices.iloc[:len(before)][before.columns], before, check_dtype=False)
        new = result.prices.iloc[len(before):]
        self.assertTrue(new.open.eq(120.).all())
        self.assertTrue(new.close.eq(125.).all())
        frt = new.loc[new.intrinio_security_id.eq("sec_XLvxMP")]
        self.assertTrue(frt.adj_open.eq(84.).all())
        self.assertTrue(frt.supplemental_adjustment_transport_provisional.eq(True).all())
        self.assertTrue(new.dividend.isna().all())
        self.assertTrue(new.split_ratio.isna().all())
        self.assertFalse(result.ledger.raw_price_interpolation.any())
        self.assertFalse(result.ledger.complete_corporate_action_feed_verified.any())

    def test_existing_source_bar_is_never_overwritten(self):
        first = self.run_case()
        second = self.run_case(prices=first.prices)
        self.assertEqual(len(first.prices), len(second.prices))
        self.assertTrue(second.ledger.reason.str.contains("already exists").all())
        pd.testing.assert_frame_equal(first.prices, second.prices)

    def test_nonunit_split_or_dividend_prevents_corresponding_append(self):
        for field, value in (("split_ratio", 2.), ("dividend", .1), ("dividend", None)):
            with self.subTest(field=field, value=value):
                prices = self.prices.copy(deep=True)
                prices.loc[prices.intrinio_security_id.eq("sec_XLvxMP"), field] = value
                result = self.run_case(prices=prices)
                self.assertEqual(len(result.prices) - len(prices), 2)
                frt = result.ledger.loc[result.ledger.intrinio_security_id.eq("sec_XLvxMP")]
                self.assertTrue(frt.status.eq("REJECTED_UNCHANGED").all())

    def test_adjustment_discontinuity_refuses_append_without_smoothing(self):
        prices = self.prices.copy(deep=True)
        mask = prices.intrinio_security_id.eq("sec_XLvxMP") & prices.trading_date.eq("2018-09-11")
        prices.loc[mask, "adj_close"] += 1.
        result = self.run_case(prices=prices)
        frt = result.ledger.loc[result.ledger.intrinio_security_id.eq("sec_XLvxMP")]
        self.assertTrue(frt.reason.str.contains("discontinuous").all())
        self.assertEqual(len(result.prices) - len(prices), 2)

    def test_missing_following_source_control_refuses_both_frt_days(self):
        prices = self.prices.loc[~(self.prices.intrinio_security_id.eq("sec_XLvxMP")
                                  & self.prices.trading_date.eq("2018-09-11"))].copy()
        result = self.run_case(prices=prices)
        frt = result.ledger.loc[result.ledger.intrinio_security_id.eq("sec_XLvxMP")]
        self.assertTrue(frt.reason.str.contains("bracket source bar").all())
        self.assertEqual(len(result.prices) - len(prices), 2)

    def test_published_control_close_conflict_refuses_append(self):
        for row in self.rows:
            if row["security_id"] == MISSING_SPECS[0][0] and row["trading_date"] == "2018-09-06":
                row["close"] = 106.
        result = self.run_case()
        frt = result.ledger.loc[result.ledger.intrinio_security_id.eq("sec_XLvxMP")]
        self.assertTrue(frt.reason.str.contains("control raw close conflicts").all())

    def test_no_unlisted_missing_key_is_automatically_admitted(self):
        source_id = MISSING_SPECS[0][0]
        self.rows.append(self.published(source_id, "2018-09-12", open=120., high=130., low=110., close=125.))
        result = self.run_case()
        self.assertEqual(len(result.prices) - len(self.prices), 4)
        self.assertFalse(result.prices.trading_date.eq("2018-09-12").any())

    def test_rejected_bracket_quote_is_not_used_as_adjustment_evidence(self):
        prices = self.prices.copy(deep=True)
        prices.loc[prices.intrinio_security_id.eq("sec_XLvxMP"), "price_quality_status"] = "provider_ohlc_anomaly"
        result = self.run_case(prices=prices)
        frt = result.ledger.loc[result.ledger.intrinio_security_id.eq("sec_XLvxMP")]
        self.assertTrue(frt.reason.str.contains("source quality").all())
        self.assertEqual(len(result.prices) - len(prices), 2)

    def test_existing_overlay_original_missing_status_and_nulls_are_preserved(self):
        prices = self.prices.copy(deep=True)
        for column in ("open", "price_quality_status"):
            prices["original_" + column] = prices[column]
        result = self.run_case(prices=prices)
        new = result.prices.iloc[len(prices):]
        self.assertTrue(new.original_open.isna().all())
        self.assertTrue(new.original_price_quality_status.eq("missing_quote").all())

    def test_frozen_source_native_mapping_mismatch_refuses_relevant_case(self):
        mapping = dict(self.mapping); mapping[MISSING_SPECS[0][0]] = "different-native"
        result = self.run_case(native_mapping=mapping)
        frt = result.ledger.loc[result.ledger.intrinio_security_id.eq("sec_XLvxMP")]
        self.assertTrue(frt.reason.str.contains("mapping differs").all())

    def test_append_does_not_introduce_optional_acceptance_column(self):
        self.assertNotIn("price_accepted", self.prices)
        result = self.run_case()
        self.assertNotIn("price_accepted", result.prices)

    def test_existing_explicit_acceptance_column_is_preserved(self):
        prices = self.prices.copy(deep=True); prices["price_accepted"] = True
        result = self.run_case(prices=prices)
        self.assertTrue(result.prices.price_accepted.eq(True).all())

    def test_action_engine_preserves_original_quality_after_append(self):
        from dmv2.action_accounting import run_action_accounting
        from dmv2.backtest_accounting import DailyAccountingPolicy
        result = self.run_case()
        selections = pd.DataFrame([
            {"signal_date": "2018-08-31", "execution_date": "2018-09-06", "source_security_id": source,
             "intrinio_security_id": sid, "side": side}
            for source, sid, side in ((MISSING_SPECS[0][0], "sec_XLvxMP", "long"),
                                      (MISSING_SPECS[2][0], "sec_geo2bz", "short"))])
        sessions = ["2018-09-06", "2018-09-07", "2018-09-10"]
        accounting = run_action_accounting(selections=selections,
            prices=result.prices.loc[result.prices.trading_date.isin(sessions)], sessions=sessions,
            rebalances=pd.DataFrame([{"signal_date": "2018-08-31", "execution_date": "2018-09-06"}]),
            terminal_date="2018-09-10", events=[], policy=DailyAccountingPolicy())
        self.assertTrue(accounting.failures.empty, accounting.failures.to_dict("records"))
        self.assertEqual(accounting.status, "complete_proxy")
        self.assertTrue(pd.notna(accounting.daily.iloc[0].nav))


if __name__ == "__main__":
    unittest.main()
