"""Synthetic source/parser contracts; no fixture enters research outputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from dmv2.alpaca_repairs import (parse_alpaca_raw_page, _selected_number,
                                prepare_alpaca_repair_declarations,
                                prepare_published_alpaca_repair_declarations)
from dmv2.ohlc_repairs import OhlcRepairPolicy, apply_reviewed_ohlc_repairs
from dmv2.universe import DataContractError


def bar(day="2018-08-02", **changes):
    value = {"t": day + "T04:00:00Z", "o": 100., "h": 110., "l": 90., "c": 105.,
             "v": 1234, "n": 42, "vw": 103.}
    value.update(changes)
    return value


class AlpacaRepairsTests(unittest.TestCase):
    def setUp(self):
        self.body = {"symbol": "SYNTHETIC", "bars": [bar()], "next_page_token": None}
        self.keys = [("SYNTHETIC", "2018-08-02")]

    def parse(self, body=None, *, raw=None, keys=None, provenance_changes=None, **limits):
        raw = json.dumps(self.body if body is None else body).encode() if raw is None else raw
        pin = "sha256:" + hashlib.sha256(raw).hexdigest()
        provenance = {"feed": "sip", "adjustment": "raw", "timeframe": "1Day",
            "source_uri": "gs://synthetic-only/raw/page.json", "source_generation": "123",
            "source_page_sha256": pin, "source_task_id": "synthetic-task",
            "source_request_id": "synthetic-request", "query_symbol": "SYNTHETIC"}
        provenance.update(provenance_changes or {})
        return parse_alpaca_raw_page(raw, self.keys if keys is None else keys,
            expected_sha256=pin, provenance=provenance, **limits)

    def test_raw_strings_complete_provenance_and_no_implicit_admission(self):
        result = self.parse()
        self.assertEqual(len(result.records), 1)
        row = result.records[0]
        self.assertEqual(row["raw_ohlc"], {"open": 100., "high": 110., "low": 90., "close": 105.})
        self.assertEqual(row["original_source_value_strings"]["volume"], "1234")
        self.assertEqual(row["original_source_value_strings"]["open"], "100.0")
        self.assertEqual(row["source_generation"], "123")
        self.assertFalse(row["source_admitted"])
        self.assertFalse(row["historical_identity_certified"])
        self.assertEqual(result.coverage[0]["status"], "SOURCE_ROW_FOUND")
        self.assertEqual(result.summary["numeric_2020_plus_conversions"], 0)

    def test_multisymbol_response_without_cross_symbol_reassociation(self):
        self.body = {"bars": {"SYNTHETIC": [bar()], "OTHER": [bar(o=-999)]}}
        result = self.parse()
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.records[0]["query_symbol"], "SYNTHETIC")
        self.assertEqual(result.summary["unrequested_metadata_rows_skipped"], 1)

    def test_2020_2023_2024_numbers_never_reach_numeric_conversion(self):
        self.body["bars"] += [bar("2020-08-02", o=-999), bar("2023-08-02", o=-999),
                             bar("2024-08-02", o=-999)]
        with patch("dmv2.alpaca_repairs._selected_number", wraps=_selected_number) as converter:
            result = self.parse()
        self.assertEqual(converter.call_count, 7)
        self.assertTrue(all(call.args[0] != "-999" for call in converter.call_args_list))
        self.assertEqual(result.summary["outside_2018_2019_metadata_rows_skipped"], 3)
        self.assertNotIn("2024-08-02", repr(result))

    def test_huge_unrequested_number_kept_as_text_without_float_overflow(self):
        raw = b'{"symbol":"SYNTHETIC","bars":[{"t":"2024-08-02T04:00:00Z","o":1e999999,"h":1e999999,"l":1e999999,"c":1e999999}]}'
        with patch("dmv2.alpaca_repairs._selected_number", side_effect=AssertionError("No numeric conversion allowed")):
            result = self.parse(raw=raw)
        self.assertEqual(result.records, ())
        self.assertEqual(result.coverage[0]["status"], "NOT_IN_THIS_RAW_PAGE")

    def test_unrequested_old_numeric_anomaly_is_not_analyzed(self):
        self.body["bars"] += [bar("2018-08-03", o=-999)]
        result = self.parse()
        self.assertEqual(len(result.records), 1)
        self.assertEqual(result.summary["unrequested_metadata_rows_skipped"], 1)

    def test_2020_request_rejected_before_json_decoding(self):
        with patch("dmv2.alpaca_repairs.json.loads", side_effect=AssertionError("JSON should not be decoded")):
            with self.assertRaisesRegex(DataContractError, "2018–2019"):
                self.parse(keys=[("SYNTHETIC", "2020-01-02")])

    def test_page_hash_and_provenance_hash_must_match(self):
        raw = json.dumps(self.body).encode()
        provenance = {"feed": "sip", "adjustment": "raw", "timeframe": "1Day",
            "source_uri": "gs://synthetic-only/raw/page.json", "source_generation": "123",
            "source_page_sha256": "sha256:" + "0" * 64, "source_task_id": "synthetic-task",
            "source_request_id": "synthetic-request", "query_symbol": "SYNTHETIC"}
        with self.assertRaisesRegex(DataContractError, "source pin"):
            parse_alpaca_raw_page(raw, self.keys, expected_sha256="sha256:" + "0" * 64, provenance=provenance)
        with self.assertRaisesRegex(DataContractError, "provenance page pin"):
            self.parse(provenance_changes={"source_page_sha256": "sha256:" + "0" * 64})

    def test_only_actual_raw_sip_daily_provenance(self):
        for key, value in (("feed", "iex"), ("adjustment", "all"), ("timeframe", "1Hour"),
                           ("source_generation", ""), ("source_uri", "https://example.invalid")):
            with self.subTest(key=key):
                with self.assertRaises(DataContractError):
                    self.parse(provenance_changes={key: value})

    def test_wrong_single_symbol_response_refused(self):
        self.body["symbol"] = "DIFFERENT"
        with self.assertRaisesRegex(DataContractError, "Response symbol"):
            self.parse()

    def test_missing_single_symbol_uses_explicit_original_request(self):
        del self.body["symbol"]
        self.assertEqual(self.parse().records[0]["query_symbol"], "SYNTHETIC")

    def test_selected_multisymbol_requires_corresponding_task_provenance(self):
        self.body = {"bars": {"OTHER": [bar()]}}
        with self.assertRaisesRegex(DataContractError, "task provenance"):
            self.parse(keys=[("OTHER", "2018-08-02")])

    def test_duplicate_selected_bars_refused_instead_of_silent_dedup(self):
        self.body["bars"].append(copy.deepcopy(self.body["bars"][0]))
        with self.assertRaisesRegex(DataContractError, "Duplicate selected"):
            self.parse()

    def test_duplicate_requested_keys_refused(self):
        with self.assertRaisesRegex(DataContractError, "Duplicate requested"):
            self.parse(keys=self.keys * 2)

    def test_unknown_envelope_and_wrong_bars_shape_refused(self):
        for body in ({"response": self.body}, {"bars": None}, {"bars": {"SYNTHETIC": None}}, []):
            with self.subTest(body=body):
                with self.assertRaises(DataContractError):
                    self.parse(body=body)

    def test_duplicate_json_fields_refused(self):
        with self.assertRaisesRegex(DataContractError, "Duplicate JSON"):
            self.parse(raw=b'{"bars":[],"bars":[]}')

    def test_selected_missing_nonfinite_bounds_and_boolean_values_refused(self):
        for changes in ({"o": None}, {"o": True}, {"o": -1}, {"o": 111.}, {"c": 89.},
                        {"v": -1}, {"vw": 0}):
            with self.subTest(changes=changes):
                self.body["bars"] = [bar(**changes)]
                with self.assertRaises(DataContractError):
                    self.parse()

    def test_nonstandard_json_nan_refused_not_materialized(self):
        with self.assertRaisesRegex(DataContractError, "Nonstandard JSON"):
            self.parse(raw=b'{"symbol":"SYNTHETIC","bars":[{"t":"2024-08-02T04:00:00Z","o":NaN}]}')

    def test_midnight_eastern_date_and_dst_are_explicit(self):
        self.body["bars"] = [bar("2019-01-18", t="2019-01-18T05:00:00Z")]
        result = self.parse(keys=[("SYNTHETIC", "2019-01-18")])
        self.assertEqual(result.records[0]["trading_date"], "2019-01-18")
        self.body["bars"] = [bar("2019-01-18", t="2019-01-18T04:00:00Z")]
        with self.assertRaisesRegex(DataContractError, "New York midnight"):
            self.parse(keys=[("SYNTHETIC", "2019-01-18")])

    def test_timezone_missing_and_intraday_timestamp_refused(self):
        for timestamp in ("2018-08-02T00:00:00", "2018-08-02T13:30:00Z", "invalid"):
            with self.subTest(timestamp=timestamp):
                self.body["bars"] = [bar(t=timestamp)]
                with self.assertRaises(DataContractError):
                    self.parse()

    def test_missing_key_is_coverage_not_replacement_or_zero_bar(self):
        result = self.parse(keys=[("SYNTHETIC", "2018-08-03")])
        self.assertEqual(result.records, ())
        self.assertEqual(result.coverage[0]["status"], "NOT_IN_THIS_RAW_PAGE")

    def test_optional_fields_can_be_absent_but_ohlc_cannot(self):
        for field in ("v", "n", "vw"):
            del self.body["bars"][0][field]
        self.assertEqual(self.parse().records[0]["optional_values"], {})
        del self.body["bars"][0]["h"]
        with self.assertRaises(DataContractError):
            self.parse()

    def test_page_and_row_bounds(self):
        with self.assertRaisesRegex(DataContractError, "byte bound"):
            self.parse(maximum_page_bytes=1)
        self.body["bars"].append(bar("2018-08-03"))
        with self.assertRaisesRegex(DataContractError, "metadata row bound"):
            self.parse(maximum_page_bars=1)
        with self.assertRaisesRegex(DataContractError, "requested symbol/date"):
            self.parse(keys=[("SYNTHETIC", "2018-08-02"), ("SYNTHETIC", "2018-08-03")],
                       maximum_selected_records=1)

    def factory_inputs(self):
        selection = self.parse()
        prices = pd.DataFrame([{"intrinio_security_id": "synthetic_native", "trading_date": "2018-08-02",
            "open": 80., "high": 110., "low": 90., "close": 105., "adj_open": 64.,
            "adj_high": 88., "adj_low": 72., "adj_close": 84., "factor": 1.,
            "raw_ohlc_bounds_valid": False, "adjusted_ohlc_bounds_valid": False,
            "price_quality_status": "provider_ohlc_anomaly"}])
        row = {"security_id": "siblis_synthetic", "trading_date": "2018-08-02", "query_symbol": "SYNTHETIC",
            "adjustment": "raw", "source_task_id": "synthetic-task", "source_request_id": "synthetic-request",
            "source_page_hash": selection.provenance["source_page_sha256"].removeprefix("sha256:")}
        return prices, selection, row

    def factory(self, *, prices=None, records=None, rows=None, mapping=None, evidence=None):
        original, selection, row = self.factory_inputs()
        pin = {"path": "source.json", "size_bytes": 1,
               "sha256": selection.provenance["source_page_sha256"], "source_uri": "gs://synthetic-only/page.json"}
        return prepare_alpaca_repair_declarations(original if prices is None else prices,
            selection.records if records is None else records, source_identity_rows=[row] if rows is None else rows,
            native_mapping={"siblis_synthetic": "synthetic_native"} if mapping is None else mapping,
            evidence_by_page_hash={pin["sha256"]: pin} if evidence is None else evidence,
            identity_evidence=pin, review_notes="Explicit synthetic-only same-source instrument review")

    def test_factory_matches_original_source_id_and_keeps_provisional_ledger(self):
        result = self.factory()
        self.assertEqual(len(result.repairs), 1)
        self.assertEqual(result.repairs[0]["intrinio_security_id"], "synthetic_native")
        self.assertEqual(result.repairs[0]["raw_ohlc"]["open"], 100.)
        ledger = result.identity_and_control_ledger[0]
        self.assertEqual(ledger["source_security_id"], ledger["alpaca_security_id"])
        self.assertEqual(ledger["status"], "PREPARED_REJECTED_OHLC_BAR")
        self.assertEqual(ledger["original_ohlc"]["open"], 80.)
        self.assertFalse(ledger["source_admitted"])

    def test_factory_does_not_insert_missing_keys(self):
        prices, _, _ = self.factory_inputs()
        result = self.factory(prices=prices.iloc[:0])
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.identity_and_control_ledger[0]["status"], "NEW_KEY_SOURCE_ADMISSION_REQUIRED")

    def test_factory_accepted_original_is_control_not_repair(self):
        prices, _, _ = self.factory_inputs(); prices.loc[0, "price_quality_status"] = "accepted"
        result = self.factory(prices=prices)
        self.assertEqual(result.repairs, ())
        self.assertEqual(result.identity_and_control_ledger[0]["status"], "CONTROL_EXISTING_ACCEPTED_BAR")

    def test_factory_wrong_or_ambiguous_original_source_identity_refused(self):
        _, _, row = self.factory_inputs()
        for rows in ([row, row], [{**row, "security_id": "different_siblis_identity"}]):
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(DataContractError, "unique admitted"):
                    self.factory(rows=rows)

    def test_factory_task_request_or_page_reassociation_refused(self):
        _, _, row = self.factory_inputs()
        for field, value in (("source_task_id", "other-task"), ("source_request_id", "other-request"),
                             ("source_page_hash", "0" * 64), ("adjustment", "all")):
            with self.subTest(field=field):
                with self.assertRaisesRegex(DataContractError, "curated date/source provenance"):
                    self.factory(rows=[{**row, field: value}])

    def test_factory_missing_actual_page_pin_refused(self):
        with self.assertRaisesRegex(DataContractError, "evidence pin missing"):
            self.factory(evidence={})

    def test_factory_duplicate_actual_record_refused(self):
        _, selection, _ = self.factory_inputs()
        with self.assertRaisesRegex(DataContractError, "Duplicate actual"):
            self.factory(records=selection.records * 2)

    def test_factory_keeps_missing_original_prices_json_serializable(self):
        prices, _, _ = self.factory_inputs(); prices.loc[0, "open"] = float("nan")
        result = self.factory(prices=prices)
        self.assertIsNone(result.identity_and_control_ledger[0]["original_ohlc"]["open"])
        json.dumps(result.identity_and_control_ledger, allow_nan=False)

    def test_factory_end_to_end_requires_real_evidence_then_converts_units(self):
        prices, selection, row = self.factory_inputs()
        raw = json.dumps(self.body).encode()
        with tempfile.TemporaryDirectory() as name:
            root = Path(name); (root / "source.json").write_bytes(raw)
            pin = {"path": "source.json", "size_bytes": len(raw),
                "sha256": selection.provenance["source_page_sha256"], "source_uri": "gs://synthetic-only/page.json"}
            declarations = prepare_alpaca_repair_declarations(prices, selection.records,
                source_identity_rows=[row], native_mapping={"siblis_synthetic": "synthetic_native"},
                evidence_by_page_hash={pin["sha256"]: pin}, identity_evidence=pin,
                review_notes="Synthetic-only independently pinned same-source review")
            overlay = apply_reviewed_ohlc_repairs(prices, declarations.repairs, evidence_root=root,
                policy=OhlcRepairPolicy(0., 0., .0001, "synthetic-frozen"))
        self.assertEqual(overlay.prices.iloc[0].open, 100.)
        self.assertEqual(overlay.prices.iloc[0].adj_open, 80.)
        self.assertEqual(overlay.prices.iloc[0].original_open, 80.)


if __name__ == "__main__":
    unittest.main()
