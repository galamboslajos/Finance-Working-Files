"""Synthetic contract checks only; no constructed financial data are research inputs."""
import copy
from dataclasses import replace
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dmv2.ohlc_repairs import (OhlcRepairPolicy, apply_reviewed_ohlc_repairs,
                               original_bar_sha256)
from dmv2.universe import DataContractError


class OhlcRepairTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.evidence = self.root / "actual-synthetic-source.json"
        self.evidence.write_bytes(b'{"synthetic_test_evidence_only":true}\n')
        self.pin = {"path": self.evidence.name, "size_bytes": self.evidence.stat().st_size,
                    "sha256": "sha256:" + hashlib.sha256(self.evidence.read_bytes()).hexdigest(),
                    "source_uri": "https://example.invalid/synthetic-test-only"}
        self.prices = pd.DataFrame([{"intrinio_security_id": "synthetic_A", "trading_date": "2018-08-02",
            "open": 80., "high": 110., "low": 95., "close": 100.,
            "adj_open": 64., "adj_high": 88., "adj_low": 76., "adj_close": 80., "factor": 1.,
            "raw_ohlc_bounds_valid": False, "adjusted_ohlc_bounds_valid": False,
            "price_quality_status": "provider_ohlc_anomaly"}])
        self.repair = {"repair_id": "synthetic-corrected-bar", "intrinio_security_id": "synthetic_A",
            "trading_date": "2018-08-02", "original_bar_sha256": original_bar_sha256(self.prices.iloc[0]),
            "raw_ohlc": {"open": 99., "high": 110., "low": 95., "close": 100.},
            "source_kind": "alternate_provider_raw_bar", "provider": "synthetic-only",
            "evidence": copy.deepcopy(self.pin), "identity_review": {
                "status": "REVIEWED_PROVISIONAL_SAME_SECURITY", "source_identifier": "SYNTHETIC-TICKER",
                "native_security_id": "synthetic_A", "valid_from": "2018-01-01", "valid_to": "2019-12-31",
                "currency": "USD", "price_basis": "unadjusted_USD_per_original_share",
                "review_notes": "Synthetic fixture tests explicit reviewed evidence declaration only.",
                "evidence": copy.deepcopy(self.pin)}}
        self.policy = OhlcRepairPolicy(0., 0., 0.0001, "synthetic-frozen-policy")

    def run_case(self, *, prices=None, repairs=None, policy=None):
        return apply_reviewed_ohlc_repairs(self.prices if prices is None else prices,
            [self.repair] if repairs is None else repairs, evidence_root=self.root,
            policy=self.policy if policy is None else policy)

    def test_overlay_keeps_original_bytes_flags_and_nonunit_adjustment_basis(self):
        before = self.prices.copy(deep=True)
        result = self.run_case()
        pd.testing.assert_frame_equal(self.prices, before)
        fixed = result.prices.iloc[0]
        self.assertEqual(fixed.open, 99.)
        self.assertEqual(fixed.adj_open, 79.2)
        self.assertEqual(fixed.factor, 1.)  # deliberately different from total multiplier .8
        self.assertEqual(fixed.original_open, 80.)
        self.assertEqual(fixed.original_price_quality_status, "provider_ohlc_anomaly")
        self.assertFalse(fixed.original_raw_ohlc_bounds_valid)
        self.assertEqual(fixed.price_quality_status, "accepted")
        self.assertEqual(fixed.ohlc_repair_source_kind, "alternate_provider_raw_bar")
        self.assertEqual(result.ledger.iloc[0].raw_to_existing_adjusted_multiplier, .8)
        self.assertFalse(result.ledger.iloc[0].knowledge_time_certified)

    def test_incomplete_original_ohlc_needs_actual_full_replacement(self):
        prices = self.prices.copy(deep=True)
        for field in ("open", "high", "low", "adj_open", "adj_high", "adj_low"):
            prices.loc[0, field] = float("nan")
        self.repair["original_bar_sha256"] = original_bar_sha256(prices.iloc[0])
        result = self.run_case(prices=prices)
        self.assertEqual(result.prices.iloc[0].adj_open, 79.2)
        self.assertTrue(pd.isna(result.prices.iloc[0].original_open))

    def test_close_conflicts_are_not_automatically_anchored(self):
        self.repair["raw_ohlc"]["close"] = 100.01
        with self.assertRaisesRegex(DataContractError, "close conflict"):
            self.run_case()

    def test_both_declared_close_tolerances_are_required(self):
        self.repair["raw_ohlc"]["close"] = 100.1
        with self.assertRaisesRegex(DataContractError, "close conflict"):
            self.run_case(policy=replace(self.policy, maximum_raw_close_absolute_difference=1.))
        with self.assertRaisesRegex(DataContractError, "close conflict"):
            self.run_case(policy=replace(self.policy, maximum_raw_close_relative_difference=.01))
        result = self.run_case(policy=replace(self.policy, maximum_raw_close_absolute_difference=1.,
            maximum_raw_close_relative_difference=.01))
        self.assertAlmostEqual(result.prices.iloc[0].adj_close, 80.08)
        self.assertAlmostEqual(result.ledger.iloc[0].raw_close_absolute_difference, .1)

    def test_source_pairs_must_reconcile_to_existing_close_basis(self):
        prices = self.prices.copy(deep=True)
        prices.loc[0, "adj_open"] = 65.
        self.repair["original_bar_sha256"] = original_bar_sha256(prices.iloc[0])
        with self.assertRaisesRegex(DataContractError, "pairs conflict"):
            self.run_case(prices=prices)

    def test_evidence_must_exist_and_match_hash_and_size(self):
        for field, value in (("sha256", "sha256:" + "0" * 64), ("size_bytes", 1), ("path", "absent.json")):
            with self.subTest(field=field):
                repair = copy.deepcopy(self.repair); repair["evidence"][field] = value
                with self.assertRaises(DataContractError):
                    self.run_case(repairs=[repair])

    def test_parent_absolute_and_symlink_paths_rejected(self):
        link = self.root / "linked.json"; link.symlink_to(self.evidence)
        for value in ("../actual-synthetic-source.json", str(self.evidence), link.name):
            with self.subTest(value=value):
                repair = copy.deepcopy(self.repair); repair["evidence"]["path"] = value
                with self.assertRaises(DataContractError):
                    self.run_case(repairs=[repair])

    def test_original_bar_changed_flags_or_prices_refused(self):
        for field, value in (("close", 101.), ("price_quality_status", "accepted"), ("factor", 2.)):
            with self.subTest(field=field):
                prices = self.prices.copy(deep=True); prices.loc[0, field] = value
                with self.assertRaisesRegex(DataContractError, "Original bar differs"):
                    self.run_case(prices=prices)

    def test_missing_or_duplicate_source_keys_not_created_or_deduplicated(self):
        with self.assertRaises(DataContractError):
            self.run_case(prices=self.prices.iloc[:0])
        with self.assertRaises(DataContractError):
            self.run_case(prices=pd.concat([self.prices, self.prices], ignore_index=True))
        with self.assertRaisesRegex(DataContractError, "Duplicate repair"):
            self.run_case(repairs=[self.repair, self.repair])

    def test_no_second_overlay(self):
        result = self.run_case()
        with self.assertRaisesRegex(DataContractError, "stack"):
            self.run_case(prices=result.prices)

    def test_accepted_original_cannot_be_replaced_under_an_ohlc_repair(self):
        prices = self.prices.copy(deep=True); prices.loc[0, "price_quality_status"] = "accepted"
        self.repair["original_bar_sha256"] = original_bar_sha256(prices.iloc[0])
        with self.assertRaisesRegex(DataContractError, "previously rejected"):
            self.run_case(prices=prices)

    def test_2020_financial_frame_refused_from_date_metadata(self):
        prices = self.prices.copy(deep=True); prices.loc[0, "trading_date"] = "2020-01-02"
        with self.assertRaisesRegex(DataContractError, "authorized"):
            self.run_case(prices=prices)

    def test_repair_date_outside2018_2019_refused(self):
        self.repair["trading_date"] = "2017-12-29"
        with self.assertRaisesRegex(DataContractError, "authorized"):
            self.run_case()

    def test_identity_raw_basis_currency_and_validity_fail_closed(self):
        for field, value in (("native_security_id", "synthetic_B"), ("currency", "EUR"),
                             ("price_basis", "dividend_adjusted"), ("valid_to", "2018-08-01"),
                             ("status", "CANDIDATE_UNREVIEWED")):
            with self.subTest(field=field):
                repair = copy.deepcopy(self.repair); repair["identity_review"][field] = value
                with self.assertRaises(DataContractError):
                    self.run_case(repairs=[repair])

    def test_incomplete_invalid_or_nonfinite_replacement_is_never_clamped(self):
        for raw in ({"open": 99., "high": 110., "low": 95.},
                    {"open": 94., "high": 110., "low": 95., "close": 100.},
                    {"open": 99., "high": float("inf"), "low": 95., "close": 100.},
                    {"open": True, "high": 110., "low": 95., "close": 100.}):
            with self.subTest(raw=raw):
                repair = copy.deepcopy(self.repair); repair["raw_ohlc"] = raw
                with self.assertRaises(DataContractError):
                    self.run_case(repairs=[repair])

    def test_empty_declaration_preserves_population_and_values(self):
        result = self.run_case(repairs=[])
        self.assertEqual(len(result.prices), len(self.prices))
        pd.testing.assert_frame_equal(result.prices[list(self.prices.columns)], self.prices)
        self.assertTrue(result.ledger.empty)

    def test_policy_has_no_implicit_tolerance_defaults(self):
        with self.assertRaises(TypeError):
            OhlcRepairPolicy()
        for value in (-1., float("nan"), True):
            with self.assertRaises(DataContractError):
                replace(self.policy, maximum_raw_close_absolute_difference=value)


if __name__ == "__main__":
    unittest.main()
