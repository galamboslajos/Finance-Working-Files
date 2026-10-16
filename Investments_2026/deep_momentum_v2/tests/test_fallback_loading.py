"""Synthetic metadata/cache barriers for the explicitly incomplete repair."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dmv2.catalog import canonical_sha256
from dmv2.fallback_loading import (FallbackReadLimits, NUMERICAL_WORKERS, PRICE_COLUMNS,
    load_fallback_year, plan_fallback_year)
from dmv2.identity_amendment import DEFINITION, IdentityAmendment, SOURCE_ROUTES, WRONG_NATIVE
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY
from dmv2.monthly import build_month_windows
from dmv2.universe import DataContractError
import dmv2.fallback_loading as loading


def pin(path, relative=None):
    return {"path": path.name if relative is None else relative,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "size_bytes": path.stat().st_size}


def amendment():
    definition = deepcopy(DEFINITION);evidence = ({"synthetic": "structural-only"},)
    return IdentityAmendment(definition, evidence,
        canonical_sha256({"definition": definition, "evidence": list(evidence)}))


class FallbackCache:
    def __init__(self, root):
        self.root = root;self.directory = root / "outputs/reference";self.year = self.directory / "years/2010"
        self.year.mkdir(parents=True);(root / "configs").mkdir()
        dates = pd.bdate_range("2007-01-01", "2011-01-03").strftime("%Y-%m-%d").tolist()
        self.calendar = pa.Table.from_pylist([{"trading_date": d, "calendar_name": "SYNTHETIC",
            "is_trading_session": True} for d in dates])
        self.windows = build_month_windows(self.calendar, start_month="2010-01", end_month="2010-12")
        self.signals = [w.signal_date for w in self.windows]
        first = dates.index(self.signals[0]);self.raw_start = dates[first-502];self.feature_end = self.signals[-1]
        self.raw_end = self.windows[-1].outcome_end;self.context_start = dates[first-251]
        self.bbby = next(iter(SOURCE_ROUTES));self.members = [self.bbby, *[f"member-{i:02d}" for i in range(11)]]
        self.ids = [WRONG_NATIVE, *["p-" + sid for sid in self.members[1:]]]
        def member(sid, day, daily=False):
            row = {"source_security_id": sid, "signal_date": day, "indices": ["sp500" if sid == self.bbby else "sp400"],
                "unified_security_id": SOURCE_ROUTES[sid] if sid == self.bbby else "u-" + sid,
                "intrinio_security_id": WRONG_NATIVE if sid == self.bbby else "p-" + sid,
                "identity_status": "resolved", "identity_review_required": False, "resolution_status": "resolved"}
            if daily: row.update(membership_effective_date="2007-01-01", membership_availability_date="2007-01-01",
                membership_end_date=None, membership_record_count=1)
            else: row.update(as_of=day, membership_tickers=["SYNTHETIC"], membership_records_json="[]")
            return row
        self.original = pa.Table.from_pylist([member(sid, day) for day in self.signals for sid in sorted(self.members)])
        self.daily = pa.Table.from_pylist([member(sid, day, True)
            for day in dates if self.context_start <= day <= self.feature_end for sid in sorted(self.members)])
        prices = []
        for day_i, day in enumerate(dates):
            if not self.raw_start <= day <= self.raw_end: continue
            for i, native in enumerate(self.ids):
                value = 100.0 + day_i * (i+1) / 1000
                prices.append({"trading_date": day, "intrinio_security_id": native,
                    "frequency": "daily", "intraperiod": False,
                    "open": value, "high": value+1, "low": value-1, "close": value,
                    "adj_open": value, "adj_high": value+1, "adj_low": value-1, "adj_close": value,
                    "volume": 1000.0+i, "adj_volume": 1000.0+i, "factor": 1.0,
                    "split_ratio": 1.0, "dividend": 0.0, "raw_ohlc_bounds_valid": True,
                    "adjusted_ohlc_bounds_valid": True, "price_quality_status": "accepted"})
        self.prices = pa.Table.from_pylist(prices)
        self.feature = self.prices.filter(pc.less_equal(self.prices["trading_date"], self.feature_end))
        self.source_pins = {"sources": [{"source_key": "SYNTHETIC", "manifest_sha256": "sha256:"+"a"*64}]}
        archived = root / "outputs/archived/src/dmv2";archived.mkdir(parents=True)
        workers = {}
        for name in NUMERICAL_WORKERS:
            source = Path(loading.__file__).with_name(name);shutil.copy2(source, archived / name)
            workers[name] = hashlib.sha256(source.read_bytes()).hexdigest()
        self.identity = {"context_policy": "available_cohort_80pct", "source_pins": self.source_pins,
            "active_reference_contract": {"workers": workers, "source_pins": self.source_pins,
                "feature_dictionary_sha256": "sha256:"+"b"*64}}
        self.identity_hash = canonical_sha256(self.identity)
        self.audit = {"year": 2010, "build_identity_sha256": self.identity_hash,
            "feature_provenance": {"identity_policy": PROVISIONAL_IDENTITY_POLICY,
                "mode": "historical_reconstruction", "requested_ids": self.ids,
                "feature_dictionary_sha256": "sha256:"+"b"*64, "signal_dates": self.signals,
                "target_windows": {w.signal_date: {"outcome_month": w.outcome_month,
                    "outcome_start": w.outcome_start, "outcome_end": w.outcome_end} for w in self.windows}},
            "loading_provenance": {"reference_loading_provenance": {"source_pins": self.source_pins,
                "request": {"raw_start": self.raw_start, "feature_end": self.feature_end,
                    "raw_end": self.raw_end, "history_sessions": 503, "signal_dates": self.signals,
                    "requested_ids": self.ids, "daily_context_start": self.context_start},
                "stages": {"raw_daily_prices": {"request": {"start": self.raw_start,
                    "end": self.raw_end, "filters": {"intrinio_security_id": self.ids}}}}}}}
        self.write()

    def write(self):
        self.audit["artifacts"] = {}
        for name, table in (("original_cohort", self.original), ("daily_cohorts", self.daily),
                            ("exchange_calendar", self.calendar), ("feature_history_daily", self.feature),
                            ("daily_history_raw", self.prices)):
            path = self.year / (name + ".parquet");pq.write_table(table, path)
            self.audit["artifacts"][name] = pin(path)
        self.rebind()

    def rebind(self):
        audit_path = self.year / "audit.json";audit_path.write_text(json.dumps(self.audit))
        self.root_audit = {"status": "COMPLETE", "build_identity_sha256": self.identity_hash,
            "years": {"2010": pin(audit_path, "years/2010/audit.json")}}
        (self.directory / "audit.json").write_text(json.dumps(self.root_audit))
        (self.directory / "build_identity.json").write_text(json.dumps(self.identity))
        self.roles = {"primary_policy": "available_cohort_80pct", "archived_recipe_directory": "outputs/archived",
            "primary": {"context_policy": "available_cohort_80pct", "directory": "outputs/reference",
                "audit_sha256": pin(self.directory / "audit.json")["sha256"],
                "build_identity_sha256": self.identity_hash}}
        (self.root / "configs/context_panel_roles.json").write_text(json.dumps(self.roles))

    def plan(self, **kwargs):
        return plan_fallback_year(self.root, 2010, amendment=amendment(),
            approved_missing_route_fallback=True, **kwargs)


class FallbackLoadingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.cache = FallbackCache(Path(self.temp.name))

    def test_planning_opens_metadata_and_footers_never_quote_values(self):
        original = pq.read_table;calls = []
        def guard(path, *args, **kwargs):
            calls.append(Path(path).name)
            self.assertNotIn(Path(path).name, ("feature_history_daily.parquet", "daily_history_raw.parquet"))
            return original(path, *args, **kwargs)
        with patch("dmv2.fallback_loading.pq.read_table", side_effect=guard): plan = self.cache.plan()
        self.assertFalse(plan.read_plan["financial_values_read"])
        self.assertEqual(len(calls), 3)
        self.assertEqual(plan.original_cohort.num_rows, self.cache.original.num_rows)
        self.assertEqual(plan.daily_cohorts.num_rows, self.cache.daily.num_rows)
        self.assertNotIn(WRONG_NATIVE, plan.requested_ids)
        self.assertEqual(plan.read_plan["monthly_wrong_route_rows"], 12)
        self.assertEqual(plan.read_plan["daily_wrong_route_rows"], len(set(self.cache.daily["signal_date"].to_pylist())))

    def test_all_historical_membership_clocks_survive_quarantine(self):
        plan = self.cache.plan()
        for old, new in zip(self.cache.daily.to_pylist(), plan.daily_cohorts.to_pylist(), strict=True):
            self.assertEqual(old["source_security_id"], new["source_security_id"])
            self.assertEqual(old["membership_availability_date"], new["membership_availability_date"])
            if old["source_security_id"] == self.cache.bbby:
                self.assertIsNone(new["intrinio_security_id"])
                self.assertFalse(new["identity_eligible"])
                self.assertEqual(new["original_intrinio_security_id"], WRONG_NATIVE)
            else: self.assertEqual(old["intrinio_security_id"], new["intrinio_security_id"])

    def test_old_admission_flags_remain_audit_only_in_all_amended_cohorts(self):
        active = ("requested", "paper_prediction_eligible", "prediction_eligible")
        self.cache.original = pa.Table.from_pylist([{**r, **{n: True for n in active}}
            for r in self.cache.original.to_pylist()])
        self.cache.daily = pa.Table.from_pylist([{**r, **{n: True for n in active}}
            for r in self.cache.daily.to_pylist()])
        self.cache.write();plan = self.cache.plan()
        for table in (plan.original_cohort, plan.daily_cohorts):
            self.assertFalse(set(active) & set(table.column_names))
            for name in active:
                self.assertTrue(all(table["original_" + name].to_pylist()))

    def test_load_preserves_source_bytes_and_separates_future_outcomes(self):
        plan = self.cache.plan();before = {p.name: pin(p) for p in self.cache.year.glob("*.parquet")}
        loaded = load_fallback_year(self.cache.root, plan=plan)
        self.assertLessEqual(max(loaded.feature_prices["trading_date"].to_pylist()), plan.feature_end)
        self.assertEqual(max(loaded.outcome_prices["trading_date"].to_pylist()), plan.raw_end)
        self.assertNotIn(WRONG_NATIVE, loaded.feature_prices["intrinio_security_id"].to_pylist())
        self.assertNotIn(WRONG_NATIVE, loaded.outcome_prices["intrinio_security_id"].to_pylist())
        self.assertEqual(before, {p.name: pin(p) for p in self.cache.year.glob("*.parquet")})
        self.assertFalse(loaded.provenance["financial_2020_plus_read"])

    def test_cache2020_metadata_refused_before_financial_projection(self):
        rows = self.cache.prices.to_pylist();rows[0]["trading_date"] = "2020-01-02"
        self.cache.prices = pa.Table.from_pylist(rows);self.cache.write();plan = self.cache.plan()
        original = pq.read_table;projections = []
        def guard(path, *args, **kwargs):
            if Path(path).name == "daily_history_raw.parquet":
                projections.append(kwargs.get("columns"))
                self.assertEqual(kwargs.get("columns"), ["intrinio_security_id", "trading_date"])
            return original(path, *args, **kwargs)
        with patch("dmv2.fallback_loading.pq.read_table", side_effect=guard):
            with self.assertRaisesRegex(DataContractError, "pre-2020"): load_fallback_year(self.cache.root, plan=plan)
        self.assertEqual(len(projections), 1)

    def test_unrequested_cache_native_is_refused_before_values(self):
        rows = self.cache.feature.to_pylist();rows[0]["intrinio_security_id"] = "unplanned"
        self.cache.feature = pa.Table.from_pylist(rows);self.cache.write()
        with self.assertRaisesRegex(DataContractError, "native ID"): load_fallback_year(self.cache.root, plan=self.cache.plan())

    def test_request_is_not_inferred_from_observed_quotes(self):
        self.cache.audit["feature_provenance"]["requested_ids"] = self.cache.ids[:-1];self.cache.rebind()
        with self.assertRaisesRegex(DataContractError, "executed request"): self.cache.plan()

    def test_daily_warmup_only_member_is_requested_without_observed_prices(self):
        rows = self.cache.daily.to_pylist()
        extra = {**rows[0], "source_security_id": "warmup-only", "unified_security_id": "u-warmup",
                 "intrinio_security_id": "p-warmup-only"}
        rows.append(extra);rows.sort(key=lambda r: (r["signal_date"], r["source_security_id"]))
        self.cache.daily = pa.Table.from_pylist(rows)
        self.cache.ids.append("p-warmup-only");self.cache.write()
        plan = self.cache.plan()
        self.assertIn("p-warmup-only", plan.requested_ids)
        self.assertNotIn("p-warmup-only", self.cache.prices["intrinio_security_id"].to_pylist())
        self.assertEqual(plan.daily_cohorts.num_rows, self.cache.daily.num_rows)

    def test_original_target_windows_and_warmup_cannot_shift(self):
        self.cache.audit["feature_provenance"]["target_windows"][self.cache.signals[0]]["outcome_end"] = "2010-02-01"
        self.cache.rebind()
        with self.assertRaisesRegex(DataContractError, "independent calendar"): self.cache.plan()

    def test_explicit_authority_and_pre2020_year_bounds_required(self):
        for year in (2009, 2020, 2023):
            with self.assertRaisesRegex(DataContractError, "2010--2019"):
                plan_fallback_year(self.cache.root, year, amendment=amendment(), approved_missing_route_fallback=True)
        with self.assertRaisesRegex(DataContractError, "approval"):
            plan_fallback_year(self.cache.root, 2010, amendment=amendment())

    def test_missing_daily_sleeve_evidence_fails_closed(self):
        for value in (None, [], ["sp600"]):
            rows = self.cache.daily.to_pylist();rows[0]["indices"] = value
            self.cache.daily = pa.Table.from_pylist(rows);self.cache.write()
            with self.assertRaisesRegex(DataContractError, "sleeve"): self.cache.plan()

    def test_byte_pin_mutation_and_mutable_plan_rejected(self):
        plan = self.cache.plan();plan.provenance["financial_read_through"] = "2020-01-01"
        with self.assertRaisesRegex(DataContractError, "plan changed"): load_fallback_year(self.cache.root, plan=plan)
        plan = self.cache.plan();path = self.cache.year / "feature_history_daily.parquet"
        path.write_bytes(path.read_bytes()+b"mutation")
        with self.assertRaisesRegex(DataContractError, "input changed"): load_fallback_year(self.cache.root, plan=plan)

    def test_source_recipe_and_archived_worker_pins_are_required(self):
        self.cache.audit["loading_provenance"]["reference_loading_provenance"]["source_pins"] = {};self.cache.rebind()
        with self.assertRaisesRegex(DataContractError, "source/identity"): self.cache.plan()
        path = self.cache.root / "outputs/archived/src/dmv2/daily_features.py"
        path.write_text(path.read_text()+"\n# mutation\n")
        with self.assertRaisesRegex(DataContractError, "Archived numerical"): self.cache.plan()

    def test_frozen_projection_budget_stops_before_financial_read(self):
        with self.assertRaisesRegex(DataContractError, "budget"):
            self.cache.plan(limits=replace(FallbackReadLimits(), max_projected_compressed_bytes=1))

    def test_same_quote_amendment_applies_independently_to_both_full_views(self):
        plan = self.cache.plan()
        class Reviewed:
            fingerprint = "sha256:"+"c"*64
            calls = [];verifications = 0
            def verify(self, root): self.verifications += 1
            def apply(self, table, *, start, end):
                self.calls.append((table.num_rows, start, end, tuple(table.column_names)))
                self_outer.assertIn("volume", table.column_names)
                return table
        self_outer = self;reviewed = Reviewed()
        loaded = load_fallback_year(self.cache.root, plan=plan, quote_amendments=reviewed)
        self.assertEqual(reviewed.verifications, 2)
        self.assertEqual([c[2] for c in reviewed.calls], [plan.feature_end, plan.raw_end])
        self.assertEqual(reviewed.calls[1][0], self.cache.prices.num_rows)
        self.assertEqual(loaded.provenance["quote_amendment_sha256"], reviewed.fingerprint)


if __name__ == "__main__": unittest.main()
