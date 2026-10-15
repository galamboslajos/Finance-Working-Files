"""Structural fixtures exercise causal clocks, never a research outcome panel."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa

from dmv2.context_features import (
    _cohorts, balanced_cross_sectional_bases, compute_context_features, context_feature_specs,
)
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, STRICT_IDENTITY_POLICY
from dmv2.universe import DataContractError


class ContextFeaturesTests(unittest.TestCase):
    def identity_fixture(self, **kwargs):
        args = self.fixture(**kwargs)
        for name in ("signal_cohorts", "daily_cohorts"):
            frame = args[name]
            frame["unified_security_id"] = "u-" + frame["source_security_id"]
            frame["identity_review_required"] = False
            frame["resolution_status"] = "resolved"
            reviewed = frame.source_security_id == "S1"
            frame.loc[reviewed, "identity_status"] = "needs_review"
            frame.loc[reviewed, "identity_review_required"] = True
        return args

    def fixture(self, n=80, signal_index=59):
        sessions = pd.bdate_range("1999-01-04", periods=n).strftime("%Y-%m-%d").tolist()
        ids = ("P1", "P2", "P3", "P4")
        signals, historical, prices = [], [], []
        for j, provider in enumerate(ids):
            for i, day in enumerate(sessions):
                member = dict(source_security_id=f"S{j+1}", signal_date=day,
                    intrinio_security_id=provider, identity_status="resolved",
                    index_id="sp500" if j < 2 else "sp400",
                    membership_effective_date="1998-01-01", membership_availability_date="1998-01-01")
                historical.append(member)
                if i == signal_index:
                    signals.append(member.copy())
                price = (50+j*10)*np.exp(.001*(j+1)*i + .004*np.sin(i*(j+1)/5))
                prices.append(dict(intrinio_security_id=provider, trading_date=day,
                    adj_close=price, close=price, volume=1000+j*100+i,
                    frequency="daily", intraperiod=False, price_quality_status="accepted"))
        features = [dict(intrinio_security_id=provider, signal_date=sessions[signal_index],
                         **{base: float(j+1) for base in balanced_cross_sectional_bases()})
                    for j, provider in enumerate(ids)]
        return dict(daily_result=SimpleNamespace(table=pd.DataFrame(features)),
            daily_prices=pd.DataFrame(prices).query("trading_date <= @sessions[@signal_index]"), session_dates=sessions,
            signal_cohorts=pd.DataFrame(signals), requested_security_ids=ids,
            daily_cohorts=pd.DataFrame(historical).query("signal_date <= @sessions[@signal_index]"))

    def test_registry_balanced_count_and_explicit_constants_aliases(self):
        specs = context_feature_specs()
        self.assertEqual(len(specs), 415)
        self.assertEqual(len({r["name"] for r in specs}), 415)
        self.assertEqual(len(balanced_cross_sectional_bases()), 80)
        counts = {role: sum(r["role"] == role for r in specs) for role in
                  ("cross_sectional", "market_state", "relative_context")}
        self.assertEqual(counts, dict(cross_sectional=320, market_state=65, relative_context=30))
        constants = [r for r in specs if r.get("structural_constant")]
        aliases = [r for r in specs if r.get("alias_of")]
        self.assertEqual(len(constants), 3)
        self.assertEqual(len(aliases), 10)
        self.assertTrue(all(not r["model_eligible"] for r in constants+aliases))
        self.assertTrue(all("sp600" not in r["name"] for r in specs))
        self.assertEqual(next(r["lookback_sessions"] for r in specs if
            r["name"] == "xs_universe_rank_pct__liq_abnormal_volume_ratio_252s"), 503)

    def test_complete_tie_midranks_robust_z_and_both_sleeves_retained(self):
        args = self.fixture()
        args["daily_result"].table["px_simple_return_5s"] = [1., 2., 2., 4.]
        result = compute_context_features(**args)
        indexed = result.table.set_index("source_security_id")
        self.assertEqual(len(indexed), 4)
        np.testing.assert_allclose(indexed["xs_universe_rank_pct__px_simple_return_5s"], [0, .5, .5, 1])
        np.testing.assert_allclose(indexed["xs_index_sleeve_rank_pct__px_simple_return_5s"], [0, 1, 0, 1])
        # Median=2 and MAD=.5; all four original members determine normalization.
        np.testing.assert_allclose(indexed["xs_universe_robust_zscore__px_simple_return_5s"],
                                   np.array([-1, 0, 0, 2])/(1.4826*.5))

    def test_partial_requested_sample_withholds_universe_normalization(self):
        args = self.fixture()
        args["requested_security_ids"] = ("P1", "P2")
        args["daily_prices"] = args["daily_prices"][args["daily_prices"].intrinio_security_id.isin(("P1", "P2"))]
        args["daily_result"].table = args["daily_result"].table.iloc[:2].copy()
        result = compute_context_features(**args)
        self.assertEqual(len(result.table), 4)
        self.assertTrue(result.table["xs_universe_rank_pct__px_simple_return_5s"].isna().all())
        self.assertTrue(result.table["market_ew_return_5s"].isna().all())
        observed = result.coverage.query("feature_name=='xs_universe_rank_pct__px_simple_return_5s'").iloc[0]
        self.assertEqual((observed.cohort_size, observed.observed_members, observed.status), (4, 2, "incomplete_cohort"))

    def test_provisional_identity_is_explicit_and_applies_to_daily_and_signal_cohorts(self):
        args = self.identity_fixture()
        original_signal = args["signal_cohorts"].copy()
        original_daily = args["daily_cohorts"].copy()
        strict = compute_context_features(**args)
        self.assertTrue(strict.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())
        self.assertTrue(strict.table.market_ew_return_5s.isna().all())
        result = compute_context_features(**args, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertTrue(result.table.xs_universe_rank_pct__px_simple_return_5s.notna().all())
        self.assertTrue(result.table.market_ew_return_5s.notna().all())
        self.assertTrue(result.table.relative_market_beta_20s.notna().all())
        checked = _cohorts(args["signal_cohorts"], "signal_cohorts",
                           identity_policy=PROVISIONAL_IDENTITY_POLICY).set_index("source_security_id")
        self.assertTrue(checked.at["S1", "identity_eligible"])
        self.assertEqual(checked.at["S1", "identity_status"], "needs_review")
        self.assertEqual(checked.at["S1", "source_identity_status"], "needs_review")
        self.assertTrue(checked.at["S1", "identity_review_required"])
        self.assertEqual(result.provenance["signal_provisional_identity_eligible_rows"], 1)
        self.assertEqual(result.provenance["identity_eligibility_policy"], PROVISIONAL_IDENTITY_POLICY)
        self.assertFalse(result.provenance["identity_continuity_certified"])
        pd.testing.assert_frame_equal(args["signal_cohorts"], original_signal)
        pd.testing.assert_frame_equal(args["daily_cohorts"], original_daily)

    def test_provisional_collisions_index_ambiguity_and_unknown_metadata_stay_blocked(self):
        for defect in ("collision", "index_ambiguous", "unknown_flag", "pending", "missing_uid"):
            with self.subTest(defect=defect):
                args = self.identity_fixture()
                for name in ("signal_cohorts", "daily_cohorts"):
                    frame = args[name]
                    mask = frame.source_security_id == "S1"
                    # Stored eligibility is not an authority to admit the row.
                    frame["identity_eligible"] = True
                    if defect == "collision":
                        frame.loc[mask, "intrinio_security_id"] = "P2"
                    elif defect == "index_ambiguous":
                        frame["indices"] = frame.index_id.map(lambda x: [x])
                        for index in frame.index[mask]:
                            frame.at[index, "indices"] = ["sp500", "sp400"]
                    elif defect == "unknown_flag":
                        frame["identity_review_required"] = frame.identity_review_required.astype(object)
                        frame.loc[mask, "identity_review_required"] = None
                    elif defect == "pending":
                        frame.loc[mask, "resolution_status"] = "pending"
                    else:
                        frame.loc[mask, "unified_security_id"] = None
                result = compute_context_features(**args, identity_policy=PROVISIONAL_IDENTITY_POLICY)
                self.assertEqual(len(result.table), 4)
                self.assertTrue(result.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())
                self.assertTrue(result.table.market_ew_return_5s.isna().all())

    def test_provisional_partial_sample_keeps_complete_original_context_policy(self):
        args = self.identity_fixture()
        args["requested_security_ids"] = ("P1", "P2")
        args["daily_prices"] = args["daily_prices"][args["daily_prices"].intrinio_security_id.isin(("P1", "P2"))]
        args["daily_result"].table = args["daily_result"].table.iloc[:2].copy()
        result = compute_context_features(**args, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(len(result.table), 4)
        self.assertTrue(result.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())
        self.assertTrue(result.table.market_ew_return_5s.isna().all())
        row = result.coverage.query("feature_name=='xs_universe_rank_pct__px_simple_return_5s'").iloc[0]
        self.assertEqual((row.cohort_size, row.observed_members), (4, 2))
        self.assertEqual(result.provenance["context_policy"], "complete_cohort")

    def test_arrow_string_null_routes_and_unknown_metadata_are_retained_without_admission(self):
        for defect in ("native_null", "flag_null", "resolution_null"):
            with self.subTest(defect=defect):
                args = self.identity_fixture()
                for name in ("signal_cohorts", "daily_cohorts"):
                    rows = args[name].to_dict("records")
                    for row in rows:
                        if row["source_security_id"] != "S1":
                            continue
                        if defect == "native_null":
                            row.update(identity_status="unmapped_intrinio", intrinio_security_id=None,
                                       unified_security_id=None, identity_review_required=None,
                                       resolution_status=None)
                        elif defect == "flag_null":
                            row.update(identity_status="resolved", identity_review_required=None)
                        else:
                            row.update(identity_status="resolved", resolution_status=None)
                    args[name] = pa.Table.from_pylist(rows)
                for identity_policy in (STRICT_IDENTITY_POLICY, PROVISIONAL_IDENTITY_POLICY):
                    checked = _cohorts(args["signal_cohorts"], "signal_cohorts",
                                       identity_policy=identity_policy).set_index("source_security_id")
                    self.assertEqual(len(checked), 4)
                    self.assertFalse(checked.at["S1", "identity_eligible"])
                    if defect == "native_null":
                        # Arrow string nulls become scalar NaN under pandas 3;
                        # retain missing identity rather than manufacturing "nan".
                        self.assertTrue(pd.isna(args["signal_cohorts"].to_pandas().intrinio_security_id.iloc[0]))
                        self.assertTrue(pd.isna(checked.at["S1", "intrinio_security_id"]))
                        self.assertFalse(isinstance(checked.at["S1", "intrinio_security_id"], str))
                    result = compute_context_features(**args, identity_policy=identity_policy)
                    self.assertEqual(len(result.table), 4)
                    self.assertTrue(result.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())
                    self.assertTrue(result.table.market_ew_return_5s.isna().all())

    def test_pandas_nullable_unknown_metadata_and_ambiguous_arrow_sleeve_stay_blocked(self):
        args = self.identity_fixture()
        for name in ("signal_cohorts", "daily_cohorts"):
            frame = args[name]
            frame["resolution_status"] = frame.resolution_status.astype("string")
            frame["identity_review_required"] = frame.identity_review_required.astype("boolean")
            mask = frame.source_security_id == "S1"
            frame.loc[mask, "resolution_status"] = pd.NA
            frame.loc[mask, "identity_review_required"] = pd.NA
        checked = _cohorts(args["signal_cohorts"], "signal_cohorts",
                           identity_policy=PROVISIONAL_IDENTITY_POLICY).set_index("source_security_id")
        self.assertFalse(checked.at["S1", "identity_eligible"])
        result = compute_context_features(**args, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertTrue(result.table.market_ew_return_5s.isna().all())
        rows = self.identity_fixture()["signal_cohorts"].to_dict("records")
        for row in rows:
            row["indices"] = [row["index_id"]] if row["source_security_id"] != "S1" else ["sp500", "sp400"]
        checked = _cohorts(pa.Table.from_pylist(rows), "signal_cohorts",
                           identity_policy=PROVISIONAL_IDENTITY_POLICY).set_index("source_security_id")
        self.assertEqual(checked.at["S1", "identity_status"], "index_ambiguous")
        self.assertEqual(checked.at["S1", "source_identity_status"], "needs_review")
        self.assertFalse(checked.at["S1", "identity_eligible"])

    def test_unknown_identity_policy_fails_before_numerical_computation(self):
        with self.assertRaises(DataContractError):
            compute_context_features(**self.fixture(), identity_policy="waive_everything")

    def test_provisional_future_collisions_and_quote_changes_do_not_change_prior_signal(self):
        args = self.identity_fixture(signal_index=79)
        day = args["session_dates"][59]
        earlier = args["daily_cohorts"][args["daily_cohorts"].signal_date == day].copy()
        indicators = args["daily_result"].table.copy()
        indicators["signal_date"] = day
        args["signal_cohorts"] = pd.concat([earlier, args["signal_cohorts"]], ignore_index=True)
        args["daily_result"].table = pd.concat([indicators, args["daily_result"].table], ignore_index=True)
        original = compute_context_features(**args, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        changed = dict(args, daily_prices=args["daily_prices"].copy(),
                       daily_cohorts=args["daily_cohorts"].copy(), signal_cohorts=args["signal_cohorts"].copy())
        mask = changed["daily_prices"].trading_date > day
        changed["daily_prices"].loc[mask, ["adj_close", "close", "volume"]] *= 987
        for name in ("signal_cohorts", "daily_cohorts"):
            mask = (changed[name].signal_date > day) & (changed[name].source_security_id == "S2")
            changed[name].loc[mask, "intrinio_security_id"] = "P1"
            changed[name].loc[mask, "identity_review_required"] = True
        updated = compute_context_features(**changed, identity_policy=PROVISIONAL_IDENTITY_POLICY)
        pd.testing.assert_frame_equal(original.table[original.table.signal_date == day],
                                      updated.table[updated.table.signal_date == day])

    def test_no_daily_context_never_invents_membership_history(self):
        args = self.fixture()
        args["daily_cohorts"] = None
        result = compute_context_features(**args)
        context = [r["name"] for r in result.specs if r["role"] != "cross_sectional"]
        self.assertTrue(result.table[context].isna().all().all())
        self.assertEqual(set(result.coverage[result.coverage.feature_name.isin(context)].status), {"context_not_loaded"})

    def test_future_price_and_membership_suffix_does_not_change_signals(self):
        args = self.fixture(signal_index=79)
        day = args["session_dates"][59]
        earlier_cohort = args["daily_cohorts"][args["daily_cohorts"].signal_date == day].copy()
        earlier_features = args["daily_result"].table.copy()
        earlier_features["signal_date"] = day
        args["signal_cohorts"] = pd.concat([earlier_cohort, args["signal_cohorts"]], ignore_index=True)
        args["daily_result"].table = pd.concat([earlier_features, args["daily_result"].table], ignore_index=True)
        original = compute_context_features(**args)
        altered = dict(args, daily_prices=args["daily_prices"].copy(), daily_cohorts=args["daily_cohorts"].copy(),
                       signal_cohorts=args["signal_cohorts"].copy())
        mask = altered["daily_prices"].trading_date > day
        altered["daily_prices"].loc[mask, ["adj_close", "close", "volume"]] *= 987
        altered["daily_cohorts"].loc[altered["daily_cohorts"].signal_date > day, "identity_status"] = "needs_review"
        altered["signal_cohorts"].loc[altered["signal_cohorts"].signal_date > day, "identity_status"] = "needs_review"
        changed = compute_context_features(**altered)
        pd.testing.assert_frame_equal(original.table[original.table.signal_date == day],
                                      changed.table[changed.table.signal_date == day])

    def test_market_metrics_and_relative_pairs_are_daily_then_snapshot(self):
        args = self.fixture()
        result = compute_context_features(**args)
        day = args["signal_cohorts"].signal_date.iloc[0]
        sessions = args["session_dates"]
        grid = args["daily_prices"].pivot(index="trading_date", columns="intrinio_security_id", values="adj_close")
        i = sessions.index(day)
        returns = grid.loc[day]/grid.loc[sessions[i-5]]-1
        self.assertAlmostEqual(result.table.market_ew_return_5s.iloc[0], returns.mean())
        self.assertAlmostEqual(result.table.market_sleeve_spread_sp400_minus_sp500_5s.iloc[0],
                               returns[["P3", "P4"]].mean()-returns[["P1", "P2"]].mean())
        daily = grid.pct_change(fill_method=None)
        history = daily.loc[sessions[i-19]:day]
        benchmark = history.mean(axis=1)
        expected = history.P1.cov(benchmark)/benchmark.var(ddof=1)
        value = result.table.set_index("source_security_id").loc["S1", "relative_market_beta_20s"]
        self.assertAlmostEqual(value, expected)
        self.assertTrue(result.table.market_new_high_breadth_1s.eq(1).all())
        self.assertTrue(result.table.market_new_low_breadth_1s.eq(1).all())
        self.assertTrue(result.table.market_volume_up_breadth_1s.eq(0).all())

    def test_sleeve_switch_has_no_cross_index_relative_window(self):
        args = self.fixture()
        day = args["signal_cohorts"].signal_date.iloc[0]
        cutoff = args["session_dates"][50]
        switch = (args["daily_cohorts"].source_security_id == "S1") & (args["daily_cohorts"].signal_date >= cutoff)
        args["daily_cohorts"].loc[switch, "index_id"] = "sp400"
        args["signal_cohorts"].loc[args["signal_cohorts"].source_security_id == "S1", "index_id"] = "sp400"
        result = compute_context_features(**args).table.set_index("source_security_id")
        self.assertTrue(pd.isna(result.loc["S1", "relative_sleeve_beta_20s"]))
        self.assertTrue(np.isfinite(result.loc["S1", "relative_market_beta_20s"]))
        self.assertTrue(np.isfinite(result.loc["S3", "relative_sleeve_beta_20s"]))

    def test_missing_daily_cohort_does_not_forward_fill_benchmark(self):
        args = self.fixture()
        day = args["session_dates"][45]
        args["daily_cohorts"] = args["daily_cohorts"][args["daily_cohorts"].signal_date != day]
        result = compute_context_features(**args)
        self.assertTrue(result.table.relative_market_beta_20s.isna().all())
        self.assertTrue(result.table.relative_relative_return_market_20s.isna().all())

    def test_identity_quarantine_invalid_parent_and_zero_mad(self):
        for defect in ("identity", "parent", "flat"):
            with self.subTest(defect=defect):
                args = self.fixture()
                if defect == "identity":
                    args["signal_cohorts"].loc[0, "identity_status"] = "needs_review"
                    mask = (args["daily_cohorts"].source_security_id == "S1") & (args["daily_cohorts"].signal_date == args["session_dates"][59])
                    args["daily_cohorts"].loc[mask, "identity_status"] = "needs_review"
                elif defect == "parent":
                    args["daily_result"].table.loc[0, "px_simple_return_5s"] = np.inf
                else:
                    args["daily_result"].table["px_simple_return_5s"] = 1.
                result = compute_context_features(**args)
                self.assertTrue(result.table.xs_universe_robust_zscore__px_simple_return_5s.isna().all())
                if defect == "flat":
                    self.assertTrue(result.table.xs_universe_rank_pct__px_simple_return_5s.eq(.5).all())
                else:
                    self.assertTrue(result.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())

    def test_bad_quality_or_missing_internal_price_blocks_complete_context(self):
        for defect in ("missing", "quality", "intraperiod", "weekly"):
            with self.subTest(defect=defect):
                args = self.fixture()
                mask = (args["daily_prices"].intrinio_security_id == "P1") & (args["daily_prices"].trading_date == args["session_dates"][57])
                if defect == "missing":
                    args["daily_prices"] = args["daily_prices"][~mask]
                elif defect == "quality":
                    args["daily_prices"].loc[mask, "price_quality_status"] = "rejected"
                elif defect == "intraperiod":
                    args["daily_prices"].loc[mask, "intraperiod"] = True
                else:
                    args["daily_prices"].loc[mask, "frequency"] = "weekly"
                result = compute_context_features(**args)
                self.assertTrue(result.table.market_ew_return_5s.isna().all())
                self.assertTrue(result.table.relative_market_beta_20s.isna().all())

    def test_outcome_contamination_rejected_before_computation(self):
        for identity_policy in (STRICT_IDENTITY_POLICY, PROVISIONAL_IDENTITY_POLICY):
            for input_name, field in (("daily_prices", "target_return"), ("signal_cohorts", "outcome_end"),
                                     ("daily_cohorts", "future_price_available"), ("daily_result", "provider_adjusted_return")):
                with self.subTest(input=input_name, identity_policy=identity_policy):
                    args = self.identity_fixture()
                    frame = args[input_name].table if input_name == "daily_result" else args[input_name]
                    frame[field] = 123.
                    with self.assertRaises(DataContractError):
                        compute_context_features(**args, identity_policy=identity_policy)

    def test_protected_metadata_duplicate_calendar_and_future_membership_fail(self):
        for defect in ("protected", "calendar", "future_membership", "missing_clock", "duplicate_price", "overshoot", "row_clock"):
            with self.subTest(defect=defect):
                args = self.fixture()
                if defect == "protected":
                    args["daily_prices"].loc[0, "trading_date"] = "2024-01-02"
                elif defect == "calendar":
                    args["session_dates"].append(args["session_dates"][-1])
                elif defect == "future_membership":
                    args["daily_cohorts"].loc[0, "membership_availability_date"] = "2001-01-01"
                elif defect == "missing_clock":
                    args["daily_cohorts"] = args["daily_cohorts"].drop(columns="membership_availability_date")
                elif defect == "duplicate_price":
                    args["daily_prices"] = pd.concat([args["daily_prices"], args["daily_prices"].iloc[:1]])
                elif defect == "overshoot":
                    args["daily_prices"].loc[0, "trading_date"] = args["session_dates"][79]
                else:
                    args["daily_cohorts"]["as_of"] = args["daily_cohorts"].signal_date
                    args["daily_cohorts"].loc[0, "as_of"] = args["session_dates"][79]
                with self.assertRaises(DataContractError):
                    compute_context_features(**args)

    def test_partial_or_disagreeing_signal_daily_cohort_is_rejected(self):
        for defect in ("omitted_member", "wrong_provider", "wrong_sleeve"):
            with self.subTest(defect=defect):
                args = self.fixture()
                mask = (args["daily_cohorts"].source_security_id == "S1") & (args["daily_cohorts"].signal_date == args["session_dates"][59])
                if defect == "omitted_member":
                    args["daily_cohorts"] = args["daily_cohorts"][~mask]
                elif defect == "wrong_provider":
                    args["daily_cohorts"].loc[mask, "intrinio_security_id"] = "OTHER"
                else:
                    args["daily_cohorts"].loc[mask, "index_id"] = "sp400"
                with self.assertRaisesRegex(DataContractError, "original signal-cohort"):
                    compute_context_features(**args)

    def test_arrow_financial_projection_occurs_after_key_guard(self):
        args = self.fixture()
        prices = args["daily_prices"].copy()
        prices.loc[0, "trading_date"] = "2024-01-02"
        class MetadataOnly:
            column_names = list(prices.columns)
            def select(self, names):
                if set(names) & {"adj_close", "close", "volume"}:
                    raise AssertionError("Financial projection happened before metadata rejection")
                return SimpleNamespace(to_pandas=lambda: prices[names].copy())
        args["daily_prices"] = MetadataOnly()
        with self.assertRaisesRegex(DataContractError, "Protected"):
            compute_context_features(**args)

    def test_numeric_text_boolean_prices_and_numeric_intraperiod_fail_closed(self):
        for field, value in (("adj_close", "60.0"), ("adj_close", True), ("intraperiod", 0)):
            with self.subTest(field=field, value=value):
                args = self.fixture()
                args["daily_prices"][field] = args["daily_prices"][field].astype(object)
                mask = (args["daily_prices"].intrinio_security_id == "P1") & (args["daily_prices"].trading_date == args["session_dates"][59])
                args["daily_prices"].loc[mask, field] = value
                result = compute_context_features(**args)
                self.assertTrue(result.table.market_ew_return_5s.isna().all())


if __name__ == "__main__":
    unittest.main()
