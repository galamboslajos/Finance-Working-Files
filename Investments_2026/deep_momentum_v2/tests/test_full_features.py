"""Structural equivalence/clock tests for the bounded annual feature path."""
import math
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

import test_daily_features as daily_fixture
import test_context_features as context_fixture
import test_features as assembly_fixture
from dmv2.daily_features import compute_daily_features
from dmv2.context_features import (
    AVAILABLE_CONTEXT_POLICY, SENSITIVITY_CONTEXT_POLICY,
    balanced_cross_sectional_bases, compute_context_features,
)
from dmv2.full_features import (
    DAILY_STATUS_LEGEND, compute_compact_daily_features,
    compute_compact_context_features, build_full_feature_panel,
)
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY
from dmv2.universe import DataContractError
from dmv2.features import build_feature_panel, predictor_frame
from dmv2.paper_features import COMMON_MOMENTUM_POLICY


class CompactDailyTests(unittest.TestCase):
    def assert_daily_equal(self, prices, dates, signals, requested=("security-a",)):
        expected = compute_daily_features(prices, dates, signal_dates=signals,
                                          requested_security_ids=requested)
        actual = compute_compact_daily_features(prices, dates, signal_dates=signals,
                                               requested_security_ids=requested)
        pd.testing.assert_frame_equal(actual.table, expected.table)
        long_codes = actual.coverage.melt(id_vars=["intrinio_security_id", "signal_date"],
            var_name="feature_name", value_name="code")
        long_codes["status"] = long_codes.code.map(DAILY_STATUS_LEGEND)
        keys = ["intrinio_security_id", "signal_date", "feature_name"]
        pd.testing.assert_frame_equal(long_codes[[*keys, "status"]].sort_values(keys).reset_index(drop=True),
            expected.coverage[[*keys, "status"]].sort_values(keys).reset_index(drop=True))
        return actual

    def test_all298_exact_values_and_statuses_multi_signal_and_missing_id(self):
        prices, dates = daily_fixture.fixture(750)
        self.assert_daily_equal(prices, dates, [dates[30], dates[250], dates[502], dates[-1]],
                                requested=("security-a", "missing-id"))

    def test_complete_dependency_spans_and_status_precedence(self):
        prices, dates = daily_fixture.fixture(750)
        prices = prices.drop(index=[10, 490, 732])
        prices.loc[15, "price_quality_status"] = "rejected"
        prices.loc[25, "volume"] = np.nan
        prices.loc[600, "intraperiod"] = True
        prices.loc[749, "adj_high"] = 1.
        self.assert_daily_equal(prices, dates, [dates[30], dates[250], dates[502], dates[-1]])

    def test_undefined_flat_zero_volume_and_missing_optional_inputs(self):
        prices, dates = daily_fixture.fixture(520)
        for field in ("adj_close", "adj_open", "adj_high", "adj_low", "close", "high", "low"):
            prices[field] = 100.
        prices["volume"] = 0.
        self.assert_daily_equal(prices, dates, [dates[250], dates[-1]])
        self.assert_daily_equal(prices.drop(columns=["adj_high", "adj_low", "adj_open", "volume"]),
                                dates, [dates[250], dates[-1]])

    def test_arrow_parity_future_guards_and_original_unchanged(self):
        prices, dates = daily_fixture.fixture(520)
        original = prices.copy(deep=True)
        self.assert_daily_equal(pa.Table.from_pandas(prices), dates, [dates[502], dates[-1]])
        pd.testing.assert_frame_equal(prices, original)
        trapped = daily_fixture.FinancialConversionTrap(pa.Table.from_pandas(prices))
        with self.assertRaisesRegex(DataContractError, "future or protected"):
            compute_compact_daily_features(trapped, dates, signal_dates=[dates[-2]],
                                           requested_security_ids=("security-a",))
        with self.assertRaisesRegex(DataContractError, "target/outcome"):
            compute_compact_daily_features(prices.assign(target=0.), dates, signal_dates=[dates[-1]],
                                           requested_security_ids=("security-a",))

    def test_future_perturbation_cannot_change_earlier_snapshot(self):
        prices, dates = daily_fixture.fixture(750)
        before = compute_compact_daily_features(prices, dates, signal_dates=[dates[502], dates[-1]],
                                                requested_security_ids=("security-a",))
        later = prices.copy()
        later.loc[later.trading_date > dates[502], "adj_close"] *= 4
        after = compute_compact_daily_features(later, dates, signal_dates=[dates[502], dates[-1]],
                                               requested_security_ids=("security-a",))
        pd.testing.assert_series_equal(before.table.iloc[0], after.table.iloc[0])
        pd.testing.assert_series_equal(before.coverage.iloc[0], after.coverage.iloc[0])


class CompactContextTests(unittest.TestCase):
    def assert_context_equal(self, args):
        expected = compute_context_features(**args)
        actual = compute_compact_context_features(**args)
        pd.testing.assert_frame_equal(actual.table, expected.table)
        pd.testing.assert_frame_equal(actual.coverage, expected.coverage)
        return actual

    def available_fixture(self, n=300, signal_index=280):
        """Twenty original members, ten per sleeve, nine observed per sleeve."""
        from types import SimpleNamespace
        original = context_fixture.ContextFeaturesTests().fixture(n=n, signal_index=signal_index)
        signal, historical, prices, indicators = [], [], [], []
        for j in range(20):
            template = "P" + str(j % 4 + 1)
            provider, source = "Q" + str(j + 1), "T" + str(j + 1)
            index = "sp500" if j < 10 else "sp400"
            for key, target in (("signal_cohorts", signal), ("daily_cohorts", historical)):
                member = original[key].loc[original[key].intrinio_security_id.eq(template)].copy()
                member["intrinio_security_id"], member["source_security_id"] = provider, source
                member["index_id"] = index
                target.append(member)
            quote = original["daily_prices"].loc[original["daily_prices"].intrinio_security_id.eq(template)].copy()
            quote["intrinio_security_id"] = provider
            if j not in (9, 19):
                prices.append(quote)
            indicators.append(dict(intrinio_security_id=provider,
                signal_date=original["signal_cohorts"].signal_date.iloc[0],
                **{base: np.nan if j in (9, 19) else float(j+1)
                   for base in balanced_cross_sectional_bases()}))
        return dict(daily_result=SimpleNamespace(table=pd.DataFrame(indicators)),
            daily_prices=pd.concat(prices, ignore_index=True),
            signal_cohorts=pd.concat(signal, ignore_index=True),
            daily_cohorts=pd.concat(historical, ignore_index=True),
            session_dates=original["session_dates"],
            requested_security_ids=tuple("Q"+str(j+1) for j in range(20)),
            context_policy=AVAILABLE_CONTEXT_POLICY)

    def test_available_context_exact90_threshold_and_every_daily_denominator(self):
        args = self.available_fixture()
        actual = self.assert_context_equal(args)
        column = "xs_universe_rank_pct__px_simple_return_5s"
        self.assertEqual(actual.table[column].notna().sum(), 18)
        self.assertTrue(actual.table.market_ew_return_20s.notna().all())
        self.assertEqual(actual.table.relative_market_beta_20s.notna().sum(), 18)
        coverage = actual.coverage.query("feature_name=='benchmark_daily_ew_return_1s'")
        self.assertEqual(len(coverage), 3 * 281)
        self.assertEqual(set(coverage.context_group), {"universe", "sp500", "sp400"})
        last = coverage[coverage.signal_date.eq(args["session_dates"][280])]
        self.assertEqual(set(last.coverage_fraction), {.9})
        self.assertTrue(last.request_exhaustive.all())
        self.assertEqual(set(last.status), {"ok"})

    def test_sensitivity_context_exact80_matches_reference_and_leaves90_gated(self):
        args = self.available_fixture()
        # The fixture already lacks Q10/Q20: remove one additional observed
        # member in each sleeve, keeping all twenty admitted IDs requested.
        indicators = args["daily_result"].table
        indicators.loc[indicators.intrinio_security_id.isin(("Q9", "Q19")),
                       list(balanced_cross_sectional_bases())] = np.nan
        prices = args["daily_prices"]
        args["daily_prices"] = prices.loc[~prices.intrinio_security_id.isin(("Q9", "Q19"))]
        args["context_policy"] = SENSITIVITY_CONTEXT_POLICY
        actual = self.assert_context_equal(args)
        self.assertEqual(actual.provenance["context_policy"], SENSITIVITY_CONTEXT_POLICY)
        self.assertEqual(actual.provenance["minimum_original_cohort_coverage"], .8)
        self.assertTrue(actual.provenance["complete_calendar_windows_required"])
        rank = "xs_universe_rank_pct__px_simple_return_5s"
        self.assertEqual(actual.table[rank].notna().sum(), 16)
        self.assertEqual(actual.table.relative_market_beta_252s.notna().sum(), 16)
        self.assertTrue(actual.table.market_ew_return_20s.notna().all())
        self.assertTrue(actual.table.market_sleeve_spread_sp400_minus_sp500_20s.notna().all())
        benchmark = actual.coverage.query("feature_name=='benchmark_daily_ew_return_1s'")
        last = benchmark.loc[benchmark.signal_date.eq(args["session_dates"][280])]
        self.assertEqual(set(last.coverage_fraction), {.8})
        self.assertEqual(set(last.minimum_coverage_fraction), {.8})
        self.assertEqual(set(last.normalization_policy), {SENSITIVITY_CONTEXT_POLICY})
        self.assertTrue(last.request_exhaustive.all())
        self.assertEqual(set(last.status), {"ok"})
        args["context_policy"] = AVAILABLE_CONTEXT_POLICY
        reference90 = compute_context_features(**args)
        self.assertTrue(reference90.table[rank].isna().all())
        self.assertTrue(reference90.table.market_ew_return_20s.isna().all())
        self.assertTrue(reference90.table.relative_market_beta_252s.isna().all())

    def test_available_context_below90_per_variable_and_per_sleeve(self):
        args = self.available_fixture()
        indicators = args["daily_result"].table
        indicators.loc[indicators.intrinio_security_id.eq("Q9"), "px_simple_return_5s"] = np.nan
        actual = self.assert_context_equal(args)
        self.assertTrue(actual.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())
        indexed = actual.table.set_index("source_security_id")
        self.assertTrue(indexed.loc[["T"+str(i) for i in range(1, 11)],
                                   "xs_index_sleeve_rank_pct__px_simple_return_5s"].isna().all())
        self.assertEqual(indexed.loc[["T"+str(i) for i in range(11, 21)],
                                    "xs_index_sleeve_rank_pct__px_simple_return_5s"].notna().sum(), 9)
        self.assertEqual(actual.table.xs_universe_rank_pct__px_simple_return_1s.notna().sum(), 18)

    def test_available_context_partial_request_cannot_qualify_as90pct(self):
        args = self.available_fixture()
        args["requested_security_ids"] = args["requested_security_ids"][:-1]
        indicators = args["daily_result"].table
        args["daily_result"].table = indicators[indicators.intrinio_security_id.ne("Q20")]
        actual = self.assert_context_equal(args)
        self.assertTrue(actual.table.xs_universe_rank_pct__px_simple_return_5s.isna().all())
        self.assertTrue(actual.table.market_ew_return_20s.isna().all())
        diagnostics = actual.coverage.query("feature_name=='benchmark_daily_ew_return_1s' and context_group=='universe'")
        self.assertEqual(set(diagnostics.status), {"incomplete_request"})
        self.assertFalse(diagnostics.request_exhaustive.any())

    def test_available_context_daily_holes_preserve_exact_relative_windows(self):
        args = self.available_fixture()
        prices = args["daily_prices"]
        missing = prices.intrinio_security_id.eq("Q9") & prices.trading_date.eq(args["session_dates"][250])
        args["daily_prices"] = prices.loc[~missing]
        actual = self.assert_context_equal(args)
        self.assertEqual(actual.table.relative_market_beta_20s.notna().sum(), 18)
        self.assertTrue(actual.table.relative_market_beta_42s.isna().all())
        daily = actual.coverage.query("feature_name=='benchmark_daily_ew_return_1s' and context_group=='universe'")
        hole = daily[daily.signal_date.eq(args["session_dates"][250])].iloc[0]
        self.assertEqual((hole.cohort_size, hole.observed_members, hole.status),
                         (20, 17, "below_coverage_threshold"))
        args["daily_cohorts"] = args["daily_cohorts"].loc[
            args["daily_cohorts"].signal_date.ne(args["session_dates"][270])]
        actual = self.assert_context_equal(args)
        self.assertTrue(actual.table.relative_market_beta_20s.isna().all())
        absent = actual.coverage.query("feature_name=='benchmark_daily_ew_return_1s'")
        absent = absent[absent.signal_date.eq(args["session_dates"][270])]
        self.assertEqual(len(absent), 3)
        self.assertEqual(set(absent.status), {"context_missing_date"})

    def test_complete_context_all415_and_relative_windows(self):
        args = context_fixture.ContextFeaturesTests().fixture(n=300, signal_index=280)
        self.assert_context_equal(args)

    def test_partial_and_missing_daily_identity_memberships_remain_missing(self):
        args = context_fixture.ContextFeaturesTests().fixture()
        args["daily_cohorts"].loc[0, "identity_status"] = "unlinked"
        self.assert_context_equal(args)
        args["daily_cohorts"] = args["daily_cohorts"].query("not(source_security_id=='S1' and signal_date==@args['session_dates'][20])")
        self.assert_context_equal(args)

    def test_provisional_flags_and_route_switches(self):
        args = context_fixture.ContextFeaturesTests().identity_fixture(n=300, signal_index=280)
        args["identity_policy"] = PROVISIONAL_IDENTITY_POLICY
        self.assert_context_equal(args)
        historical = args["daily_cohorts"]
        selected = (historical.source_security_id == "S1") & (historical.signal_date == args["session_dates"][270])
        historical.loc[selected, "intrinio_security_id"] = "absent-old-route"
        self.assert_context_equal(args)

    def test_sleeve_changes_and_price_gap(self):
        args = context_fixture.ContextFeaturesTests().fixture(n=300, signal_index=280)
        historical = args["daily_cohorts"]
        selected = (historical.source_security_id == "S1") & (historical.signal_date < args["session_dates"][260])
        historical.loc[selected, "index_id"] = "sp400"
        prices = args["daily_prices"]
        args["daily_prices"] = prices[~((prices.intrinio_security_id == "P2") & (prices.trading_date == args["session_dates"][250]))]
        self.assert_context_equal(args)

    def test_absent_daily_context_keeps_explicit_unavailability(self):
        args = context_fixture.ContextFeaturesTests().fixture()
        args["daily_cohorts"] = None
        actual = self.assert_context_equal(args)
        self.assertTrue(actual.table.market_ew_return_1s.isna().all())

    def test_missing_daily_date_and_ambiguous_original_sleeve_match_legacy(self):
        args = context_fixture.ContextFeaturesTests().fixture()
        historical = args["daily_cohorts"]
        args["daily_cohorts"] = historical[historical.signal_date != args["session_dates"][45]]
        self.assert_context_equal(args)
        args = context_fixture.ContextFeaturesTests().fixture()
        for label in ("signal_cohorts", "daily_cohorts"):
            frame = args[label]
            frame["indices"] = frame.index_id.map(lambda x: [x])
            frame.loc[frame.source_security_id == "S1", "indices"] = pd.Series(
                [["sp500", "sp400"] for _ in range((frame.source_security_id == "S1").sum())],
                index=frame.index[frame.source_security_id == "S1"])
        self.assert_context_equal(args)

    def test_context_future_membership_and_financial_clocks_fail_closed(self):
        args = context_fixture.ContextFeaturesTests().fixture()
        args["daily_cohorts"].loc[0, "membership_availability_date"] = "2000-01-01"
        with self.assertRaisesRegex(DataContractError, "future membership"):
            compute_compact_context_features(**args)
        args = context_fixture.ContextFeaturesTests().fixture()
        protected = args["daily_prices"].copy()
        protected.loc[protected.index[-1], "trading_date"] = "2024-01-02"
        args["daily_prices"] = protected
        with self.assertRaisesRegex(DataContractError, "Protected 2024"):
            compute_compact_context_features(**args)


class FullAssemblyTests(unittest.TestCase):
    def fixture(self):
        fixture = assembly_fixture.AssemblyIntegrationTests()
        fixture.setUp()
        return fixture

    def test_all728_values_original_denominators_and_active_common_population(self):
        fixture = self.fixture()
        snapshots = [fixture.cohort()]
        kwargs = dict(requested_security_ids=("id-a", "id-b"),
                      normalization_policy=COMMON_MOMENTUM_POLICY,
                      identity_policy=PROVISIONAL_IDENTITY_POLICY)
        baseline = build_feature_panel(fixture.prices, fixture.calendar, snapshots, **kwargs)
        result = build_full_feature_panel(fixture.prices, fixture.calendar, snapshots, **kwargs)
        pd.testing.assert_frame_equal(result.table.to_pandas(), baseline.table.to_pandas())
        pd.testing.assert_frame_equal(result.coverage.to_pandas(), baseline.coverage.to_pandas())
        pd.testing.assert_frame_equal(result.diagnostics["paper"], baseline.diagnostics["paper"])
        pd.testing.assert_frame_equal(result.diagnostics["context"], baseline.diagnostics["context"])
        self.assertEqual(result.summary, baseline.summary)
        self.assertEqual(len(predictor_frame(result).columns), 715)
        self.assertFalse(result.provenance["outcomes_consumed"])
        aggregate = result.diagnostics["daily"]
        self.assertTrue(aggregate.groupby(["signal_date", "feature_name"]).original_members.sum().eq(2).all())
        self.assertEqual(result.provenance["target_windows"], baseline.provenance["target_windows"])
        self.assertEqual(result.provenance["feature_dictionary_sha256"], baseline.provenance["feature_dictionary_sha256"])

    def test_context_policy_is_explicit_and_paper15_daily_inputs_unchanged(self):
        from dmv2.paper_features import PAPER_FEATURE_NAMES
        fixture = self.fixture()
        kwargs = dict(requested_security_ids=("id-a", "id-b"),
                      normalization_policy=COMMON_MOMENTUM_POLICY,
                      identity_policy=PROVISIONAL_IDENTITY_POLICY)
        legacy = build_full_feature_panel(fixture.prices, fixture.calendar, [fixture.cohort()], **kwargs)
        actual = build_full_feature_panel(fixture.prices, fixture.calendar, [fixture.cohort()],
            context_policy=AVAILABLE_CONTEXT_POLICY, **kwargs)
        baseline = build_feature_panel(fixture.prices, fixture.calendar, [fixture.cohort()],
            context_policy=AVAILABLE_CONTEXT_POLICY, **kwargs)
        pd.testing.assert_frame_equal(actual.table.to_pandas(), baseline.table.to_pandas())
        pd.testing.assert_frame_equal(actual.table.select([*PAPER_FEATURE_NAMES]).to_pandas(),
                                      legacy.table.select([*PAPER_FEATURE_NAMES]).to_pandas())
        for name in ("paper", "paper_raw_momentum", "daily"):
            pd.testing.assert_frame_equal(actual.diagnostics[name], legacy.diagnostics[name])
        self.assertEqual(actual.provenance["paper"], legacy.provenance["paper"])
        self.assertEqual(actual.provenance["context"]["context_policy"], AVAILABLE_CONTEXT_POLICY)
        self.assertIn("daily_benchmark", actual.diagnostics)
        self.assertFalse(actual.diagnostics["context"].feature_name.eq("benchmark_daily_ew_return_1s").any())

    def test_blocked_members_remain_in_original_status_denominators(self):
        from test_paper_features import member
        fixture = self.fixture()
        snapshots = [fixture.cohort(members=[member("a", "id-a"), member("b", "id-a")])]
        result = build_full_feature_panel(fixture.prices, fixture.calendar, snapshots,
            requested_security_ids=("id-a", "id-b"), identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(result.table.num_rows, 2)
        self.assertTrue(result.table.to_pandas().iloc[:, 2:].isna().all().all())
        status = result.diagnostics["daily"]
        self.assertEqual(set(status.status), {"identity_unresolved"})
        self.assertTrue(status.original_members.eq(2).all())

    def test_unchanged_daily_source_and_target_contamination_rejected(self):
        fixture = self.fixture()
        original = fixture.prices
        build_full_feature_panel(original, fixture.calendar, [fixture.cohort()],
            requested_security_ids=("id-a", "id-b"))
        self.assertIs(original, fixture.prices)
        self.assertTrue(original.equals(fixture.prices))
        contaminated = original.append_column("next_month_return", pa.array([0.] * original.num_rows))
        with self.assertRaisesRegex(DataContractError, "target/outcome"):
            build_full_feature_panel(contaminated, fixture.calendar, [fixture.cohort()],
                requested_security_ids=("id-a", "id-b"))


if __name__ == "__main__":
    unittest.main()
