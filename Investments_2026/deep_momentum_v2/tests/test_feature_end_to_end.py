"""Complete tiny-cohort integration with long synthetic calendar warmup.

These artificial quotes test timing and worker interfaces, never financial
research, strategy performance or historical data coverage.
"""
import json
import math
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa

from dmv2.features import build_feature_panel, predictor_frame, registered_feature_specs, training_slice
from dmv2.universe import DataContractError, EXCLUDED_SCHEMA, UNIVERSE_SCHEMA, UniverseSnapshot


SIGNALS = ("2000-01-31", "2000-02-29")
IDS = ("provider-a", "provider-b")


def member(day, source, provider, index):
    clocks = {"membership_effective_date": "1997-01-01",
              "membership_availability_date": "1997-01-01", "membership_end_date": None}
    return dict(source_security_id=source, intrinio_security_id=provider,
                as_of=day, signal_date=day, identity_status="resolved",
                indices=[index], index_id=index, identity_review_required=False,
                resolution_status="resolved", unified_security_id="u-" + source,
                membership_tickers=["STRUCTURAL_ONLY"],
                membership_records_json=json.dumps([clocks]), **clocks)


def complete_fixture():
    # This weekday grid is TEST_ONLY, not an exchange-calendar certification.
    sessions = pd.bdate_range("1997-01-01", "2000-04-03").strftime("%Y-%m-%d").tolist()
    calendar = pa.Table.from_pylist([dict(trading_date=day, calendar_name="TEST_ONLY",
                                        is_trading_session=True) for day in sessions])
    rows, historical = [], []
    for j, provider in enumerate(IDS):
        for i, day in enumerate(sessions):
            if day > max(SIGNALS):
                continue
            value = (70 + j * 30) * math.exp((.0006 + j * .0002) * i +
                                           (.008 + j * .002) * math.sin((i + j * 4) / (3 + j)))
            rows.append(dict(intrinio_security_id=provider, trading_date=day,
                frequency="daily", intraperiod=False, price_quality_status="accepted",
                adj_close=value, adj_open=value * .998, adj_high=value * 1.015,
                adj_low=value * .985, close=value, high=value * 1.015,
                low=value * .985, volume=1000. + j * 100 + i * (1 + j),
                factor=1., dividend=0., split_ratio=1.))
            historical.append(member(day, "source-" + "ab"[j], provider, ("sp500", "sp400")[j]))
    snapshots = []
    for day in SIGNALS:
        members = [member(day, "source-" + "ab"[j], provider, ("sp500", "sp400")[j])
                   for j, provider in enumerate(IDS)]
        snapshots.append(UniverseSnapshot(
            pa.Table.from_pylist(members, schema=UNIVERSE_SCHEMA),
            pa.Table.from_pylist([], schema=EXCLUDED_SCHEMA), day, "combined",
            {"mode": "historical_reconstruction", "knowledge_time_verified": False}, {}))
    return dict(daily_prices=pd.DataFrame(rows), calendar=calendar,
                signal_cohorts=snapshots, requested_security_ids=IDS,
                daily_cohorts=pd.DataFrame(historical))


class FeatureEndToEndTests(unittest.TestCase):
    def test_complete_pipeline_preserves_dictionary_and_paper_fifteen(self):
        args = complete_fixture()
        panel = build_feature_panel(**args)
        frame = panel.table.to_pandas()
        self.assertEqual(panel.table.num_rows, 4)
        self.assertEqual(panel.summary["registered_features"], 728)
        self.assertEqual(len(registered_feature_specs()), 728)
        paper = [f"{prefix}{h}" for prefix in ("zMOM", "meanMOM", "stdMOM") for h in (1, 3, 6, 9, 12)]
        self.assertTrue(np.isfinite(frame[paper].to_numpy()).all())
        self.assertFalse(any("target" in name or "outcome" in name or "status" in name for name in frame.columns))
        self.assertEqual(panel.coverage.num_rows, 4)
        self.assertFalse(panel.summary["model_ready"])
        self.assertFalse(panel.provenance["outcomes_consumed"])
        diagnostics = panel.diagnostics["daily"]
        finite_dates = diagnostics.latest_input_date.notna()
        self.assertTrue((diagnostics.loc[finite_dates, "latest_input_date"] <=
                         diagnostics.loc[finite_dates, "signal_date"]).all())
        for signal, window in panel.provenance["target_windows"].items():
            self.assertLess(signal, window["outcome_start"])
            self.assertLessEqual(window["outcome_start"], window["outcome_end"])
        # Exclude three structural constants and ten explicitly identical ranks.
        X = predictor_frame(panel)
        self.assertEqual(X.shape, (4, 715))
        constants = [s["name"] for s in panel.specs if s.get("structural_constant")]
        aliases = [s for s in panel.specs if s.get("alias_of")]
        self.assertEqual((len(constants), len(aliases)), (3, 10))
        self.assertFalse(set(constants) & set(X.columns))
        for spec in aliases:
            self.assertNotIn(spec["name"], X)
            # One member per sleeve gives undefined sleeve ranks; equality with
            # matching NaNs is still explicit. Universe ranks have two members.
            np.testing.assert_allclose(frame[spec["name"]], frame[spec["alias_of"]], equal_nan=True)

    def test_market_and_paper_values_agree_with_manual_past_formulas(self):
        args = complete_fixture()
        panel = build_feature_panel(**args)
        frame = panel.table.to_pandas().set_index(["source_security_id", "signal_date"])
        prices = args["daily_prices"].pivot(index="trading_date", columns="intrinio_security_id", values="adj_close")
        signal = SIGNALS[0]
        sessions = args["calendar"].column("trading_date").to_pylist()
        i = sessions.index(signal)
        past_returns = prices.loc[signal] / prices.loc[sessions[i - 20]] - 1
        self.assertAlmostEqual(frame.loc[("source-a", signal), "market_ew_return_20s"], past_returns.mean())
        self.assertAlmostEqual(frame.loc[("source-b", signal), "market_sleeve_spread_sp400_minus_sp500_20s"],
                               past_returns["provider-b"] - past_returns["provider-a"])
        grouped = {}
        for day in sessions: grouped[day[:7]] = day
        current = prices.loc[signal] / prices.loc[grouped["1999-12"]] - 1
        mean, sd = current.mean(), current.std(ddof=1)
        self.assertAlmostEqual(frame.loc[("source-a", signal), "meanMOM1"], mean)
        self.assertAlmostEqual(frame.loc[("source-a", signal), "stdMOM1"], sd)
        self.assertAlmostEqual(frame.loc[("source-a", signal), "zMOM1"], (current["provider-a"] - mean) / sd)
        # MOM3 uses November/December returns and excludes January.
        longer = prices.loc[grouped["1999-12"]] / prices.loc[grouped["1999-10"]] - 1
        self.assertAlmostEqual(frame.loc[("source-a", signal), "meanMOM3"], longer.mean())
        daily = prices.pct_change(fill_method=None)
        paired = daily.loc[sessions[i - 19]:signal]
        benchmark = paired.mean(axis=1)
        beta = paired["provider-a"].cov(benchmark) / benchmark.var(ddof=1)
        self.assertAlmostEqual(frame.loc[("source-a", signal), "relative_market_beta_20s"], beta)

    def test_future_suffix_inside_request_changes_only_later_signal(self):
        args = complete_fixture()
        original = build_feature_panel(**args).table.to_pandas()
        later = complete_fixture()
        mask = later["daily_prices"].trading_date > SIGNALS[0]
        for field in ("adj_close", "adj_open", "adj_high", "adj_low", "close", "high", "low", "volume"):
            later["daily_prices"].loc[mask, field] *= 19.
        changed = build_feature_panel(**later).table.to_pandas()
        pd.testing.assert_frame_equal(original[original.signal_date == SIGNALS[0]].reset_index(drop=True),
                                      changed[changed.signal_date == SIGNALS[0]].reset_index(drop=True))
        self.assertNotEqual(original[original.signal_date == SIGNALS[1]].px_simple_return_20s.iloc[0],
                            changed[changed.signal_date == SIGNALS[1]].px_simple_return_20s.iloc[0])

    def test_missing_daily_path_withholds_context_without_erasing_members(self):
        args = complete_fixture()
        days = args["calendar"].column("trading_date").to_pylist()
        hole = days[days.index(SIGNALS[0]) - 2]
        args["daily_prices"] = args["daily_prices"][~((args["daily_prices"].intrinio_security_id == IDS[0])
                                                                    & (args["daily_prices"].trading_date == hole))]
        panel = build_feature_panel(**args)
        frame = panel.table.to_pandas()
        self.assertEqual(len(frame), 4)
        signal_rows = frame[frame.signal_date == SIGNALS[0]].set_index("source_security_id")
        self.assertTrue(math.isnan(signal_rows.loc["source-a", "px_simple_return_20s"]))
        self.assertTrue(math.isfinite(signal_rows.loc["source-b", "px_simple_return_20s"]))
        self.assertTrue(signal_rows.market_ew_return_20s.isna().all())
        self.assertTrue(signal_rows.xs_universe_rank_pct__px_simple_return_20s.isna().all())
        self.assertTrue(signal_rows.relative_market_beta_20s.isna().all())
        # Daily gap is not an absent monthly endpoint; the paper explicitly uses
        # monthly returns, so its monthly arithmetic remains independently valid.
        self.assertTrue(np.isfinite(signal_rows.zMOM1).all())

    def test_future_target_suffix_cannot_be_feature_input(self):
        args = complete_fixture()
        for field in ("next_month_class", "outcome_end", "target_return", "provider_adjusted_return"):
            with self.subTest(field=field), self.assertRaisesRegex(DataContractError, "target/outcome"):
                build_feature_panel(**{**args, "daily_prices": args["daily_prices"].assign(**{field: np.nan})})
        row = args["daily_prices"].iloc[[0]].copy()
        row["trading_date"] = "2000-03-01"
        with self.assertRaisesRegex(DataContractError, "future or protected"):
            build_feature_panel(**{**args, "daily_prices": pd.concat([args["daily_prices"], row])})

    def test_future_removal_and_unrequested_entrant_do_not_select_past_cohort(self):
        args = complete_fixture()
        original = build_feature_panel(**args).table.to_pandas()
        second = args["signal_cohorts"][1]
        rows = second.table.to_pylist()
        rows = [row for row in rows if row["source_security_id"] != "source-a"]
        entrant = member(SIGNALS[1], "source-c", "provider-c", "sp500")
        entrant["membership_records_json"] = json.dumps([{
            "membership_effective_date": "2000-02-01",
            "membership_availability_date": "2000-02-01", "membership_end_date": None}])
        rows.append(entrant)
        args["signal_cohorts"][1] = UniverseSnapshot(
            pa.Table.from_pylist(rows, schema=UNIVERSE_SCHEMA), second.excluded,
            second.as_of, second.universe, second.provenance, second.summary)
        historical = args["daily_cohorts"]
        args["daily_cohorts"] = historical[~((historical.source_security_id == "source-a")
                                            & (historical.signal_date > SIGNALS[0]))].copy()
        new_rows = []
        for day in sorted(historical[historical.signal_date > SIGNALS[0]].signal_date.unique()):
            added = member(day, "source-c", "provider-c", "sp500")
            added["membership_effective_date"] = "2000-02-01"
            added["membership_availability_date"] = "2000-02-01"
            added["membership_records_json"] = entrant["membership_records_json"]
            new_rows.append(added)
        args["daily_cohorts"] = pd.concat([args["daily_cohorts"], pd.DataFrame(new_rows)], ignore_index=True)
        panel = build_feature_panel(**args)
        changed = panel.table.to_pandas()
        pd.testing.assert_frame_equal(original[original.signal_date == SIGNALS[0]].reset_index(drop=True),
                                      changed[changed.signal_date == SIGNALS[0]].reset_index(drop=True))
        self.assertEqual(set(changed[changed.signal_date == SIGNALS[0]].source_security_id), {"source-a", "source-b"})
        self.assertEqual(set(changed[changed.signal_date == SIGNALS[1]].source_security_id), {"source-b", "source-c"})
        coverage = panel.coverage.to_pandas().set_index(["source_security_id", "signal_date"])
        self.assertEqual(coverage.loc[("source-c", SIGNALS[1]), "feature_status"], "not_requested")

    def test_next_month_targets_join_separately_and_missing_y_preserves_row(self):
        panel = build_feature_panel(**complete_fixture())
        rows = []
        for signal in SIGNALS:
            window = panel.provenance["target_windows"][signal]
            for j, source in enumerate(("source-a", "source-b")):
                missing = j == 1
                rows.append(dict(source_security_id=source, signal_date=signal, **window,
                    return_status="missing_end" if missing else "ok",
                    provider_adjusted_return=np.nan if missing else .05))
        targets = pd.DataFrame(rows)
        earlier_X = predictor_frame(panel)
        result = training_slice(panel, targets, train_start="2000-02-01", train_end="2000-02-29",
                                label_observed_through="2000-02-29")
        self.assertEqual(len(result.X), 2)
        self.assertEqual(len(result.y), 2)
        self.assertEqual(result.y.notna().sum(), 1)
        self.assertEqual(set(result.X.index.get_level_values("signal_date")), {SIGNALS[0]})
        self.assertNotIn("provider_adjusted_return", result.X)
        altered = targets.copy()
        altered.loc[altered.signal_date == SIGNALS[1], "provider_adjusted_return"] = 1000.
        later_y = training_slice(panel, altered, train_start="2000-02-01", train_end="2000-02-29",
                                 label_observed_through="2000-02-29")
        pd.testing.assert_frame_equal(result.X, later_y.X)
        pd.testing.assert_series_equal(result.y, later_y.y)
        pd.testing.assert_frame_equal(earlier_X, predictor_frame(panel))


if __name__ == "__main__":
    unittest.main()
