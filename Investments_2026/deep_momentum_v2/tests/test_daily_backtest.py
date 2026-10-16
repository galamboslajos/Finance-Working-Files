import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from dmv2.daily_backtest import configuration, summarize_daily, read_daily_backtest
from dmv2.universe import DataContractError

ROOT = Path(__file__).resolve().parents[1]


class DailyBacktestTests(unittest.TestCase):
    def result(self, complete=True):
        nav = [1., 1.05, 1.04, 1.08] if complete else [1., 1.05, np.nan, np.nan]
        return SimpleNamespace(status="complete_proxy" if complete else "failed_closed",
            daily=pd.DataFrame({"date": pd.to_datetime(["2018-01-02", "2018-01-02", "2018-01-03", "2018-01-03"]),
                "nav": nav, "turnover_one_way": [1., 0., 0., 1.]}),
            summary={"first_failure_date": None if complete else "2018-01-03", "stop_count": 0, "trade_count": 4},
            trades=pd.DataFrame({"cost": [0., 0., 0., 0.]}))

    def summary(self, result):
        return summarize_daily(result, arm="rf_paper15", basket="class_1_10", variant="unstopped",
                               cost_bps=0, expected_sessions=2)

    def test_closes_are_daily_observations_not_two_marks_per_day(self):
        row, monthly = self.summary(self.result())
        self.assertEqual(row["expected_sessions"], 2)
        self.assertAlmostEqual(row["total_return"], .08)
        daily = np.array([.05, 1.08 / 1.05 - 1])
        self.assertAlmostEqual(row["annualized_arithmetic_mean"], daily.mean() * 252)
        self.assertAlmostEqual(monthly.iloc[0]["return"], .08)

    def test_failed_path_never_reports_prefix_performance(self):
        row, monthly = self.summary(self.result(False))
        for field in ("total_return", "cagr", "annualized_arithmetic_mean", "sharpe_zero_cash_rate", "final_nav"):
            self.assertIsNone(row[field])
        self.assertEqual(row["valid_close_sessions"], 1)
        self.assertTrue(monthly["return"].isna().all())

    def test_frozen_configuration_rejects_unimplemented_semantic_changes(self):
        cfg = configuration(ROOT)
        self.assertEqual(cfg["selection"]["top_k_per_direction"], 10)
        for path, value in ((["accounting", "target_weights_basis"], "postcost"),
                            (["variants", "asset_stop_10pct", "trailing"], True),
                            (["selection", "top_k_per_direction"], 90), (["end"], "2020-12-31")):
            with tempfile.TemporaryDirectory() as directory:
                target = Path(directory) / "configs/daily_backtest.json"
                target.parent.mkdir()
                altered = json.loads(json.dumps(cfg))
                parent = altered
                for component in path[:-1]:
                    parent = parent[component]
                parent[path[-1]] = value
                target.write_text(json.dumps(altered))
                with self.assertRaisesRegex(DataContractError, "Frozen"):
                    configuration(directory)

    def test_missing_saved_report_does_not_plan_load_or_execute(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch("dmv2.daily_backtest.plan_daily_backtest") as plan, \
             patch("dmv2.daily_backtest.load_daily_backtest") as load, \
             patch("dmv2.daily_backtest.run_daily_accounting") as execute:
            with self.assertRaises(FileNotFoundError):
                read_daily_backtest(directory)
            plan.assert_not_called()
            load.assert_not_called()
            execute.assert_not_called()

    def cell(self):
        notebook = json.loads((ROOT / "notebooks/00_research_master.ipynb").read_text())
        return next("".join(c["source"]) for c in notebook["cells"]
                    if c["cell_type"] == "code" and "daily_backtest_run = read_daily_backtest(catalog)" in "".join(c["source"]))

    def test_notebook_failed_rerun_clears_all_saved_results(self):
        names = ("daily_backtest_run", "daily_backtest_comparison_df", "daily_backtest_selection_df",
                 "daily_backtest_ret_readiness_df", "daily_backtest_failures_df")
        state = {"catalog": Mock(), **{name: object() for name in names}}
        with patch("dmv2.daily_backtest.read_daily_backtest", side_effect=ValueError("stale")):
            with self.assertRaisesRegex(ValueError, "stale"):
                exec(self.cell(), state)
        self.assertTrue(all(state[name] is None for name in names))

    def test_notebook_missing_run_never_starts_backtest_or_fit(self):
        state = {"catalog": Mock()}
        with patch("dmv2.daily_backtest.read_daily_backtest", side_effect=FileNotFoundError), \
             patch("dmv2.daily_backtest.run_daily_backtest") as run, \
             patch("dmv2.daily_backtest.load_daily_backtest") as load:
            exec(self.cell(), state)
        run.assert_not_called()
        load.assert_not_called()
        self.assertIsNone(state["daily_backtest_run"])


if __name__ == "__main__":
    unittest.main()
