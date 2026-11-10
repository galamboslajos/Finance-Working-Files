import sys
from pathlib import Path
import unittest
import tempfile
import json
from unittest import mock
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from rich_5min_features import calculate_grid, PATH_FEATURES, MODEL_FEATURES, security_grid
from joint_features import MODEL_FEATURES as JOINT_FEATURES
import train_xgb_ret_ensemble as engine


def fixture(days=24):
    n = np.full(days, 78)
    n[8] = 42
    shape = (days, 78)
    t = np.arange(78)[None, :]
    d = np.arange(days)[:, None]
    r = .0005*np.sin(t*.73+d*.17)+.00015
    start = 100+np.arange(days)*.1
    close = start[:, None]*np.exp(np.cumsum(r, axis=1))
    opening = np.concatenate([start[:, None], close[:, :-1]], axis=1)
    volume = np.broadcast_to(1000+30*(t-39)**2, shape).astype(float).copy()
    grid = dict(open=opening, close=close, high=np.maximum(opening, close)*1.001,
        low=np.minimum(opening, close)*.999, volume=volume, vwap=(opening+close)/2,
        source=np.ones(shape, int), count=np.ones(shape, int), time_valid=np.ones(shape, bool),
        minute_bars=np.full(shape, 5.0))
    for k, a in grid.items():
        mask = np.arange(78)[None, :] >= n[:, None]
        a[mask] = False if a.dtype == bool else 0 if k in ("source", "count") else np.nan
    return grid, n


class RichPaths(unittest.TestCase):
    def test_contract_units_and_domains(self):
        grid, n = fixture()
        a = calculate_grid(grid, n)
        self.assertEqual(MODEL_FEATURES[:30], JOINT_FEATURES)
        self.assertEqual(len(set(MODEL_FEATURES)), 42)
        self.assertTrue(a[PATH_FEATURES].iloc[10:].notna().all().all())
        for name in ("path_downside_variance_share", "path_vwap_above_fraction"):
            self.assertTrue(a[name].between(0, 1).all())
        for name in ("path_signed_efficiency", "path_return_autocorr_1", "path_extrema_time_balance", "path_signed_late_relative_activity", "path_return_volume_alignment"):
            self.assertTrue(a[name].dropna().between(-1, 1).all())
        scaled = {k:v.copy() for k,v in grid.items()}
        for k in ("open", "high", "low", "close", "vwap"):
            scaled[k] /= 4
        scaled["volume"] *= 4
        np.testing.assert_allclose(a[PATH_FEATURES], calculate_grid(scaled, n)[PATH_FEATURES], rtol=1e-9, atol=1e-10, equal_nan=True)
        scaled["volume"] *= 1000
        np.testing.assert_allclose(a[PATH_FEATURES], calculate_grid(scaled, n)[PATH_FEATURES], rtol=1e-9, atol=1e-10, equal_nan=True)

    def test_future_perturbation_and_prefix(self):
        grid, n = fixture()
        a = calculate_grid(grid, n)
        poisoned = {k:v.copy() for k,v in grid.items()}
        for name in ("open", "high", "low", "close", "vwap", "volume"):
            poisoned[name][18:] *= 50
        b = calculate_grid(poisoned, n)
        pd.testing.assert_frame_equal(a.iloc[:18], b.iloc[:18])
        pd.testing.assert_frame_equal(a.iloc[:18], calculate_grid({k:v[:18] for k,v in grid.items()}, n[:18]))

    def test_current_day_not_own_reference(self):
        grid, n = fixture()
        a = calculate_grid(grid, n)
        grid["volume"][15, 66:] *= 10
        b = calculate_grid(grid, n)
        self.assertGreater(b.path_late_relative_activity.iloc[15], a.path_late_relative_activity.iloc[15])
        np.testing.assert_allclose(a.iloc[:15][PATH_FEATURES], b.iloc[:15][PATH_FEATURES], equal_nan=True)

    def test_missing_slot_and_whole_calendar_day(self):
        grid, n = fixture()
        grid["count"][12, 30] = 0
        grid["count"][14] = 0
        a = calculate_grid(grid, n)
        self.assertTrue(a.loc[[12, 14], PATH_FEATURES].isna().all().all())
        self.assertTrue(a.path_rv_relative_10d.iloc[13:].isna().all())
        self.assertTrue(a.path_late_relative_activity.iloc[13:].isna().all())
        self.assertTrue(np.isfinite(a.path_signed_efficiency.iloc[13]))
        self.assertEqual(a.path_scheduled_missing_slots.iloc[14], 78)

    def test_early_close_reference_and_last_hour(self):
        grid, n = fixture()
        # Exactly constant per-slot dollar activity; early-close afternoons are
        # structurally absent, yet subsequent afternoon references remain valid.
        for day in range(len(n)):
            grid["volume"][day, :n[day]] = 100000/grid["vwap"][day, :n[day]]
        a = calculate_grid(grid, n)
        np.testing.assert_allclose(a.path_late_relative_activity.iloc[10:], 1, atol=1e-12)
        grid, n = fixture(); n[15] = 42
        a = calculate_grid(grid, n)
        self.assertTrue(np.isfinite(a.path_late_vs_early_return.iloc[15]))
        r = np.log(grid["close"][15, :42]/np.r_[grid["open"][15, 0], grid["close"][15, :41]])
        self.assertAlmostEqual(a.path_late_variance_concentration.iloc[15], (r[-12:]**2).sum()/(r*r).sum()*42/12)

    def test_provenance_availability_and_duplicate(self):
        for key, value in [("source", 2), ("time_valid", False), ("count", 2)]:
            grid, n = fixture(); grid[key][12, 20] = value
            a = calculate_grid(grid, n)
            self.assertTrue(a.loc[12, PATH_FEATURES].isna().all())
        grid, n = fixture(); grid["source"][12:] = 2
        a = calculate_grid(grid, n)
        self.assertTrue(a.path_rv_relative_10d.iloc[12:22].isna().all())
        self.assertTrue(np.isfinite(a.path_rv_relative_10d.iloc[22]))

    def test_zero_activity_and_flat_prices_are_undefined(self):
        grid, n = fixture()
        for key in ("open", "high", "low", "close", "vwap"):
            grid[key][:] = 100
        grid["volume"][:] = 0
        a = calculate_grid(grid, n)
        self.assertTrue(a[PATH_FEATURES].drop(columns="path_extrema_time_balance").isna().all().all())
        self.assertTrue(a.path_extrema_time_balance.eq(0).all())

    def test_grid_does_not_compress_missing_dates(self):
        calendar = pd.DataFrame(dict(session_num=[0, 1, 2]))
        frame = pd.DataFrame(dict(session_num=[0, 2], slot=[0, 0], open=[1., 2.], high=[1., 2.], low=[1., 2.], close=[1., 2.],
            volume=[1., 1.], vwap=[1., 2.], minute_bars=[5., 5.], source=[1, 1], time_valid=[True, True]))
        grid = security_grid(frame, calendar)
        self.assertTrue(np.isnan(grid["close"][1]).all())
        self.assertEqual(grid["close"][2, 0], 2)

    def test_native_missing_only_for_new_features_and_gap(self):
        matrix = np.ones((3, 42), np.float32)
        matrix[:, 29:] = np.nan
        engine.validate_matrix(matrix, MODEL_FEATURES)
        matrix[0, 0] = np.nan
        with self.assertRaises(ValueError):
            engine.validate_matrix(matrix, MODEL_FEATURES)
        matrix[0, 0] = 1; matrix[0, 40] = np.inf
        with self.assertRaises(ValueError):
            engine.validate_matrix(matrix, MODEL_FEATURES)

    def test_learner_and_uncapped_wfo_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            project = Path(folder); run = project/"data/models/path"
            release = project/"data/features/10_rich_5min_path_v1"
            release.mkdir(parents=True); (release/"MANIFEST.json").write_text("{}")
            with mock.patch.object(engine, "prepare_path_panel"), mock.patch.object(engine, "wait_for_start", side_effect=InterruptedError("before fitting")):
                with self.assertRaises(InterruptedError):
                    engine.full_run(project, run, experiment="path42")
            config = json.loads((run/"CONFIG.json").read_text())
            for key in ("max_rounds", "memory_limit_gib", "hard_stop_utc", "not_before_utc"):
                self.assertIsNone(config[key])
            self.assertEqual(config["minimum_available_memory_gib"], 0)
            self.assertEqual(config["parameters"], dict(engine.base.PARAMETERS, nthread=1))
            self.assertEqual((config["members_per_quarter"], config["total_quarters"], config["train_months"], config["test_months"], config["early_stopping_patience"]), (100, 40, 12, 3, 30))
            self.assertEqual(config["n_per_side"], [2, 3])
            self.assertFalse(list(run.rglob("*.ubj")))

    def test_timestamp_delivery_and_early_close_raw_scan(self):
        import duckdb
        from rich_5min_features import read_month
        opening = pd.Timestamp("2025-11-28T14:30:00Z")
        calendar = pd.DataFrame(dict(signal_date=[pd.Timestamp("2025-11-28")], session_num=[0],
            market_open_utc=[opening], market_close_utc=[opening+pd.Timedelta(minutes=210)]))
        rows = []
        for i in range(43):
            ts = opening+pd.Timedelta(minutes=5*i)
            rows.append(dict(security_id="A", trading_date="2025-11-28", bar_timestamp_utc=ts.isoformat(),
                available_at_utc=(ts+pd.Timedelta(minutes=6 if i==41 else 5)).isoformat(),
                open=100., high=101., low=99., close=100., volume=1000, vwap=100., minute_bars=5, source="test"))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"bars.parquet"; pd.DataFrame(rows).to_parquet(path)
            con = duckdb.connect(); con.register("sessions", calendar)
            frame = read_month(con, path); con.close()
        self.assertEqual(len(frame), 42)
        self.assertEqual(frame.slot.max(), 41)
        self.assertFalse(frame.time_valid.iloc[-1])


if __name__ == "__main__":
    unittest.main()
