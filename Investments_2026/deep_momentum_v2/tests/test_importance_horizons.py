import json
import unittest

from dmv2.importance_horizons import build_horizon_mapping


def spec(name, family, sessions=None, **extra):
    value = dict(name=name, family=family, role="invented", required_inputs=["fixture"], **extra)
    if sessions is not None:
        value["lookback_sessions"] = sessions
    return value


class HorizonMappingTests(unittest.TestCase):
    def test_declared_window_separate_from_raw_span(self):
        specs = [spec("risk_realized_vol_5s", "security_return_distribution", 6),
                 spec("risk_return_autocorr_1_5s", "security_return_distribution", 7)]
        rows = build_horizon_mapping(specs, [x["name"] for x in specs])
        self.assertEqual([x["group"] for x in rows], ["sessions_5", "sessions_5"])
        self.assertEqual([x["actual_required_history"]["span_sessions"] for x in rows], [6, 7])

    def test_multiscale_baseline_not_hidden_in_short_window(self):
        specs = [spec("liq_abnormal_volume_ratio_5s", "security_liquidity", 256),
                 spec("xs_universe_robust_zscore__liq_abnormal_volume_ratio_5s", "cross_sectional", 256)]
        rows = build_horizon_mapping(specs, [x["name"] for x in specs])
        self.assertEqual(len({x["group"] for x in rows}), 1)
        self.assertTrue(all(x["multiscale"] for x in rows))
        self.assertEqual(rows[0]["declared_horizon"], {"outer_average_sessions": 5, "volume_baseline_sessions": 252})
        self.assertNotEqual(rows[0]["group"], "sessions_5")

    def test_skip_is_single_separate_assignment(self):
        x = spec("px_skip_return_63_21s", "security_momentum", 64,
                 _spans=(("adj_close", 21, 63),))
        row, = build_horizon_mapping([x], [x["name"]])
        self.assertEqual(row["group"], "skip_sessions_63_21")
        self.assertEqual(row["declared_horizon"]["return_interval_sessions"], 42)
        self.assertEqual(row["actual_required_history"]["native_input_spans"][0]["required_observation_count"], 43)

    def test_paper_nominal_horizon_and_exact_endpoints(self):
        specs = [spec("zMOM1", "paper_momentum_state", lookback_months=1),
                 spec("meanMOM12", "paper_momentum_state", lookback_months=12)]
        one, twelve = build_horizon_mapping(specs, [x["name"] for x in specs])
        self.assertEqual(one["declared_horizon"]["included_month_return_count"], 1)
        self.assertEqual(twelve["declared_horizon"]["included_month_return_count"], 11)
        self.assertEqual(twelve["actual_required_history"]["required_month_end_lags"], list(range(12, 0, -1)))
        self.assertIsNone(twelve["actual_required_history"]["calendar_to_session_conversion"])

    def test_current_event_distinguishes_prior_session(self):
        specs = [spec("event_split_event", "security_event_and_level", 1),
                 spec("event_adjustment_factor_change", "security_event_and_level", 2)]
        rows = build_horizon_mapping(specs, [x["name"] for x in specs])
        self.assertEqual(len({x["group"] for x in rows}), 1)
        self.assertEqual([x["declared_horizon"]["previous_session_required"] for x in rows], [False, True])

    def test_json_order_unique_and_exact_name_preservation(self):
        specs = [spec("px_simple_return_5s", "security_return", 6),
                 spec("px_simple_return_1s", "security_return", 2)]
        names = [x["name"] for x in specs][::-1]
        rows = build_horizon_mapping(specs, names)
        self.assertEqual([x["name"] for x in json.loads(json.dumps(rows, allow_nan=False))], names)
        self.assertEqual([x["feature_ordinal"] for x in rows], [0, 1])

    def test_unknown_missing_duplicate_and_excluded_fail(self):
        good = spec("px_simple_return_5s", "security_return", 6)
        for names, specs in [(["unknown"], [good]),
            ([good["name"], good["name"]], [good]),
            ([good["name"]], [good, good]),
            ([good["name"]], [{**good, "model_eligible": False}]),
            ([good["name"]], [{**good, "family": "mystery"}])]:
            with self.subTest(names=names, specs=specs):
                with self.assertRaises(ValueError):
                    build_horizon_mapping(specs, names)

    def test_cross_section_span_disagreement_fails(self):
        base = spec("px_simple_return_5s", "security_return", 6)
        xs = spec("xs_universe_rank_pct__px_simple_return_5s", "cross_sectional", 5)
        with self.assertRaises(ValueError):
            build_horizon_mapping([base, xs], [xs["name"]])

    def test_inconsistent_composite_or_native_history_fails(self):
        for x in [spec("liq_abnormal_volume_ratio_5s", "security_liquidity", 5),
                  spec("px_skip_return_21_63s", "security_momentum", 22),
                  spec("px_simple_return_5s", "security_return", 6, _spans=(("adj_close", 0, 3),))]:
            with self.assertRaises(ValueError):
                build_horizon_mapping([x], [x["name"]])


if __name__ == "__main__":
    unittest.main()
