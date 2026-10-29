"""Structural tests for membership timing, identity and daily-return boundaries."""
import unittest

import numpy as np
import pandas as pd

from prepare_sp500_daily import (
    attach_identity, identity_mapping, join_panel, prepare_prices, qualify_membership,
)


SESSIONS = pd.DatetimeIndex(["2019-12-30", "2019-12-31", "2020-01-02",
                            "2020-01-03", "2020-01-06"])


def prices(days, closes, native="n1"):
    return pd.DataFrame([dict(
        intrinio_security_id=native, trading_date=day, ticker="TEST",
        frequency="daily", intraperiod=False, open=value, high=value,
        low=value, close=value, adj_open=value, adj_high=value, adj_low=value,
        adj_close=value, volume=100., adj_volume=100., factor=1.,
        split_ratio=1., dividend=0., raw_ohlc_bounds_valid=True,
        adjusted_ohlc_bounds_valid=True, price_quality_status="accepted",
    ) for day, value in zip(days, closes)])


def mapping(ids=("s1",), natives=("n1",)):
    return pd.DataFrame(dict(source_security_id=ids, intrinio_security_id=natives,
                             identity_status="resolved", identity_eligible=True))


def members(days, ids=None):
    n = len(days)
    return pd.DataFrame(dict(
        security_id=ids or ["s1"] * n, trading_date=days, is_member=[True] * n,
        membership_effective_date=["2019-12-30"] * n,
        membership_end_date=["2020-01-06"] * n,
        membership_availability_date=days,
    ))


class PreparationTests(unittest.TestCase):
    def test_inclusive_end_effective_and_future_availability(self):
        m = members(["2019-12-30", "2019-12-31", "2020-01-02", "2020-01-03"])
        m["membership_effective_date"] = "2019-12-31"
        m["membership_end_date"] = "2020-01-02"
        result = qualify_membership(m, SESSIONS)
        self.assertEqual(result.membership_status.tolist(),
                         ["future_effective", "eligible", "eligible", "membership_ended"])
        m.loc[1, "membership_availability_date"] = "2020-01-02"
        self.assertEqual(qualify_membership(m, SESSIONS).iloc[1].membership_status,
                         "future_availability")

    def test_missing_availability_and_holidays_remain_auditable(self):
        m = members(["2020-01-01", "2020-01-02"])
        m.loc[1, "membership_availability_date"] = None
        self.assertEqual(qualify_membership(m, SESSIONS).membership_status.tolist(),
                         ["non_session", "availability_missing"])

    def test_malformed_dates_fail(self):
        m = members(["2020-1-02"])
        with self.assertRaises(ValueError):
            qualify_membership(m, SESSIONS)

    def test_missing_prices_and_delisted_members_not_dropped(self):
        m = qualify_membership(members(["2020-01-02", "2020-01-03"], ["s1", "delisted"]), SESSIONS)
        m = attach_identity(m, mapping(("s1", "delisted"), ("n1", "gone")))
        p = prepare_prices(prices(["2019-12-31", "2020-01-02"], [100., 101.]), SESSIONS)
        panel = join_panel(m, p)
        self.assertEqual(len(panel), 2)
        self.assertEqual(panel.iloc[1].quote_status, "missing_price")
        self.assertTrue(np.isnan(panel.iloc[1].provider_adjusted_return_1d))

    def test_same_day_native_collision_but_not_disjoint_aliases(self):
        m = qualify_membership(members(["2020-01-02", "2020-01-02"], ["s1", "s2"]), SESSIONS)
        routes = mapping(("s1", "s2"), ("n1", "n1"))
        out = attach_identity(m, routes)
        self.assertFalse(out.identity_eligible.any())
        self.assertEqual(len(out), 2)
        m.loc[1, "date"] = pd.Timestamp("2020-01-03")
        self.assertTrue(attach_identity(m, routes).identity_eligible.all())

    def test_previous_session_return_before_entry_and_across_year(self):
        p = prepare_prices(prices(["2019-12-31", "2020-01-02"], [100., 110.]), SESSIONS)
        m = attach_identity(qualify_membership(members(["2020-01-02"]), SESSIONS), mapping())
        self.assertAlmostEqual(join_panel(m, p).iloc[0].provider_adjusted_return_1d, .1)

    def test_reentry_does_not_bridge_gap(self):
        p = prepare_prices(prices(["2019-12-31", "2020-01-03"], [100., 110.]), SESSIONS)
        self.assertTrue(np.isnan(p.iloc[-1].provider_adjusted_return_1d))
        self.assertEqual(p.iloc[-1].return_status, "missing_previous_session")

    def test_non_session_quote_never_supplies_endpoint(self):
        p = prepare_prices(prices(["2019-12-31", "2020-01-01", "2020-01-02"],
                                  [100., 900., 110.]), SESSIONS)
        self.assertEqual(p.iloc[1].quote_status, "non_session")
        self.assertAlmostEqual(p.iloc[-1].provider_adjusted_return_1d, .1)

    def test_invalid_endpoint_blocks_two_returns_without_clipping(self):
        raw = prices(["2019-12-31", "2020-01-02", "2020-01-03", "2020-01-06"],
                     [100., 1., 2., 10.])
        raw.loc[1, "price_quality_status"] = "rejected"
        out = prepare_prices(raw, SESSIONS)
        self.assertTrue(out.iloc[1:3].provider_adjusted_return_1d.isna().all())
        self.assertAlmostEqual(out.iloc[3].provider_adjusted_return_1d, 4.)

    def test_duplicate_prices_fail_instead_of_deduplicating(self):
        with self.assertRaises(ValueError):
            prepare_prices(prices(["2020-01-02"] * 2, [100., 100.]), SESSIONS)

    def test_identity_product_constraint_review_and_known_wrong_issuer(self):
        old_bbby = "siblis_isin_c297a5f81a23c9f90503"
        b = pd.DataFrame([
            dict(source_product_id=product, source_security_id=sid,
                 unified_security_id=uid, bridge_method="isin", bridge_confidence="high")
            for product, sid, uid in [("sp_composite1500", "s1", "u1"),
                                      ("nasdaq100_sp100", "s1", "different"),
                                      ("sp_composite1500", old_bbby, "u2")]])
        links = pd.DataFrame([
            dict(unified_security_id=uid, intrinio_security_id=native,
                 identity_review_required=True, resolution_status="resolved",
                 query_identifier_type="isin")
            for uid, native in [("u1", "n1"), ("u2", "wrong")]])
        provisional = identity_mapping(b, links, "provisional").set_index("source_security_id")
        strict = identity_mapping(b, links, "strict").set_index("source_security_id")
        self.assertTrue(provisional.loc["s1", "identity_eligible"])
        self.assertFalse(strict.loc["s1", "identity_eligible"])
        self.assertFalse(provisional.loc[old_bbby, "identity_eligible"])
        self.assertEqual(provisional.loc[old_bbby, "identity_status"], "known_wrong_issuer")

    def test_wrong_issuer_price_never_attaches(self):
        m = attach_identity(qualify_membership(members(["2020-01-02"]), SESSIONS), mapping())
        m["identity_eligible"] = False
        p = prepare_prices(prices(["2020-01-02"], [900.]), SESSIONS)
        out = join_panel(m, p)
        self.assertTrue(out.adj_close.isna().all())
        self.assertEqual(out.iloc[0].quote_status, "identity_unresolved")

    def test_empty_price_source_retains_members(self):
        p = prepare_prices(prices(["2020-01-02"], [100.]).iloc[:0], SESSIONS)
        m = attach_identity(qualify_membership(members(["2020-01-02"]), SESSIONS), mapping())
        self.assertEqual(join_panel(m, p).iloc[0].quote_status, "missing_price")


if __name__ == "__main__":
    unittest.main()
