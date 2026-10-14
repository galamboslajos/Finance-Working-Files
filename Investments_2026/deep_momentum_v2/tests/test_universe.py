"""Adversarial contracts for historical membership and left-preserving quotes.

These tiny, fabricated metadata and quote rows are test fixtures only. They are
not market data, a research panel, or a simulated financial experiment. The
tests exercise rejection/quarantine/coverage behavior without any cloud reads.
"""
from __future__ import annotations

from datetime import date
import math
import unittest

import pyarrow as pa

from dmv2.universe import (
    DataContractError,
    KnowledgeTimeUnavailable,
    attach_quotes,
    build_snapshot,
)


AS_OF = "2020-01-31"
DAY = date(2020, 1, 31)

MEMBERSHIP_SCHEMA = pa.schema([
    ("index_id", pa.string()),
    ("trading_date", pa.string()),
    ("security_id", pa.string()),
    ("is_member", pa.bool_()),
    ("membership_effective_date", pa.string()),
    ("membership_end_date", pa.string()),
    ("membership_availability_date", pa.string()),
    ("pit_policy", pa.string()),
    ("identity_confidence", pa.string()),
    ("membership_ticker", pa.string()),
])
BRIDGE_SCHEMA = pa.schema([
    ("source_product_id", pa.string()),
    ("source_security_id", pa.string()),
    ("unified_security_id", pa.string()),
    ("bridge_method", pa.string()),
    ("bridge_confidence", pa.string()),
])
LINK_SCHEMA = pa.schema([
    ("unified_security_id", pa.string()),
    ("intrinio_security_id", pa.string()),
    ("identity_review_required", pa.bool_()),
    ("resolution_status", pa.string()),
])
PRICE_SCHEMA = pa.schema([
    ("intrinio_security_id", pa.string()),
    ("trading_date", pa.string()),
    ("close", pa.float64()),
    ("volume", pa.float64()),
    ("adj_close", pa.float64()),
    ("adj_volume", pa.float64()),
])


def native_dates(row):
    return {key: value.isoformat() if isinstance(value, date) else value
            for key, value in row.items()}


def member(source_id="S1", index="sp500", **changes):
    row = {
        "index_id": index,
        "trading_date": DAY,
        "security_id": source_id,
        "is_member": True,
        "membership_effective_date": date(2019, 1, 2),
        "membership_end_date": None,
        "membership_availability_date": date(2019, 1, 1),
        "pit_policy": "test_effective_time",
        "identity_confidence": "high",
        "membership_ticker": "TEST",
    }
    row.update(changes)
    return native_dates(row)


def bridge(source_id="S1", unified_id="U1", **changes):
    row = {
        "source_product_id": "sp_composite1500",
        "source_security_id": source_id,
        "unified_security_id": unified_id,
        "bridge_method": "test_stable_id",
        "bridge_confidence": "high",
    }
    row.update(changes)
    return row


def link(unified_id="U1", intrinio_id="I1", **changes):
    row = {
        "unified_security_id": unified_id,
        "intrinio_security_id": intrinio_id,
        "identity_review_required": False,
        "resolution_status": "resolved",
    }
    row.update(changes)
    return row


def quote(intrinio_id="I1", **changes):
    row = {
        "intrinio_security_id": intrinio_id,
        "trading_date": DAY,
        "close": 10.0,
        "volume": 20.0,
        "adj_close": 10.0,
        "adj_volume": 20.0,
    }
    row.update(changes)
    return native_dates(row)


def snapshot(members=None, bridges=None, links=None, **changes):
    options = {
        "as_of": AS_OF,
        "universe": "sp500",
        "mode": "historical_reconstruction",
    }
    options.update(changes)
    return build_snapshot(
        pa.Table.from_pylist([member()] if members is None else members,
                            schema=MEMBERSHIP_SCHEMA),
        pa.Table.from_pylist([bridge()] if bridges is None else bridges,
                            schema=BRIDGE_SCHEMA),
        pa.Table.from_pylist([link()] if links is None else links,
                            schema=LINK_SCHEMA),
        **options,
    )


def keyed_rows(table):
    return {row["source_security_id"]: row for row in table.to_pylist()}


class HistoricalUniverseTests(unittest.TestCase):
    def assert_rejected_or_quarantined(self, bridge_table, link_table):
        try:
            value = build_snapshot(
                pa.Table.from_pylist([member()], schema=MEMBERSHIP_SCHEMA),
                bridge_table, link_table, as_of=AS_OF, universe="sp500",
                mode="historical_reconstruction")
        except DataContractError:
            return
        self.assertEqual(value.table.num_rows, 1)
        self.assertNotEqual(keyed_rows(value.table)["S1"]["identity_status"], "resolved")
        attached = attach_quotes(value, pa.Table.from_pylist([quote()],
                                                            schema=PRICE_SCHEMA))
        self.assertEqual(keyed_rows(attached.table)["S1"]["quote_status"],
                         "identity_unresolved")

    def test_resolves_stable_ids_and_preserves_snapshot_contract(self):
        value = snapshot()
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"S1"})
        self.assertEqual(rows["S1"]["indices"], ["sp500"])
        self.assertEqual(rows["S1"]["unified_security_id"], "U1")
        self.assertEqual(rows["S1"]["intrinio_security_id"], "I1")
        self.assertEqual(rows["S1"]["identity_status"], "resolved")
        self.assertEqual(value.as_of, AS_OF)
        self.assertEqual(value.universe, "sp500")
        self.assertIsInstance(value.provenance, dict)
        self.assertIsInstance(value.summary, dict)

    def test_end_date_is_inclusive_and_null_end_is_open(self):
        value = snapshot(members=[
            member("ENDS_TODAY", membership_end_date=DAY),
            member("ENDED", membership_end_date=date(2020, 1, 30)),
            member("OPEN", membership_end_date=None),
        ], bridges=[], links=[])
        self.assertEqual(set(keyed_rows(value.table)), {"ENDS_TODAY", "OPEN"})

    def test_effective_and_available_dates_equal_to_asof_are_eligible(self):
        value = snapshot(members=[member(membership_effective_date=DAY,
                                          membership_availability_date=DAY)])
        self.assertEqual(set(keyed_rows(value.table)), {"S1"})

    def test_future_effective_membership_never_enters_snapshot(self):
        value = snapshot(members=[member(), member(
            "FUTURE", membership_effective_date=date(2020, 2, 3))],
            bridges=[bridge(), bridge("FUTURE", "UF")],
            links=[link(), link("UF", "IF")])
        self.assertEqual(set(keyed_rows(value.table)), {"S1"})

    def test_future_recorded_availability_is_excluded_even_in_reconstruction(self):
        value = snapshot(members=[member(), member(
            "ANNOUNCED_LATER", membership_availability_date=date(2020, 2, 3))],
            bridges=[bridge(), bridge("ANNOUNCED_LATER", "UF")],
            links=[link(), link("UF", "IF")])
        self.assertEqual(set(keyed_rows(value.table)), {"S1"})

    def test_missing_required_dates_have_visible_exclusions_and_denominators(self):
        value = snapshot(members=[
            member(),
            member("NO_EFFECTIVE", membership_effective_date=None),
            member("NO_AVAILABILITY", membership_availability_date=None),
        ], bridges=[], links=[])
        self.assertEqual(set(keyed_rows(value.table)), {"S1"})
        excluded = value.excluded.to_pylist()
        self.assertEqual({row["source_security_id"] for row in excluded},
                         {"NO_EFFECTIVE", "NO_AVAILABILITY"})
        self.assertTrue(all(row["reason"] for row in excluded))
        self.assertEqual(value.summary["source_rows_at_date"], 3)
        self.assertEqual(value.summary["excluded_membership_rows"], 2)
        self.assertEqual(value.summary["constituents"], 1)

    def test_end_before_effective_date_is_rejected_not_treated_as_expired(self):
        with self.assertRaises(DataContractError):
            snapshot(members=[member(membership_effective_date=DAY,
                                     membership_end_date=date(2020, 1, 30))])

    def test_duplicate_dated_membership_key_is_rejected(self):
        with self.assertRaises(DataContractError):
            snapshot(members=[member(), member(membership_ticker="OTHER")])

    def test_false_membership_other_index_and_other_dates_are_excluded(self):
        value = snapshot(members=[
            member(),
            member("FALSE", is_member=False),
            member("MID", index="sp400"),
            member("SMALL", index="sp600"),
            member("OLD", trading_date=date(2020, 1, 30)),
            member("LATER", trading_date=date(2020, 2, 3)),
        ], bridges=[], links=[])
        self.assertEqual(set(keyed_rows(value.table)), {"S1"})

    def test_sp400_selection_uses_historical_index_not_latest_master(self):
        value = snapshot(members=[member(), member("MID", index="sp400")],
                         bridges=[], links=[], universe="sp400")
        self.assertEqual(set(keyed_rows(value.table)), {"MID"})
        self.assertEqual(keyed_rows(value.table)["MID"]["indices"], ["sp400"])

    def test_combined_deduplicates_only_same_source_identity(self):
        value = snapshot(members=[
            member("OVERLAP", "sp500", membership_ticker="SHARED"),
            member("OVERLAP", "sp400", membership_ticker="SHARED"),
            member("DISTINCT", "sp400", membership_ticker="SHARED"),
            member("SMALL", "sp600"),
        ], bridges=[], links=[], universe="combined")
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"OVERLAP", "DISTINCT"})
        self.assertEqual(set(rows["OVERLAP"]["indices"]), {"sp500", "sp400"})
        self.assertEqual(rows["DISTINCT"]["indices"], ["sp400"])

    def test_combined_does_not_promote_future_other_index_membership(self):
        value = snapshot(members=[
            member("S1", "sp500"),
            member("S1", "sp400", membership_availability_date=date(2020, 2, 3)),
        ], universe="combined")
        self.assertEqual(keyed_rows(value.table)["S1"]["indices"], ["sp500"])
        self.assertEqual(value.excluded.num_rows, 1)

    def test_future_quote_availability_is_not_membership_eligibility(self):
        # Delisted/current-active flags and future price coverage are not inputs.
        value = snapshot(members=[member("GONE")], bridges=[], links=[])
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"GONE"})
        self.assertEqual(rows["GONE"]["identity_status"], "unmapped_bridge")

    def test_foreign_product_bridge_cannot_hijack_source_identity(self):
        value = snapshot(bridges=[bridge(), bridge(
            "S1", "FOREIGN", source_product_id="other_product")],
            links=[link(), link("FOREIGN", "FOREIGN_I")])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["identity_status"], "resolved")
        self.assertEqual(row["intrinio_security_id"], "I1")

    def test_foreign_product_only_bridge_leaves_member_unmapped(self):
        value = snapshot(bridges=[bridge(source_product_id="other_product")])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["identity_status"], "unmapped_bridge")
        self.assertIsNone(row["intrinio_security_id"])

    def test_missing_bridge_retains_member_in_denominator(self):
        value = snapshot(members=[member(), member("NO_MAP")])
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"S1", "NO_MAP"})
        self.assertEqual(rows["NO_MAP"]["identity_status"], "unmapped_bridge")
        self.assertIsNone(rows["NO_MAP"]["intrinio_security_id"])

    def test_ambiguous_bridge_quarantines_without_arbitrary_choice(self):
        value = snapshot(bridges=[bridge(), bridge("S1", "U2")],
                         links=[link(), link("U2", "I2")])
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"S1"})
        self.assertEqual(rows["S1"]["identity_status"], "ambiguous_bridge")
        self.assertIsNone(rows["S1"]["intrinio_security_id"])

    def test_valid_plus_null_bridge_candidate_is_quarantined(self):
        value = snapshot(bridges=[bridge(), bridge("S1", None)])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["identity_status"], "ambiguous_bridge")
        attached = attach_quotes(value, pa.Table.from_pylist([quote()],
                                                            schema=PRICE_SCHEMA))
        self.assertEqual(keyed_rows(attached.table)["S1"]["quote_status"],
                         "identity_unresolved")

    def test_blank_unified_id_is_rejected_or_quarantined(self):
        for bad_id in ["", " ", "\t"]:
            with self.subTest(bad_id=repr(bad_id)):
                self.assert_rejected_or_quarantined(
                    pa.Table.from_pylist([bridge("S1", bad_id)], schema=BRIDGE_SCHEMA),
                    pa.Table.from_pylist([link()], schema=LINK_SCHEMA))

    def test_nonstring_unified_id_is_rejected_or_quarantined(self):
        bridges = pa.Table.from_pylist([bridge()], schema=BRIDGE_SCHEMA)
        bridges = bridges.set_column(bridges.schema.get_field_index("unified_security_id"),
                                     "unified_security_id", pa.array([123], type=pa.int64()))
        self.assert_rejected_or_quarantined(
            bridges, pa.Table.from_pylist([link()], schema=LINK_SCHEMA))

    def test_blank_intrinio_id_is_rejected_or_quarantined(self):
        for bad_id in ["", " ", "\t"]:
            with self.subTest(bad_id=repr(bad_id)):
                self.assert_rejected_or_quarantined(
                    pa.Table.from_pylist([bridge()], schema=BRIDGE_SCHEMA),
                    pa.Table.from_pylist([link("U1", bad_id)], schema=LINK_SCHEMA))

    def test_nonstring_intrinio_id_is_rejected_or_quarantined(self):
        links = pa.Table.from_pylist([link()], schema=LINK_SCHEMA)
        links = links.set_column(links.schema.get_field_index("intrinio_security_id"),
                                 "intrinio_security_id", pa.array([123], type=pa.int64()))
        self.assert_rejected_or_quarantined(
            pa.Table.from_pylist([bridge()], schema=BRIDGE_SCHEMA), links)

    def test_missing_intrinio_link_keeps_member(self):
        value = snapshot(links=[])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["identity_status"], "unmapped_intrinio")
        self.assertEqual(row["unified_security_id"], "U1")
        self.assertIsNone(row["intrinio_security_id"])

    def test_review_required_identity_is_quarantined(self):
        value = snapshot(links=[link(identity_review_required=True)])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["identity_status"], "needs_review")
        attached = attach_quotes(value, pa.Table.from_pylist([quote()],
                                                            schema=PRICE_SCHEMA))
        self.assertEqual(keyed_rows(attached.table)["S1"]["quote_status"],
                         "identity_unresolved")

    def test_ambiguous_intrinio_link_does_not_pick_first(self):
        value = snapshot(links=[link(), link("U1", "I2")])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["identity_status"], "ambiguous_intrinio")
        self.assertIsNone(row["intrinio_security_id"])

    def test_target_collision_quarantines_all_sources(self):
        value = snapshot(members=[member(), member("S2")],
                         bridges=[bridge(), bridge("S2", "U2")],
                         links=[link(), link("U2", "I1")])
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"S1", "S2"})
        self.assertEqual({row["identity_status"] for row in rows.values()},
                         {"target_collision"})
        attached = attach_quotes(value, pa.Table.from_pylist([quote()],
                                                            schema=PRICE_SCHEMA))
        self.assertEqual({row["quote_status"] for row in attached.table.to_pylist()},
                         {"identity_unresolved"})

    def test_knowledge_time_refuses_undated_identity_bridge(self):
        with self.assertRaises(KnowledgeTimeUnavailable):
            snapshot(mode="knowledge_time")

    def test_invalid_dates_universe_and_mode_fail_closed(self):
        for changes in [
            {"as_of": "2020-02-30"},
            {"as_of": "2020-1-31"},
            {"as_of": "2020-01-31T00:00:00"},
            {"universe": "sp600"},
            {"universe": "SP500"},
            {"universe": "sp500 OR 1=1"},
            {"mode": "current_members"},
        ]:
            with self.subTest(changes=changes):
                with self.assertRaises(DataContractError):
                    snapshot(**changes)

    def test_missing_required_membership_column_fails_closed(self):
        incomplete = pa.Table.from_pylist([member()], schema=MEMBERSHIP_SCHEMA).drop(
            ["membership_effective_date"])
        with self.assertRaises(DataContractError):
            build_snapshot(incomplete,
                           pa.Table.from_pylist([bridge()], schema=BRIDGE_SCHEMA),
                           pa.Table.from_pylist([link()], schema=LINK_SCHEMA),
                           as_of=AS_OF, universe="sp500",
                           mode="historical_reconstruction")


class QuoteCoverageTests(unittest.TestCase):
    def attach(self, value, rows):
        return attach_quotes(value, pa.Table.from_pylist(rows, schema=PRICE_SCHEMA))

    def test_exact_asof_quote_attaches_to_resolved_member(self):
        value = self.attach(snapshot(), [quote()])
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["quote_status"], "ok")
        self.assertEqual(row["quote_close"], 10.0)
        self.assertIsInstance(value.summary, dict)

    def test_missing_quote_never_drops_historical_member(self):
        value = self.attach(snapshot(), [])
        rows = keyed_rows(value.table)
        self.assertEqual(set(rows), {"S1"})
        self.assertEqual(rows["S1"]["quote_status"], "missing_price")
        self.assertIsNone(rows["S1"].get("quote_close"))

    def test_partial_declared_request_preserves_unqueried_missing_and_unresolved(self):
        value = snapshot(members=[member("GOOD"), member("MISSING"),
                                  member("UNQUERIED"), member("UNRESOLVED")],
                         bridges=[bridge("GOOD", "UG"), bridge("MISSING", "UM"),
                                  bridge("UNQUERIED", "UQ")],
                         links=[link("UG", "IG"), link("UM", "IM"),
                                link("UQ", "IQ")])
        attached = attach_quotes(
            value, pa.Table.from_pylist([quote("IG")], schema=PRICE_SCHEMA),
            requested_security_ids=("IG", "IM"))
        rows = keyed_rows(attached.table)
        self.assertEqual({key: row["quote_status"] for key, row in rows.items()},
                         {"GOOD": "ok", "MISSING": "missing_price",
                          "UNQUERIED": "not_requested",
                          "UNRESOLVED": "identity_unresolved"})
        self.assertIsNone(rows["UNQUERIED"]["quote_close"])
        self.assertEqual(attached.summary["constituents"], 4)
        self.assertEqual(attached.summary["quote_status_counts"], {
            "ok": 1, "missing_price": 1, "not_requested": 1,
            "identity_unresolved": 1})

    def test_quote_outside_declared_request_is_rejected(self):
        with self.assertRaises(DataContractError):
            attach_quotes(
                snapshot(), pa.Table.from_pylist([quote(), quote("OUTSIDE")],
                                                 schema=PRICE_SCHEMA),
                requested_security_ids=("I1",))

    def test_empty_declared_price_request_is_unqueried_not_missing(self):
        attached = attach_quotes(
            snapshot(), pa.Table.from_pylist([], schema=PRICE_SCHEMA),
            requested_security_ids=())
        row = keyed_rows(attached.table)["S1"]
        self.assertEqual(row["quote_status"], "not_requested")
        self.assertIsNone(row["quote_close"])
        self.assertEqual(attached.summary["quote_status_counts"], {"not_requested": 1})

    def test_unresolved_identity_never_drops_or_receives_unrelated_quote(self):
        value = self.attach(snapshot(members=[member("NO_MAP")],
                                     bridges=[], links=[]), [quote()])
        row = keyed_rows(value.table)["NO_MAP"]
        self.assertEqual(row["quote_status"], "identity_unresolved")
        self.assertIsNone(row.get("quote_close"))

    def test_good_missing_unresolved_and_invalid_denominators_survive_together(self):
        value = snapshot(members=[member("GOOD"), member("MISSING"),
                                  member("BAD"), member("UNMAPPED")],
                         bridges=[bridge("GOOD", "UG"), bridge("MISSING", "UM"),
                                  bridge("BAD", "UB")],
                         links=[link("UG", "IG"), link("UM", "IM"),
                                link("UB", "IB")])
        attached = self.attach(value, [quote("IG"), quote("IB", close=0.0)])
        rows = keyed_rows(attached.table)
        self.assertEqual(set(rows), {"GOOD", "MISSING", "BAD", "UNMAPPED"})
        self.assertEqual({key: row["quote_status"] for key, row in rows.items()},
                         {"GOOD": "ok", "MISSING": "missing_price",
                          "BAD": "invalid_price", "UNMAPPED": "identity_unresolved"})

    def test_any_non_asof_quote_rejected_including_unrelated_security(self):
        for wrong_day in [date(2020, 1, 30), date(2020, 2, 3)]:
            with self.subTest(wrong_day=wrong_day):
                with self.assertRaises(DataContractError):
                    self.attach(snapshot(), [quote(), quote(
                        "UNRELATED", trading_date=wrong_day)])

    def test_duplicate_quote_keys_are_rejected_not_aggregated(self):
        with self.assertRaises(DataContractError):
            self.attach(snapshot(), [quote(), quote(close=11.0)])

    def test_invalid_close_gets_explicit_coverage_status(self):
        for close in [None, 0.0, -1.0, math.nan, math.inf, -math.inf]:
            with self.subTest(close=close):
                value = self.attach(snapshot(), [quote(close=close)])
                rows = keyed_rows(value.table)
                self.assertEqual(set(rows), {"S1"})
                self.assertEqual(rows["S1"]["quote_status"], "invalid_price")

    def test_source_quality_warning_is_not_silently_treated_as_clean(self):
        prices = pa.Table.from_pylist([quote()], schema=PRICE_SCHEMA).append_column(
            "price_quality_status", pa.array(["test_known_anomaly"]))
        value = attach_quotes(snapshot(), prices)
        row = keyed_rows(value.table)["S1"]
        self.assertEqual(row["quote_status"], "source_quality_flag")

    def test_empty_membership_returns_empty_snapshot_not_unrelated_prices(self):
        value = snapshot(members=[], bridges=[], links=[])
        self.assertEqual(value.table.num_rows, 0)
        self.assertEqual(self.attach(value, [quote()]).table.num_rows, 0)

    def test_protected_holdout_quote_attachment_fails_before_financial_use(self):
        value = snapshot(as_of="2024-01-31", members=[member(
            trading_date="2024-01-31")])
        with self.assertRaises(DataContractError):
            self.attach(value, [quote(trading_date="2024-01-31")])


if __name__ == "__main__":
    unittest.main()
