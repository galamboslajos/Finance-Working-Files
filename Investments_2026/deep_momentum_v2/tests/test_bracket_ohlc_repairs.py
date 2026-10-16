"""Synthetic regression tests for the separate, fixed three-key amendment.

Every numerical quote, local projection and receipt is invented in temporary
files. Fixed reviewed source identifiers are contract constants only. No private
financial files, absolute workspace paths, models or backtest results are read.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import inspect
import tempfile
import unittest

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
def parameterize(names, values):
    def decorate(function):
        function._parameters = (names.split(","), values)
        return function
    return decorate


def raises(exception, *, match):
    return unittest.TestCase().assertRaisesRegex(exception, match)

from dmv2 import bracket_ohlc_repairs as m


def case(tmp_path):
    """Invented numerical quotes; real closed-case IDs carry no prices."""
    rows, sip_rows = [], []
    for ordinal, (symbol, source, native) in enumerate(m.CASES):
        base = float(100 + 10 * ordinal)
        raw_by_day = {
            m.PRECEDING_DATE: dict(open=base, high=base+1., low=base-1., close=base+.5),
            m.TARGET_DATE: dict(open=base, high=base+.2, low=base-1., close=base+1.5),
            m.FOLLOWING_DATE: dict(open=base+.2, high=base+1.5, low=base-.8, close=base+.6),
        }
        for day, raw in raw_by_day.items():
            rejected = day == m.TARGET_DATE
            rows.append({"trading_date":day,"intrinio_security_id":native,
                "frequency":"daily","intraperiod":False, **raw,
                **{"adj_"+name:value*.8 for name,value in raw.items()},
                "volume":1000.+ordinal, "adj_volume":1250.+ordinal,
                "factor":1., "split_ratio":1.,
                "dividend":.5 if ordinal == 0 and day == m.PRECEDING_DATE else 0.,
                "raw_ohlc_bounds_valid":not rejected, "adjusted_ohlc_bounds_valid":not rejected,
                "price_quality_status":"provider_ohlc_anomaly" if rejected else "accepted"})
            independent = (dict(open=base+.3, high=base+2., low=base-1., close=base+1.)
                           if rejected else dict(raw))
            sip_rows.append({"security_id":source,"trading_date":day,"query_symbol":symbol,
                "feed":"sip","adjustment":"raw","timeframe":"1Day",
                "source_task_id":"SYNTHETIC_TASK","source_request_id":"SYNTHETIC_REQUEST",
                "source_page_hash":"SYNTHETIC_UNVERIFIED_RAW_LINEAGE", **independent})
    prices = pd.DataFrame(rows)
    projection_path = tmp_path / "synthetic_sip.parquet"
    pq.write_table(pa.Table.from_pylist(sip_rows),projection_path)
    artifact_hash = hashlib.sha256(projection_path.read_bytes()).hexdigest()
    receipt = {"object":{"uri":m.SOURCE_URI,"generation":m.TABLE_GENERATION,
        "size_bytes":m.TABLE_SIZE,"sha256":m.TABLE_DECLARED_SHA256},
        "source_publication_manifest_sha256":m.PUBLICATION_SHA256,
        "selected_artifact_sha256":artifact_hash,
        "only_2018_2019_financial_rows_returned_or_used":True,
        "excluded_financial_rows_converted_to_python_or_returned":False,
        "whole_source_table_sha256_reread":False,"raw_page_hashes_verified":False}
    receipt_path = tmp_path / "synthetic_receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    def pin(path):
        return {"path":path.name,"size_bytes":path.stat().st_size,
            "sha256":"sha256:"+hashlib.sha256(path.read_bytes()).hexdigest(),"source_uri":m.SOURCE_URI}
    provenance = {"source_table_uri":m.SOURCE_URI,"source_table_generation":m.TABLE_GENERATION,
        "source_table_size_bytes":m.TABLE_SIZE,"source_table_sha256_declared":m.TABLE_DECLARED_SHA256,
        "publication_manifest_uri":m.PUBLICATION_URI,"publication_manifest_generation":m.PUBLICATION_GENERATION,
        "publication_manifest_canonical_sha256":m.PUBLICATION_SHA256,
        "publication_manifest_canonical_pin_verified":True,
        "whole_source_table_sha256_reread":False,"raw_page_hash_verified":False,
        "retained_manifest_pin_verified":False,"mixed_row_group_numeric_decoding_possible":True,
        "projection_policy":"date_and_id_filter_in_Arrow_before_Python_conversion",
        "projected_compressed_bytes":100,"maximum_projected_compressed_bytes":64*1024*1024,
        "range_receipt":pin(receipt_path)}
    cases = []
    for symbol,source,native in m.CASES:
        by_date = prices.loc[prices.intrinio_security_id == native].set_index("trading_date")
        cases.append({"query_symbol":symbol,"source_security_id":source,"intrinio_security_id":native,
            "trading_date":m.TARGET_DATE,"preceding_date":m.PRECEDING_DATE,"following_date":m.FOLLOWING_DATE,
            "original_target_bar_sha256":m.original_bar_sha256(by_date.loc[m.TARGET_DATE]),
            "preceding_control_bar_sha256":m.original_bar_sha256(by_date.loc[m.PRECEDING_DATE]),
            "following_control_bar_sha256":m.original_bar_sha256(by_date.loc[m.FOLLOWING_DATE])})
    manifest = {"schema":"dmv2-three-bracketed-sip-ohlc/v1","policy_id":m.POLICY_ID,
        "projection_evidence":pin(projection_path),"source_provenance":provenance,
        "cases":cases,"review_notes":"SYNTHETIC UNIT TEST ONLY: all prices, receipt and local files are invented. "
        "External publication constants exercise the closed contract, not real source verification."}
    return prices, manifest, tmp_path



def run(case):
    prices, manifest, root = case
    return m.apply_bracketed_sip_ohlc_repairs(prices, evidence_root=root, manifest=manifest)


def index(prices, symbol, day=m.TARGET_DATE):
    native = dict((s, native) for s, _, native in m.CASES)[symbol]
    return prices.index[(prices.intrinio_security_id == native) & (prices.trading_date == day)][0]


def repin_original(case, symbol, day):
    prices, manifest, _ = case
    name = {m.TARGET_DATE: "original_target_bar_sha256", m.PRECEDING_DATE: "preceding_control_bar_sha256",
            m.FOLLOWING_DATE: "following_control_bar_sha256"}[day]
    next(x for x in manifest["cases"] if x["query_symbol"] == symbol)[name] = m.original_bar_sha256(prices.loc[index(prices,symbol,day)])


def mutate_evidence(case, transform):
    _, manifest, root = case
    projection = manifest["projection_evidence"]
    path = root / projection["path"]
    table = pq.read_table(path)
    updated = transform(table)
    pq.write_table(updated, path)
    projection["size_bytes"] = path.stat().st_size
    projection["sha256"] = "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
    receipt_pin = manifest["source_provenance"]["range_receipt"]
    receipt_path = root / receipt_pin["path"]
    receipt = json.loads(receipt_path.read_text())
    receipt["selected_artifact_sha256"] = projection["sha256"].removeprefix("sha256:")
    receipt_path.write_text(json.dumps(receipt))
    receipt_pin["size_bytes"] = receipt_path.stat().st_size
    receipt_pin["sha256"] = "sha256:" + hashlib.sha256(receipt_path.read_bytes()).hexdigest()


def change_column(table, name, predicate, value):
    rows = table.select(["security_id", "trading_date"]).to_pylist()
    values = table[name].to_pylist()
    values = [value if predicate(row) else old for row,old in zip(rows, values, strict=True)]
    return table.set_column(table.schema.get_field_index(name), name, pa.array(values, type=table[name].type))


def check_synthetic_full_bars_and_controls_are_preserved(case):
    before = case[0].copy(deep=True)
    result = run(case)
    assert len(result.prices) == 9 and len(result.ledger) == 3
    assert "price_accepted" not in result.prices
    for symbol in ("ACC","K","SO"):
        at = index(before,symbol)
        entry = result.ledger.loc[result.ledger.query_symbol == symbol].iloc[0]
        assert result.prices.loc[at,"original_price_quality_status"] == "provider_ohlc_anomaly"
        assert result.prices.loc[at,"price_quality_status"] == "accepted"
        assert result.prices.loc[at,"close"] == {"ACC":101.,"K":111.,"SO":121.}[symbol]
        assert entry.multiplier_source_date == m.PRECEDING_DATE
        assert entry.maximum_adjusted_pair_residual < .0002
        assert entry.raw_prices_interpolated == False
        assert entry.knowledge_time_certified == False
        for field in ("volume","adj_volume","factor","dividend","split_ratio"):
            assert result.prices.loc[at,field] == before.loc[at,field]
    controls = before.trading_date != m.TARGET_DATE
    pd.testing.assert_frame_equal(result.prices.loc[controls,before.columns], before.loc[controls])
    pd.testing.assert_frame_equal(case[0], before)


@parameterize("field,value",[("frequency","weekly"),("intraperiod",True),("intraperiod",None),("intraperiod",0)])
def check_target_frequency_and_intraperiod_are_independently_guarded(case,field,value):
    prices = case[0]
    prices[field] = prices[field].astype(object)
    prices.loc[index(prices,"ACC"),field] = value
    # These fields deliberately are absent from legacy original_bar_sha256.
    with raises(m.DataContractError, match="already-rejected"):
        run(case)


def check_optional_price_acceptance_is_only_set_for_targets_without_nans(case):
    prices = case[0]
    prices["price_accepted"] = prices.trading_date != m.TARGET_DATE
    result = run(case)
    assert result.prices.price_accepted.notna().all()
    assert result.prices.price_accepted.all()


def check_target_bad_close_cannot_supply_adjustment_units(case):
    expected = run(case).ledger
    case[0].loc[index(case[0],"ACC"),"close"] += .00001
    repin_original(case,"ACC",m.TARGET_DATE)
    result = run(case)
    assert result.ledger.adjustment_multiplier.tolist() == expected.adjustment_multiplier.tolist()
    assert result.ledger.replacement_adjusted_ohlc.tolist() == expected.replacement_adjusted_ohlc.tolist()


def check_following_control_is_qa_only_and_not_the_multiplier(case):
    expected = run(case).ledger
    at = index(case[0],"ACC",m.FOLLOWING_DATE)
    for field in m.OHLC:
        case[0].loc[at,"adj_" + field] += .000001
    repin_original(case,"ACC",m.FOLLOWING_DATE)
    result = run(case)
    assert result.ledger.adjustment_multiplier.tolist() == expected.adjustment_multiplier.tolist()
    assert result.ledger.replacement_adjusted_ohlc.tolist() == expected.replacement_adjusted_ohlc.tolist()


def check_preceding_ex_dividend_pair_is_allowed_without_reapplying_dividend(case):
    result = run(case)
    entry = result.ledger.loc[result.ledger.query_symbol == "ACC"].iloc[0]
    assert entry.preceding_dividend_already_in_paired_units == .5
    before = case[0].loc[index(case[0],"ACC",m.PRECEDING_DATE)]
    assert entry.adjustment_multiplier == before.adj_close / before.close


@parameterize("day,field,value",[
    (m.TARGET_DATE,"dividend",.1),(m.TARGET_DATE,"split_ratio",2.),
    (m.PRECEDING_DATE,"split_ratio",2.),(m.FOLLOWING_DATE,"dividend",.1),
    (m.FOLLOWING_DATE,"split_ratio",2.)])
def check_action_ambiguity_refuses_instead_of_manufacturing_units(case,day,field,value):
    case[0].loc[index(case[0],"ACC",day),field] = value
    repin_original(case,"ACC",day)
    with raises(m.DataContractError, match="action|split"):
        run(case)


@parameterize("field,value",[
    ("price_quality_status","accepted"),
    ("raw_ohlc_bounds_valid",True),
    ("adjusted_ohlc_bounds_valid",True)])
def check_only_already_rejected_target_can_be_replaced(case,field,value):
    case[0].loc[index(case[0],"ACC"),field] = value
    repin_original(case,"ACC",m.TARGET_DATE)
    with raises(m.DataContractError, match="already-rejected"):
        run(case)


def check_original_target_values_and_control_values_are_hash_pinned(case):
    case[0].loc[index(case[0],"ACC"),"high"] += .01
    with raises(m.DataContractError, match="target differs"):
        run(case)


def check_following_paired_rounding_conflict_refuses(case):
    case[0].loc[index(case[0],"K",m.FOLLOWING_DATE),"adj_high"] += .001
    repin_original(case,"K",m.FOLLOWING_DATE)
    with raises(m.DataContractError, match="adjustment units"):
        run(case)


def check_raw_control_mismatch_refuses_even_with_compatible_close(case):
    source = m.CASES[0][1]
    mutate_evidence(case, lambda t: change_column(t,"high",
        lambda r: r["security_id"] == source and r["trading_date"] == m.FOLLOWING_DATE,105.))
    with raises(m.DataContractError, match="full raw OHLC"):
        run(case)


@parameterize("field,value",[("feed","iex"),("adjustment","all"),("timeframe","1Hour"),("query_symbol","WRONG")])
def check_independent_source_basis_must_match(case,field,value):
    source = m.CASES[0][1]
    mutate_evidence(case, lambda t: change_column(t,field,
        lambda r: r["security_id"] == source and r["trading_date"] == m.TARGET_DATE,value))
    with raises(m.DataContractError, match="SIP query"):
        run(case)


def check_source_evidence_bytes_cannot_change_unpinned(case):
    path = case[2] / case[1]["projection_evidence"]["path"]
    path.write_bytes(path.read_bytes() + b"x")
    with raises(m.DataContractError, match="wrong size"):
        run(case)


def check_future_metadata_refuses_before_any_financial_read(case,monkeypatch):
    source = m.CASES[0][1]
    mutate_evidence(case, lambda t: change_column(t,"trading_date",
        lambda r: r["security_id"] == source and r["trading_date"] == m.TARGET_DATE,"2020-01-02"))
    constructor = m.pq.ParquetFile
    read_columns = []
    class Guard:
        def __init__(self,*args,**kwargs):
            self.real = constructor(*args,**kwargs)
            self.schema_arrow, self.metadata = self.real.schema_arrow, self.real.metadata
        def read(self,columns):
            read_columns.append(columns)
            assert not set(columns) & set(m.OHLC), "Financial read occurred before date guard"
            return self.real.read(columns=columns)
    monkeypatch.setattr(m.pq,"ParquetFile",Guard)
    with raises(m.DataContractError, match="authorized"):
        run(case)
    assert len(read_columns) == 1


@parameterize("kind",["other_source","extra_case","other_date"])
def check_fixed_scope_cannot_expand(case,kind):
    manifest = case[1]
    if kind == "other_source":
        manifest["cases"][0]["intrinio_security_id"] = "some_new_stock"
    elif kind == "extra_case":
        manifest["cases"].append(deepcopy(manifest["cases"][0]))
    else:
        manifest["cases"][0]["trading_date"] = "2019-08-14"
    with raises(m.DataContractError, match="three|expand"):
        run(case)


def check_publication_source_generation_and_limits_are_explicit(case):
    case[1]["source_provenance"]["raw_page_hash_verified"] = True
    with raises(m.DataContractError, match="limitations"):
        run(case)


def check_extra_unaffected_rows_remain_in_population(case):
    prices,manifest,root = case
    extra = prices.loc[[index(prices,"ACC",m.PRECEDING_DATE)]].copy()
    extra.intrinio_security_id = "unrelated_native"
    prices = pd.concat([prices,extra],ignore_index=True)
    result = run((prices,manifest,root))
    assert len(result.prices) == len(prices)
    pd.testing.assert_frame_equal(result.prices.loc[[9],prices.columns],prices.loc[[9]])


def check_existing_original_markers_and_unrelated_rows_are_preserved(case):
    prices,manifest,root = case
    for field in m.ORIGINAL_FIELDS:
        prices["original_" + field] = prices[field]
    prices["original_close"] = -999.  # Synthetic pre-existing provenance marker.
    extra = prices.loc[[index(prices,"ACC",m.PRECEDING_DATE)]].copy()
    extra.intrinio_security_id = "unrelated_native"
    prices = pd.concat([prices,extra],ignore_index=True)
    before = prices.copy(deep=True)
    result = run((prices,manifest,root))
    pd.testing.assert_frame_equal(prices,before)
    assert result.prices.original_close.eq(-999.).all()
    assert len(result.prices) == len(prices)
    pd.testing.assert_frame_equal(result.prices.loc[[9],prices.columns],prices.loc[[9]])


class TestBracketedOhlcRepairContract(unittest.TestCase):
    pass


def _register_tests():
    class MonkeyPatch:
        def __init__(self): self.changes = []
        def setattr(self, obj, name, value):
            self.changes.append((obj,name,getattr(obj,name)))
            setattr(obj,name,value)
        def undo(self):
            for obj,name,value in reversed(self.changes): setattr(obj,name,value)
    for name,function in sorted(globals().items()):
        if not name.startswith("check_") or not callable(function): continue
        names,values = getattr(function,"_parameters",([], [()]))
        for ordinal,value in enumerate(values):
            values_tuple = value if isinstance(value,tuple) else (value,)
            kwargs = dict(zip(names,values_tuple))
            def invoke(self, function=function, kwargs=kwargs):
                with tempfile.TemporaryDirectory(prefix="dmv2-bracket-test-") as temporary:
                    patch = MonkeyPatch()
                    arguments = {"case":case(Path(temporary)), **kwargs}
                    if "monkeypatch" in inspect.signature(function).parameters:
                        arguments["monkeypatch"] = patch
                    try: function(**arguments)
                    finally: patch.undo()
            setattr(TestBracketedOhlcRepairContract, "test_" + name.removeprefix("check_") + "_" + str(ordinal), invoke)


_register_tests()


if __name__ == "__main__":
    unittest.main(verbosity=2)

