"""Synthetic keyed amendments used identically for features/outcomes/execution."""
import json
import unittest

import pandas as pd
import pyarrow as pa

from dmv2.fallback_quotes import build_quote_amendments
from dmv2.universe import DataContractError


def original():
    return pd.DataFrame([dict(intrinio_security_id="A", trading_date=day,
        open=10.,high=9. if index == 0 else 11.,low=9.,close=10.,
        adj_open=5.,adj_high=4.5 if index == 0 else 5.5,adj_low=4.5,adj_close=5.,
        volume=123.,adj_volume=246.,factor=.5,split_ratio=1.,dividend=0.,
        frequency="daily",intraperiod=False,price_quality_status="provider_ohlc_anomaly" if index == 0 else "accepted",
        raw_ohlc_bounds_valid=index != 0,adjusted_ohlc_bounds_valid=index != 0)
        for index,day in enumerate(("2018-01-02","2018-01-03"))])


def amended(append=False):
    result=original()
    result.loc[0,["high","adj_high"]]=[11.,5.5]
    result.loc[0,"price_quality_status"]="accepted"
    result.loc[0,["raw_ohlc_bounds_valid","adjusted_ohlc_bounds_valid"]]=True
    if append:
        row=result.iloc[1].copy()
        row["trading_date"]="2018-01-04"
        for column in ("volume","adj_volume","factor","split_ratio","dividend"):
            row[column]=None
        result=pd.concat([result,pd.DataFrame([row])],ignore_index=True)
    return result


class FallbackQuoteTests(unittest.TestCase):
    def bundle(self, append=False):
        return build_quote_amendments(original(),amended(append),ledger=pd.DataFrame([{"status":"synthetic_only"}]),
            sessions=("2018-01-02","2018-01-03","2018-01-04"))

    def test_ohlc_overlay_preserves_schema_rows_order_volume_and_source(self):
        raw=pa.Table.from_pandas(original(),preserve_index=False)
        before=raw.to_pylist()
        output=self.bundle().apply(raw,start="2018-01-02",end="2018-01-03")
        self.assertEqual(output.schema,raw.schema)
        self.assertEqual(len(output),len(raw))
        self.assertEqual(output["high"].to_pylist(),[11.,11.])
        self.assertEqual(output["adj_high"].to_pylist(),[5.5,5.5])
        self.assertEqual(output["volume"].to_pylist(),[123.,123.])
        self.assertEqual(output["adj_volume"].to_pylist(),[246.,246.])
        self.assertEqual(raw.to_pylist(),before)

    def test_verified_append_is_inside_bounds_and_original_id_request(self):
        raw=pa.Table.from_pandas(original(),preserve_index=False)
        bundle=self.bundle(append=True)
        output=bundle.apply(raw,start="2018-01-01",end="2018-01-04")
        self.assertEqual(len(output),3)
        self.assertEqual(output["trading_date"].to_pylist(),["2018-01-02","2018-01-03","2018-01-04"])
        self.assertIsNone(output["volume"].to_pylist()[-1])
        self.assertIsNone(output["split_ratio"].to_pylist()[-1])
        restricted=bundle.apply(raw,start="2018-01-01",end="2018-01-03")
        self.assertEqual(len(restricted),2)
        other=original()
        other["intrinio_security_id"]="B"
        self.assertTrue(bundle.apply(pa.Table.from_pandas(other,preserve_index=False)).equals(pa.Table.from_pandas(other,preserve_index=False)))

    def test_pre2018_history_unchanged_and_no_full_financial_conversion(self):
        source=original()
        source["trading_date"]=["2009-01-02","2009-01-05"]
        raw=pa.Table.from_pandas(source,preserve_index=False)
        self.assertTrue(self.bundle().apply(raw,start="2009-01-01",end="2009-12-31").equals(raw))

    def test_changed_original_source_refused(self):
        source=original()
        source.loc[0,"close"]=999.
        with self.assertRaisesRegex(DataContractError,"differs"):
            self.bundle().apply(pa.Table.from_pandas(source,preserve_index=False))

    def test_replacement_can_reuse_identical_already_amended_values(self):
        table=pa.Table.from_pandas(amended(),preserve_index=False)
        self.assertTrue(self.bundle().apply(table).equals(table))

    def test_append_collisions_refused(self):
        with self.assertRaisesRegex(DataContractError,"collides"):
            self.bundle(True).apply(pa.Table.from_pandas(amended(True),preserve_index=False))

    def test_tampered_amendment_identity_refused(self):
        bundle=self.bundle()
        object.__setattr__(bundle,"fingerprint","sha256:"+"0"*64)
        with self.assertRaisesRegex(DataContractError,"identity changed"):
            bundle.apply(pa.Table.from_pandas(original(),preserve_index=False))

    def test_future_dates_and_request_bounds_refused(self):
        source=original()
        source.loc[0,"trading_date"]="2020-01-02"
        with self.assertRaisesRegex(DataContractError,"2020"):
            self.bundle().apply(pa.Table.from_pandas(source,preserve_index=False))
        with self.assertRaises(DataContractError):
            self.bundle().apply(pa.Table.from_pandas(original(),preserve_index=False),end="2018-01-02")

    def test_native_duplicate_and_dropped_original_keys_refused(self):
        source=original()
        with self.assertRaises(DataContractError):
            self.bundle().apply(pa.Table.from_pandas(pd.concat([source,source]),preserve_index=False))
        with self.assertRaisesRegex(DataContractError,"drop"):
            build_quote_amendments(source,amended().iloc[:1],ledger=pd.DataFrame())

    def test_replacement_cannot_change_volume_or_action_values(self):
        for field in ("volume","adj_volume","factor","split_ratio","dividend"):
            target=amended()
            target.loc[0,field]=777.
            with self.subTest(field=field),self.assertRaisesRegex(DataContractError,"volume/action"):
                build_quote_amendments(original(),target,ledger=pd.DataFrame())


if __name__=="__main__":
    unittest.main()
