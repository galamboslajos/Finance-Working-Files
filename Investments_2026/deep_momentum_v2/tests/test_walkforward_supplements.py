"""Execution-only successors are bounded/pinned before monetary decoding."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import pyarrow.parquet as pq

from dmv2.action_loading import ACTION_PRICE_COLUMNS
from dmv2.backtest_loading import _pin
from dmv2.universe import DataContractError
from dmv2.walkforward_backtest import _supplement_plan
from dmv2.walkforward_clock import walkforward_clock


def fixture(root):
    directory=root/'data/derived/synthetic_successor'; directory.mkdir(parents=True)
    rows=[]
    for day in ['2020-06-01','2020-06-02']:
        row={key:100. for key in ACTION_PRICE_COLUMNS}
        row.update(trading_date=day,intrinio_security_id='NEW',frequency='daily',intraperiod=False,
            raw_ohlc_bounds_valid=True,adjusted_ohlc_bounds_valid=True,price_quality_status='accepted')
        rows.append(row)
    path=directory/'quotes.parquet'; pd.DataFrame(rows).to_parquet(path,index=False)
    source={'pin':_pin(root,path),'allowed_native_ids':['NEW'],'start':'2020-06-01','end':'2020-06-02',
        'evidence_reference':'https://www.sec.gov/synthetic-closing',
        'identity_evidence_reference':'https://www.sec.gov/synthetic-successor'}
    registry={'events':[{'successor_id':'NEW'}],'supplemental_quote_sources':[source]}
    metadata=pd.DataFrame({'trading_date':['2020-06-01'],'intrinio_security_id':['OLD']})
    return path,registry,metadata


class SupplementalQuoteTests(unittest.TestCase):
    def test_pinned_successor_metadata_is_planned_without_ohlc_decode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); path,registry,metadata=fixture(root); pins=[]
            original=pq.ParquetFile
            class Guard:
                def __init__(self,*args,**kwargs):self.pf=original(*args,**kwargs)
                def __getattr__(self,name):return getattr(self.pf,name)
                def read(self,columns=None,*args,**kwargs):
                    if columns is None or {'open','close','adj_open','adj_close'}&set(columns):
                        raise AssertionError('Monetary decode before successor metadata approval')
                    return self.pf.read(columns=columns,*args,**kwargs)
            with patch('dmv2.walkforward_backtest.pq.ParquetFile',side_effect=Guard):
                plans,keys,stats=_supplement_plan(root,registry,metadata,['OLD'],walkforward_clock()[0],pins)
            self.assertEqual(plans,[path]); self.assertEqual(len(keys),2)
            self.assertEqual(pins,[_pin(root,path)]); self.assertTrue(stats['execution_only'])
            self.assertFalse(stats['existing_requested_native_ids_extended'])
            self.assertFalse(stats['existing_quote_keys_overwritten'])

    def test_existing_requested_native_is_never_extended_after_fitting(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); path,registry,metadata=fixture(root)
            # Even an absent historical date for the ID cannot be patched here:
            # original feature/target consequences require a different input build.
            with self.assertRaisesRegex(DataContractError,'existing feature requests'):
                _supplement_plan(root,registry,metadata,['OLD','NEW'],walkforward_clock()[0],[])

    def test_scope_identity_overlap_hash_and_schema_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); path,registry,metadata=fixture(root); sessions=walkforward_clock()[0]
            for mutation in ('wrong_native','future','non_iso','http','non_successor','overlap'):
                with self.subTest(mutation=mutation):
                    changed=deepcopy(registry); source=changed['supplemental_quote_sources'][0]
                    if mutation=='wrong_native':source['allowed_native_ids']=['OTHER']; changed['events']=[{'successor_id':'OTHER'}]
                    if mutation=='future':source['end']='2024-01-02'
                    if mutation=='non_iso':source['start']='2020-6-1'
                    if mutation=='http':source['evidence_reference']='http://www.sec.gov/synthetic'
                    if mutation=='non_successor':changed['events']=[]
                    if mutation=='overlap':changed['supplemental_quote_sources'].append(deepcopy(source))
                    with self.assertRaises(DataContractError):_supplement_plan(root,changed,metadata,['OLD'],sessions,[])
            original=path.read_bytes(); path.write_bytes(original+b'\n')
            with self.assertRaisesRegex(DataContractError,'hash/size'):_supplement_plan(root,registry,metadata,['OLD'],sessions,[])
            path.write_bytes(original)
            frame=pd.read_parquet(path).drop(columns=['adj_open']); frame.to_parquet(path,index=False)
            registry['supplemental_quote_sources'][0]['pin']=_pin(root,path)
            with self.assertRaisesRegex(DataContractError,'paired raw/adjusted'):_supplement_plan(root,registry,metadata,['OLD'],sessions,[])

    def test_exact_date_and_session_bounds_precede_financial_projection(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary).resolve(); path,registry,metadata=fixture(root)
            for day in ('2020-06-03','2020-05-31'):
                frame=pd.read_parquet(path); frame.loc[0,'trading_date']=day; frame.to_parquet(path,index=False)
                registry['supplemental_quote_sources'][0]['pin']=_pin(root,path)
                with self.assertRaisesRegex(DataContractError,'dates/natives/sessions'):
                    _supplement_plan(root,registry,metadata,['OLD'],walkforward_clock()[0],[])


if __name__=='__main__':unittest.main()
