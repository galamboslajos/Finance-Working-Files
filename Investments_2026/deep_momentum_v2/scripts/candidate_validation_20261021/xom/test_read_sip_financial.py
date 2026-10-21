import hashlib
import io
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

import read_sip_financial as worker
from prepare_financial_plan import prepare, FINANCIAL, PROVENANCE
from read_sip_keys import pin, sha, write, select_keys

HERE=Path(__file__).parent
QA=HERE.parent/'xom_qa'/'xom_field_audit.py'
SOURCE='siblis_isin_b284bb7497616f5afba0'


class FinancialIntegrationTests(unittest.TestCase):
    def fixture(self, directory, bad_feed=False):
        rows=[]
        for d in ('2016-01-04','2016-01-05','2016-01-09','2024-01-02'):
            for adjustment in ('raw','all'):
                row=dict(security_id=SOURCE,trading_date=d,query_symbol='XOM',adjustment=adjustment,
                    open=10.,high=12.,low=9.,close=11.,volume=100.,timestamp=d+'T00:00:00Z',feed='sip',timeframe='1Day',
                    asof='source_asof',source_run_id='run',source_raw_run_root='rawroot',source_task_id='task',
                    source_page_hash='hash',source_request_id='request')
                if d>'2023-12-31' or d=='2016-01-09':
                    for c in FINANCIAL:row[c]=-1.e300 # poisoned future rows must never reach QA/output
                rows.append(row)
        if bad_feed:rows[0]['feed']='iex'
        table=pa.Table.from_pylist(rows);sink=io.BytesIO();pq.write_table(table,sink,row_group_size=4,compression='snappy')
        raw=sink.getvalue();n=struct.unpack('<I',raw[-8:-4])[0];footer=raw[-n-8:]
        pf=pq.ParquetFile(io.BytesIO(raw));groups=[];positions=[];all_keys=[]
        keyplan={'security_id':SOURCE,'query_symbol':'XOM','start':'2016-01-04','end':'2023-12-31'}
        for index in range(pf.metadata.num_row_groups):
            rg=pf.metadata.row_group(index);chunks={}
            for ci in range(rg.num_columns):
                c=rg.column(ci);chunks[c.path_in_schema]={'dictionary_page_offset':c.dictionary_page_offset,
                    'data_page_offset':c.data_page_offset,'compressed_bytes':c.total_compressed_size}
            groups.append({'index':index,'rows':rg.num_rows,'column_chunks':chunks})
            meta=table.slice(index*4,rg.num_rows).select(worker.KEYS)
            selected,indices=select_keys(meta,keyplan)
            if selected.num_rows:
                keys=selected.to_pylist();positions.append({'index':index,'positions':indices.to_pylist(),'keys':keys});all_keys+=keys
        obj={'size_bytes':len(raw),'uri':'gs://synthetic-only/object','generation':1,'sha256':'synthetic-not-remote'}
        write(directory/'layout.json',{'object':obj,'row_groups':groups});(directory/'footer.bin').write_bytes(footer)
        write(directory/'protocol.json',{'synthetic':True});write(directory/'footer_receipt.json',{'actual_cli_stdout_bytes':20,'object':obj})
        helper_pin=pin(HERE/'range_source.py')
        keyplan.update(schema='dmv2-xom-sip-key-plan/v1',layout_pin=pin(directory/'layout.json'),footer_pin=pin(directory/'footer.bin'),
            pins=[pin(directory/'protocol.json'),pin(directory/'footer_receipt.json'),helper_pin])
        write(directory/'key_plan.json',keyplan);write(directory/'positions.json',positions)
        write(directory/'keys.json',sorted(all_keys,key=lambda r:(r['trading_date'],r['adjustment'])));write(directory/'key_ranges.json',[])
        write(directory/'key_receipt.json',{'status':'PASS_METADATA_KEYS_ONLY','financial_values_decoded':False,
            'plan_pin':pin(directory/'key_plan.json'),'positions_pin':pin(directory/'positions.json'),'selected_key_pin':pin(directory/'keys.json'),
            'range_pin':pin(directory/'key_ranges.json'),'helper_pin':helper_pin,'selected_row_groups':[p['index'] for p in positions],
            'selected_rows':len(all_keys),'actual_http_range_payload_bytes':10,'object':obj,'worker_pin':pin(HERE/'read_sip_keys.py')})
        write(directory/'calendar.json',{'sessions':['2007-01-03','2016-01-04','2016-01-05','2023-12-29']})
        prepare(directory/'key_receipt.json',QA,directory/'calendar.json',directory/'financial_plan.json')
        self.seal(directory)
        return raw

    def seal(self,directory):
        path=directory/'launch.json'
        if path.exists():path.unlink() # scratch synthetic fixture only
        write(path,{'schema':'dmv2-xom-sip-financial-launch/v1','status':'SEALED_BEFORE_FINANCIAL_READ',
            'plan_sha256':sha(directory/'financial_plan.json'),'worker_sha256':sha(HERE/'read_sip_financial.py'),
            'planner_sha256':sha(HERE/'prepare_financial_plan.py')})

    def fake(self,raw,log):
        class FakeTransport:
            def __init__(self,obj,*,max_bytes):self.limit=max_bytes;self.total=0;self.ranges=[]
            def fetch(self,start,length,label):
                self.total+=length
                if self.total>self.limit:raise AssertionError('Synthetic range budget overrun')
                data=raw[start:start+length];log.append(label)
                self.ranges.append(dict(start=start,bytes=length,sha256=hashlib.sha256(data).hexdigest(),label=label))
                return data
        return FakeTransport

    def test_end_to_end_filtered_quotes_and_independent_calendar(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=self.fixture(p);log=[]
            with patch.object(worker,'AuthenticatedRanges',self.fake(raw,log)):
                receipt=worker.run(p/'financial_plan.json',p/'launch.json',p/'output')
            quotes=pq.read_table(p/'output'/'quotes.parquet').to_pylist()
            self.assertEqual(len(quotes),4)
            self.assertTrue(all(r['trading_date']<'2024-01-01' for r in quotes))
            self.assertTrue(all(r['trading_date']!='2016-01-09' and r['open']>0 for r in quotes))
            self.assertEqual(receipt['session_rows_audited'],4)
            self.assertEqual(receipt['extra_non_session_keys_metadata_only'],2)
            self.assertEqual(receipt['financial_2024_plus_python_rows'],0)
            self.assertFalse(receipt['original_source_chunks_retained'])
            self.assertFalse(any('rg1/' in x for x in log))
            summary=json.loads((p/'output'/'field_summary.json').read_text())
            self.assertTrue(all(set(row)==set(worker.KEYS) for row in summary['extra_non_XNYS_keys_metadata_only']))
            self.assertEqual({x.split('/')[1] for x in log},set(FINANCIAL+PROVENANCE))
            self.assertFalse(any(p.name.endswith('.bin') for p in (p/'output').rglob('*')))

    def test_bad_selected_feed_fails_before_financial_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=self.fixture(p,bad_feed=True);log=[]
            with patch.object(worker,'AuthenticatedRanges',self.fake(raw,log)):
                with self.assertRaisesRegex(ValueError,'feed/timeframe'):
                    worker.run(p/'financial_plan.json',p/'launch.json',p/'output')
            self.assertFalse(any(x.split('/')[1] in FINANCIAL for x in log))

    def test_unsealed_worker_fails_before_transport(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            launch=json.loads((p/'launch.json').read_text());launch['worker_sha256']='wrong';(p/'launch.json').write_text(json.dumps(launch))
            with patch.object(worker,'AuthenticatedRanges',side_effect=AssertionError('Network not allowed')):
                with self.assertRaisesRegex(ValueError,'worker/plan changed'):
                    worker.run(p/'financial_plan.json',p/'launch.json',p/'output')

    def test_forbidden_financial_projection_fails_before_transport(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);self.fixture(p)
            plan=json.loads((p/'financial_plan.json').read_text());plan['financial_columns'].append('vwap');(p/'financial_plan.json').write_text(json.dumps(plan));self.seal(p)
            with patch.object(worker,'AuthenticatedRanges',side_effect=AssertionError('Network not allowed')):
                with self.assertRaisesRegex(ValueError,'projection differs'):
                    worker.run(p/'financial_plan.json',p/'launch.json',p/'output')

    def test_invalid_positions_rejected_before_take(self):
        table=pa.table({'open':[1.,2.]})
        for positions in ([1,0],[0,0],[-1],[2],[True]):
            with self.assertRaisesRegex(ValueError,'Invalid immutable'):
                worker.selected_values(table,{'positions':positions,'keys':[{}]*len(positions)})

    def test_position_length_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError,'length mismatch'):
            worker.selected_values(pa.table({'open':[1.]}),{'positions':[0],'keys':[]})

    def test_postread_pin_change_prevents_pass_receipt(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=self.fixture(p);log=[];base=self.fake(raw,log)
            class MutatingTransport(base):
                def fetch(self,start,length,label):
                    result=super().fetch(start,length,label)
                    if label.endswith('/volume'):
                        (p/'calendar.json').write_text('{"changed_after_read":true}')
                    return result
            with patch.object(worker,'AuthenticatedRanges',MutatingTransport):
                with self.assertRaisesRegex(ValueError,'Pinned metadata changed'):
                    worker.run(p/'financial_plan.json',p/'launch.json',p/'output')
            self.assertFalse((p/'output'/'receipt.json').exists())


if __name__=='__main__':unittest.main()
