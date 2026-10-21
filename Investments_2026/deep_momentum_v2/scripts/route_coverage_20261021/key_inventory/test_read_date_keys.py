"""Small fabricated key-only contracts; no research files or financial columns."""
import calendar
from datetime import date
from pathlib import Path
import hashlib,struct,tempfile
import unittest
import pyarrow as pa
import pyarrow.parquet as pq
from read_date_keys import (COLUMNS,IDS,START,END,ContractError,anchor_summary,
    cumulative_usage,filtered_keys,included_group,inspect_footer,validate_keys)


def table(rows):return pa.Table.from_pylist(rows,schema=pa.schema([(c,pa.string()) for c in COLUMNS]))

def row(native,day):return dict(zip(COLUMNS,(native,day)))

class PruningContracts(unittest.TestCase):
    def test_frozen_plan_footer_hash_excludes_eight_byte_trailer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'keys.parquet';pq.write_table(table([row('sec_zq1J05','2020-01-31')]),path)
            raw=path.read_bytes();length=struct.unpack('<I',raw[-8:-4])[0]
            offset=len(raw)-8-length;metadata=raw[offset:offset+length]
            pf=pq.ParquetFile(path);rg=pf.metadata.row_group(0)
            group={'date_min':'2020-01-31','date_max':'2020-01-31','native_min':'sec_zq1J05',
                'native_max':'sec_zq1J05','rows':1,'index':0,'included':True,
                'projected_compressed_bytes':sum(rg.column(i).total_compressed_size for i in range(2))}
            obj={'path':str(path),'size_bytes':len(raw),'footer_offset':offset,'footer_size_bytes':length,
                'footer_sha256':hashlib.sha256(metadata).hexdigest(),'key_schema':[[c,'string'] for c in COLUMNS],
                'groups':[group]}
            _,got=inspect_footer(obj);self.assertEqual(got,[group])
            with self.assertRaisesRegex(ContractError,'Raw footer'):
                inspect_footer({**obj,'footer_sha256':hashlib.sha256(raw[offset:]).hexdigest()})

    def test_exact_native_bounds_no_ticker_or_substitution(self):
        group={'date_min':'1973-05-03','date_max':'2026-07-24','native_min':'sec_zq1J05','native_max':'sec_zq1J05'}
        self.assertTrue(included_group(group,IDS,START,END))
        self.assertFalse(included_group({**group,'native_min':'sec_other','native_max':'sec_other'},IDS,START,END))

    def test_missing_stats_include_fail_safe_and_invalid_bounds_reject(self):
        self.assertTrue(included_group({},IDS,START,END))
        self.assertFalse(included_group({'date_min':'2024-01-01','date_max':'2026-07-24'},IDS,START,END))
        with self.assertRaises(ContractError):included_group({'date_min':'bad','date_max':'2026-07-24'},IDS,START,END)
        with self.assertRaises(ContractError):included_group({'native_min':'z','native_max':'a'},IDS,START,END)

    def test_arrow_exact_filter_and_date_bounds_before_python(self):
        got=filtered_keys(table([row('sec_zq1J05','2007-01-01'),row('sec_zq1J05','2023-12-31'),
            row('sec_zq1J05','2024-01-02'),row('sec_z5YpGE','2026-07-06'),row('BK','2020-01-31')]),IDS,START,END)
        self.assertEqual(got.to_pylist(),[row('sec_zq1J05','2007-01-01'),row('sec_zq1J05','2023-12-31')])
        self.assertEqual(filtered_keys(table([]),IDS,START,END).num_rows,0)

    def test_no_extra_field_native_substitution_or_date_widening(self):
        with self.assertRaises(ContractError):filtered_keys(table([row('sec_zq1J05','2020-01-31')]).append_column('close',pa.array([10.])),IDS,START,END)
        with self.assertRaises(ContractError):filtered_keys(table([]),IDS|{'other'},START,END)
        with self.assertRaises(ContractError):filtered_keys(table([]),IDS,'2000-01-01',END)
        with self.assertRaises(ContractError):validate_keys([row('other','2020-01-31')],IDS,START,END)
        with self.assertRaises(ContractError):validate_keys([row('sec_zq1J05','2024-01-01')],IDS,START,END)

    def test_duplicates_null_keys_and_malformed_dates_fail_no_truncation(self):
        with self.assertRaises(ContractError):validate_keys([row('sec_zq1J05','2020-01-31')]*2,IDS,START,END)
        with self.assertRaises(ContractError):filtered_keys(table([row(None,'2020-01-31')]),IDS,START,END)
        with self.assertRaises(ContractError):filtered_keys(table([row('sec_zq1J05','20200131')]),IDS,START,END)
        with self.assertRaises(ContractError):filtered_keys(table([row('sec_zq1J05','2020-02-30')]),IDS,START,END)
        with self.assertRaises(ContractError):validate_keys([row('sec_zq1J05','2020-02-30')],IDS,START,END)

    def test_cumulative_files_returned_rows_and_projection_budgets(self):
        prior={'candidate_rows':85045,'data_files':6,'footer_files':6,'projected_compressed_bytes':6495846,'metadata_rows_converted':85045}
        new={'candidate_rows':759396,'data_files':3,'footer_files':105,'projected_compressed_bytes':1456535,'metadata_rows_converted':13445}
        limits={'max_candidate_rows':20000000,'max_data_files':9,'max_footer_files':128,
            'max_projected_column_bytes':134217728,'max_returned_metadata_rows':100000}
        got=cumulative_usage(prior,new,limits)
        self.assertEqual(got['data_files'],9);self.assertEqual(got['metadata_rows_converted'],98490)
        self.assertEqual(got['projected_compressed_bytes'],7952381)
        for changes in ({'data_files':4},{'metadata_rows_converted':14956},{'projected_compressed_bytes':134217728}):
            with self.subTest(changes=changes),self.assertRaises(ContractError):cumulative_usage(prior,{**new,**changes},limits)

    def test_source_anchors_exact_original_keys_not_nearest_substitute(self):
        windows=[];previous='2009-12-31'
        for year in range(2010,2024):
            for month in range(1,13):
                last=f'{year}-{month:02d}-{calendar.monthrange(year,month)[1]:02d}'
                windows.append({'outcome_month':f'{year}-{month:02d}','signal_date':previous,
                    'outcome_start':f'{year}-{month:02d}-01','outcome_end':last});previous=last
        cases=[{'ticker_label':'BK','native_security_id':'sec_zq1J05','role':'unpromoted_candidate'},
            {'ticker_label':'XOM','native_security_id':'sec_z5YpGE','role':'current_route'}]
        got=anchor_summary([('sec_zq1J05','2009-12-31'),('sec_zq1J05','2010-01-30')],windows,cases)
        self.assertEqual(got[0]['matched_anchor_dates'],1)
        self.assertFalse(got[0]['month_endpoints'][0]['end_key_exists'])
        self.assertEqual(got[1]['date_key_rows'],0);self.assertEqual(len(got[1]['missing_anchor_dates']),169)
        self.assertFalse(got[0]['historical_identity_or_quote_validity_certified'])
        with self.assertRaises(ContractError):anchor_summary([],windows[:-1],cases)

if __name__=='__main__':unittest.main()
