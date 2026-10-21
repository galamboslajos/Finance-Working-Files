import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('read_stage1', HERE / 'read_stage1.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def case(label='XOM'):
    return {'ticker_label': label, 'source_security_id': 'source_' + label,
        'native_security_id': 'native_' + label}


def sample():
    return {
        'security_master_siblis': [{'security_id': 'source_XOM', 'primary_ticker': 'XOM', 'isin': 'ISIN_XOM'}],
        'security_identifier_history': [{'security_id': 'source_XOM', 'identifier_type': 'isin',
            'identifier_value': 'ISIN_XOM', 'valid_from': None, 'valid_to': None}],
        'security_identity_bridge': [{'source_product_id': 'sp_composite1500',
            'source_security_id': 'source_XOM', 'unified_security_id': 'unified_XOM',
            'isin': 'ISIN_XOM', 'primary_ticker': 'XOM'}],
        'intrinio_identity_links': [{'unified_security_id': 'unified_XOM',
            'intrinio_security_id': 'native_XOM', 'query_identifier': 'ISIN_XOM',
            'query_identifier_type': 'isin', 'observed_tickers_json': '["XOM"]',
            'observed_isins_json': '["ISIN_XOM"]'}],
        'intrinio_security_master': [{'intrinio_security_id': 'native_XOM', 'company_id': 'company_XOM',
            'ticker': 'XOM', 'composite_ticker': 'XOM:US', 'figi': 'FIGI_XOM'}],
        'identifier_resolution_evidence': [{'requested_identifier': 'native_XOM',
            'intrinio_security_id': 'native_XOM', 'http_status': 200, 'status': 'complete',
            'price_row_count': 0, 'price_page_count': 1}],
    }


class CandidateMetadataTests(unittest.TestCase):
    def test_exact_route_and_empty_acquisition_remain_unexplained(self):
        result = m.candidates(sample(), [case()])[0]
        self.assertEqual(result['candidate_native_ids'], ['native_XOM'])
        self.assertEqual(result['candidate_summaries'][0]['reported_acquisition_rows'], [0])
        self.assertFalse(result['price_history_absence_proven'])
        self.assertFalse(result['canonical_mapping_promoted'])

    def test_source_alias_match_is_not_independent_confirmation(self):
        result = m.candidates(sample(), [case()])[0]
        self.assertFalse(result['source_alias_agreement_is_independent_confirmation'])
        self.assertFalse(result['candidate_summaries'][0]['independent_source_isin_provider_identity_verified'])

    def test_same_company_alternative_retained_not_selected(self):
        tables = sample()
        tables['intrinio_security_master'].append({'intrinio_security_id': 'alternate',
            'company_id': 'company_XOM', 'ticker': 'XOM-A', 'type': 'preferred'})
        result = m.candidates(tables, [case()])[0]
        self.assertEqual(result['candidate_native_ids'], ['alternate', 'native_XOM'])
        self.assertTrue(all(r['qualification'] == 'CANDIDATE_ONLY_NOT_PROMOTED' for r in result['candidate_summaries']))

    def test_same_figi_alternative_is_candidate_only(self):
        tables = sample()
        tables['intrinio_security_master'].append({'intrinio_security_id': 'other',
            'company_id': 'different', 'ticker': 'OTHER', 'figi': 'FIGI_XOM'})
        result = m.candidates(tables, [case()])[0]
        self.assertIn('other', result['candidate_native_ids'])
        self.assertFalse(result['canonical_mapping_promoted'])

    def test_unrelated_issuer_not_returned(self):
        tables = sample()
        tables['intrinio_security_master'].append({'intrinio_security_id': 'unrelated',
            'company_id': 'another_company', 'ticker': 'ZZZ', 'figi': 'otherfigi'})
        result = m.candidates(tables, [case()])[0]
        self.assertNotIn('unrelated', result['candidate_native_ids'])

    def test_wrong_source_product_not_exact_route(self):
        tables = sample(); tables['security_identity_bridge'][0]['source_product_id'] = 'russell'
        result = m.candidates(tables, [case()])[0]
        self.assertFalse(result['exact_sp1500_bridge_rows'])
        self.assertEqual(len(result['related_source_alias_bridge_rows']), 1)

    def test_missing_identity_dates_are_not_filled(self):
        result = m.candidates(sample(), [case()])[0]
        self.assertIsNone(result['source_identifier_history'][0]['valid_from'])
        self.assertIsNone(result['source_identifier_history'][0]['valid_to'])

    def test_malformed_aliases_rejected(self):
        tables = sample(); tables['intrinio_identity_links'][0]['observed_tickers_json'] = '{"XOM":1}'
        with self.assertRaisesRegex(ValueError, 'Malformed'):
            m.candidates(tables, [case()])

    def test_unbounded_candidates_rejected_whole_not_truncated(self):
        tables = sample()
        tables['intrinio_security_master'] += [{'intrinio_security_id': str(i),
            'company_id': 'company_XOM', 'ticker': 'OTHER'} for i in range(201)]
        with self.assertRaisesRegex(ValueError, 'Unbounded'):
            m.candidates(tables, [case()])

    def test_namespaced_symbol_metadata_normalized_without_mapping_promotion(self):
        self.assertEqual(m.base_ticker('xom:us'), 'XOM')
        self.assertEqual(m.base_ticker('BK-A'), 'BK')
        self.assertEqual(m.base_ticker(None), '')

    def test_config_mutation_rejected_before_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'; p.write_text('original')
            item = m.pin(p); p.write_text('modified')
            with self.assertRaisesRegex(ValueError, 'bytes changed'):
                m.checked(item)

    def test_synthetic_frozen_six_table_roundtrip(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as d:
            top = Path(d); tables = sample(); entries = []
            for table, rows in tables.items():
                path = top / (table + '.parquet'); pq.write_table(pa.Table.from_pylist(rows), path)
                pf = pq.ParquetFile(path); cols = pf.schema_arrow.names
                with path.open('rb') as f:
                    f.seek(-8, 2); tail = f.read(8); n = struct.unpack('<I', tail[:4])[0]
                    f.seek(-(n + 8), 2); footer = f.read(n + 8)
                obj = {**m.pin(path), 'rows': len(rows), 'columns': cols,
                    'footer_offset': path.stat().st_size - len(footer),
                    'footer_size_bytes': len(footer), 'footer_sha256': hashlib.sha256(footer).hexdigest(),
                    'schema': [[name, str(pf.schema_arrow.field(name).type)] for name in cols]}
                entries.append({'source': 'synthetic', 'table': table, 'columns': cols,
                    'physical_rows': len(rows), 'objects': [obj], 'projected_compressed_bytes': path.stat().st_size})
            marker = top / 'marker'; marker.write_text('synthetic helper')
            protocol = top / 'protocol.json'; protocol.write_text(json.dumps({
                'status': 'FROZEN_METADATA_ONLY', 'cases': [case()], 'claim_limits': ['candidate only'],
                'budget': {'max_projected_column_bytes': 128 * 1024**2, 'max_candidate_rows': 20000000,
                    'max_returned_metadata_rows': 100000, 'max_data_files': 8, 'max_footer_files': 128,
                    'max_rss_bytes': 2 * 1024**3, 'max_wall_seconds': 600, 'max_output_bytes': 128 * 1024**2}}))
            plan = {'schema': 'dmv2-xom-bk-source-stage1-plan/v1', 'cases': [case()],
                'tables': entries, 'protocol_pin': m.pin(protocol),
                'planner_pin': m.pin(marker), 'footer_helper_pin': m.pin(marker),
                'source_config_pin': m.pin(marker), 'manifest_pins': [m.pin(marker)],
                'projected_compressed_bytes': sum(e['projected_compressed_bytes'] for e in entries),
                'physical_rows': 6, 'returned_metadata_rows_upper_bound': 6, 'data_files': 6, 'footer_files': 6}
            plan_path = top / 'plan.json'; m.write(plan_path, plan)
            config = top / 'config.json'; m.write(config, {
                'schema': 'dmv2-xom-bk-source-stage1-config/v1', 'plan_sha256': m.sha(plan_path),
                'worker_sha256': m.sha(HERE / 'read_stage1.py'), 'protocol_sha256': m.sha(protocol)})
            receipt = m.run(plan_path, config, top / 'result')
            self.assertEqual(receipt['metadata_rows_converted'], 6)
            self.assertEqual(receipt['financial_columns_decoded'], [])
            self.assertFalse(receipt['canonical_mappings_promoted'])
            self.assertEqual(receipt['case_summary'][0]['candidate_native_ids'], ['native_XOM'])


if __name__ == '__main__':
    unittest.main()
