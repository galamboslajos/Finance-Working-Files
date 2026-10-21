import importlib.util
import io
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('xom_footer', HERE / 'inspect_sip_footer.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def fixture():
    table = pa.Table.from_pydict({'security_id': ['target', 'target', 'zzz'],
        'trading_date': ['2020-01-02', '2024-01-02', '2020-01-02'],
        'query_symbol': ['XOM', 'XOM', 'ZZZ'], 'adjustment': ['raw', 'raw', 'raw'],
        'open': [9876543.21, 9876543.21, 9876543.21],
        'high': [9876543.21, 9876543.21, 9876543.21],
        'low': [9876543.21, 9876543.21, 9876543.21],
        'close': [9876543.21, 9876543.21, 9876543.21], 'volume': [9, 9, 9]})
    buf = io.BytesIO(); pq.write_table(table, buf, row_group_size=1)
    raw = buf.getvalue(); length = struct.unpack('<I', raw[-8:-4])[0]
    footer = raw[-(length + 8):]
    plan = {'object': {'declared_rows': 3}, 'expected_columns': table.column_names,
        'security_id': 'target', 'start': '2016-01-04', 'end': '2023-12-31',
        'metadata_columns': ['security_id', 'trading_date', 'query_symbol', 'adjustment'],
        'financial_cost_columns': ['open', 'high', 'low', 'close', 'volume']}
    return footer, plan


class FooterOnlyTests(unittest.TestCase):
    def test_metadata_source_date_statistics_select_exact_candidate_group(self):
        footer, plan = fixture(); result = m.inspect_footer(footer, plan)
        self.assertEqual(result['candidate_row_groups'], [0])
        self.assertEqual(result['rows'], 3)

    def test_financial_statistics_and_values_never_exposed(self):
        footer, plan = fixture(); result = m.inspect_footer(footer, plan)
        self.assertNotIn('9876543', json.dumps(result))
        self.assertFalse(result['financial_min_max_statistics_accessed'])
        self.assertFalse(result['column_values_decoded'])
        for group in result['row_groups']:
            self.assertEqual(set(group['safe_metadata_statistics']), {'security_id', 'trading_date'})

    def test_compressed_costs_present_without_column_data_access(self):
        footer, plan = fixture(); result = m.inspect_footer(footer, plan)
        self.assertGreater(result['candidate_metadata_projection_bytes'], 0)
        self.assertGreater(result['candidate_ohlcv_projection_bytes_cost_only'], 0)

    def test_invalid_footer_magic_rejected(self):
        footer, plan = fixture()
        with self.assertRaisesRegex(ValueError, 'magic'):
            m.inspect_footer(footer[:-4] + b'NOPE', plan)

    def test_invalid_footer_length_rejected(self):
        footer, plan = fixture()
        with self.assertRaisesRegex(ValueError, 'length'):
            m.inspect_footer(b'X' + footer, plan)

    def test_manifest_schema_mismatch_rejected(self):
        footer, plan = fixture(); plan['expected_columns'] = ['wrong']
        with self.assertRaisesRegex(ValueError, 'schema'):
            m.inspect_footer(footer, plan)

    def test_cli_stdout_is_bounded_during_execution(self):
        with self.assertRaisesRegex(ValueError, 'payload bound'):
            m.bounded_cli([sys.executable, '-c', "import sys;sys.stdout.write('x'*10000)"], 8)

    def test_cli_failure_does_not_reveal_raw_stderr(self):
        with self.assertRaises(ValueError) as ctx:
            m.bounded_cli([sys.executable, '-c', "import sys;sys.stderr.write('dummy_secret_text');sys.exit(1)"], 8)
        self.assertNotIn('dummy_secret_text', str(ctx.exception))
        self.assertIn('raw stderr not logged', str(ctx.exception))

    def test_changed_pin_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'x'; p.write_text('one'); item = m.pin(p); p.write_text('two')
            with self.assertRaisesRegex(ValueError, 'Pinned'):
                m.check(item)


if __name__ == '__main__':
    unittest.main()
