import io
import struct
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from range_source import VirtualParquet, chunk_range, load_group
from read_sip_keys import select_keys


PLAN = {'security_id': 'source_xom', 'start': '2016-01-04', 'end': '2023-12-31', 'query_symbol': 'XOM'}


class KeyTests(unittest.TestCase):
    def table(self, **changes):
        rows = [dict(security_id='source_other', trading_date='2021-01-04', query_symbol='OTHER', adjustment='all'),
                dict(security_id='source_xom', trading_date='2021-01-04', query_symbol='XOM', adjustment='raw'),
                dict(security_id='source_xom', trading_date='2021-01-04', query_symbol='XOM', adjustment='all'),
                dict(security_id='source_xom', trading_date='2024-01-02', query_symbol='XOM', adjustment='all')]
        rows[1].update(changes)
        return pa.Table.from_pylist(rows)

    def test_exact_keys_keep_positions_and_exclude_future(self):
        selected, indices = select_keys(self.table(), PLAN)
        self.assertEqual(indices.to_pylist(), [1, 2])
        self.assertEqual(selected.num_rows, 2)

    def test_null_date_cannot_vanish(self):
        with self.assertRaisesRegex(ValueError, 'Null key'):
            select_keys(self.table(trading_date=None), PLAN)

    def test_null_symbol_cannot_pass_arrow_all(self):
        with self.assertRaisesRegex(ValueError, 'Null key'):
            select_keys(self.table(query_symbol=None), PLAN)

    def test_null_adjustment_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Null key'):
            select_keys(self.table(adjustment=None), PLAN)

    def test_wrong_symbol_rejected(self):
        with self.assertRaisesRegex(ValueError, 'conflicting'):
            select_keys(self.table(query_symbol='BAD'), PLAN)

    def test_wrong_adjustment_rejected(self):
        with self.assertRaisesRegex(ValueError, 'adjustment'):
            select_keys(self.table(adjustment='split'), PLAN)

    def test_malformed_date_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Noncanonical'):
            select_keys(self.table(trading_date='2021-1-4'), PLAN)

    def test_invalid_calendar_date_rejected(self):
        with self.assertRaises((ValueError, pa.ArrowInvalid)):
            select_keys(self.table(trading_date='2021-02-31'), PLAN)

    def test_unrelated_null_keys_do_not_admit_other_stock(self):
        rows = self.table().to_pylist()
        rows[0]['trading_date'] = None
        self.assertEqual(select_keys(pa.Table.from_pylist(rows), PLAN)[0].num_rows, 2)


class RangeTests(unittest.TestCase):
    def fixture(self):
        table = pa.table({'security_id': ['source_xom']*12, 'trading_date': ['2021-01-04']*12,
                          'query_symbol': ['XOM']*12, 'adjustment': ['all']*12,
                          'open': [1.25]*12})
        sink = io.BytesIO()
        pq.write_table(table, sink, row_group_size=4, compression='snappy')
        raw = sink.getvalue()
        footer_len = struct.unpack('<I', raw[-8:-4])[0]
        footer = raw[-footer_len-8:]
        pf = pq.ParquetFile(io.BytesIO(raw))
        groups = []
        for index in range(pf.metadata.num_row_groups):
            rg = pf.metadata.row_group(index)
            chunks = {}
            for ci in range(rg.num_columns):
                c = rg.column(ci)
                chunks[c.path_in_schema] = {'dictionary_page_offset': c.dictionary_page_offset,
                    'data_page_offset': c.data_page_offset, 'compressed_bytes': c.total_compressed_size}
            groups.append({'index': index, 'rows': rg.num_rows, 'column_chunks': chunks})
        return table, raw, footer, groups

    def test_exact_selected_chunks_reconstruct_rowgroup_without_financial(self):
        expected, raw, footer, groups = self.fixture()
        class FakeTransport:
            def __init__(self): self.labels = []
            def fetch(self, start, length, label):
                self.labels.append(label)
                return raw[start:start+length]
        transport = FakeTransport()
        cols = ['security_id', 'trading_date', 'query_symbol', 'adjustment']
        result = load_group(footer, {'size_bytes': len(raw)}, groups[1], cols, transport)
        self.assertEqual(result.to_pylist(), expected.slice(4, 4).select(cols).to_pylist())
        self.assertEqual(transport.labels, ['rg1/'+c for c in cols])
        self.assertNotIn('open', result.column_names)

    def test_missing_column_ranges_raise_after_footer_open(self):
        _, raw, footer, groups = self.fixture()
        view = VirtualParquet(len(raw), footer, [])
        pq.ParquetFile(view, pre_buffer=False)
        view.opening = False
        start, count = chunk_range(groups[0]['column_chunks']['open'])
        view.seek(start)
        with self.assertRaisesRegex(ValueError, 'Unmapped'):
            view.read(count)

    def test_range_seek_and_unbounded_read_guard(self):
        view = VirtualParquet(100, b'12345678', [])
        with self.assertRaisesRegex(ValueError, 'outside'):
            view.seek(-1)
        view.seek(0)
        with self.assertRaisesRegex(ValueError, 'Unbounded'):
            view.read()

    def test_dictionary_or_data_page_offsets(self):
        self.assertEqual(chunk_range({'dictionary_page_offset': 12, 'data_page_offset': 19, 'compressed_bytes': 4}), (12,4))
        self.assertEqual(chunk_range({'dictionary_page_offset': None, 'data_page_offset': 19, 'compressed_bytes': 4}), (19,4))


if __name__ == '__main__': unittest.main()
