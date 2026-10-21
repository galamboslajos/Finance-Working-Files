"""Pinned range transport and a strict in-memory Parquet envelope."""
import hashlib
import io
import ssl
import subprocess
import urllib.error
import urllib.parse
import urllib.request

import pyarrow.parquet as pq


def need(test, message):
    if not test:
        raise ValueError(message)


def chunk_range(column):
    start = column['dictionary_page_offset']
    if start is None or start <= 0:
        start = column['data_page_offset']
    return int(start), int(column['compressed_bytes'])


class AuthenticatedRanges:
    def __init__(self, obj, *, max_bytes, max_single=16777216):
        self.obj, self.limit, self.max_single = obj, max_bytes, max_single
        self.total, self.ranges = 0, []
        command = ['/Users/lajosgalambos/google-cloud-sdk/bin/gcloud',
            '--configuration=deep-momentum-research', '--account=info@galamboscapital.com',
            '--project=trusty-agility-492109-v0', '--verbosity=error', 'auth', 'print-access-token']
        auth = subprocess.run(command, capture_output=True, text=True, timeout=30)
        need(auth.returncode == 0, 'Dedicated authentication failed; credentials not logged')
        self._token = auth.stdout.strip()
        need(bool(self._token) and '\n' not in self._token, 'Invalid authentication result; credentials not logged')
        bucket, key = obj['uri'][5:].split('/', 1)
        self.url = ('https://storage.googleapis.com/storage/v1/b/' + bucket + '/o/' +
            urllib.parse.quote(key, safe='') + '?alt=media&generation=' + str(obj['generation']))
        self.context = ssl.create_default_context(
            cafile='/Users/lajosgalambos/google-cloud-sdk/lib/third_party/certifi/cacert.pem')

    def fetch(self, start, length, label):
        need(0 < length <= self.max_single and 0 <= start < start + length <= self.obj['size_bytes'], 'Invalid exact column range')
        need(self.total + length <= self.limit, 'Frozen cumulative range payload exceeded')
        request = urllib.request.Request(self.url, headers={'Authorization': 'Bearer ' + self._token,
            'Range': f'bytes={start}-{start+length-1}'})
        try:
            with urllib.request.urlopen(request, context=self.context, timeout=60) as reply:
                need(reply.status == 206 and reply.headers.get('Content-Range') ==
                    f'bytes {start}-{start+length-1}/{self.obj["size_bytes"]}', 'Exact generation/range response mismatch')
                generation = reply.headers.get('x-goog-generation')
                need(generation is None or int(generation) == self.obj['generation'], 'Wrong immutable object generation')
                raw = reply.read(length + 1)
                need(len(raw) == length, 'Range response payload length mismatch')
        except urllib.error.HTTPError as error:
            raise ValueError(f'Authenticated range HTTP status{error.code}; credentials not logged') from None
        self.total += length
        self.ranges.append({'start': start, 'bytes': length, 'sha256': hashlib.sha256(raw).hexdigest(), 'label': label})
        return raw


class VirtualParquet(io.RawIOBase):
    """Missing data ranges fail; only opening footer padding may be zero bytes."""
    def __init__(self, size, footer, chunks):
        self.size, self.position, self.opening = size, 0, True
        self.blocks = [(0, b'PAR1'), (size-len(footer), footer), *chunks]

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        need(0 <= self.position <= self.size, 'Virtual seek outside pinned object')
        return self.position

    def read(self, length=-1):
        need(length >= 0, 'Unbounded virtual read forbidden')
        start = self.position; length = min(length, self.size-start)
        if not length: return b''
        end = start + length
        for offset, raw in self.blocks:
            if offset <= start and end <= offset + len(raw):
                self.position = end
                return raw[start-offset:end-offset]
        # Arrow may request a64KiB tail to locate the footer. Outside the footer,
        # padding is never consumed as column values; later missing reads fail.
        if self.opening and start >= max(0, self.size-65536) and end == self.size:
            result = bytearray(length)
            for offset, raw in self.blocks:
                lo, hi = max(start, offset), min(end, offset+len(raw))
                if lo < hi: result[lo-start:hi-start] = raw[lo-offset:hi-offset]
            self.position = end
            return bytes(result)
        raise ValueError('Unmapped column data read denied')

    def readinto(self, buffer):
        raw = self.read(len(buffer)); buffer[:len(raw)] = raw
        return len(raw)


def load_group(footer, obj, group, columns, transport):
    chunks = []
    for column in columns:
        start, length = chunk_range(group['column_chunks'][column])
        chunks.append((start, transport.fetch(start, length, f'rg{group["index"]}/{column}')))
    source = VirtualParquet(obj['size_bytes'], footer, chunks)
    pf = pq.ParquetFile(source, pre_buffer=False)
    source.opening = False
    result = pf.read_row_group(group['index'], columns=columns, use_threads=False)
    need(result.num_rows == group['rows'], 'Row-group source count changed')
    return result
