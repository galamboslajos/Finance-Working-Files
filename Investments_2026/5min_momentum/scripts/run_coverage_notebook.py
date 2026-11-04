"""Execute the coverage notebook and save outputs only in the private data directory."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / 'data/reports/01_load_sp500_5min'
PRIVATE.mkdir(parents=True, exist_ok=True)
for name, relative in [('IPYTHONDIR', '.ipython'), ('MPLCONFIGDIR', '.matplotlib'),
                       ('JUPYTER_RUNTIME_DIR', '.jupyter_runtime')]:
    folder = PRIVATE / relative
    folder.mkdir(exist_ok=True)
    os.environ[name] = str(folder)

import nbformat
from nbclient import NotebookClient

source = ROOT / '01_load_sp500_5min.ipynb'
nb = nbformat.read(source, as_version=4)
destination = PRIVATE / '01_load_sp500_5min.executed.ipynb'


def progress(cell, cell_index, **kwargs):
    if cell.cell_type == 'code':
        print(f'Executing cell {cell_index + 1}/{len(nb.cells)}: {cell.source.splitlines()[0]}', flush=True)


client = NotebookClient(nb, timeout=600, kernel_name='python3',
                        resources={'metadata': {'path': str(ROOT)}},
                        on_cell_start=progress, record_timing=True)
try:
    client.execute()
except Exception:
    failed = PRIVATE / '01_load_sp500_5min.failed.ipynb'
    nbformat.write(nb, failed)
    raise
nbformat.write(nb, destination)
assert not any(o.get('output_type') == 'error' for c in nb.cells
               if c.cell_type == 'code' for o in c.get('outputs', []))
print('Executed successfully:', destination, flush=True)
print((PRIVATE / 'SUMMARY.json').read_text(), flush=True)
