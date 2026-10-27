#!/usr/bin/env python3
"""Execute passive notebook cells without source reads, estimators or subprocesses."""
from pathlib import Path
import builtins,json,sys
import numpy as np
import pyarrow.parquet as pq
import IPython.display as ipd
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
nb=json.loads((ROOT/'notebooks/17_full_period_monthly_daily_report.ipynb').read_text())
outputs=[];namespace={};original=builtins.__import__
def imports(name,*args,**kwargs):
    if name.split('.')[0] in ('sklearn','xgboost','joblib'):raise AssertionError('Estimator import in passive notebook')
    return original(name,*args,**kwargs)
def forbid(*args,**kwargs):raise AssertionError('Source/array read or subprocess in passive notebook')
with patch('builtins.__import__',imports),patch.object(np,'load',forbid),patch.object(pq,'read_table',forbid),patch('subprocess.Popen',forbid),patch.object(ipd,'display',lambda value:outputs.append(type(value).__name__)):
    for cell in nb['cells']:
        if cell['cell_type']=='code':exec(compile(''.join(cell['source']),'notebook17','exec'),namespace)
assert namespace['report']['status']=='VERIFIED_MONTHLY_DAILY_DISPLAY' and namespace['saved']['status']=='COMPLETE_VERIFIED' and len(outputs)==6
assert all(not c.get('outputs') and c.get('execution_count') is None for c in nb['cells'] if c['cell_type']=='code')
print(json.dumps(dict(status='PASS_PASSIVE_RUNALL',code_cells=sum(c['cell_type']=='code' for c in nb['cells']),display_objects=outputs,estimator_imports=0,source_array_or_table_reads=0,subprocesses=0,committed_outputs_clear=True)))
