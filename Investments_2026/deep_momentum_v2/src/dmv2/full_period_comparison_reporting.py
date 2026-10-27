"""Passive JSON reader for the distinct monthly-engine/daily-engine report."""
from pathlib import Path
import hashlib,json


def read_full_period_comparison(directory):
    root=Path(directory).resolve()
    def read(path):
        p=Path(path)
        if p.stat().st_size>2*1024**2:raise ValueError('Small display JSON required')
        return json.loads(p.read_text())
    def check(item):
        p=Path(item['path'])
        if not p.is_absolute() or item['size_bytes']>8*1024**2 or p.stat().st_size!=item['size_bytes'] or hashlib.sha256(p.read_bytes()).hexdigest()!=item['sha256']:raise ValueError('Changed verified display artifact')
        return p
    proof=read(root/'verification.json')
    if proof['status']!='PASS_VERIFIED_MONTHLY_AND_DAILY_DISPLAY' or proof['monthly_tables_verified']!=16 or proof['new_fits']!=0 or proof['new_backtests']!=0:raise ValueError('Display proof incomplete')
    independent=read(check(proof['independent_proof_pin']))
    if independent['status']!='PASS' or independent['complete_paths']!=48 or independent['withheld_paths']!=0 or independent['monthly_references_verified']!=16:raise ValueError('Independent full comparison missing')
    check(proof['worker_pin'])
    for item in proof['source_pins'].values():check(item)
    value=read(check(proof['summary_pin']))
    if value['status']!='VERIFIED_MONTHLY_DAILY_DISPLAY' or value['months']!=75 or value['sessions']!=1569 or len(value['comparison'])!=4:raise ValueError('Display period/grid mismatch')
    return value
