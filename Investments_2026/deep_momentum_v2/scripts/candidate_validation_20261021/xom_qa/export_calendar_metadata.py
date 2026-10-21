"""Write only independent XNYS date metadata and calendar implementation pins."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import exchange_calendars as xc

HERE = Path(__file__).resolve().parent
plan = json.loads((HERE / 'calendar_metadata_plan.json').read_text())
version = importlib.metadata.version(plan['calendar_package'])
assert version == plan['required_version'], 'Calendar package drift'
root = Path(xc.__file__).resolve().parent
pins = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob('*.py'))}
calendar = xc.get_calendar(plan['calendar'], start=plan['start'], end=plan['end'])
sessions = [stamp.date().isoformat()
            for stamp in calendar.sessions_in_range(plan['start'], plan['end'])]
assert sessions and sessions == sorted(set(sessions))
assert all(plan['start'] <= d <= plan['end'] for d in sessions)
out = {'schema': 'dmv2-XOM-independent-calendar-metadata/v1',
       'calendar': 'XNYS', 'start': plan['start'], 'end': plan['end'],
       'package': plan['calendar_package'], 'package_version': version,
       'package_code_sha256': hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest(),
       'package_source_hashes': pins, 'sessions': sessions,
       'session_count': len(sessions), 'source_quotes_read': False,
       'financial_values_read': False,
       'plan_sha256': hashlib.sha256((HERE / 'calendar_metadata_plan.json').read_bytes()).hexdigest(),
       'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
(HERE / 'xnys_sessions_2016_2023.json').write_text(json.dumps(out, sort_keys=True, indent=2) + '\n')
print(json.dumps({'package_version': version, 'session_count': len(sessions),
                  'first': sessions[0], 'last': sessions[-1],
                  'output_sha256': hashlib.sha256((HERE / 'xnys_sessions_2016_2023.json').read_bytes()).hexdigest()}))
