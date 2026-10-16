"""Explicit passive rendering of completed, independently verified saved reports."""
import argparse
from pathlib import Path
import sys
p=argparse.ArgumentParser(description=__doc__)
for name in ['root','bindings','output','old-snapshot']:
 p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args()
sys.path.insert(0,str(a.root.resolve()/'src'))
from dmv2.source_corrected_reporting import render
render(a.root.resolve(),a.bindings.resolve(),a.output.resolve(),a.old_snapshot.resolve())
