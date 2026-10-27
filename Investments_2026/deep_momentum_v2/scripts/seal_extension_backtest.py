#!/usr/bin/env python3
"""Metadata-only planner for fresh bounded realization attempts.

The original sealed controller's plan subcommand writes a valid plan and then
references an obsolete diagnostic key. Preserve its bytes for the existing run;
this additive planner reports the correct reservation key and starts no replay.
"""
import argparse,json
from pathlib import Path


def main():
    from dmv2.extension_backtest import seal
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--protocol',required=True);p.add_argument('--output',required=True);p.add_argument('--plan',required=True);a=p.parse_args()
    plan=seal(Path(__file__).resolve().parents[1],a.data_root,a.protocol,a.output,a.plan)
    print(json.dumps(dict(status=plan['status'],scenarios=48,new_fits=0,source_read_bytes_reserved=plan['source_read_bytes_reserved'])))


if __name__=='__main__':main()
