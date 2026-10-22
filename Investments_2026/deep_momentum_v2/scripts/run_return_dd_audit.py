#!/usr/bin/env python3
"""Explicit metadata planning or one immutable return/DD audit materialization."""
import argparse
import json
from dmv2.return_dd_audit import plan_audit, run_audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('plan')
    plan.add_argument('--source-audit', required=True)
    plan.add_argument('--output-directory', required=True)
    plan.add_argument('--plan', required=True)
    plan.add_argument('--protocol', required=True)
    plan.add_argument('--source-ceiling-mib', type=int, choices=(64, 128), default=64)
    run = sub.add_parser('run'); run.add_argument('--plan', required=True)
    args = parser.parse_args()
    if args.action == 'plan':
        result = plan_audit(args.source_audit, args.output_directory, args.plan,
            protocol_path=args.protocol, source_projection_bytes=args.source_ceiling_mib * 1024**2)
        print(json.dumps({'status': result['status'], 'plan_sha256': result['plan_sha256'],
            'source_projection_bytes': result['source_projection_bytes'], 'months': len(result['months'])}))
    else:
        result = run_audit(args.plan)
        print(json.dumps({k: result[k] for k in ('status', 'months', 'original_rows', 'direction_rows',
            'active_seconds', 'peak_rss_bytes')}))


if __name__ == '__main__':
    main()
