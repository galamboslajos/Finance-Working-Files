#!/usr/bin/env python3
"""Explicit RF-expanded tuning lifecycle; no notebook or import-time execution."""
from __future__ import annotations
import argparse
import json
import signal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from dmv2.rf_tuning_control import TuningControlError, prepare_run, request_pause, run_attempt, status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prepare = sub.add_parser("prepare", help="Verify metadata pins and create a fresh private attempt; zero fits")
    prepare.add_argument("--project-root", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--amendment", type=Path, required=True)
    for action in ("run", "resume", "status", "pause", "interrupt"):
        command = sub.add_parser(action)
        command.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action in {"run", "resume"}:
        def terminate_gracefully(signum, frame):
            raise KeyboardInterrupt("Supervisor terminated; child interrupted with no automatic retry")
        signal.signal(signal.SIGTERM, terminate_gracefully)
    try:
        if args.action == "prepare":
            value = {"run_root": str(prepare_run(args.project_root, args.protocol, args.amendment)), "status": "PREPARED"}
        elif args.action in {"run", "resume"}:
            run_attempt(args.run_root, resume=args.action == "resume")
            value = status(args.run_root)
        elif args.action == "status":
            value = status(args.run_root)
        else:
            value = request_pause(args.run_root, immediate=args.action == "interrupt")
        print(json.dumps(value, sort_keys=True, indent=2))
        return 0
    except TuningControlError as exc:
        print(json.dumps({"status": "REFUSED", "reason": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
