"""Explicit zero-fit tuned-RF prepare/preflight/replay entry point."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'backtest'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--configuration', type=Path)
    parser.add_argument('--output-directory', type=Path)
    parser.add_argument('--run', type=Path)
    parser.add_argument('--allow-withheld-paths', action='store_true')
    args = parser.parse_args(argv)
    root = args.root.resolve()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(root / 'src'))
    from dmv2.rf_tuned_backtest import prepare, preflight, backtest
    if args.phase == 'prepare':
        if args.configuration is None or args.output_directory is None or args.run is not None or args.allow_withheld_paths:
            parser.error('Prepare requires configuration and fresh output only')
        output = prepare(root, configuration_path=(root / args.configuration).resolve(),
                         output_directory=(root / args.output_directory).resolve())
    else:
        if args.run is None or args.configuration is not None or args.output_directory is not None:
            parser.error('Preflight/backtest requires only the exact prepared run')
        if args.phase == 'preflight':
            if args.allow_withheld_paths: parser.error('Withheld-path approval belongs to backtest after preflight review')
            output = preflight(root, (root / args.run).resolve())
        else:
            output = backtest(root, (root / args.run).resolve(), allow_withheld_paths=args.allow_withheld_paths)
    print(json.dumps({'phase': args.phase, 'output': str(output), 'new_estimator_fits': 0,
                      'new_preprocessor_fits': 0, 'new_inference_calls': 0}, sort_keys=True))


if __name__ == '__main__':
    main()
