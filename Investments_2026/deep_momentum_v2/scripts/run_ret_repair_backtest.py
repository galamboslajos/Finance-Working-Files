"""Explicit prepare/preflight/replay for an additive, zero-fit RET repair."""
import argparse
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('prepare', 'preflight', 'backtest'))
    for name in ('root', 'configuration', 'output-directory', 'run'):
        parser.add_argument('--' + name, type=Path, required=name == 'root')
    parser.add_argument('--allow-withheld-paths', action='store_true'); args = parser.parse_args(argv)
    root = args.root.resolve()
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    sys.path.insert(0, str(root / 'src'))
    from dmv2.ret_repair_backtest import prepare, preflight, backtest
    if args.phase == 'prepare':
        if args.configuration is None or args.output_directory is None or args.run is not None or args.allow_withheld_paths:
            parser.error('Prepare requires configuration and fresh output only')
        result = prepare(root, configuration_path=(root / args.configuration).resolve(), output_directory=(root / args.output_directory).resolve())
    else:
        if args.run is None or args.configuration is not None or args.output_directory is not None:
            parser.error('Preflight/replay requires only the exact prepared run')
        if args.phase == 'preflight' and args.allow_withheld_paths: parser.error('Withheld review belongs after preflight')
        result = (preflight(root, (root / args.run).resolve()) if args.phase == 'preflight' else
            backtest(root, (root / args.run).resolve(), allow_withheld_paths=args.allow_withheld_paths))
    print(json.dumps({'phase': args.phase, 'output': str(result), 'new_estimator_fits': 0, 'new_inference_calls': 0}))


if __name__ == '__main__': main()
