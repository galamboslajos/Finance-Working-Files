"""Run only an additive, explicitly versioned accounting/preflight phase; zero fits."""
import argparse
import json
from pathlib import Path
import sys


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('preflight','backtest'))
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--identity-sha256',required=True)
    parser.add_argument('--backtest-output',type=Path,required=True)
    parser.add_argument('--failed-attempt',type=Path,required=True)
    parser.add_argument('--event-registry',type=Path)
    parser.add_argument('--identity-handover-registry',type=Path)
    parser.add_argument('--identity-handover-decision',type=Path)
    parser.add_argument('--allow-withheld-paths',action='store_true')
    args=parser.parse_args(argv)
    root=args.root.resolve(); run=(root/args.run).resolve()
    if not run.is_relative_to(root/'outputs/walkforward'):
        parser.error('Run must remain within the existing private walk-forward outputs')
    output=(root/args.backtest_output).resolve();prior=(root/args.failed_attempt).resolve()
    if (not output.is_relative_to(run) or not prior.is_relative_to(run)
            or output==run or prior==run or output.exists()):
        parser.error('Explicit existing failed attempt and new separate output are required inside the run')
    registry_paths=[]
    for value in (args.event_registry,args.identity_handover_registry,args.identity_handover_decision):
        path=None if value is None else (root/value).resolve()
        if path is not None and not path.is_relative_to(root/'data/derived'):
            parser.error('Execution registries must be private derived-data evidence')
        registry_paths.append(path)
    sys.path.insert(0,str(root/'src'))
    from dmv2.catalog import canonical_sha256
    cfg=json.loads((root/'configs/walkforward_2020_2023.json').read_text())
    identity=json.loads((run/'build_identity.json').read_text())
    if canonical_sha256(identity)!=args.identity_sha256 or identity.get('configuration')!=cfg:
        parser.error('Explicit parent identity and frozen configuration must match')
    from dmv2.walkforward_backtest_v2 import run_walkforward_backtest_v2
    result=run_walkforward_backtest_v2(root,run,cfg,args.identity_sha256,
        event_registry=registry_paths[0],identity_handover_registry=registry_paths[1],
        identity_handover_decision=registry_paths[2],
        output_directory=output,failed_attempt_directory=prior,
        preflight_only=args.phase=='preflight',allow_withheld_paths=args.allow_withheld_paths)
    print(json.dumps({'phase':args.phase,'implementation_revision':'identity_route_failure_v2',
        'output':str(result),'new_model_fits':0,'source_identity_promoted':False},sort_keys=True))


if __name__=='__main__':main()
