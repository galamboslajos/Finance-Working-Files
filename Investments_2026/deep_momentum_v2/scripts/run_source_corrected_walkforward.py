"""Explicit fixed-recipe corrected walk-forward; notebook imports never fit."""
import argparse
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['prepare','plan-models','fold','finalize-models'])
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--year',type=int,choices=[2020,2021,2022,2023])
    args=parser.parse_args();root=args.root.resolve();run=(root/args.run).resolve()
    if not run.is_relative_to(root/'outputs/walkforward_source_corrected'):
        parser.error('Use only a separate source-corrected private run')
    sys.path.insert(0,str(root/'src'))
    if args.phase=='prepare':
        from dmv2.source_corrected_replay import prepare_source_corrected_run
        directory,identity=prepare_source_corrected_run(root,run,approved_replay=True)
        print(json.dumps({'phase':'prepared_corrected_inputs','directory':str(directory),
                          'identity_sha256':identity,'model_fits':0}),flush=True)
        return
    from dmv2.catalog import canonical_sha256
    from dmv2.walkforward_models import plan_walkforward_fold,run_walkforward_fold,finalize_walkforward_models
    identity=json.loads((run/'build_identity.json').read_text());pin=canonical_sha256(identity)
    cfg=json.loads((root/'configs/walkforward_2020_2023.json').read_text())
    if cfg!=identity['configuration']:parser.error('Frozen configuration changed')
    if args.phase=='plan-models':
        plans={}
        for year in [2020,2021,2022,2023]:
            plan=plan_walkforward_fold(root,run,cfg,pin,year)
            p=plan['provenance']
            if (p['projected_compressed_bytes']>cfg['execution']['model_maximum_projected_read_bytes']
                    or p['estimated_peak_memory_bytes']>cfg['execution']['model_maximum_peak_rss_bytes']):
                parser.error('Annual fold exceeds the fixed resource ceilings')
            plans[str(year)]={'fingerprint':plan['fingerprint'],'provenance':p}
            print(json.dumps({'year':year,'projected_compressed_bytes':p['projected_compressed_bytes'],
                              'estimated_peak_memory_bytes':p['estimated_peak_memory_bytes'],
                              'model_fits':0}),flush=True)
        destination=run/'model_read_plan.json'
        if destination.exists():
            if json.loads(destination.read_text())!=plans:parser.error('Existing metadata plan changed')
        else:destination.write_text(json.dumps(plans,indent=2)+'\n')
    elif args.phase=='fold':
        if args.year is None:parser.error('--year required')
        plan=plan_walkforward_fold(root,run,cfg,pin,args.year)
        frozen=json.loads((run/'model_read_plan.json').read_text())[str(args.year)]
        if plan['fingerprint']!=frozen['fingerprint']:parser.error('Pre-fit metadata plan changed')
        result=run_walkforward_fold(root,run,cfg,pin,args.year)
        print(json.dumps({'phase':'fold_complete','year':args.year,'output':str(result)}),flush=True)
    else:
        result=finalize_walkforward_models(root,run,cfg,pin)
        print(json.dumps({'phase':'models_complete','output':str(result)}),flush=True)


if __name__=='__main__':main()
