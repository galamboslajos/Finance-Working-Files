"""Stdlib-only exact backtest control builder; no metric JSON or table decoding."""
from pathlib import Path
import argparse
import json
from control import ROOT,check,pin,read,save_new,need
from backtest_integration import ARMS,BASKETS,YEARS

# Previously independently checked execution reference; hashes are frozen evidence,
# never discovered by parsing performance/metric JSON during control construction.
EXECUTION_REFERENCE={
 'execution_verification_pin':('outputs/ret_repair_backtest/2020-2023-20261020-v1/verification/ret_repair_backtest.json','8253c45415a9766172c6955b58906c6d5762e53853e50ab50048996259ebf3a0',176259),
 'execution_phase_pin':('outputs/ret_repair_backtest/2020-2023-20261020-v1/backtest/build_identity.json','7e7b771d601d5a223d0240423e2980ffc98a81ca4ff3191d7ab8df9792d09bb1',458456),
 'execution_backtest_audit_pin':('outputs/ret_repair_backtest/2020-2023-20261020-v1/backtest/audit.json','85396c655ae5577f5af2dcfdd4079cedc0363ccd128addc89f936f4670098627',110290),
 'execution_manifest_pin':('data/derived/ret_repairs/20261020-v1/repair_manifest.json','7474b3c62a19d636bcfaea05d27553315721cb0fd65ce1301ac572440ba8b9fb',43533),
 'base_configuration_pin':('configs/walkforward_2020_2023.json','3e37a9f616d3c27ed83682c0087ca15db117bc272409b474fa5ea0ecbabc5d11',17923)}


def make_control(project,model_protocol,output):
    project=Path(project).resolve();protocol_path=Path(model_protocol).resolve();p=read(protocol_path)
    need(project.name=='deep_momentum_v2' and protocol_path.is_relative_to(ROOT),'Exact V2/local protocol required')
    need(p['schema']=='dmv2-corrected-refit-protocol/v1' and p['status']=='SEALED_BEFORE_FINANCIAL_READ'
         and p['outcome_years']==list(YEARS) and p['parameter_search'] is False,'Frozen20-fit source protocol required')
    run=Path(p['run_directory']).resolve();need(run.is_relative_to(ROOT/'run'),'Model run outside approved staging')
    # Pipeline progress/identity contain only control metadata, not model scores.
    progress_pin=pin(run/'pipeline/progress.json');progress=read(progress_pin['path'])
    identity_pin=pin(run/'pipeline/identity.json');identity=read(identity_pin['path'])
    need(progress['schema']=='dmv2-corrected-refit-pipeline-progress/v1'
         and progress['status']=='COMPLETE_VERIFIED' and progress['identity_pin']==identity_pin
         and identity['protocol_pin']==pin(protocol_path),'All50model phases must finish independently verified')
    final_pin=pin(run.parent/'verification/models.json')
    need(progress['final_verification_pin']==final_pin,'Pipeline final verification path/pin changed')
    entries=[]
    for year in YEARS:
        for arm in ARMS:
            directory=run/'folds'/str(year)/'arms'/arm
            entries.append({'year':year,'arm':arm,'predictions_pin':pin(directory/'predictions.parquet'),
                            'prediction_receipt_pin':pin(directory/'PREDICT_COMPLETE.json')})
    existing={}
    for name,(relative,sha,size) in EXECUTION_REFERENCE.items():
        expected={'path':str(project/relative),'sha256':sha,'size_bytes':size};check(expected);existing[name]=expected
    control={'schema':'dmv2-corrected-forecasts-backtest-control/v1',
        'status':'FROZEN_AFTER_TWENTY_MODEL_VERIFICATIONS','project_root':str(project),
        'model_protocol_pin':pin(protocol_path),'model_pipeline_identity_pin':identity_pin,
        'model_pipeline_progress_pin':progress_pin,'catalog_pin':p['catalog_pin'],
        'model_verification_pins':{str(y):pin(run.parent/'verification'/f'models_{y}.json') for y in YEARS},
        'final_model_verification_pin':final_pin,'forecast_entries':entries,
        **existing,'arm_ids':list(ARMS),'baskets':list(BASKETS),'scenarios':120,
        'RET_source_policy':'48corrected_catalog_states_120completed_calendar_months',
        'financial_2024_plus_authorized':False,'new_estimator_fits':0,'new_inference_calls':0,
        'new_postselection_hindsight_exclusions':False,'old_selection_reuse':False,
        'code_builder_pin':pin(__file__),'runtime':p['runtime'],
        'construction_reads':{'metric_JSON_parsed':False,'source_or_forecast_columns_decoded':False,
            'existing_reference_files':'raw SHA/size checks only','numeric_libraries_imported':False}}
    check(progress_pin);check(identity_pin);check(pin(protocol_path));check(p['catalog_pin'])
    return save_new(output,control)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project-root',required=True)
    parser.add_argument('--model-protocol',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();print(json.dumps(make_control(args.project_root,args.model_protocol,args.output)))
