"""Stdlib-only separate backtest protocol; existing model protocol stays immutable."""
from pathlib import Path
import argparse
import copy
import json
from control import ROOT,CAPS,check,pin,read,save_new,need


def make_protocol(model_protocol,source_plan,output):
    model=read(model_protocol);source=read(source_plan)
    need(source['schema']=='dmv2-corrected-forecasts-backtest-source-plan/v1'
         and source['model_protocol_pin']==pin(model_protocol) and source['runtime']==model['runtime']
         and source['scenario_count']==120,'Exact corrected forecast/source/runtime plan required')
    need(model['status']=='SEALED_BEFORE_FINANCIAL_READ' and model['parameter_search'] is False,'Frozen model authority required')
    for key in ('final_model_verification_pin','input_verification_pin','terminal_guard_pin'):check(source[key])
    limits=copy.deepcopy(model['limits'])
    need(set(limits['active_seconds_by_category'])=={'models','verification','repairs','rebuild','backtest'},'Shared active-time categories changed')
    limits.update(rss_warning_bytes=3*1024**3,rss_stop_bytes=4*1024**3,
        available_proxy_minimum_bytes=7*1024**3,output_maximum_bytes=20*1024**3,max_fits=0)
    limits['active_seconds_by_category']['backtest']=7200
    return save_new(output,{'schema':'dmv2-corrected-forecasts-backtest-protocol/v1',
        'status':'SEALED_BEFORE_FINANCIAL_READ','source_plan_pin':pin(source_plan),
        'model_protocol_pin':pin(model_protocol),'python_path':model['python_path'],
        'runtime':model['runtime'],'limits':limits,'shared_read_caps':CAPS,
        'maximum_backtest_output_bytes':1024**3,'scenarios':120,'selection_sets':15,
        'outcome_months':48,'sessions':1006,'allow_withheld_paths':True,
        'financial_2024_plus_authorized':False,'new_estimator_fits':0,'new_inference_calls':0,
        'changed_portfolio_rules':False,'changed_stop_or_cost_rules':False,
        'old_execution_repairs_and_events_unchanged':True,'builder_pin':pin(__file__),
        'phase_maximum_seconds':{'selection':600,'preflight':600,'replay':1800,'verification':3600},
        'instructions':'Use the existing supervisor and backtest active-time category; original model/source protocols and checkpoints remain immutable.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model-protocol',required=True);p.add_argument('--source-plan',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(make_protocol(a.model_protocol,a.source_plan,a.output)))
