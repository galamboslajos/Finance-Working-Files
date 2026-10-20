"""Prepare a receipt-bound notebook candidate or private saved-result report.

No model, matrix, price panel or numerical workflow is opened. Notebook mode
checks audit JSON only, writes a separate candidate plus an installation plan,
and never edits the installed notebook. Report mode uses the passive reader
only after the same completed independent-receipt gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import sys

YEARS=(2020,2021,2022,2023)
ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
RECEIPT_FILES={'feature_verification':'features.json','model_verification':'models.json',
    'backtest_verification':'backtest.json'}


def require(ok,message):
    if not ok:raise AssertionError(message)


def digest(path):
    state=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):state.update(block)
    return state.hexdigest()


def canonical(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return 'sha256:'+hashlib.sha256(raw.encode()).hexdigest()


def json_file(path):
    path=Path(path); require(path.stat().st_size<=16*1024**2,'Unbounded handoff JSON')
    return json.loads(path.read_text())


def write_json(path,value):
    require(not path.exists(),'Preserve an existing handoff artifact: '+str(path))
    path.write_text(json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def within(base,path):
    path=Path(path).resolve(); require(path.is_relative_to(Path(base).resolve()),'Handoff input leaves its declared directory')
    return path


def final_bindings(root,run,expected_identity,*,verification_directory=None):
    """Require all three completed audits and all three PASS receipts first.

    Missing verification files stop before opening any saved run audit. This
    metadata-only gate does not import a reporting/model/feature worker or read
    Parquet. Raw hashes become literal notebook bindings only after the gate.
    """
    root=Path(root).resolve(); run=within(root/'outputs/walkforward',run)
    verification=run/'verification' if verification_directory is None else Path(verification_directory).resolve()
    paths={name:verification/filename for name,filename in RECEIPT_FILES.items()}
    for path in paths.values():
        require(path.is_file(),'All three completed independent verification receipts are required: '+str(path))
    receipts={name:json_file(path) for name,path in paths.items()}
    require(receipts['feature_verification'].get('status') in {'PASS','PASSED'}
        and receipts['model_verification'].get('status')=='PASSED'
        and receipts['backtest_verification'].get('status')=='PASS',
        'Notebook binding requires completed independent verification, including full model inference reconciliation')
    identity=json_file(run/'build_identity.json'); pin=canonical(identity)
    require(pin==expected_identity and run.name=='2020-2023-'+pin[7:23], 'Explicit walk-forward identity/path changed')
    require(all(receipt.get('build_identity_sha256')==pin for receipt in receipts.values()),
        'Independent receipts refer to different research builds')
    declared_backtest=receipts['backtest_verification'].get('backtest_directory')
    backtest=run/'backtest' if declared_backtest is None else within(run,root/declared_backtest)
    audit_paths={'features':run/'audit.json','models':run/'models/audit.json','backtest':backtest/'audit.json'}
    audits={name:json_file(path) for name,path in audit_paths.items()}
    require(all(audit.get('status')=='COMPLETE' and audit.get('build_identity_sha256')==pin for audit in audits.values()),
        'Research audit is incomplete or changed')
    require(set(audits['features'].get('years',{}))=={str(y) for y in YEARS}
        and audits['models'].get('summary',{}).get('fitted_arms')==16
        and audits['models']['summary'].get('preprocessor_fits')==8
        and audits['models']['summary'].get('months')==48
        and audits['backtest'].get('scenarios')==64 and audits['backtest'].get('sessions')==1006
        and audits['backtest'].get('outcome_months')==48,'Research audit changed the declared fold/month/scenario grid')
    raw={name:digest(path) for name,path in audit_paths.items()}
    require(all(receipt.get('root_audit_sha256')==raw['features'] for receipt in receipts.values()),
        'Independent receipt does not bind the exact feature audit')
    model=receipts['model_verification']; bt=receipts['backtest_verification']; feature=receipts['feature_verification']
    require(model.get('models_audit_sha256')==raw['models']
        and model.get('model_identity_sha256')==audits['models'].get('model_identity_sha256')
        and model.get('annual_fits_reconciled')==16 and model.get('stitched_prediction_months')==48
        and model.get('models_fitted_by_verifier')==0 and model.get('preprocessors_fitted_by_verifier')==0
        and model.get('financial_2024_plus_decoded') is False,'Independent model receipt is partial or changed')
    require(bt.get('backtest_audit_sha256')==raw['backtest']
        and bt.get('backtest_phase_identity_sha256')==audits['backtest'].get('phase_identity_sha256')
        and bt.get('models_fitted')==0 and bt.get('preprocessors_fitted')==0
        and bt.get('financial_2024_plus_read') is False and bt.get('source_pins_unchanged') is True,
        'Independent backtest receipt is partial or changed')
    require(feature.get('models_fitted',feature.get('model_fits'))==0
        and feature.get('financial_2024_plus_read') is False,'Feature verification changed fit/protected scope')
    expected={name:'sha256:'+value for name,value in raw.items()}
    expected.update({name:'sha256:'+digest(path) for name,path in paths.items()})
    return {'root':root,'run':run,'identity_sha256':pin,'expected_receipts':expected,
        'verification_directory':verification,'backtest_directory':backtest,'audit_paths':audit_paths,
        'verification_paths':paths}


def _output_free(notebook):
    for cell in notebook.get('cells',[]):
        if cell['cell_type']=='code':
            require(cell.get('execution_count') is None and cell.get('outputs',[])==[],
                'Clear notebook outputs separately before preparing a preserved-cell handoff')


def bound_section(template,bindings):
    cells=deepcopy(json_file(template)); require(len(cells)==2,'Section 23 template must contain one markdown and one code cell')
    require(cells[0]['cell_type']=='markdown' and cells[1]['cell_type']=='code','Unexpected section 23 cell ordering')
    code=''.join(cells[1]['source']); replacements={'__PENDING_VALIDATED_RUN__':bindings['run'].name,
        '__PENDING_VALIDATED_IDENTITY__':bindings['identity_sha256'],
        '__PENDING_FEATURE_AUDIT_SHA256__':bindings['expected_receipts']['features'],
        '__PENDING_MODEL_AUDIT_SHA256__':bindings['expected_receipts']['models'],
        '__PENDING_BACKTEST_AUDIT_SHA256__':bindings['expected_receipts']['backtest'],
        '__PENDING_FEATURE_VERIFICATION_SHA256__':bindings['expected_receipts']['feature_verification'],
        '__PENDING_MODEL_VERIFICATION_SHA256__':bindings['expected_receipts']['model_verification'],
        '__PENDING_BACKTEST_VERIFICATION_SHA256__':bindings['expected_receipts']['backtest_verification']}
    for placeholder,value in replacements.items():
        require(code.count(placeholder)==1,'Section 23 template has a missing or ambiguous literal binding '+placeholder)
        code=code.replace(placeholder,value)
    if bindings['verification_directory']!=bindings['run']/'verification':
        old='expected_receipts=WALKFORWARD_RECEIPTS, verify_sources=False)'
        new='expected_receipts=WALKFORWARD_RECEIPTS, verification_directory='+repr(str(bindings['verification_directory']))+', verify_sources=False)'
        require(code.count(old)==1,'Cannot bind an explicit verification directory in section 23')
        code=code.replace(old,new)
    parsed=ast.parse(code)
    forbidden={'fit','partial_fit','fit_transform','load','load_model','run_walkforward_fold',
        'build_walkforward_panel','run_walkforward_backtest','finalize_walkforward_models'}
    for node in ast.walk(parsed):
        if isinstance(node,ast.Call):
            name=getattr(node.func,'id',getattr(node.func,'attr',''))
            require(name not in forbidden,'Section 23 template contains a numerical/model-loading call '+name)
    compile(code,'walkforward_saved_section 23','exec')
    cells[0]['id']='walkforward-23-explanation'; cells[1]['id']='walkforward-23-saved-reports'
    cells[1].update(source=code.splitlines(keepends=True),execution_count=None,outputs=[])
    return cells


def prepare_notebook(root,run,expected_identity,template,output_directory,*,verification_directory=None,notebook=None):
    """Write only a candidate and install plan outside the installed workspace."""
    bindings=final_bindings(root,run,expected_identity,verification_directory=verification_directory)
    root=bindings['root']; notebook=root/'notebooks/00_research_master.ipynb' if notebook is None else within(root,notebook)
    original_bytes=notebook.read_bytes(); original=json.loads(original_bytes); _output_free(original)
    require(not any('## 23.' in ''.join(c.get('source',[])) or 'read_walkforward_run' in ''.join(c.get('source',[]))
        for c in original['cells']),'Preserve an existing section 23; inspect its bindings before another handoff')
    candidate=deepcopy(original); prefix=deepcopy(original['cells'])
    candidate['cells'].extend(bound_section(template,bindings)); _output_free(candidate)
    require(candidate['cells'][:len(prefix)]==prefix,'Preparing section 23 changed an earlier notebook cell')
    output=Path(output_directory).resolve()
    require(not output.is_relative_to(root),'Notebook preparation must write to a separate scratch directory')
    require(not output.exists(),'Preserve an existing notebook handoff directory')
    output.mkdir(parents=True)
    path=output/notebook.name; write_json(path,candidate)
    snapshot=output/'original_notebook.ipynb'; snapshot.write_bytes(original_bytes)
    plan={'status':'READY_FOR_SCOPED_INSTALL','original_notebook':str(notebook),
        'original_notebook_sha256':hashlib.sha256(original_bytes).hexdigest(),
        'private_original_snapshot':str(snapshot),
        'candidate_notebook':str(path),'candidate_notebook_sha256':digest(path),
        'original_cell_count':len(prefix),'candidate_cell_count':len(candidate['cells']),
        'previous_cells_preserved':True,'all_outputs_empty':True,'all_execution_counts_null':True,
        'build_identity_sha256':bindings['identity_sha256'],'run_directory':str(bindings['run']),
        'verification_directory':str(bindings['verification_directory']),
        'expected_receipts':bindings['expected_receipts'],'section_template_sha256':digest(template),
        'preparation_script_sha256':digest(Path(__file__).resolve()),'installed_notebook_changed':False,
        'models_loaded':False,'matrices_loaded':False,'reports_or_source_prices_decoded':False,
        'remaining_actions':['Before installation verify the original notebook still matches original_notebook_sha256.',
            'Install only the candidate notebook with the scoped workspace writer; preserve the private original snapshot.',
            'Run the added passive report cell in a fresh kernel with model/matrix/build calls blocked.',
            'Keep the committed notebook output-free; save execution evidence only in ignored private outputs.']}
    require(notebook.read_bytes()==original_bytes,'Installed notebook changed during preparation')
    write_json(output/'install_plan.json',plan)
    return plan


def _markdown(frame,columns,*,percent=()):
    frame=frame[[name for name in columns if name in frame]].copy()
    def value(item,name):
        import pandas as pd
        if pd.isna(item):return 'unavailable'
        if name in percent:return f'{float(item):.2%}'
        if isinstance(item,float):return f'{item:.4f}'
        return str(item).replace('|','\\|').replace('\n',' ')
    lines=['| '+' | '.join(frame.columns)+' |','| '+' | '.join('---' for _ in frame.columns)+' |']
    lines.extend('| '+' | '.join(value(item,name) for name,item in zip(frame.columns,row,strict=True))+' |'
        for row in frame.itertuples(index=False,name=None))
    return '\n'.join(lines)


def prepare_report(root,run,expected_identity,output_directory,*,verification_directory=None):
    """Write one private Markdown/CSV snapshot from passive report tables only."""
    bindings=final_bindings(root,run,expected_identity,verification_directory=verification_directory)
    sys.path.insert(0,str(bindings['root']/'src'))
    from dmv2.walkforward_reporting import read_walkforward_run
    saved=read_walkforward_run(bindings['root'],bindings['run'],identity_sha256=bindings['identity_sha256'],
        expected_receipts=bindings['expected_receipts'],verification_directory=bindings['verification_directory'],verify_sources=False)
    # Export exact calendar metadata with the small tables so a separate
    # renderer needs no strategy worker, Parquet loader or calendar package.
    backtest_phase=json_file(bindings['backtest_directory']/'build_identity.json')
    model_phase=json_file(bindings['run']/'models/build_identity.json')
    sessions=backtest_phase['clock']['selection_sessions']
    observed=[day for day in sessions if '2020-01-01'<=day<='2023-12-31']
    require(len(observed)==1006 and observed==sorted(set(observed))
        and observed[0]=='2020-01-02' and observed[-1]=='2023-12-29',
        'Snapshot calendar differs from the exact continuous daily horizon')
    monthly_dates={month:max(day for day in observed if day[:7]==month)
        for month in saved['monthly_10bps'].outcome_month.unique()}
    require(len(monthly_dates)==48,'Snapshot calendar lost an outcome month')
    output=Path(output_directory).resolve(); root=bindings['root']
    require(not output.is_relative_to(root) or output.is_relative_to(bindings['run']/'reports'),
        'Generated results must remain in this run’s private reports or separate scratch')
    require(not output.exists(),'Preserve an existing private report snapshot'); output.mkdir(parents=True)
    tables={'model_comparison':saved['model_comparison'],'backtest_comparison':saved['backtest_comparison'],
        'backtest_yearly':saved['backtest_yearly'],'selection_coverage':saved['selection_coverage'],
        'monthly_10bps':saved['monthly_10bps'],'model_classes_1_2_9_10':saved['model_classes_1_2_9_10'],
        'feature_target_coverage':saved['year_summaries']}
    artifacts={}
    for name,frame in tables.items():
        path=output/(name+'.csv'); frame.to_csv(path,index=False)
        artifacts[path.name]={'sha256':digest(path),'size_bytes':path.stat().st_size}
    summary=saved['backtest_summary']; dev=saved['backtest_comparison'].loc[lambda frame:frame.cost_bps.eq(10)]
    annual=saved['backtest_yearly'].loc[lambda frame:frame.cost_bps.eq(10)]
    classes=saved['model_classes_1_2_9_10']
    class_table='No optional saved class-mode table is available.' if classes.empty else _markdown(
        classes.loc[classes.arm_id.isin(ARMS)],
        ['arm_id','class_id','precision','recall','actual_support','labelled_predicted_support','missing_label_predicted_support'],
        percent=['precision','recall'])
    text=['# Annual walk-forward research snapshot, 2020–2023','',
        '48 monthly prediction periods and 1,006 XNYS daily sessions: January 2, 2020–December 29, 2023. '
        'Each test year uses only its preceding ten completed outcome years, with four fixed estimator/input arms per year. '
        'The daily ledger carries NAV continuously across all four years; annual retraining adds no liquidation or capital reset.','',
        f"Audited paths: {summary['complete_proxy_paths']} complete proxies and {summary['failed_closed_paths']} withheld out of 64. "
        'A withheld path retains its unsupported exposure and has unavailable full-period performance. '
        'PASS verification certifies audited behavior, not economic completion of every path.','',
        '## Model probabilities','',
        'Log loss and accuracy average monthly observed-label scores equally. Prediction rows with missing outcomes remain in the forecast cohort. '
        'Paper15 omits the paper’s size input by user direction; the expanded candidate bank has 715 features with training-only retention and imputation.','',
        _markdown(saved['model_comparison'],['arm_id','mean_monthly_log_loss','mean_monthly_accuracy','prediction_rows','labelled_rows'],percent=['mean_monthly_accuracy']),'',
        '## Separate class 1, 2, 9 and 10 diagnostics','',
        'Precision and recall below use all eligible class-mode forecasts with observed outcomes. '
        'They do not measure only the smaller selected trading baskets, and missing outcomes are not counted as correct or incorrect.','',
        class_table,'',
        '## Portfolio accounting at 10 bps per one-way traded notional','',
        'Class 1/10 and 2/9 baskets remain separate. Intended weights are 50% long/50% short with '
        'N=min(10,long candidates,short candidates) per side. Retired new orders and stopped names remain cash without replacements. '
        'Monthly predictions execute at the next session’s open. The 10% stop is per asset; gaps may exceed it.','',
        _markdown(dev,['arm','basket','variant','status','total_return','cagr','maximum_drawdown','annualized_volatility','sharpe_zero_cash_rate','trading_cost_paid'],
            percent=['total_return','cagr','maximum_drawdown','annualized_volatility']),'',
        '## Annual slices of the same continuous NAV','',
        'For complete paths, each annual initial NAV equals the preceding year’s final NAV. These are slices of one continuous strategy, '
        'rather than separately funded annual experiments. The CSV includes all 256 scenario/year rows; this table shows the 10 bps paths.','',
        _markdown(annual,['arm','basket','variant','year','initial_nav','final_nav','total_return'],percent=['total_return']),'',
        '## Original cohort and target coverage','',
        _markdown(saved['year_summaries'],['year','original_rows','prediction_eligible_rows','labelled_rows','feature_history_rows','separate_outcome_daily_rows']),'',
        '## Interpretation limits','',
        '- Historical identities remain provisional. The known wrong old-BBBY routes stay quarantined with original rows retained; this is not a claim of fully certified survivorship-free history.',
        '- DF/RDC trading restrictions are fixed retrospective coverage restrictions. They remain in valid training and prediction rows and can affect representativeness.',
        '- Vendor adjustment and source knowledge-time completeness remain uncertified. Monthly endpoint labels and daily legal-entitlement accounting serve different purposes.',
        '- Independent quote repairs preserve existing volume/action fields; appended unknown volume, split and dividend fields stay missing. No interpolation supplies missing prices.',
        '- Daily stops use an idealized adverse high/low threshold, with gap fills at the observed open. Borrow and funding charges are zero; actual execution/borrow availability is not certified.',
        '- All 64 preset cost/stop/basket/model paths are retained. No result-driven refitting, feature selection or new exclusion is implied. Protected 2024+ financial values remain outside this evaluation.',
        '- The feature verifier checks all 15 paper formulas and deterministic daily samples. Certification of every expanded formula remains incomplete.','',
        '## Exact evidence','',f"Build identity: `{bindings['identity_sha256']}`.",'',
        'The snapshot is bound to these raw audit and independent-verification receipts:','',
        *[f'- {name}: `{sha}`.' for name,sha in sorted(bindings['expected_receipts'].items())],'']
    path=output/'RESULTS.md'; path.write_text('\n'.join(text))
    artifacts[path.name]={'sha256':digest(path),'size_bytes':path.stat().st_size}
    result={'status':'SAVED_PASSIVE_REPORT_SNAPSHOT','build_identity_sha256':bindings['identity_sha256'],
        'run_directory':str(bindings['run']),'backtest_directory':str(bindings['backtest_directory']),
        'verification_directory':str(bindings['verification_directory']),
        'expected_receipts':bindings['expected_receipts'],'artifacts':artifacts,
        'plot_calendar':{'first_session':'2020-01-02','terminal_session':'2023-12-29',
            'initial_nav_anchor':'2019-12-31','sessions':1006,'months':48,'monthly_session_end':monthly_dates},
        'frozen_model_runtime':model_phase.get('runtime',{}),
        'generator_sha256':digest(Path(__file__).resolve()),'report_tables_only':True,
        'models_loaded':False,'matrices_loaded':False,'source_prices_decoded':False,'models_fitted':0}
    write_json(output/'report_receipt.json',result); return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('notebook','report')); parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--identity-sha256',required=True)
    parser.add_argument('--verification-directory',type=Path); parser.add_argument('--output-directory',type=Path,required=True)
    parser.add_argument('--section-template',type=Path)
    args=parser.parse_args(argv)
    if args.mode=='notebook':
        require(args.section_template is not None,'Notebook mode requires the reviewed section 23 JSON template')
        result=prepare_notebook(args.root,args.run,args.identity_sha256,args.section_template,args.output_directory,
            verification_directory=args.verification_directory)
    else:
        result=prepare_report(args.root,args.run,args.identity_sha256,args.output_directory,
            verification_directory=args.verification_directory)
    print(json.dumps({'status':result['status'],'build_identity_sha256':result['build_identity_sha256'],
        'output_directory':str(args.output_directory.resolve())},sort_keys=True))


if __name__=='__main__':main()
