"""Passive final handoff from exact precharged verified JSON reports only."""
from pathlib import Path
import argparse,collections,json
import control as c
ROOT=c.ROOT
ARMS=['rf_paper15','xgb_paper15','rf_extended','xgb_extended','rf_tuned_extended']
INPUTS={'models':'run/verification/models.json','backtest':'run/verification/backtest.json','model_report':'run/reports/models_v1/comparison.json','model_receipt':'run/reports/models_v1/receipt.json','portfolio_report':'run/reports/backtest_v1/report.json','portfolio_receipt':'run/reports/backtest_v1/receipt.json'}
def pct(v):return '—' if v is None else f'{100*v:+.2f}%'
def num(v):return '—' if v is None else f'{v:.2f}'
def plan():
    files={k:c.pin(ROOT/v) for k,v in INPUTS.items()}; amount=sum(p['size_bytes'] for p in files.values())
    c.need(amount<=8*1024**2,'Handoff JSON read cap exceeded')
    return c.save_new(ROOT/'plans/root_final_handoff_v1.json',{'schema':'dmv2-passive-final-handoff-plan/v1','files':files,'worker_pin':c.pin(__file__),'projected_bytes':amount,'financial_table_reads':0,'model_deserializations':0,'maximum_output_bytes':2*1024**2,'purpose':'Copy verified saved metrics into final README and user summary; no new selection or estimation'})
def run():
    p=c.pin(ROOT/'plans/root_final_handoff_v1.json'); z=c.read(p['path']);c.check(z['worker_pin'])
    c.require_reservation(ROOT,'root_final_handoff_v1',z['projected_bytes'],p)
    c.need(not (ROOT/'README.md').exists() and not (ROOT/'run/reports/root_handoff_v1.json').exists(),'Fresh final handoff outputs required')
    started=c.save_new(ROOT/'run/reports/root_handoff_v1.STARTED.json',{'status':'ATTEMPT_CONSUMED','plan_pin':p,'reservation':'root_final_handoff_v1'})
    d={k:c.read(c.check(v)) for k,v in z['files'].items()}; m,b,r,q=(d[k] for k in ['models','backtest','model_report','portfolio_report'])
    c.need(m['status']=='PASS' and m['verified_estimator_fits']==20 and m['verified_training_only_preprocessing_states']==8,'Model proof mismatch')
    c.need(b['status']=='PASS' and b['scenarios']==120 and b['all_scenarios_retained'] and b['sessions']==1006 and b['outcome_months']==48,'Backtest proof mismatch')
    c.need(q['status']=='FROM_INDEPENDENTLY_VERIFIED_SAVED_OUTPUTS' and q['independent_verification_pin']==z['files']['backtest'] and q['complete_proxy_paths']==b['complete_proxy_paths'] and q['withheld_paths']==b['withheld_paths'],'Portfolio report detached')
    c.need(r['status']=='COMPLETE_FROM_INDEPENDENTLY_VERIFIED_OUTPUTS' and r['same_observed_keys_all_arms'] and r['final_verification_pin']==z['files']['models'],'Model report detached')
    mr,br=d['model_receipt'],d['portfolio_receipt']
    c.need(mr['status']=='COMPLETE' and mr['comparison_pin']==z['files']['model_report'] and mr['final_verification_pin']==z['files']['models'],'Model receipt detached')
    c.need(br['status']=='COMPLETE_FROM_INDEPENDENTLY_VERIFIED_OUTPUTS' and br['artifacts']['report']==z['files']['portfolio_report'] and br['independent_verification_pin']==z['files']['backtest'],'Portfolio receipt detached')
    c.check(br['artifacts']['markdown']);c.check(mr['markdown_pin'])
    rows={(a['arm'],a['basket'],a['variant'],a['cost_bps']):a for a in q['rows']}; labels=q['labels']
    main=[]
    lines=['# Corrected-input model refit and RET daily backtest, 2020–2023','','This is the completed, independently verified conditional research baseline. The previous runs, stopped/failed attempts and checkpoints are preserved. No parameter search, target-family change or trading-rule change was performed.','',
    'Annual refits use the previous ten completed outcome years: 2010–2019 for 2020, 2011–2020 for 2021, 2012–2021 for 2022, and 2013–2022 for 2023. The historical S&P 500 + S&P 400 union, primary 80% feature-availability policy, Paper15 and expanded 715 predictors remain fixed. Five arms (four baseline RF/XGBoost combinations plus RF expanded using prior annual CV settings) require 20 estimator fits and eight shared training-only preprocessing states. Prior CV parameters were reused; they were not retuned on corrected inputs.','',
    '**Backtest period: 2 January 2020–29 December 2023; 48 monthly outcomes, 1,006 sessions.** Monthly predictions feed fresh RET, class1/10 and class2/9 selections. NAV continues across annual refits. The 120 predefined paths are five arms × three baskets × two stop variants × four costs. Independent verification retained all120: **'+str(b['complete_proxy_paths'])+' complete / '+str(b['withheld_paths'])+' withheld**. Missing exposure remains in the ledgers; withheld full-period and annual performance stays null.','',
    '## RET with the 10% per-asset stop','',
    'The stated focus is RF expanded with SL; the baseline and prior-CV RF remain distinct. Gross uses0bp. Net uses10bp per one-way traded notional. Borrow/funding costs are0 in this frozen proxy. Returns and drawdowns cover the full four-year continuous path; annualization uses252 sessions/year, and Sharpe uses daily returns with a zero cash rate. A dash is unavailable, never zero.','',
    '| Model | Gross total | Net total | Net CAGR | Net volatility | Net Sharpe | Net max DD | Status |','|---|---:|---:|---:|---:|---:|---:|---|']
    for arm in ARMS:
        g,n=(rows[(arm,'ret','asset_stop_10pct',cost)] for cost in [0,10]); status='complete' if n['complete_proxy_path'] else 'withheld'
        item={'arm':arm,'label':labels[arm],'gross_total_return':g['total_return'],'net_total_return':n['total_return'],'net_cagr':n['cagr'],'net_volatility':n['annualized_volatility'],'net_sharpe':n['sharpe_zero_cash_rate'],'net_maximum_drawdown':n['maximum_drawdown'],'status':status};main.append(item)
        lines.append(f"| {labels[arm]} | {pct(g['total_return'])} | {pct(n['total_return'])} | {pct(n['cagr'])} | {pct(n['annualized_volatility'])} | {num(n['sharpe_zero_cash_rate'])} | {pct(n['maximum_drawdown'])} | {status} |")
    lines+=['','Full class1/10 and class2/9 comparisons, unstopped comparators, all0/5/10/25bp cost assumptions, continuous annual slices, coverage, stop/trade counts and turnover/cost totals are saved in run/reports/backtest_v1/report.md and report.json. These are simulated historical proxies, not live or investability-certified results.','',
    '## Prediction diagnostics','',
    'All five arms share prediction and observed-label keys. Accuracy is ten-class mode accuracy, separate from portfolio performance. Class1 is best and class10 worst. Observed-label coverage is measured among prediction-eligible stock-months, not the entire historical universe.','',
    '| Model | Pooled log loss | Equal-month log loss | Pooled accuracy | Observed labels / predictions |','|---|---:|---:|---:|---:|']
    model_summary=[]
    for arm in ARMS:
        a=r['arms'][arm]['overall'];model_summary.append({'arm':arm,**a})
        lines.append(f"| {labels[arm]} | {a['pooled_log_loss']:.6f} | {a['mean_monthly_log_loss']:.6f} | {100*a['pooled_accuracy']:.2f}% | {a['labelled_rows']:,} / {a['prediction_rows']:,} |")
    lines+=['','Separate class1/2/9/10 precision, recall and support, monthly diagnostics and annual metrics are retained in run/reports/models_v1/comparison.md and comparison.json. No paired improvement claim against old runs is made because input/cohort changes differ. This period was already reviewed during research; it is not a new untouched holdout. No 2024+ financial evaluation is included.','',
    '## Selection, remaining gaps and limits','',
    'RET ranks the full predicted probability distribution against the shared causal120-completed-calendar-month class payoffs. The separate class baskets select within their predicted classes. Both use up to10 names per side, equal counts per side, 50% long/50% short, equal weights within a side and next-session open execution. Stops use the unchanged daily-OHLC observation/fill assumptions; gaps can exceed10%, and stopped capital remains cash until the next scheduled rebalance. Future quote or target availability never chooses the basket.','',
    'Monthly training labels remain provider-adjusted close-to-close endpoint returns. RET therefore predicts that label payoff, not the exact next-open, stop-dependent realized portfolio payoff. This definitional difference remains a target-alignment research question; the completed run does not establish that label and daily execution returns coincide.','',
    'All15 fresh selections have zero unavailable selection months. The maximum-horizon preflight finds127 unique missing native/date keys across nine IDs, plus one quality-flagged key; these affect168 and2 episode rows respectively. The saved failure state, rather than this maximum-horizon check alone, determines each path’s availability. SELECTED_HORIZON_GAPS_20261021.md lists the exact identities, dates, affected models and remaining evidence needs. The descriptive names are PBCT, EV, NBL, HRC, PSB, WWE, WRI, FBIN and HWM; a name match does not independently establish an entitlement or historical identity.','',
    'The verified actual first failures are44 missing-quote paths,16 source-quality-flag paths and4 selected-provider/held-source identity-collision paths. The collision affects XGBoost expanded, unstopped class1/10, on2022-10-03; it is separate from quote availability. ACTUAL_FAILURE_DIAGNOSTICS_20261021.md and its hash-pinned JSON retain exact first-failure keys and costs. No tolerance or identity override was added.','',
    'The reviewed BK/XOM routes and prior40 execution-only quote repairs remain pinned. RCM/DISCA learning-bar candidates were rejected by the frozen volume-compatibility control; no such learning bars were admitted. XOM’s69 retained precision warnings remain audit evidence and no tolerance was widened. Provisional identity decisions, historical reconstruction/knowledge-time, provider adjustment vintage, terminal-return coverage and survivorship limitations remain unresolved. Independent checks establish the saved timing, preprocessing, forecasts, selections and accounting within this scope; they do not certify every source observation or eliminate these research biases.','',
    'Next priority: verify the remaining selected-horizon quotes, dated corporate-action entitlements and selected/held identity handover collision, then replay the same saved forecasts/selections in a separately versioned execution repair. Audit any resulting feature/label/RET-payoff impact before changing labels. A further model refit is warranted only if that audited learning input changes; this list does not authorize a new search or alternative label experiment.','',
    '## Evidence and archive','',
    'The installed package is outputs/corrected_refit/2020-2023-20261021-v1/. installed_catalog.json lists the actual archive locations; relocation_index.json translates the staging prefix while preserving embedded provenance byte-for-byte. No active pointer is promoted. The private archive includes model artifacts, predictions, all120 ledgers, failed attempts/checkpoints, frozen plans, source decisions, reviews and proofs. Reproduction requires a fresh attempt and frozen plan; never execute into this sealed archive.','',
    '- Model proof: run/verification/models.json, SHA256 '+z['files']['models']['sha256']+'.',
    '- Backtest proof: run/verification/backtest.json, SHA256 '+z['files']['backtest']['sha256']+'.',
    '- Passive report receipts: run/reports/models_v1/receipt.json and run/reports/backtest_v1/receipt.json.',
    '- Input decisions: RECOVERY_INPUT_DECISIONS.md and MONTHLY_ENDPOINT_LABEL_POLICY.md.',
    '- Integration/attempt provenance: RUNTIME_ENTRYPOINTS_20261021.md and run/control/.',
    '- The installer adds archive_receipt.json only after opaque copy/hash verification; this pre-install document makes no earlier archive-success claim.','']
    prose='\n'.join(lines);c.need(len(prose.encode())<=z['maximum_output_bytes'],'Handoff output cap exceeded')
    with (ROOT/'README.md').open('x') as f:f.write(prose)
    failures=[{k:a.get(k) for k in ['arm','basket','variant','cost_bps','first_failure_date','first_failure_mark','withheld_marks','retained_unvalued_lots']} for a in b['scenario_checks'] if not a['economic_performance_available'] and a['cost_bps']==0]
    result={'status':'COMPLETE_FROM_VERIFIED_REPORTS','started_pin':started,'plan_pin':p,'input_pins':z['files'],'complete_paths':b['complete_proxy_paths'],'withheld_paths':b['withheld_paths'],'RET_SL_primary':main,'model_metrics':model_summary,'failure_path_summaries':failures,'identity_failure_evidence':[{k:a.get(k) for k in ['arm','basket','variant','cost_bps','first_failure_date','identity_failure_proof']} for a in b['scenario_checks'] if a.get('identity_failure_proof') and a['cost_bps']==0],'README_pin':c.pin(ROOT/'README.md'),'financial_tables_read':0,'model_deserializations':0}
    receipt=c.save_new(ROOT/'run/reports/root_handoff_v1.json',result)
    print(json.dumps(result,indent=2));print(receipt)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['plan','run']);v=a.parse_args();print(plan() if v.command=='plan' else run())
