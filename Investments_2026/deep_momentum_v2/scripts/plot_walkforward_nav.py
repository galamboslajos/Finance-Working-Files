"""Render saved monthly NAV from a six-receipt, hash-bound report snapshot.

Stage one is prepare_walkforward_handoff.py report in the frozen V2 runtime.
This stage imports no research worker, model, matrix, Parquet or source price.
An existing renderer environment may supply Matplotlib without changing the
frozen scientific environment. No returns or performance metrics are computed.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
import hashlib
import importlib.metadata
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import platform
import sys
import tempfile

ARMS=('rf_paper15','xgb_paper15','rf_extended','xgb_extended')
BASKETS=('class_1_10','class_2_9')
VARIANTS=('unstopped','asset_stop_10pct')
MONTHS=tuple(f'{year}-{month:02d}' for year in range(2020,2024) for month in range(1,13))
LABELS={'rf_paper15':'RF · Paper15','xgb_paper15':'XGBoost · Paper15',
    'rf_extended':'RF · expanded','xgb_extended':'XGBoost · expanded'}
COLORS=dict(zip(ARMS,('#0072B2','#E69F00','#009E73','#CC79A7'),strict=True))
STYLES=dict(zip(ARMS,('-','--','-.',':'),strict=True))
METRICS=('final_nav','total_return','cagr','annualized_arithmetic_mean',
    'annualized_volatility','sharpe_zero_cash_rate','maximum_drawdown',
    'total_one_way_turnover','trading_cost_paid')
MAX_BYTES=16*1024**2


def require(ok,message):
    if not ok:raise AssertionError(message)


def digest(path):
    state=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''):state.update(block)
    return state.hexdigest()


def bounded_json(path):
    path=Path(path); require(path.stat().st_size<=MAX_BYTES,'Unbounded plot JSON')
    return json.loads(path.read_text())


def _handoff():
    path=Path(__file__).resolve().with_name('prepare_walkforward_handoff.py')
    spec=importlib.util.spec_from_file_location('_walkforward_plot_metadata_gate',path)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def _csv(path,limit):
    require(Path(path).stat().st_size<=MAX_BYTES,'Unbounded plot CSV')
    with Path(path).open(newline='') as stream:
        reader=csv.DictReader(stream); require(reader.fieldnames is not None,'Missing plot CSV header')
        require(len(reader.fieldnames)==len(set(reader.fieldnames)),'Duplicate plot CSV columns')
        rows=[]
        for row in reader:
            require(None not in row and all(value is not None for value in row.values()),'Malformed plot CSV')
            rows.append(row); require(len(rows)<=limit,'Plot CSV exceeds its exact grid')
    require(len(rows)==limit,'Plot CSV lost rows from its exact grid')
    return rows


def _boolean(value):
    # bool('False') is True: never use generic truthiness for a CSV flag.
    require(value in {'True','False'},'CSV completeness must be literal True or False')
    return value=='True'


def read_snapshot(root,snapshot,*,snapshot_receipt_sha256):
    """Authenticate metadata and small CSVs before converting any NAV values."""
    root=Path(root).resolve(); snapshot=Path(snapshot).resolve()
    receipt_path=snapshot/'report_receipt.json'
    require(digest(receipt_path)==snapshot_receipt_sha256.removeprefix('sha256:'),
        'Explicit report snapshot receipt hash changed')
    receipt=bounded_json(receipt_path)
    require(receipt.get('status')=='SAVED_PASSIVE_REPORT_SNAPSHOT'
        and receipt.get('report_tables_only') is True and receipt.get('models_loaded') is False
        and receipt.get('matrices_loaded') is False and receipt.get('source_prices_decoded') is False
        and receipt.get('models_fitted')==0,'Plot input is not a passive saved-report snapshot')
    bindings=_handoff().final_bindings(root,receipt['run_directory'],receipt['build_identity_sha256'],
        verification_directory=receipt['verification_directory'])
    require(receipt['expected_receipts']==bindings['expected_receipts'],
        'Snapshot six receipts differ from the current independent PASS evidence')
    require(Path(receipt['backtest_directory']).resolve()==bindings['backtest_directory'],
        'Snapshot refers to another backtest phase')
    allowed={'model_comparison.csv','backtest_comparison.csv','backtest_yearly.csv',
        'selection_coverage.csv','monthly_10bps.csv','model_classes_1_2_9_10.csv',
        'feature_target_coverage.csv','RESULTS.md'}
    require(set(receipt['artifacts'])==allowed,'Snapshot artifact manifest changed')
    for name,pin in receipt['artifacts'].items():
        path=(snapshot/name).resolve()
        require(path.parent==snapshot and path.stat().st_size==pin['size_bytes']
            and path.stat().st_size<=MAX_BYTES and digest(path)==pin['sha256'],
            'Saved snapshot artifact hash/size changed: '+name)
    calendar=receipt['plot_calendar']
    require(calendar.get('first_session')=='2020-01-02' and calendar.get('terminal_session')=='2023-12-29'
        and calendar.get('initial_nav_anchor')=='2019-12-31' and calendar.get('sessions')==1006
        and calendar.get('months')==48,'Plot calendar changed the four-year evaluation horizon')
    ends=calendar['monthly_session_end']; require(set(ends)==set(MONTHS),'Plot calendar lost a month')
    for month in MONTHS:
        require(date.fromisoformat(ends[month]).isoformat()==ends[month] and ends[month][:7]==month,
            'Plot calendar monthly endpoint is malformed or outside its month')
    require(ends['2023-12']=='2023-12-29','Plot calendar lost the exact terminal session')
    comparison=_csv(snapshot/'backtest_comparison.csv',64)
    monthly=_csv(snapshot/'monthly_10bps.csv',768)
    expected=set(itertools.product(ARMS,BASKETS,VARIANTS,('0','5','10','25')))
    grid=[tuple(row[key] for key in ('arm','basket','variant','cost_bps')) for row in comparison]
    require(len(set(grid))==64 and set(grid)==expected,'Comparison lost the exact 64-scenario grid')
    by_scenario={}
    for row,key in zip(comparison,grid,strict=True):
        complete=_boolean(row['complete_proxy_path'])
        require(row['status']==('complete_proxy' if complete else 'failed_closed'),
            'CSV status contradicts the strict completeness flag')
        require(row['expected_sessions']=='1006','Comparison changed the expected daily horizon')
        if not complete:
            require(all(row.get(name,'') in {'','nan','NaN'} for name in METRICS),
                'Withheld scenario claims full-period performance')
        by_scenario[key]=(row,complete)
    month_grid=[tuple(row[key] for key in ('arm','basket','variant','cost_bps','outcome_month')) for row in monthly]
    require(len(set(month_grid))==768
        and set(month_grid)==set(itertools.product(ARMS,BASKETS,VARIANTS,('10',),MONTHS)),
        'Monthly plot input lost the exact 768-row grid')
    month_rows={key:row for key,row in zip(month_grid,monthly,strict=True)}
    series={}; withheld=[]
    for arm,basket,variant in itertools.product(ARMS,BASKETS,VARIANTS):
        key=(arm,basket,variant); complete=by_scenario[(*key,'10')][1]
        if not complete:
            # Prefix values, even apparently valid numbers, never enter float()
            # or a Matplotlib data line for an economically incomplete path.
            series[key]=None; withheld.append(key); continue
        values=[float(month_rows[(*key,'10',month)]['nav']) for month in MONTHS]
        require(len(values)==48 and all(math.isfinite(value) and value>0 for value in values),
            'Complete path requires 48 finite positive saved monthly NAVs')
        series[key]=values
    require(digest(receipt_path)==snapshot_receipt_sha256.removeprefix('sha256:'),
        'Report snapshot receipt changed during authentication')
    return {'root':root,'snapshot':snapshot,'receipt':receipt,'receipt_path':receipt_path,
        'bindings':bindings,'series':series,'withheld':withheld,
        'dates':[date.fromisoformat(ends[month]) for month in MONTHS]}


def build_figure(prepared):
    """Display existing monthly NAV; no cumulative-return reconstruction."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.lines import Line2D
    figure,axes=plt.subplots(2,2,figsize=(14.0,9.0),sharex=True,sharey=True)
    panel_audit=[]; anchor=date(2019,12,31); terminal=date(2023,12,29)
    for i,basket in enumerate(BASKETS):
        for j,variant in enumerate(VARIANTS):
            axis=axes[i,j]; handles=[]; failed=[]; plotted=[]
            for arm in ARMS:
                values=prepared['series'][(arm,basket,variant)]
                if values is None:
                    failed.append(arm)
                    handles.append(Line2D([],[],color=COLORS[arm],marker='x',linestyle='None',
                        label=LABELS[arm]+' · WITHHELD'))
                else:
                    line,=axis.plot([anchor,*prepared['dates']],[1.0,*values],color=COLORS[arm],
                        linestyle=STYLES[arm],linewidth=1.8,label=LABELS[arm])
                    handles.append(line); plotted.append(arm)
            axis.axhline(1.0,color='#7b8390',linewidth=.7,alpha=.55,zorder=0)
            for year in (2021,2022,2023):
                axis.axvline(date(year,1,1),color='#9ca3af',linestyle='--',linewidth=.8,alpha=.7,zorder=0)
            axis.grid(axis='y',alpha=.18); axis.spines[['top','right']].set_visible(False)
            axis.set_xlim(anchor,terminal); axis.xaxis.set_major_locator(mdates.YearLocator())
            axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            axis.legend(handles=handles,loc='upper left',fontsize=8.4,frameon=False,ncol=2)
            axis.set_title(('Class 1/10' if basket=='class_1_10' else 'Class 2/9')+' · '
                +('unstopped' if variant=='unstopped' else '10% stop per asset'),fontsize=12,pad=12)
            if j==0:axis.set_ylabel('Portfolio NAV · initial capital = 1')
            if i==1:axis.set_xlabel('Monthly last XNYS session · 2020–2023')
            if failed:
                axis.text(.015,.025,'WITHHELD: '+', '.join(LABELS[arm] for arm in failed)
                    +'\nNo prefix line or full-period return is shown.',transform=axis.transAxes,
                    fontsize=8.3,color='#5b2634',va='bottom',
                    bbox={'facecolor':'white','edgecolor':'none','alpha':.85,'pad':4})
            if not plotted:axis.set_ylim(.9,1.1)
            panel_audit.append({'basket':basket,'variant':variant,'plotted_arms':plotted,
                'withheld_arms':failed,'legend_arms':list(ARMS),'data_points_per_complete_path':49})
    synthetic=prepared['receipt'].get('frozen_model_runtime',{}).get('synthetic') is True
    title=('SYNTHETIC LAYOUT CHECK · ' if synthetic else '')+'Annual walk-forward · monthly-end NAV at 10 bps'
    figure.suptitle(title,fontsize=14 if synthetic else 17,x=.07,ha='left',y=.975)
    subtitle='January 2, 2020–December 29, 2023 · 48 months · 1,006 daily XNYS sessions'
    figure.text(.07,.931,subtitle+(' · No research results' if synthetic else ''),fontsize=11,color='#465263')
    figure.text(.07,.056,'Monthly endpoints of one continuous daily backtest; annual model changes do not reset capital.\n'
        'December 31, 2019 NAV = 1 is a capital anchor. Price/accounting proxy; zero borrow and funding charges.\n'
        '10 bps per one-way traded notional. All four preset model arms shown; incomplete paths are withheld in full.',
        fontsize=9.0,color='#465263',va='bottom',linespacing=1.5)
    figure.subplots_adjust(left=.07,right=.98,top=.88,bottom=.17,hspace=.30,wspace=.14)
    return figure,panel_audit


def render_snapshot(root,snapshot,*,snapshot_receipt_sha256,output_directory):
    prepared=read_snapshot(root,snapshot,snapshot_receipt_sha256=snapshot_receipt_sha256)
    output=Path(output_directory).resolve(); root=prepared['root']; run=prepared['bindings']['run']
    require(output.is_relative_to(run/'reports') or output.is_relative_to(Path('/private/tmp').resolve()),
        'Plot outputs must remain in this run’s reports or private scratch')
    require(not output.exists(),'Preserve an existing plot output directory')
    # Matplotlib caches are temporary scratch artifacts, never new home files.
    old_environment={key:os.environ.get(key) for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','MPLBACKEND')}
    with tempfile.TemporaryDirectory(prefix='dmv2-nav-render-',dir='/private/tmp') as cache:
        os.environ.update(MPLCONFIGDIR=str(Path(cache)/'matplotlib'),XDG_CACHE_HOME=str(Path(cache)/'xdg'),MPLBACKEND='Agg')
        try:
            import matplotlib
            matplotlib.use('Agg',force=True)
            import matplotlib.pyplot as plt
            figure,panels=build_figure(prepared)
            output.mkdir(parents=True); png=output/'walkforward_monthly_nav_10bps.png'
            svg=output/'walkforward_monthly_nav_10bps.svg'
            figure.savefig(png,dpi=180,facecolor='white')
            figure.savefig(svg,format='svg',facecolor='white',metadata={'Date':None})
            plt.close(figure)
        finally:
            for key,value in old_environment.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value
    # Repeat source gates after rendering to detect concurrent evidence changes.
    current=_handoff().final_bindings(root,run,prepared['receipt']['build_identity_sha256'],
        verification_directory=prepared['receipt']['verification_directory'])
    require(current['expected_receipts']==prepared['receipt']['expected_receipts']
        and digest(prepared['receipt_path'])==snapshot_receipt_sha256.removeprefix('sha256:'),
        'Source evidence changed during rendering')
    for name,pin in prepared['receipt']['artifacts'].items():
        path=prepared['snapshot']/name
        require(path.stat().st_size==pin['size_bytes'] and digest(path)==pin['sha256'],
            'Saved snapshot artifact changed during rendering: '+name)
    result={'status':'SAVED_NAV_DISPLAY_ONLY','build_identity_sha256':prepared['receipt']['build_identity_sha256'],
        'snapshot_directory':str(prepared['snapshot']),'snapshot_receipt_sha256':digest(prepared['receipt_path']),
        'source_receipts':prepared['receipt']['expected_receipts'],
        'csv_pins':{name:prepared['receipt']['artifacts'][name] for name in ('backtest_comparison.csv','monthly_10bps.csv')},
        'comparison_rows':64,'monthly_rows':768,'outcome_months':48,'daily_sessions':1006,
        'first_session':'2020-01-02','terminal_session':'2023-12-29','initial_capital_anchor':'2019-12-31',
        'panels':panels,'frozen_model_runtime':prepared['receipt'].get('frozen_model_runtime',{}),
        'renderer_runtime':{'executable':sys.executable,'python':platform.python_version(),
            'matplotlib':importlib.metadata.version('matplotlib'),'numpy':importlib.metadata.version('numpy')},
        'renderer_script_sha256':digest(Path(__file__).resolve()),
        'report_snapshot_only':True,'models_loaded':False,'matrices_loaded':False,
        'source_prices_decoded':False,'models_fitted':0,'new_metrics_computed':False,
        'withheld_prefixes_plotted':False,'borrow_and_funding_charges':0,
        'artifacts':{path.name:{'sha256':digest(path),'size_bytes':path.stat().st_size} for path in (png,svg)}}
    (output/'plot_receipt.json').write_text(json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--snapshot-receipt-sha256',required=True); parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args(argv)
    result=render_snapshot(args.root,args.snapshot,snapshot_receipt_sha256=args.snapshot_receipt_sha256,
        output_directory=args.output_directory)
    print(json.dumps({'status':result['status'],'output_directory':str(args.output_directory.resolve()),
        'complete_10bps_paths':sum(len(panel['plotted_arms']) for panel in result['panels']),
        'withheld_10bps_paths':sum(len(panel['withheld_arms']) for panel in result['panels'])},sort_keys=True))


if __name__=='__main__':main()
