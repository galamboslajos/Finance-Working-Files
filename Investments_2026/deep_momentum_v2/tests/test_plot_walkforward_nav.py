"""Synthetic receipt/CSV-only plotting; no model or dataset imports."""
import calendar
import csv
from datetime import date,timedelta
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/plot_walkforward_nav.py'
spec=importlib.util.spec_from_file_location('plot_walkforward_nav',SCRIPT)
plot=importlib.util.module_from_spec(spec); spec.loader.exec_module(plot)


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n')


def canonical(value):
    raw=json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False)
    return 'sha256:'+hashlib.sha256(raw.encode()).hexdigest()


def csv_file(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def repin(snapshot):
    path=snapshot/'report_receipt.json'; value=json.loads(path.read_text())
    for name in value['artifacts']:
        item=snapshot/name; value['artifacts'][name]={'sha256':plot.digest(item),'size_bytes':item.stat().st_size}
    save(path,value); return plot.digest(path)


def synthetic_fixture(root,*,withheld=True):
    identity={'purpose':'SYNTHETIC TEST ONLY; no research data'}; identity_pin=canonical(identity)
    run=root/'outputs/walkforward'/('2020-2023-'+identity_pin[7:23]); run.mkdir(parents=True)
    save(run/'build_identity.json',identity)
    save(run/'audit.json',{'status':'COMPLETE','build_identity_sha256':identity_pin,
        'years':{str(year):{} for year in range(2020,2024)}})
    save(run/'models/audit.json',{'status':'COMPLETE','build_identity_sha256':identity_pin,
        'model_identity_sha256':'sha256:'+'1'*64,'summary':{'fitted_arms':16,'preprocessor_fits':8,'months':48}})
    save(run/'backtest/audit.json',{'status':'COMPLETE','build_identity_sha256':identity_pin,
        'phase_identity_sha256':'sha256:'+'2'*64,'scenarios':64,'sessions':1006,'outcome_months':48})
    raw={key:plot.digest(run/path) for key,path in (('features','audit.json'),('models','models/audit.json'),('backtest','backtest/audit.json'))}
    shared={'build_identity_sha256':identity_pin,'root_audit_sha256':raw['features']}
    save(run/'verification/features.json',{**shared,'status':'PASS','models_fitted':0,'financial_2024_plus_read':False})
    save(run/'verification/models.json',{**shared,'status':'PASSED','models_audit_sha256':raw['models'],
        'model_identity_sha256':'sha256:'+'1'*64,'annual_fits_reconciled':16,'stitched_prediction_months':48,
        'models_fitted_by_verifier':0,'preprocessors_fitted_by_verifier':0,'financial_2024_plus_decoded':False})
    save(run/'verification/backtest.json',{**shared,'status':'PASS','backtest_audit_sha256':raw['backtest'],
        'backtest_phase_identity_sha256':'sha256:'+'2'*64,'backtest_directory':str((run/'backtest').relative_to(root)),
        'models_fitted':0,'preprocessors_fitted':0,'financial_2024_plus_read':False,'source_pins_unchanged':True})
    expected={key:'sha256:'+value for key,value in raw.items()}
    expected.update({key:'sha256:'+plot.digest(run/'verification'/name)
        for key,name in (('feature_verification','features.json'),('model_verification','models.json'),('backtest_verification','backtest.json'))})
    snapshot=run/'reports/SYNTHETIC_snapshot'; snapshot.mkdir(parents=True)
    comparisons=[]; monthly=[]
    for arm,basket,variant,cost in itertools.product(plot.ARMS,plot.BASKETS,plot.VARIANTS,(0,5,10,25)):
        complete=not (withheld and arm.endswith('extended'))
        row={'arm':arm,'basket':basket,'variant':variant,'cost_bps':cost,
            'complete_proxy_path':complete,'status':'complete_proxy' if complete else 'failed_closed',
            'expected_sessions':1006,'total_return':.15 if complete else None}
        comparisons.append(row)
        if cost==10:
            for index,month in enumerate(plot.MONTHS):
                # Deliberately nonnumeric withheld prefixes: converting these
                # would fail even before any accidental plotted line appears.
                value=1.+index*.003+plot.ARMS.index(arm)*.008 if complete else 'PREFIX_MUST_NOT_BE_CONVERTED'
                monthly.append({'arm':arm,'basket':basket,'variant':variant,'cost_bps':10,
                    'outcome_month':month,'nav':value,'return':'IGNORED_NOT_RECOMPUTED'})
    csv_file(snapshot/'backtest_comparison.csv',comparisons); csv_file(snapshot/'monthly_10bps.csv',monthly)
    for name in ('model_comparison.csv','backtest_yearly.csv','selection_coverage.csv',
        'model_classes_1_2_9_10.csv','feature_target_coverage.csv'):
        (snapshot/name).write_text('synthetic_only\nno_dataset_values\n')
    (snapshot/'RESULTS.md').write_text('SYNTHETIC FIXTURE ONLY, NOT RESEARCH RESULTS.\n')
    endpoints={}
    for month in plot.MONTHS:
        year,number=map(int,month.split('-')); day=date(year,number,calendar.monthrange(year,number)[1])
        while day.weekday()>4:day-=timedelta(days=1)
        endpoints[month]=day.isoformat()
    endpoints['2023-12']='2023-12-29'
    receipt={'status':'SAVED_PASSIVE_REPORT_SNAPSHOT','build_identity_sha256':identity_pin,
        'run_directory':str(run),'backtest_directory':str(run/'backtest'),'verification_directory':str(run/'verification'),
        'expected_receipts':expected,'report_tables_only':True,'models_loaded':False,'matrices_loaded':False,
        'source_prices_decoded':False,'models_fitted':0,'frozen_model_runtime':{'synthetic':True},
        'plot_calendar':{'first_session':'2020-01-02','terminal_session':'2023-12-29','initial_nav_anchor':'2019-12-31',
            'sessions':1006,'months':48,'monthly_session_end':endpoints},
        'artifacts':{path.name:{} for path in snapshot.iterdir()}}
    save(snapshot/'report_receipt.json',receipt)
    return run,snapshot,repin(snapshot)


class SavedNAVPlotTests(unittest.TestCase):
    def test_six_receipts_grid_and_withheld_poison_values(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
            root=Path(temporary); run,snapshot,pin=synthetic_fixture(root)
            prepared=plot.read_snapshot(root,snapshot,snapshot_receipt_sha256=pin)
            self.assertEqual(len(prepared['series']),16); self.assertEqual(len(prepared['withheld']),8)
            self.assertTrue(all(len(values)==48 for values in prepared['series'].values() if values is not None))

    def test_missing_receipt_stops_before_any_csv_read_or_renderer_import(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
            root=Path(temporary); run,snapshot,pin=synthetic_fixture(root)
            (run/'verification/models.json').unlink()
            with patch.object(plot,'_csv',side_effect=AssertionError('CSV opened before receipt gate')):
                with self.assertRaisesRegex(AssertionError,'All three completed independent'):
                    plot.read_snapshot(root,snapshot,snapshot_receipt_sha256=pin)

    def test_snapshot_and_csv_tampering_refused(self):
        for target in ('receipt','monthly'):
            with self.subTest(target=target),tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
                root=Path(temporary); run,snapshot,pin=synthetic_fixture(root)
                path=snapshot/('report_receipt.json' if target=='receipt' else 'monthly_10bps.csv')
                path.write_text(path.read_text()+'\n')
                with self.assertRaisesRegex(AssertionError,'hash'):
                    plot.read_snapshot(root,snapshot,snapshot_receipt_sha256=pin)

    def test_boolean_status_duplicates_and_positive_nav_guards(self):
        for kind in ('string_false','contradict_status','duplicate','nan_complete','negative_complete','withheld_return'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
                root=Path(temporary); run,snapshot,pin=synthetic_fixture(root)
                name='monthly_10bps.csv' if kind in {'duplicate','nan_complete','negative_complete'} else 'backtest_comparison.csv'
                with (snapshot/name).open(newline='') as stream:rows=list(csv.DictReader(stream))
                if kind=='string_false':rows[0]['complete_proxy_path']='false'
                elif kind=='contradict_status':rows[0]['status']='failed_closed'
                elif kind=='duplicate':rows[-1]=rows[0]
                elif kind=='nan_complete':rows[0]['nav']='nan'
                elif kind=='negative_complete':rows[0]['nav']='-1'
                elif kind=='withheld_return':next(row for row in rows if row['complete_proxy_path']=='False')['total_return']='.2'
                csv_file(snapshot/name,rows); pin=repin(snapshot)
                with self.assertRaises(AssertionError):plot.read_snapshot(root,snapshot,snapshot_receipt_sha256=pin)

    @unittest.skipUnless(importlib.util.find_spec('matplotlib'),'Existing Matplotlib renderer required; never install')
    def test_render_synthetic_four_panels_all_arms_and_no_withheld_line(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
            root=Path(temporary); run,snapshot,pin=synthetic_fixture(root)
            output=run/'reports/SYNTHETIC_plot'; old={key:os.environ.get(key) for key in ('MPLCONFIGDIR','XDG_CACHE_HOME','MPLBACKEND')}
            original=plot.build_figure
            def check_actual_figure(prepared):
                figure,panels=original(prepared)
                for axis in figure.axes:
                    labelled=[line for line in axis.lines if line.get_label() in plot.LABELS.values()]
                    self.assertEqual({line.get_label() for line in labelled},
                        {plot.LABELS['rf_paper15'],plot.LABELS['xgb_paper15']})
                    self.assertEqual(len(axis.get_legend().get_texts()),4)
                    self.assertEqual(sum('WITHHELD' in text.get_text() for text in axis.get_legend().get_texts()),2)
                    for line in labelled:
                        self.assertEqual(len(line.get_ydata()),49); self.assertEqual(line.get_ydata()[0],1.)
                        self.assertEqual(line.get_xdata()[0],date(2019,12,31))
                return figure,panels
            with patch.object(plot,'build_figure',side_effect=check_actual_figure):
                result=plot.render_snapshot(root,snapshot,snapshot_receipt_sha256=pin,output_directory=output)
            self.assertEqual(len(result['panels']),4); self.assertFalse(result['withheld_prefixes_plotted'])
            self.assertEqual(sum(len(panel['plotted_arms']) for panel in result['panels']),8)
            self.assertEqual(sum(len(panel['withheld_arms']) for panel in result['panels']),8)
            for panel in result['panels']:self.assertEqual(panel['legend_arms'],list(plot.ARMS))
            self.assertEqual(set(path.suffix for path in output.iterdir()),{'.png','.svg','.json'})
            self.assertEqual({key:os.environ.get(key) for key in old},old)
            self.assertEqual(result['frozen_model_runtime'],{'synthetic':True})
            self.assertIn('matplotlib',result['renderer_runtime']); self.assertFalse(result['new_metrics_computed'])
            with self.assertRaisesRegex(AssertionError,'existing plot output'):
                plot.render_snapshot(root,snapshot,snapshot_receipt_sha256=pin,output_directory=output)


if __name__=='__main__':unittest.main()
