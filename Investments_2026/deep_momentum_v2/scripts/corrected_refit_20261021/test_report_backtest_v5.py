"""Tiny invented saved-summary tests; stdlib only, no research/model reads."""
from copy import deepcopy
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import control as c
import report_backtest_v5 as r


def fabricated():
    comparison=[];annuals=[];selection=[];preflight=[]
    bad=('rf_paper15','class_1_10','unstopped',25)
    for arm,basket,variant,cost in itertools.product(r.ARMS,r.BASKETS,r.VARIANTS,r.COSTS):
        complete=(arm,basket,variant,cost)!=bad;final=1.05**4
        row=dict(arm=arm,basket=basket,variant=variant,cost_bps=cost,
            status='complete_proxy' if complete else 'failed_closed',expected_sessions=1006,
            valid_close_sessions=1006 if complete else 200,first_failure_date=None if complete else '2020-10-15',
            complete_proxy_path=complete,outcome_start='2020-01-01',outcome_end='2023-12-31',outcome_months=48,
            economic_accounting_certified=False,initial_nav=1.,stop_count=3,trade_count=100)
        row.update(dict(zip(r.PERFORMANCE,(final,final-1,final**(252/1006)-1,.1,.4,-.2,12.,.01))) if complete
                   else {name:None for name in r.PERFORMANCE})
        comparison.append(row)
        previous=1.
        for year in range(2020,2024):
            end=previous*1.05
            a=dict(arm=arm,basket=basket,variant=variant,cost_bps=cost,year=year,
                expected_sessions={2020:253,2021:252,2022:251,2023:250}[year],complete_full_period_proxy=complete)
            a.update(dict(zip(r.YEAR_FINANCIAL,(previous,end,.05,.1,.4,-.1,.0025))) if complete
                     else {name:None for name in r.YEAR_FINANCIAL})
            annuals.append(a);previous=end
    for arm,basket in itertools.product(r.ARMS,r.BASKETS):
        for year,month in itertools.product(range(2020,2024),range(1,13)):
            previous=f'{year-1}-12' if month==1 else f'{year}-{month-1:02d}'
            selection.append(dict(arm=arm,basket=basket,outcome_month=f'{year}-{month:02d}',
                signal_date=previous+'-28',execution_date=f'{year}-{month:02d}-02',status='available',
                original_rows=100,prediction_eligible=90,long_candidates=20,short_candidates=20,
                long_selected=10,short_selected=10,selected_names_per_side=10))
        preflight.append(dict(arm=arm,basket=basket,selected_episode_rows=960,distinct_maximum_horizon_quote_keys=2000,
            missing_key_episode_rows=0,quality_flag_episode_rows=0,unique_missing_keys=0,unique_quality_flag_keys=0,
            retired_new_orders=0,ready_for_accounting=True,withheld_selection_months=0,source_metadata_only=True,
            prices_or_outcomes_read=False,selection_changed=False,
            unstopped_maximum_horizons_reviewed_for_both_stop_variants=True))
    return comparison,annuals,selection,preflight,{'complete_proxy_paths':119,'withheld_paths':1}


class SavedSummaryTests(unittest.TestCase):
    def test_all_costs_models_variants_and_withheld_preserved(self):
        report=r.build_report(*fabricated())
        self.assertEqual(len(report['rows']),120);self.assertEqual(len(report['yearly']),480)
        self.assertEqual(len(report['coverage']),15);self.assertEqual(report['withheld_paths'],1)
        self.assertIsNone(next(x for x in report['rows'] if not x['complete_proxy_path'])['total_return'])
        self.assertFalse(report['old_result_comparison_performed'])

    def test_withheld_performance_is_never_displayed_as_available(self):
        args=list(fabricated());failed=next(x for x in args[0] if not x['complete_proxy_path'])
        failed['total_return']=.8
        with self.assertRaisesRegex(ValueError,'Withheld path exposed'):r.build_report(*args)

    def test_zero_complete_paths_is_valid_and_preserves_nulls(self):
        args=list(fabricated())
        for row in args[0]:
            row.update(complete_proxy_path=False,status='failed_closed',valid_close_sessions=100,
                first_failure_date='2020-06-01',**{n:None for n in r.PERFORMANCE})
        for row in args[1]:row.update(complete_full_period_proxy=False,**{n:None for n in r.YEAR_FINANCIAL})
        args[4]={'complete_proxy_paths':0,'withheld_paths':120}
        report=r.build_report(*args)
        self.assertEqual(report['complete_proxy_paths'],0);self.assertEqual(report['withheld_paths'],120)
        self.assertTrue(all(row['total_return'] is None for row in report['rows']))
        self.assertIn('0 complete / 120 withheld',r.markdown(report))

    def test_failed_annual_prefix_is_rejected(self):
        args=list(fabricated());failed=next(x for x in args[1] if not x['complete_full_period_proxy'])
        failed['total_return']=.05
        with self.assertRaisesRegex(ValueError,'annual prefix'):r.build_report(*args)

    def test_annual_capital_cannot_reset(self):
        args=list(fabricated());annual=next(x for x in args[1] if x['year']==2021 and x['complete_full_period_proxy'])
        annual['initial_nav']=1.
        with self.assertRaisesRegex(ValueError,'resets capital'):r.build_report(*args)

    def test_metadata_protected_future_and_unknown_completeness(self):
        args=list(fabricated());args[0][0]['outcome_end']='2024-12-31'
        with self.assertRaisesRegex(ValueError,'period'):r.metadata(*args[:4])
        args=list(fabricated());args[0][0]['complete_proxy_path']=None
        with self.assertRaisesRegex(ValueError,'period'):r.metadata(*args[:4])

    def test_markdown_has_period_cost_status_and_no_old_improvement_claim(self):
        prose=r.markdown(r.build_report(*fabricated()))
        for word in ('1,006','48 monthly outcomes','four years','prior CV settings','RF · expanded baseline',
                     '0bp','5bp','10bp','25bp','withheld','Unstopped comparator','no new tuning',
                     'not a zero return','continuous NAV','no full-period or annual performance'):
            self.assertIn(word,prose)
        self.assertIn('does not fit, infer, replay',prose)
        self.assertIn('claim improvement against an older run',prose)

    def test_missing_reported_scenario_or_selection_month_rejected(self):
        args=list(fabricated());args[0].pop()
        with self.assertRaisesRegex(ValueError,'120 unique'):r.build_report(*args)
        args=list(fabricated());args[2].pop()
        with self.assertRaisesRegex(ValueError,'15-by48'):r.build_report(*args)

    def test_readiness_never_parses_financial_proof(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();(root/'run/verification').mkdir(parents=True)
            (root/'run/verification/backtest.json').write_text('invalid JSON intentionally not parsed')
            with patch.object(r,'ROOT',root),patch.object(r,'BASE',root/'run/backtest'):
                result=r.readiness()
            self.assertFalse(result['proof_contents_parsed'])
            self.assertEqual(result['status'],'WAITING_FOR_VERIFIED_OUTPUT_FILES')

    def test_plan_json_hashes_invalid_metric_json_before_any_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();base=root/'run/backtest'
            for path in (root/'run/verification/backtest.json',base/'replay/audit.json',base/'preflight/audit.json'):
                path.parent.mkdir(parents=True,exist_ok=True);path.write_text('invalid metric JSON, no parse allowed yet')
            source=c.save_new(root/'source.json',{'schema':'dmv2-corrected-forecasts-backtest-source-plan/v1','scenario_count':120})
            protocol=c.save_new(root/'protocol.json',{'schema':'dmv2-corrected-forecasts-backtest-protocol/v1',
                'status':'SEALED_BEFORE_FINANCIAL_READ','source_plan_pin':source})
            for name in ('control.py','supervise.py'):(root/name).write_text('# invented code bytepin\n')
            with patch.object(r,'ROOT',root),patch.object(r,'BASE',base):
                result=r.plan_json(source['path'],protocol['path'],root/'plans/first.json')
            z=c.read(result['path']);self.assertEqual(z['projected_reads']['compressed_source'],r.json_budget(z))
            self.assertEqual(z['status'],'SEALED_BEFORE_FINANCIAL_JSON_PARSE')

    def test_false_is_not_a_count_or_numeric_metric(self):
        args=list(fabricated());args[2][0]['long_candidates']=False
        with self.assertRaisesRegex(ValueError,'coverage count'):r.build_report(*args)
        args=list(fabricated());args[0][0]['cagr']=False
        with self.assertRaisesRegex(ValueError,'unknown performance'):r.build_report(*args)

    def test_failed_planning_attempt_cannot_reparse_financial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();named={}
            for name in ('source','protocol','verification','replay','preflight'):
                named[name]=c.save_new(root/(name+'.json'),{'invented':True})
            unique=r.unique_pins(named);amount=sum(p['size_bytes'] for p in unique)
            first=c.save_new(root/'plans/json.json',{'schema':'dmv2-backtest-report-JSON-plan/v1',
                'status':'SEALED_BEFORE_FINANCIAL_JSON_PARSE','worker_pin':c.pin(r.__file__),
                'JSON_pins':named,'unique_JSON_pins':unique,
                'projected_reads':{'compressed_source':amount,'derived_matrix':0}})
            out=root/'plans/tables.json'
            with patch.object(r,'ROOT',root),patch.object(r,'require_reservation'),\
                 patch.object(r,'parsed_json',side_effect=ValueError('invented authority stop')) as parse:
                with self.assertRaisesRegex(ValueError,'authority stop'):r.plan(first['path'],first['sha256'],'invented',out)
                self.assertTrue(out.with_name(out.name+'.STARTED.json').exists())
                with self.assertRaisesRegex(ValueError,'Fresh scoped'):r.plan(first['path'],first['sha256'],'invented',out)
                self.assertEqual(parse.call_count,1)

    def test_failed_run_authority_preserves_consumed_attempt_before_parse(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory).resolve();stub=c.save_new(root/'stub.json',{'invented':True})
            named={'stub':stub};amount=stub['size_bytes']
            z={'schema':'dmv2-backtest-report-table-plan/v1','status':'SEALED_BEFORE_REPORT_VALUES',
                'worker_pin':c.pin(r.__file__),'control_pin':stub,'supervisor_pin':stub,'JSON_plan_pin':stub,
                'JSON_pins':named,'unique_JSON_pins':[stub],'tables':{},'maximum_output_bytes':r.MAX_OUTPUT,
                'maximum_seconds':r.MAX_SECONDS,'projected_reads':{'compressed_source':amount,'derived_matrix':0}}
            p=c.save_new(root/'plan.json',z);out=root/'run/reports/backtest_v1'
            with patch.object(r,'ROOT',root),patch.object(r,'require_reservation'),patch.object(r,'require_guard'),\
                 patch.object(r,'parsed_json',side_effect=ValueError('invented authority stop')) as parse:
                with self.assertRaisesRegex(ValueError,'authority stop'):r.run(p['path'],p['sha256'],'invented',out)
                self.assertTrue((out/'STARTED.json').exists())
                with self.assertRaisesRegex(ValueError,'Fresh exact'):r.run(p['path'],p['sha256'],'invented',out)
                self.assertEqual(parse.call_count,1)

    def test_table_plan_uses_only_exact_four_audit_artifacts(self):
        artifacts={n:{'path':'invented/'+n} for n in ('comparison.parquet','yearly.parquet','selection_coverage.parquet')}
        data={'replay':{'artifacts':artifacts},'preflight':{'artifacts':{'summary.parquet':{'path':'invented/summary'}}}}
        with patch.object(r,'projection',side_effect=lambda p,c,n:{'pin':p,'columns':list(c),'rows':n}) as projected:
            result=r.table_plan(data)
        self.assertEqual(projected.call_count,6)
        self.assertEqual(result['comparison']['metadata']['pin'],artifacts['comparison.parquet'])
        self.assertEqual(result['comparison']['values']['columns'],list(r.COMPARISON))
        self.assertEqual(result['selection']['metadata']['rows'],720)
        self.assertEqual(result['preflight']['metadata']['pin'],data['preflight']['artifacts']['summary.parquet'])


if __name__=='__main__':unittest.main()
