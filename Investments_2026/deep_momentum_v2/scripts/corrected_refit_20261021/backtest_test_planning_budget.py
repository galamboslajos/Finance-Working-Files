"""Invented byte pins and metadata only; no Arrow/source data or ledger writes."""
from pathlib import Path
import tempfile
import sys
import types
import unittest
from unittest.mock import patch
import control as c
import backtest_integration as b
import backtest_runner as r


class PlanningBudget(unittest.TestCase):
    def test_planning_hashes_receipts_without_parsing_metrics(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);opaque=root/'opaque.json';opaque.write_text('NOT JSON: must never be parsed before reservation')
            item=c.pin(opaque)
            control={key:item for key in ('catalog_pin','final_model_verification_pin','model_protocol_pin','execution_phase_pin',
                'execution_verification_pin','execution_backtest_audit_pin','execution_manifest_pin','base_configuration_pin')}
            control['model_verification_pins']={str(y):item for y in b.YEARS}
            control['forecast_entries']=[{'prediction_receipt_pin':item} for _ in range(20)]
            c.save_new(root/'control.json',control)
            planned=b.planning_json_plan(root/'control.json',root/'planning.json')
            self.assertEqual(c.read(planned['path'])['projected_bytes'],opaque.stat().st_size)
            with patch.object(b,'require_reservation',side_effect=ValueError('NOT PRECHARGED')):
                with self.assertRaisesRegex(ValueError,'NOT PRECHARGED'):
                    b.plan_sources(root/'control.json',root/'sources.json',planning_plan=root/'planning.json',planning_reservation='synthetic')
            self.assertFalse((root/'sources.json.STARTED.json').exists())
            malformed=c.read(planned['path']);malformed['projected_bytes']=0
            c.save_new(root/'malformed.json',malformed)
            with patch.object(b,'require_reservation') as reservation:
                with self.assertRaisesRegex(ValueError,'Planning JSON exact set/bytes changed'):
                    b.plan_sources(root/'control.json',root/'sources2.json',planning_plan=root/'malformed.json',planning_reservation='synthetic')
                reservation.assert_not_called()
    def test_nested_fields_fail_instead_of_undercharging(self):
        group=types.SimpleNamespace(num_columns=1,column=lambda i:types.SimpleNamespace(path_in_schema='nested.a',total_compressed_size=10))
        arrow=types.SimpleNamespace(names=['nested'],field=lambda n:types.SimpleNamespace(type='struct'))
        source=types.SimpleNamespace(schema_arrow=arrow,metadata=types.SimpleNamespace(num_rows=1,num_row_groups=1,row_group=lambda i:group))
        parquet=types.ModuleType('pyarrow.parquet');parquet.ParquetFile=lambda p:source
        package=types.ModuleType('pyarrow');package.__path__=[];package.parquet=parquet
        with patch.dict(sys.modules,{'pyarrow':package,'pyarrow.parquet':parquet}),patch.object(b,'check',return_value=Path('/synthetic')):
            with self.assertRaisesRegex(ValueError,'Nested projected'):
                b.footer_projection({'path':'/synthetic','sha256':'x','size_bytes':1},['nested'],1)
    def test_legacy_unicode_canonical_hash(self):
        import hashlib,json
        expected='sha256:'+hashlib.sha256(json.dumps({'evidence':'á'},sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(b.canonical({'evidence':'á'}),expected)
    def fixture(self,root):
        pins={}
        for key in ('control','base','phase','manifest'):
            pins[key]=c.save_new(root/(key+'.json'),{'metadata_only':key})
        source={'schema':'dmv2-corrected-forecasts-backtest-source-plan/v1','scenario_count':120,
            'control_pin':pins['control'],'base_configuration_pin':pins['base'],
            'execution_phase_pin':pins['phase'],'execution_manifest_pin':pins['manifest'],
            'execution_authority':{},'runtime':{},'worker_pins':[],'source_read_projections':{'selection':100,'quote_preflight':200,'quote_replay':500}}
        sp=c.save_new(root/'source.json',source)
        protocol={'schema':'dmv2-corrected-forecasts-backtest-protocol/v1','status':'SEALED_BEFORE_FINANCIAL_READ',
            'source_plan_pin':sp,'allow_withheld_paths':True,'scenarios':120,'financial_2024_plus_authorized':False,
            'runtime':{},'maximum_backtest_output_bytes':1024**3,'limits':{'rss_stop_bytes':4*1024**3,
            'rss_warning_bytes':3*1024**3,'active_seconds_by_category':{'backtest':7200}}}
        c.save_new(root/'protocol.json',protocol)
        return source,sp,pins
    def test_selection_exact_json_plus_projection_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source,sp,pins=self.fixture(root)
            with patch.object(r,'source_pins',return_value=[]):
                result=r.make_plan(root/'source.json',root/'protocol.json','selection',root/'selection_plan.json')
            p=c.read(result['path'])
            self.assertEqual(p['projected_read_bytes'],100+pins['control']['size_bytes']+pins['base']['size_bytes'])
    def test_replay_reserves_metadata_and_values_without_duplicate_json_parse(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);source,sp,pins=self.fixture(root);base=root/'backtest';(base/'selection').mkdir(parents=True);(base/'preflight').mkdir()
            selection={'status':'COMPLETE','source_plan_pin':sp,'selection_sets':15,'artifacts':{}}
            sap=c.save_new(base/'selection/audit.json',selection)
            c.save_new(base/'preflight/audit.json',{'status':'COMPLETE','source_plan_pin':sp,'selection_audit_pin':sap})
            with patch.object(r,'BASE',base),patch.object(r,'source_pins',return_value=[]),patch.object(r,'saved_selection_plans',return_value=({},33)):
                result=r.make_plan(root/'source.json',root/'protocol.json','replay',root/'replay_plan.json')
            p=c.read(result['path']);json_cost=sum(pins[k]['size_bytes'] for k in ('control','phase','manifest'))
            self.assertEqual(p['projected_read_bytes'],33+500+json_cost)
            self.assertEqual(len(p['JSON_pins']),3)

if __name__=='__main__':unittest.main()
