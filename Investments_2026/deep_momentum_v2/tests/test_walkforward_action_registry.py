"""Synthetic-only proposal and pinned-unit-PASS finalization contracts."""
from pathlib import Path
import importlib.util
import json
import shutil
import tempfile
import unittest
from unittest.mock import patch

BASE=Path(__file__).parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
m=module('registry_builder',BASE/'scripts/build_walkforward_action_registry.py')
fixture=module('unit_review_fixture',Path(__file__).with_name('test_walkforward_action_units.py'))
qa=fixture.m


def setup(root):
    run,unit_plan,unit_path=fixture.completed_fixture(root)
    reviewer=root/'scripts/review_walkforward_action_units.py';reviewer.parent.mkdir()
    shutil.copy2(BASE/'scripts/review_walkforward_action_units.py',reviewer)
    primary=fixture.write(root/'evidence/primary.json',{'source_url':'https://example.test/primary','legal_terms_reviewed':True})
    plan={'schema':m.PLAN,'project_root':str(root),'run_directory':'run',
        'build_identity_sha256':unit_plan['build_identity_sha256'],'configuration':unit_plan['configuration'],
        'unit_reviewer_script':fixture.localpin(root,reviewer),
        'events':[{'draft':fixture.localpin(root,root/'evidence/draft.json'),'evidence':[fixture.localpin(root,primary)]}]}
    plan_path=fixture.write(root/'data/derived/proposal_plan.json',plan)
    proposal_path=root/'data/derived/proposal.json'
    proposal=m.propose(plan_path,m.sha(plan_path),proposal_path)
    return run,unit_path,proposal_path,proposal


class RegistryTests(unittest.TestCase):
    def test_explicit_spinoff_or_unknown_cash_stock_type_is_rejected(self):
        for action_type in ('spinoff','unknown_cash_stock',None):
            with self.subTest(action_type=action_type):
                terms=fixture.event();terms['action_type']=action_type
                with self.assertRaisesRegex(m.RegistryError,'unsupported action_type'):m.normalize(terms)
        m.normalize(fixture.event()) # Preserve missing source type for CMD.
        for action_type in ('cash_merger','stock_merger','mixed_merger'):
            terms=fixture.event();terms['action_type']=action_type;m.normalize(terms)

    def test_source_proposal_never_promotes_original_verified_flag(self):
        original=fixture.event();original['verified']=True
        normalized=m.normalize(original)
        self.assertFalse(normalized['verified']);self.assertTrue(normalized['source_documented_event']['verified'])
        self.assertEqual(normalized['paired_unit_review_status'],'PENDING')
        self.assertEqual(normalized['cash_per_raw_share'],original['cash_per_raw_share'])
        self.assertEqual(normalized['execution_review_conventions'],m.CONVENTIONS)

    def test_real_small_unit_pass_finalizes_new_registry_without_source_mutation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,unit_plan,proposal_path,proposal=setup(root)
            original=(root/'evidence/draft.json').read_bytes();frozen=proposal_path.read_bytes()
            qa.execute(unit_plan,qa.sha(unit_plan),root/'data/derived/units',authorize_postfit=True)
            receipt=root/'data/derived/units/receipt.json'
            result=m.finalize(proposal_path,m.sha(proposal_path),[fixture.localpin(root,receipt)],root/'data/derived/registry.json')
            self.assertEqual(result['schema'],'dmv2-walkforward-actions/v1')
            self.assertTrue(result['events'][0]['verified'])
            self.assertEqual(result['events'][0]['paired_unit_review_status'],'PASS')
            self.assertEqual(result['events'][0]['stock_ratio'],.5)
            self.assertFalse(result['portfolio_performance_certified'])
            self.assertFalse(result['physical_settlement_or_identity_or_source_action_completeness_certified'])
            self.assertEqual((root/'evidence/draft.json').read_bytes(),original)
            self.assertEqual(proposal_path.read_bytes(),frozen)
            with self.assertRaisesRegex(m.RegistryError,'Preserve original'):m.finalize(proposal_path,m.sha(proposal_path),
                [fixture.localpin(root,receipt)],root/'data/derived/registry.json')

    def test_incomplete_model_phase_blocks_even_unit_receipt_read(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,unit_plan,proposal_path,proposal=setup(root)
            model=run/'models/audit.json';value=json.loads(model.read_text());value['summary']['fitted_arms']=15;fixture.write(model,value)
            original=m.read
            def guard(path):
                self.assertNotEqual(Path(path).name,'not_read.json','Unit financial receipt read before finalized-model guard')
                return original(path)
            with patch.object(m,'read',side_effect=guard):
                with self.assertRaisesRegex(m.RegistryError,'All16completed'):m.finalize(proposal_path,m.sha(proposal_path),
                    [{'path':'not_read.json','sha256':'0'*64,'size_bytes':1}],root/'data/derived/registry.json')

    def test_failed_or_widened_unit_gate_cannot_promote(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,unit_plan,proposal_path,proposal=setup(root)
            qa.execute(unit_plan,qa.sha(unit_plan),root/'data/derived/units',authorize_postfit=True)
            receipt=root/'data/derived/units/receipt.json';value=json.loads(receipt.read_text())
            value['event_reviews'][0]['predecessor_basis']['maximum_absolute_residual']=.000201
            fixture.write(receipt,value)
            with self.assertRaisesRegex(m.RegistryError,'gate was widened or failed'):m.finalize(proposal_path,m.sha(proposal_path),
                [fixture.localpin(root,receipt)],root/'data/derived/registry.json')
            value['status']='REFUSED';fixture.write(receipt,value)
            with self.assertRaisesRegex(m.RegistryError,'zero-fit PASS'):m.finalize(proposal_path,m.sha(proposal_path),
                [fixture.localpin(root,receipt)],root/'data/derived/registry.json')

    def test_stale_draft_or_different_reviewer_script_fails_closed(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);run,unit_plan,proposal_path,proposal=setup(root)
            qa.execute(unit_plan,qa.sha(unit_plan),root/'data/derived/units',authorize_postfit=True)
            receipt=root/'data/derived/units/receipt.json';value=json.loads(receipt.read_text())
            value['reviewer_script_sha256']='0'*64;fixture.write(receipt,value)
            with self.assertRaisesRegex(m.RegistryError,'zero-fit PASS'):m.finalize(proposal_path,m.sha(proposal_path),
                [fixture.localpin(root,receipt)],root/'data/derived/registry.json')
            draft=root/'evidence/draft.json';value=json.loads(draft.read_text());value['cash_per_raw_share']=6.;fixture.write(draft,value)
            with self.assertRaisesRegex(m.RegistryError,'Pinned evidence changed'):m.finalize(proposal_path,m.sha(proposal_path),
                [fixture.localpin(root,receipt)],root/'data/derived/registry.json')

    def test_external_namespace_and_past_scope_are_explicit(self):
        terms=fixture.event();terms.update(successor_id='execution_cusip:48214T305',identity_namespace='execution_cusip',
            permanent_identifier='48214T305',successor_provider_identity_asserted=False)
        self.assertEqual(m.normalize(terms)['successor_id'],'execution_cusip:48214T305')
        terms['successor_provider_identity_asserted']=True
        with self.assertRaisesRegex(m.RegistryError,'nonprovider'):m.normalize(terms)
        terms=fixture.event();terms['effective_date']='2019-09-18'
        with self.assertRaisesRegex(m.RegistryError,'outside2020'):m.normalize(terms)


if __name__=='__main__':unittest.main()
