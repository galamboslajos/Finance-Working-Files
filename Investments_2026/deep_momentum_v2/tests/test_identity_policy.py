import unittest
from types import SimpleNamespace
import pyarrow as pa
import pandas as pd

from dmv2.identity_policy import (STRICT_IDENTITY_POLICY, PROVISIONAL_IDENTITY_POLICY,
    qualify_identity_records, identity_policy_provenance, select_engineering_ids)
from dmv2.identity_policy import qualify_audit_ledger
from dmv2.universe import DataContractError


def row(sid='a',pid='p-a',status='needs_review',flag=True,day='2000-01-31'):
    return dict(source_security_id=sid,as_of=day,intrinio_security_id=pid,
        unified_security_id='u-'+sid,identity_status=status,identity_review_required=flag,
        resolution_status='resolved')


class IdentityPolicyTests(unittest.TestCase):
    def test_flag_accepted_only_explicitly_and_original_record_is_unchanged(self):
        source=row();original=dict(source)
        self.assertFalse(qualify_identity_records([source])[0]['identity_eligible'])
        p=qualify_identity_records([source],identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]
        self.assertTrue(p['identity_eligible']);self.assertEqual(p['identity_status'],'needs_review')
        self.assertTrue(p['identity_review_required']);self.assertEqual(p['source_identity_status'],'needs_review')
        self.assertEqual(source,original)

    def test_missing_or_unresolved_metadata_remain_blocked(self):
        for key,value in [('unified_security_id',None),('intrinio_security_id',None),
                          ('identity_review_required',None),('identity_review_required',1),
                          ('resolution_status','not_resolved')]:
            with self.subTest(key=key,value=value):
                r=row();r[key]=value
                self.assertFalse(qualify_identity_records([r],identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]['identity_eligible'])
        self.assertFalse(qualify_identity_records([row(flag=False)],identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]['identity_eligible'])

    def test_pandas_missing_identifier_scalars_remain_visible_unlinked_members(self):
        for value in (float('nan'),pd.NA,None):
            r=row(status='unmapped_intrinio');r['intrinio_security_id']=value
            out=qualify_identity_records([r],identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]
            self.assertFalse(out['identity_eligible']);self.assertIsNone(out['intrinio_security_id'])

    def test_every_same_date_collision_member_stays_blocked(self):
        rows=[row('a','p',flag=False,status='resolved'),row('b','p')]
        out=qualify_identity_records(rows,identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertTrue(all(r['identity_status']=='target_collision' and not r['identity_eligible'] for r in out))

    def test_alias_on_other_date_is_not_a_future_collision(self):
        out=qualify_identity_records([row(),row('b','p-a',day='2001-01-31')],identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertTrue(all(r['identity_eligible'] for r in out))

    def test_claimed_eligibility_cannot_override_structural_status(self):
        for status in ('ambiguous_bridge','ambiguous_intrinio','unmapped_bridge','target_collision','index_ambiguous'):
            r=row(status=status);r['identity_eligible']=True;r['source_identity_status']='resolved'
            self.assertFalse(qualify_identity_records([r],identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]['identity_eligible'])

    def test_strict_rechecks_claimed_resolved_and_allows_reduced_legacy_record(self):
        r=row(status='resolved')
        self.assertFalse(qualify_identity_records([r])[0]['identity_eligible'])
        for c in ('identity_review_required','resolution_status','unified_security_id'):r.pop(c)
        self.assertTrue(qualify_identity_records([r])[0]['identity_eligible'])

    def test_provisional_claim_flags_remain_unverified(self):
        p=identity_policy_provenance(PROVISIONAL_IDENTITY_POLICY)
        self.assertFalse(p['knowledge_time_verified']);self.assertFalse(p['survivorship_verified'])
        self.assertTrue(p['rerun_after_identity_repair']);self.assertTrue(p['identity_uncertainty_accepted'])

    def test_unknown_policy_protected_date_duplicate_keys_fail(self):
        for rows,policy in [([row()],'ignore_all'),([row(day='2024-01-31')],PROVISIONAL_IDENTITY_POLICY),
                            ([row(),row()],PROVISIONAL_IDENTITY_POLICY)]:
            with self.assertRaises(DataContractError):qualify_identity_records(rows,identity_policy=policy)

    def test_engineering_smoke_chooses_clean_and_flagged_before_price_availability(self):
        rows=[row('clean-b','p-b',status='resolved',flag=False),row('clean-a','p-a',status='resolved',flag=False),
              row('flagged','p-z'),row('missing',None,status='unmapped_intrinio',flag=None)]
        snapshot=SimpleNamespace(table=pa.Table.from_pylist(rows))
        self.assertEqual(select_engineering_ids(snapshot,2),('p-a','p-b'))
        self.assertEqual(select_engineering_ids(snapshot,2,identity_policy=PROVISIONAL_IDENTITY_POLICY),('p-a','p-z'))

    def test_eligibility_has_no_price_survival_or_source_alias_dependency(self):
        original=row()
        for extras in ({'last_price_date':'2026-03-31','delisted':False},
                       {'price_row_count':0,'observed_isins_json':'[]'},
                       {'price_row_count':999,'last_price_date':'2000-01-01'}):
            result=qualify_identity_records([{**original,**extras}],identity_policy=PROVISIONAL_IDENTITY_POLICY)[0]
            self.assertTrue(result['identity_eligible'])

    def test_separate_audit_view_retains_originals_and_blocks_changed_route(self):
        table=pa.Table.from_pylist([dict(source_security_id=sid,signal_date='2000-01-31',
            unified_security_id='u-'+sid,intrinio_security_id='p-'+sid,
            original_identity_status='needs_review',provider_review_required=True,
            link_resolution_status='resolved',route_agrees_with_snapshot=agrees)
            for sid,agrees in [('a',True),('b',False)]])
        original=table.to_pylist()
        rows,summary=qualify_audit_ledger(table,identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertEqual(table.to_pylist(),original)
        self.assertTrue(rows[0]['identity_eligible']);self.assertFalse(rows[1]['identity_eligible'])
        self.assertEqual(rows[1]['source_identity_status'],'needs_review')
        self.assertEqual(summary[0]['original_members'],2)
        self.assertEqual(summary[0]['identity_eligible_members'],1)


if __name__=='__main__':unittest.main()
