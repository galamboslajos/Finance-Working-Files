"""Fabricated29-event metadata cases only; no financial source/model reads."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).parent))
import terminal_guard_v2 as g


def fixture():
    events=[];original=[];targets=[]
    for i in range(29):
        sid=f'source_{i}';native=f'native_{i}'
        original.append(dict(source_security_id=sid,signal_date='2019-12-31',intrinio_security_id=native))
        target=dict(source_security_id=sid,signal_date='2019-12-31',outcome_month='2020-01',
            prediction_eligible=True,target_available=False,source_return_status='missing_end',
            return_observation_status='source_return_status:missing_end',target_status='source_return_status:missing_end')
        targets.append(target)
        events.append(dict(event_id=f'event_{i}',family='current_reviewed' if i<18 else 'historical_retirement',
            effective_date='2020-01-03',predecessor_id=native,overlap_rows=[dict(source_security_id=sid,
            signal_date='2019-12-31',outcome_month='2020-01',prediction_eligible=True,target_available=False,return_status='missing_end')],
            event_outcome_overlap_rows=1,missing_target_event_overlap_rows=1,post_effective_original_cohort_rows=0,
            post_effective_rows=[],original_predecessor_cohort_rows=1))
    audit=dict(status='PASS_METADATA_AUDIT',known_legal_event_count=29,all_original_keys_preserved=True,new_fits=0)
    return events,original,targets,audit


class TerminalGuardTests(unittest.TestCase):
    def test_all29_events_missing_and_all_original_rows_retained(self):
        events,original,targets,audit=fixture();g.registry_checks(events,audit)
        reports=g.audit_events(events,original,copy.deepcopy(original),targets,copy.deepcopy(targets))
        self.assertEqual(len(reports),29);self.assertTrue(all(not x['target_available'] for x in reports))
        self.assertEqual(sum(x['post_effective_original_rows'] for x in reports),0)

    def test_new_terminal_label_is_not_admitted_as_a_price_continuity_fix(self):
        e,o,t,a=fixture();new=copy.deepcopy(t);new[0]['target_available']=True
        with self.assertRaisesRegex(ValueError,'economic-return policy'):
            g.audit_events(e,o,o,t,new)

    def test_missing_target_cannot_remove_prediction_eligibility(self):
        e,o,t,a=fixture();new=copy.deepcopy(t);new[0]['prediction_eligible']=False
        with self.assertRaisesRegex(ValueError,'prediction cohort'):
            g.audit_events(e,o,o,t,new)

    def test_row_deletion_and_silent_route_rebind_fail(self):
        e,o,t,a=fixture()
        with self.assertRaisesRegex(ValueError,'key set'):g.audit_events(e,o,o[1:],t,t[1:])
        changed=copy.deepcopy(o);changed[0]['intrinio_security_id']='successor'
        with self.assertRaisesRegex(ValueError,'membership routes'):g.audit_events(e,o,changed,t,t)

    def test_post_effective_membership_is_independently_rejected(self):
        e,o,t,a=fixture();e[0]['original_predecessor_cohort_rows']=2
        extra={**o[0],'signal_date':'2020-01-31'};o=o+[extra]
        t=t+[{**t[0],'signal_date':'2020-01-31','outcome_month':'2020-02'}]
        with self.assertRaisesRegex(ValueError,'Post-effective'):g.audit_events(e,o,o,t,t)

    def test_duplicate_family_or_changed_registry_count_rejected(self):
        e,o,t,a=fixture();e[0]['event_id']=e[1]['event_id']
        with self.assertRaisesRegex(ValueError,'29 unique'):g.registry_checks(e,a)
        e,o,t,a=fixture();e[0]['family']='historical_retirement'
        with self.assertRaisesRegex(ValueError,'29 unique'):g.registry_checks(e,a)

    def test_explicit_status_cannot_be_changed_while_staying_unlabelled(self):
        e,o,t,a=fixture();new=copy.deepcopy(t);new[0]['source_return_status']='ok'
        with self.assertRaisesRegex(ValueError,'endpoint status'):g.audit_events(e,o,o,t,new)

    def test_target_metadata_calendar_requires_complete_next_month(self):
        rows=[]
        for month in range(1,13):
            prevyear=2019 if month==1 else 2020;prevmonth=12 if month==1 else month-1
            rows.append(dict(source_security_id='A',signal_date=f'{prevyear}-{prevmonth:02d}-28',outcome_month=f'2020-{month:02d}',
                prediction_eligible=True,target_available=False,source_return_status='missing_end',return_observation_status='missing',target_status='missing'))
        g.validate_target_rows(rows,2020);rows[-1]['outcome_month']='2024-01'
        with self.assertRaisesRegex(ValueError,'annual target month'):g.validate_target_rows(rows,2020)

    def test_footer_projection_never_requests_return_or_class_fields(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            path=Path(d)/'toy.parquet';e,o,t,a=fixture()
            pq.write_table(pa.Table.from_pylist([{**r,'target_class':10,'label_return':-0.999} for r in t]),path)
            with patch.object(pq.ParquetFile,'read',side_effect=AssertionError('Footer pass must not read values')):
                spec=g.project(g.pin(path),g.TARGET)
            self.assertEqual(spec['columns'],list(g.TARGET))
            self.assertNotIn('target_class',spec['columns']);self.assertNotIn('label_return',spec['columns'])

    def test_nullable_source_status_remains_unknown_on_unavailable_ineligible_rows(self):
        rows=[]
        for month in range(1,13):
            py=2019 if month==1 else 2020;pm=12 if month==1 else month-1
            rows.append(dict(source_security_id='A',signal_date=f'{py}-{pm:02d}-28',outcome_month=f'2020-{month:02d}',
                prediction_eligible=False,target_available=False,source_return_status=None,
                return_observation_status=None,target_status='identity_ineligible'))
        g.validate_target_rows(rows,2020)
        self.assertIsNone(rows[0]['source_return_status'])
        rows[0]['prediction_eligible']=True;rows[0]['target_available']=True
        with self.assertRaisesRegex(ValueError,'unknown source status'):g.validate_target_rows(rows,2020)

    def test_actual_terminal_rows_cannot_pass_unknown_missingness_evidence(self):
        e,o,t,a=fixture();new=copy.deepcopy(t);new[0]['return_observation_status']=None
        with self.assertRaisesRegex(ValueError,'explicit missingness evidence'):g.audit_events(e,o,o,t,new)

    def test_event_overlap_must_match_predecessor_not_only_family_count(self):
        e,o,t,a=fixture();e[0]['predecessor_id']=e[1]['predecessor_id']
        with self.assertRaisesRegex(ValueError,'declared predecessor'):g.audit_events(e,o,o,t,t)


if __name__=='__main__':unittest.main()
