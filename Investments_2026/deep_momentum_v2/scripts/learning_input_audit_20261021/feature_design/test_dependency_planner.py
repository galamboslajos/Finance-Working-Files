"""Pure structural adversarial checks; all dates/sessions are synthetic."""
import calendar
from datetime import date,timedelta
import unittest
from dependency_planner import exact_span_dependencies,paper_candidates,classify_patch_fields,fold_map


def days(start,count):
    first=date.fromisoformat(start); return [(first+timedelta(days=i)).isoformat() for i in range(count)]


class StructuralDependencyTests(unittest.TestCase):
    def test_exact_503_session_boundary_is_inclusive_and_then_expires(self):
        sessions=days('2018-01-01',505)
        windows=[dict(signal_date=sessions[i],outcome_month=f'month{i}') for i in (501,502,503,504)]
        actual=exact_span_dependencies(sessions,windows,[sessions[0]],[('volume',0,502)])
        self.assertEqual([r['signal_date'] for r in actual],sessions[501:503])
        self.assertEqual(actual[-1]['patched_field_dates'],{'volume':[sessions[0]]})

    def test_future_repair_never_enters_a_signal_even_with_long_lookback(self):
        sessions=days('2020-01-01',510)
        self.assertEqual(exact_span_dependencies(sessions,[dict(signal_date=sessions[500])],[sessions[501]],[('adj_close',0,502)]),[])

    def test_skip_window_requires_exact_near_and_far_bounds(self):
        sessions=days('2020-01-01',30); window=[dict(signal_date=sessions[25])]
        result=exact_span_dependencies(sessions,window,[sessions[i] for i in (3,4,24,25)],[('adj_close',1,21)])
        self.assertEqual(result[0]['patched_field_dates']['adj_close'],[sessions[4],sessions[24]])

    def test_paper_endpoint_propagates_nov2021_through_nov2022_only(self):
        sessions=days('2020-01-01',365+365+365+365)
        windows=[]
        for year,month in [(2021,11),(2021,12)]+[(2022,m) for m in range(1,13)]:
            signal=f'{year}-{month:02d}-{calendar.monthrange(year,month)[1]:02d}'
            windows.append(dict(signal_date=signal,outcome_month='next'))
        hits=paper_candidates(sessions,windows,['2021-11-30'])
        self.assertEqual(len(hits),13)
        self.assertEqual((hits[0]['signal_date'],hits[-1]['signal_date']),('2021-11-30','2022-11-30'))
        self.assertEqual(set(hits[0]['raw_momentum_horizons']),{'1'})
        self.assertEqual(len(hits[0]['potential_affected_paper_features']),15)
        self.assertEqual(set(hits[-1]['raw_momentum_horizons']),{'12'})

    def test_interior_disca_dates_do_not_enter_paper_endpoint_formulas(self):
        sessions=days('2020-01-01',1460)
        windows=[dict(signal_date=f'2022-{m:02d}-{calendar.monthrange(2022,m)[1]:02d}') for m in range(4,13)]
        self.assertEqual(paper_candidates(sessions,windows,['2022-04-04','2022-04-05','2022-04-06','2022-04-07','2022-04-08']),[])

    def test_missing_volume_is_checked_only_at_dates_actually_referenced(self):
        sessions=days('2020-01-01',4)
        hits=exact_span_dependencies(sessions,[dict(signal_date=sessions[3])],[sessions[0]],[('adj_close',0,3),('volume',0,2)])
        self.assertEqual(set(hits[0]['patched_field_dates']),{'adj_close'})
        self.assertTrue(classify_patch_fields(hits[0],{'adj_close':'present_all_nonnull','volume':'missing'})['execution_patch_supplies_referenced_fields'])
        hits=exact_span_dependencies(sessions,[dict(signal_date=sessions[3])],[sessions[1]],[('adj_close',0,3),('volume',0,2)])
        status=classify_patch_fields(hits[0],{'adj_close':'present_all_nonnull','volume':'missing'})
        self.assertFalse(status['execution_patch_supplies_referenced_fields'])
        self.assertEqual(status['blocked_referenced_patch_fields'],['volume'])

    def test_fold_assignment_uses_following_outcome_month_and_completed_train_year(self):
        windows=[dict(signal_date='2021-11-30',outcome_month='2021-12'),dict(signal_date='2021-12-31',outcome_month='2022-01')]
        folds=[dict(outer_outcome_year=2022,train_outcome_start='2012-01-01',train_outcome_end='2021-12-31',inner_splits=[
            dict(split_id='v2021',train_outcome_start='2012-01-01',train_outcome_end='2020-12-31',validation_outcome_start='2021-01-01',validation_outcome_end='2021-12-31')])]
        result=fold_map(windows,['2021-11-30','2021-12-31'],folds)[0]
        self.assertEqual(result['outer_training_hit_months'],['2021-12'])
        self.assertEqual(result['outer_prediction_hit_months'],['2022-01'])
        self.assertEqual(result['inner_splits'][0]['train_hit_months'],[])
        self.assertEqual(result['inner_splits'][0]['validation_hit_months'],['2021-12'])

    def test_negative_lag_definition_or_noncalendar_patch_is_rejected(self):
        sessions=days('2020-01-01',4)
        with self.assertRaises(ValueError): exact_span_dependencies(sessions,[dict(signal_date=sessions[-1])],[sessions[0]],[('adj_close',-1,2)])
        with self.assertRaises(ValueError): exact_span_dependencies(sessions,[dict(signal_date=sessions[-1])],['2023-01-01'],[('adj_close',0,2)])

if __name__=='__main__': unittest.main()
