"""Structural metadata fixtures; no research financial values are consumed."""
import unittest

import pandas as pd

from dmv2.identity_impact import plan_identity_impact
from dmv2.universe import DataContractError


class IdentityImpactTests(unittest.TestCase):
    def setUp(self):
        self.sessions=tuple(pd.bdate_range('2018-01-02',periods=400).strftime('%Y-%m-%d'))
        self.days=[self.sessions[i] for i in (100,119,120,351,352)]
        self.monthly=pd.DataFrame([dict(source_security_id=sid,signal_date=day,indices=[sleeve],
            unified_security_id='uid-'+sid,intrinio_security_id='provider-'+sid)
            for day in self.days for sid,sleeve in [('other500','sp500'),('other400','sp400')]])
        self.monthly=pd.concat([self.monthly,pd.DataFrame([dict(source_security_id='BBBY',signal_date=self.days[0],indices=['sp500'],
            unified_security_id='uid-BBBY',intrinio_security_id='wrong')])],ignore_index=True)
        self.daily=self.monthly.loc[self.monthly.source_security_id.eq('BBBY')].copy()

    def plan(self,**changes):
        args=dict(signal_cohorts=self.monthly,daily_cohorts=self.daily,sessions=self.sessions,
            affected_source_ids=['BBBY'],amendment_start=self.days[0],amendment_end=self.days[0])
        args.update(changes)
        return plan_identity_impact(**args)

    def test_local_298_are_only_on_amended_source(self):
        result=self.plan()
        rows=result.dependent_keys.loc[result.dependent_keys.scope.eq('direct_source')]
        self.assertEqual(set(rows.source_security_id),{'BBBY'})
        self.assertEqual(rows.potential_columns_per_row.sum(),298)
        self.assertFalse(result.summary['research_financial_values_read'])
        self.assertTrue(result.summary['potential_not_actual_changed_cells'])

    def test_paper15_and_universe_context_have_nonlocal_same_signal_scope(self):
        rows=self.plan().dependent_keys
        paper=rows.loc[rows.family.eq('paper_momentum_state')]
        self.assertEqual(len(paper),3)
        self.assertTrue(paper.potential_columns_per_row.eq(15).all())
        universe=rows.loc[rows.scope.eq('universe_cross_section')]
        sleeve=rows.loc[rows.scope.eq('same_historical_sleeve')]
        self.assertEqual(len(universe),3)
        self.assertTrue(universe.potential_columns_per_row.eq(155).all())
        self.assertEqual(set(sleeve.source_security_id),{'BBBY','other500'})
        self.assertTrue(sleeve.potential_columns_per_row.eq(155).all())

    def test_relative_tail_includes_251_later_sessions_but_excludes_252(self):
        rows=self.plan().dependent_keys
        market=rows.loc[rows.scope.eq('market_252s')]
        self.assertIn(self.sessions[351],set(market.signal_date))
        self.assertNotIn(self.sessions[352],set(market.signal_date))
        self.assertIn('other400',set(market.loc[market.signal_date.eq(self.sessions[351]),'source_security_id']))
        sleeve=rows.loc[rows.scope.eq('sleeve_252s') & rows.signal_date.eq(self.sessions[351])]
        self.assertEqual(set(sleeve.source_security_id),{'other500'})

    def test_shorter_relative_window_exact_boundary(self):
        rows=self.plan().dependent_keys
        market=rows.loc[rows.scope.eq('market_20s')]
        self.assertIn(self.sessions[119],set(market.signal_date))
        self.assertNotIn(self.sessions[120],set(market.signal_date))

    def test_current_market_state_has_no_forward_24_month_memory(self):
        result=self.plan()
        market=result.dependent_keys.loc[result.dependent_keys.family.eq('market_state')]
        self.assertEqual(set(market.signal_date),{self.days[0]})
        self.assertTrue(market.potential_columns_per_row.eq(62).all())
        self.assertFalse(result.summary['forward_24_month_market_memory'])

    def test_daily_absence_is_unknown_not_filled_from_monthly(self):
        result=self.plan(daily_cohorts=None)
        self.assertTrue(result.summary['nonlocal_relative_closure_unknown'])
        relative=result.dependent_keys.loc[result.dependent_keys.family.eq('relative_context')]
        self.assertEqual(set(relative.source_security_id),{'BBBY'})
        self.assertEqual(set(relative.coverage_basis),{'direct_only_nonlocal_daily_tail_unknown'})
        self.assertFalse(result.summary['daily_membership_filled_from_monthly'])

    def test_identical_annual_warmup_overlap_counted_once(self):
        daily=pd.concat([self.daily.assign(year_artifact=2018),self.daily.assign(year_artifact=2019)],ignore_index=True)
        result=self.plan(daily_cohorts=daily)
        self.assertEqual(result.summary['distinct_affected_daily_membership_days'],1)
        pd.testing.assert_frame_equal(result.scope_counts,self.plan().scope_counts)

    def test_disagreeing_annual_routes_are_rejected(self):
        daily=pd.concat([self.daily,self.daily.assign(intrinio_security_id='different')],ignore_index=True)
        with self.assertRaisesRegex(DataContractError,'disagrees'):
            self.plan(daily_cohorts=daily)

    def test_financial_columns_rejected_without_conversion(self):
        class Poison:
            def __float__(self):raise AssertionError('Financial conversion')
        for frame,name in [(self.monthly,'signal_cohorts'),(self.daily,'daily_cohorts')]:
            with self.assertRaisesRegex(DataContractError,'financial columns'):
                self.plan(**{name:frame.assign(adj_close=Poison())})

    def test_broader_metadata_scope_does_not_authorize_financial_values(self):
        result=self.plan(financial_read_through='2018-01-01')
        self.assertTrue(result.scope_counts.metadata_rows_after_financial_boundary.gt(0).any())
        self.assertFalse(result.summary['research_financial_values_read'])
        self.assertFalse(result.summary['saved_target_or_RET_financial_values_consumed'])

    def test_target_rank_scope_preserves_all_same_signal_rows(self):
        result=self.plan()
        targets=result.dependent_keys.loc[result.dependent_keys.artifact_type.eq('target')]
        self.assertEqual(set(targets.source_security_id),{'BBBY','other500','other400'})
        monthly=result.dependent_keys.loc[result.dependent_keys.artifact_type.eq('monthly_return')]
        self.assertEqual(set(monthly.source_security_id),{'BBBY'})
        window=result.payoff_signal_windows.iloc[0]
        expected=pd.Period(self.days[0][:7],'M')+1
        self.assertEqual(window.first_payoff_signal_month,str(expected))
        self.assertEqual(window.last_possible_payoff_signal_month,str(expected+119))
        self.assertTrue(window.completed_history_gate_unchanged)

    def test_original_inputs_are_unchanged(self):
        monthly,daily=self.monthly.copy(deep=True),self.daily.copy(deep=True)
        self.plan()
        pd.testing.assert_frame_equal(self.monthly,monthly)
        pd.testing.assert_frame_equal(self.daily,daily)

    def test_no_amended_member_means_no_claimed_impact(self):
        result=self.plan(affected_source_ids=['absent'])
        self.assertTrue(result.direct_keys.empty)
        self.assertTrue(result.dependent_keys.empty)
        self.assertEqual(result.summary['predictor_potential_cells_upper_bound'],0)

    def test_protected_dates_and_noncalendar_cohort_dates_rejected(self):
        frame=self.monthly.copy();frame.loc[0,'signal_date']='2024-01-02'
        with self.assertRaisesRegex(DataContractError,'Protected'):
            self.plan(signal_cohorts=frame)
        with self.assertRaisesRegex(DataContractError,'absent'):
            self.plan(sessions=self.sessions[:300])

    def test_count_only_plan_matches_materialized_bounds(self):
        complete=self.plan()
        counted=self.plan(materialize_dependent_keys=False,daily_metadata_provenance={'synthetic_original_key_extract':True})
        self.assertTrue(counted.dependent_keys.empty)
        pd.testing.assert_frame_equal(counted.scope_counts,complete.scope_counts)
        self.assertEqual(counted.summary['predictor_potential_cells_upper_bound'],complete.summary['predictor_potential_cells_upper_bound'])
        self.assertFalse(counted.summary['dependent_keys_materialized'])

    def test_preexisting_admission_flags_are_metadata_not_new_population_filters(self):
        frame=self.monthly.assign(identity_eligible=False,paper_prediction_eligible=False,requested=False)
        result=self.plan(signal_cohorts=frame)
        self.assertEqual(result.summary['original_monthly_rows'],len(frame))
        self.assertEqual(result.summary['direct_source_signal_rows'],1)
        self.assertEqual(result.dependent_keys.loc[result.dependent_keys.family.eq('paper_momentum_state')].shape[0],3)


    def test_missing_affected_daily_sleeve_does_not_silently_erase_tail(self):
        for daily in (self.daily.drop(columns=['indices']),self.daily.assign(indices=None),self.daily.assign(indices=pd.NA)):
            with self.assertRaisesRegex(DataContractError,'Affected daily.*nonempty valid sleeve'):
                self.plan(daily_cohorts=daily)
        daily=self.daily.copy();daily.at[daily.index[0],'indices']=[]
        with self.assertRaisesRegex(DataContractError,'Affected daily.*nonempty valid sleeve'):
            self.plan(daily_cohorts=daily)

    def test_missing_affected_monthly_sleeve_is_rejected(self):
        for value in (None,[],pd.NA):
            monthly=self.monthly.copy(deep=True)
            monthly.at[monthly.index[monthly.source_security_id.eq('BBBY')][0],'indices']=value
            with self.assertRaisesRegex(DataContractError,'Affected monthly.*nonempty valid sleeve'):
                self.plan(signal_cohorts=monthly)

    def test_unknown_affected_sleeve_is_rejected(self):
        daily=self.daily.copy();daily.at[daily.index[0],'indices']=['sp600']
        with self.assertRaisesRegex(DataContractError,'only original'):
            self.plan(daily_cohorts=daily)
        monthly=self.monthly.copy();monthly.at[monthly.index[monthly.source_security_id.eq('BBBY')][0],'indices']=['unknown']
        with self.assertRaisesRegex(DataContractError,'only original'):
            self.plan(signal_cohorts=monthly)

    def test_irrelevant_daily_missing_sleeve_does_not_block_known_scope(self):
        extra=self.daily.copy().assign(source_security_id='unrelated',indices=None)
        actual=self.plan(daily_cohorts=pd.concat([self.daily,extra],ignore_index=True))
        pd.testing.assert_frame_equal(actual.scope_counts,self.plan().scope_counts)
        self.assertFalse(actual.summary['nonlocal_relative_closure_unknown'])

    def test_unknown_other_monthly_sleeve_is_conservative_not_missing_exposure(self):
        monthly=self.monthly.copy(deep=True)
        monthly.loc[monthly.source_security_id.eq('other500'),'indices']=None
        result=self.plan(signal_cohorts=monthly)
        relative=result.dependent_keys.loc[result.dependent_keys.scope.eq('sleeve_252s') & result.dependent_keys.signal_date.eq(self.sessions[351])]
        self.assertIn('other500',set(relative.source_security_id))
        self.assertEqual(result.summary['other_monthly_missing_sleeve_policy'],'conservative_union_of_both_sleeves_for_dependency_bounds')

    def test_explicit_index_id_can_supply_missing_indices_evidence(self):
        daily=self.daily.drop(columns=['indices']).assign(index_id='sp500')
        pd.testing.assert_frame_equal(self.plan(daily_cohorts=daily).scope_counts,self.plan().scope_counts)



if __name__=='__main__':unittest.main()
