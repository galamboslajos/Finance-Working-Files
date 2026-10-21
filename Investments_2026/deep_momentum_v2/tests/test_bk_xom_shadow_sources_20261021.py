import copy
import unittest

import pyarrow as pa

from dmv2.bk_xom_shadow_sources import (BK_KEY, BK_OLD, BK_SOURCE, BK_UNIFIED, XOM_KEY, XOM_OLD,
    XOM_SOURCE, XOM_UNIFIED, PRICE_COLUMNS, PRICE_SCHEMA, WARNING, compose_price_views,
    normalize_candidates, ordered_keys, route_cohorts)
from dmv2.identity_policy import PROVISIONAL_IDENTITY_POLICY, qualify_identity_records
from dmv2.universe import DataContractError


def member(source='other', native='sec_other', unified='equity_other', d='2016-01-05', **extra):
    return dict(source_security_id=source, signal_date=d, as_of=d,
        unified_security_id=unified, intrinio_security_id=native, indices=['sp500'],
        identity_status='needs_review', source_identity_status='needs_review',
        identity_review_required=True, resolution_status='resolved', identity_eligible=False,
        requested=False, paper_prediction_eligible=False, **extra)


def cohorts(rows=None):
    if rows is None:
        rows = [member(BK_SOURCE, BK_OLD, BK_UNIFIED), member(XOM_SOURCE, XOM_OLD, XOM_UNIFIED), member()]
    return pa.Table.from_pylist(rows)


def routed(rows=None):
    table = cohorts(rows)
    return route_cohorts(table, table, expected_monthly_keys=ordered_keys(table), expected_daily_keys=ordered_keys(table))


def price(native='sec_other', d='2016-01-04', value=100.):
    return dict(intrinio_security_id=native, trading_date=d, frequency='daily', intraperiod=False,
        open=value, high=value+1, low=value-1, close=value, volume=1000.,
        adj_open=value/2, adj_high=(value+1)/2, adj_low=(value-1)/2, adj_close=value/2,
        adj_volume=1000., factor=1., split_ratio=1., dividend=0.,
        raw_ohlc_bounds_valid=True, adjusted_ohlc_bounds_valid=True, price_quality_status='accepted')


def prices(rows):
    return pa.Table.from_pylist(rows, schema=PRICE_SCHEMA)


def sip_rows(dates=('2016-01-04','2016-01-05','2016-01-06')):
    return [dict(security_id=XOM_SOURCE, trading_date=d, query_symbol='XOM', adjustment=mode,
        open=100./scale, high=102./scale, low=99./scale, close=101./scale, volume=1000.)
        for d in dates for mode, scale in [('raw',1),('all',2)]]


SESSIONS = ('2016-01-04','2016-01-05','2016-01-06','2024-01-02')


def candidate(warnings=('2016-01-05',)):
    quality = [dict(trading_date=d, flags=[WARNING] if d in warnings else []) for d in SESSIONS[:3]]
    return normalize_candidates(prices([price(BK_KEY,d) for d in SESSIONS[:3]]),
        pa.Table.from_pylist(sip_rows()), sessions=SESSIONS, xom_quality_metadata=quality,
        expected_xom_warning_dates=warnings)


class RouteContract(unittest.TestCase):
    def test_original_keys_flags_and_typed_provider_preserved(self):
        result = routed(); rows = result.monthly_cohort.to_pylist()
        self.assertEqual(result.requested_ids, tuple(sorted([BK_KEY, XOM_KEY, 'sec_other'])))
        self.assertEqual(rows[0]['shadow_original_intrinio_security_id'], BK_OLD)
        self.assertEqual(rows[1]['shadow_original_intrinio_security_id'], XOM_OLD)
        self.assertEqual(rows[1]['price_route_provider'], 'Alpaca/SIP')
        self.assertIsNone(rows[1]['provider_intrinio_security_id'])
        self.assertTrue(all(r['identity_review_required'] is True and r['identity_eligible'] is True for r in rows))
        self.assertEqual(ordered_keys(result.monthly_cohort), ordered_keys(cohorts()))
        self.assertFalse(result.provenance['quote_availability_used'])

    def test_complete_original_population_required(self):
        original=cohorts(); filtered=original.slice(0,2)
        with self.assertRaises(DataContractError):
            route_cohorts(filtered, original, expected_monthly_keys=ordered_keys(original), expected_daily_keys=ordered_keys(original))

    def test_collision_counts_blocked_other_original_member(self):
        rows=[member(BK_SOURCE,BK_OLD,BK_UNIFIED),member(native=BK_KEY)]
        rows[1].update(identity_review_required=None, identity_status='needs_review')
        result=routed(rows)
        self.assertEqual(len(result.monthly_cohort),2)
        self.assertTrue(all(not r['identity_eligible'] for r in result.monthly_cohort.to_pylist()))
        self.assertNotIn(BK_KEY,result.requested_ids)

    def test_same_unified_alias_blocks_both_even_with_distinct_native(self):
        result=routed([member(XOM_SOURCE,XOM_OLD,XOM_UNIFIED),member(unified=XOM_UNIFIED)])
        rows=result.monthly_cohort.to_pylist()
        self.assertTrue(all(not r['identity_eligible'] for r in rows))
        again=qualify_identity_records(rows,identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertTrue(all(not r['identity_eligible'] for r in again))

    def test_collision_recomputed_not_stale_after_reviewed_move(self):
        rows=[member(BK_SOURCE,BK_OLD,BK_UNIFIED),member(native=BK_OLD)]
        for r in rows:r['identity_status']='target_collision'
        result=routed(rows)
        self.assertTrue(all(r['identity_eligible'] for r in result.monthly_cohort.to_pylist()))
        self.assertTrue(all(r['shadow_original_identity_status']=='target_collision' for r in result.monthly_cohort.to_pylist()))

    def test_no_candidate_history_or_early_route_splice(self):
        result=routed([member(XOM_SOURCE,XOM_OLD,XOM_UNIFIED,d='2015-12-31')])
        row=result.monthly_cohort.to_pylist()[0]
        self.assertEqual(row['intrinio_security_id'],XOM_OLD)
        self.assertEqual(row['price_route_status'],'original_route_no_candidate_interval_or_history_fill')

    def test_ambiguity_and_quarantine_not_revived(self):
        rows=[member()]; rows[0].update(identity_status='historical_identity_conflict',intrinio_security_id=None)
        result=routed(rows)
        self.assertFalse(result.monthly_cohort['identity_eligible'][0].as_py())
        self.assertEqual(result.requested_ids,())

    def test_unknown_source_mapping_protected_dates_and_reapplication_rejected(self):
        for mutate in [lambda r:r.update(unified_security_id='different'),lambda r:r.update(intrinio_security_id=BK_KEY),
                       lambda r:r.update(signal_date='2024-01-02',as_of='2024-01-02')]:
            r=member(BK_SOURCE,BK_OLD,BK_UNIFIED); mutate(r)
            with self.assertRaises(DataContractError):routed([r])
        result=routed()
        with self.assertRaises(DataContractError):
            route_cohorts(result.monthly_cohort,result.daily_cohorts,
                expected_monthly_keys=ordered_keys(result.monthly_cohort),expected_daily_keys=ordered_keys(result.daily_cohorts))

    def test_financial_columns_rejected_from_route_metadata(self):
        table=cohorts().append_column('adj_open',pa.array([1.,1.,1.]))
        with self.assertRaises(DataContractError):
            route_cohorts(table,table,expected_monthly_keys=ordered_keys(table),expected_daily_keys=ordered_keys(table))


class QuoteViews(unittest.TestCase):
    def test_raw_all_pair_values_units_null_actions_and_warning_ledger(self):
        table,ledger=candidate(); rows=table.to_pylist(); xom=[r for r in rows if r['intrinio_security_id']==XOM_KEY]
        self.assertEqual(xom[0]['close'],101.)
        self.assertEqual(xom[0]['adj_close'],50.5)
        self.assertTrue(all(r['factor'] is None and r['dividend'] is None and r['split_ratio'] is None for r in xom))
        flagged=[r for r in ledger if r['retained_precision_warning']]
        self.assertEqual([r['trading_date'] for r in flagged],['2016-01-05'])
        self.assertIsNone(flagged[0]['provider_intrinio_security_id'])
        self.assertTrue(all(r['price_quality_status']=='accepted' for r in xom))
        self.assertFalse(any('warning' in name or 'provider' in name for name in table.column_names))

    def test_erased_warning_missing_pair_or_source_mismatch_rejected(self):
        rows=sip_rows(); quality=[dict(trading_date=d,flags=[]) for d in SESSIONS[:3]]
        def go(items=rows,q=quality,expected=()):
            return normalize_candidates(prices([]),pa.Table.from_pylist(items),sessions=SESSIONS,
                xom_quality_metadata=q,expected_xom_warning_dates=expected)
        with self.assertRaises(DataContractError):go(expected=('2016-01-05',))
        with self.assertRaises(DataContractError):go(items=rows[:-1])
        changed=copy.deepcopy(rows);changed[-1]['security_id']='alias'
        with self.assertRaises(DataContractError):go(items=changed)
        with self.assertRaises(DataContractError):go(items=rows+[rows[0]])

    def test_non_session_and_protected_candidate_keys_rejected(self):
        for d in ('2016-01-09','2024-01-02'):
            with self.assertRaises(DataContractError):
                normalize_candidates(prices([price(BK_KEY,d)]),pa.Table.from_pylist(sip_rows()),
                    sessions=SESSIONS,xom_quality_metadata=[],expected_xom_warning_dates=())

    def compose(self,**overrides):
        candidates,_=candidate()
        args=dict(base_feature_history=prices([price(d=d) for d in SESSIONS[:2]]),
            base_outcome_history=prices([price(d=d) for d in SESSIONS[:3]]), candidates=candidates,
            routing=routed(),raw_start=SESSIONS[0],feature_end=SESSIONS[1],raw_end=SESSIONS[2],
            sessions=SESSIONS,outcome_dates=SESSIONS[1:3],original_requested_ids=('sec_other',BK_OLD,XOM_OLD))
        args.update(overrides);return compose_price_views(**args)

    def test_past_feature_cutoff_separate_outcomes_daily_and_no_value_replacements(self):
        result=self.compose()
        self.assertEqual(result.feature_prices.num_rows,6)
        self.assertEqual(result.outcome_prices.num_rows,6)
        self.assertTrue(all(d<='2016-01-05' for d in result.feature_prices['trading_date'].to_pylist()))
        self.assertIn('2016-01-06',result.outcome_prices['trading_date'].to_pylist())
        self.assertEqual(result.feature_history_raw.slice(0,2).to_pylist(),prices([price(d=d) for d in SESSIONS[:2]]).to_pylist())
        self.assertEqual(result.provenance['base_values_replaced'],0)
        self.assertEqual(tuple(result.feature_prices.column_names),PRICE_COLUMNS)

    def test_blocked_original_quotes_retained_raw_not_builder_views(self):
        result=self.compose(routing=routed([member(BK_SOURCE,BK_OLD,BK_UNIFIED),member(XOM_SOURCE,XOM_OLD,XOM_UNIFIED)]))
        self.assertIn('sec_other',result.feature_history_raw['intrinio_security_id'].to_pylist())
        self.assertNotIn('sec_other',result.feature_prices['intrinio_security_id'].to_pylist())

    def test_full_candidate_bundle_clipped_before_annual_session_validation(self):
        table,_=candidate()
        table=pa.concat_tables([prices([price(BK_KEY,'2007-07-02')]),table,
                                prices([price(BK_KEY,'2023-12-29')])])
        result=self.compose(candidates=table)
        self.assertTrue(all('2016-01-04'<=d<='2016-01-05' for d in result.feature_prices['trading_date'].to_pylist()))
        self.assertNotIn('2007-07-02',result.outcome_history_raw['trading_date'].to_pylist())

    def test_existing_candidate_overlap_or_old_route_prices_fail_closed(self):
        with self.assertRaises(DataContractError):
            self.compose(base_feature_history=prices([price(BK_KEY)]),original_requested_ids=(BK_KEY,))
        for native in (BK_OLD,XOM_OLD):
            with self.assertRaises(DataContractError):self.compose(base_feature_history=prices([price(native)]))

    def test_row_cap_and_future_base_dates_fail_without_truncation(self):
        with self.assertRaises(DataContractError):self.compose(max_rows=1)
        with self.assertRaises(DataContractError):self.compose(base_outcome_history=prices([price(d='2024-01-02')]))


if __name__=='__main__':unittest.main()
