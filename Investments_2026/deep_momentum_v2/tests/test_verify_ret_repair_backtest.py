"""Synthetic independent repair checks; no local financial panels."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1]/'scripts/verify_ret_repair_backtest.py'
spec = importlib.util.spec_from_file_location('independent_ret_repair_under_test', SCRIPT)
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def bar(day='2021-11-01', native='target', price=10., factor=.5):
    return {'trading_date': day, 'intrinio_security_id': native,
        'open': price, 'high': price+1., 'low': price-1., 'close': price+.2,
        'adj_open': price*factor, 'adj_high': (price+1.)*factor,
        'adj_low': (price-1.)*factor, 'adj_close': (price+.2)*factor,
        'frequency': 'daily', 'intraperiod': False, 'price_quality_status': 'accepted',
        'adjusted_ohlc_bounds_valid': True, 'raw_ohlc_bounds_valid': True,
        'factor': factor, 'split_ratio': 1., 'dividend': 0.}


def amendment(row, *, kind='append', previous=None):
    return {'intrinio_security_id': row['intrinio_security_id'], 'trading_date': row['trading_date'],
        'kind': kind, 'original_bar_sha256': None if previous is None else m.bar_hash(previous),
        'replacement_bar_sha256': m.bar_hash(row)}


def unit_fixture():
    target, native = 'target', 'historical'
    controls = ['2021-10-29', '2021-12-02']
    source = pd.DataFrame([{**bar(day, native=native), 'original_source_native': native}
        for day in [controls[0], '2021-11-01', controls[-1]]])
    repairs = pd.DataFrame([bar('2021-11-01')])
    original = pd.DataFrame([bar(day) for day in controls])
    proof = {'intrinio_security_id': target, 'source_native': native,
        'method': 'same_provider_dated_route_with_paired_controls',
        'factor_method': 'per_row_observed_source_adj_over_raw', 'control_dates': controls,
        'max_paired_residual': 2e-4, 'max_control_relative_residual': 2e-4,
        'identity_evidence_pin': {'path': 'synthetic_identity.json', 'sha256': '0'*64, 'size_bytes': 1}}
    return proof, source, repairs, original


class PairedOverlayTests(unittest.TestCase):
    def test_canonical_numpy_and_missing_values_match(self):
        a = bar(); b = deepcopy(a)
        b['factor'] = np.float64(b['factor']); b['intraperiod'] = np.bool_(False)
        a['dividend'] = None; b['dividend'] = np.nan
        self.assertEqual(m.bar_hash(a), m.bar_hash(b))

    def test_append_preserves_every_original_and_has_exact_after_hash(self):
        original = pd.DataFrame([bar('2021-10-29')]); repair = bar()
        result, report = m.reconstruct_overlay(original, pd.DataFrame([repair]), [amendment(repair)],
            sessions=['2021-10-29', '2021-11-01'])
        self.assertEqual(len(result), 2)
        self.assertEqual(m.bar_hash(result.iloc[0]), m.bar_hash(original.iloc[0]))
        self.assertEqual(report[0]['after_bar_sha256'], m.bar_hash(repair))

    def test_append_cannot_overwrite_an_existing_accepted_quote(self):
        row = bar()
        with self.assertRaisesRegex(m.VerificationError, 'overwrite'):
            m.reconstruct_overlay(pd.DataFrame([row]), pd.DataFrame([row]), [amendment(row)], sessions=[row['trading_date']])

    def test_undeclared_or_duplicate_repair_rows_rejected(self):
        original = pd.DataFrame([bar('2021-10-29')]); row = bar()
        for rows, declarations in (([row], []), ([row, row], [amendment(row)])):
            with self.subTest(rows=len(rows)), self.assertRaises(m.VerificationError):
                m.reconstruct_overlay(original, pd.DataFrame(rows), declarations, sessions=['2021-10-29', '2021-11-01'])

    def test_after_hash_and_before_hash_tampering_rejected(self):
        old, row = bar(price=9.), bar()
        a = amendment(row); a['replacement_bar_sha256'] = 'sha256:'+'0'*64
        with self.assertRaisesRegex(m.VerificationError, 'Replacement paired quote hash'):
            m.reconstruct_overlay(pd.DataFrame([bar('2021-10-29')]), pd.DataFrame([row]), [a], sessions=['2021-11-01'])
        a = amendment(row, kind='replace', previous=old); a['original_bar_sha256'] = 'sha256:'+'0'*64
        with self.assertRaisesRegex(m.VerificationError, 'before-bar'):
            m.reconstruct_overlay(pd.DataFrame([old]), pd.DataFrame([row]), [a], sessions=['2021-11-01'])

    def test_accepted_quote_replacement_needs_actual_rejection_callback(self):
        old, row = bar(price=9.), bar(); a = amendment(row, kind='replace', previous=old)
        with self.assertRaisesRegex(m.VerificationError, 'rejection evidence'):
            m.reconstruct_overlay(pd.DataFrame([old]), pd.DataFrame([row]), [a], sessions=['2021-11-01'])
        a['rejection_evidence_pin'] = {'path': 'reviewed_rejection.json'}
        called=[]
        result,_=m.reconstruct_overlay(pd.DataFrame([old]), pd.DataFrame([row]), [a], sessions=['2021-11-01'],
            authenticate_rejection=lambda declaration:called.append(declaration))
        self.assertEqual(len(called),1)
        self.assertEqual(m.bar_hash(result.iloc[0]),m.bar_hash(row))

    def test_future_key_rejected(self):
        row=bar('2024-01-02')
        with self.assertRaises(m.VerificationError):
            m.reconstruct_overlay(pd.DataFrame([bar()]),pd.DataFrame([row]),[amendment(row)],sessions=['2024-01-02'])

    def test_nonfinite_or_invalid_paired_ohlc_rejected(self):
        for field,value in [('open',float('inf')),('adj_close',0.),('low',12.),('intraperiod',0)]:
            row=bar();row[field]=value
            with self.subTest(field=field): self.assertFalse(m.acceptable(row))


class LegalEventTests(unittest.TestCase):
    def fixture(self):
        old=[{'event_id':'old','predecessor_id':'old_native','effective_date':'2020-01-03','cash_per_raw_share':10.,'stock_ratio':0.,'successor_id':None}]
        new={'event_id':'new','predecessor_id':'new_native','effective_date':'2021-11-01','cash_per_raw_share':50.,
             'stock_ratio':0.,'successor_id':None,'verified':True,'evidence_reference':'https://issuer.example/legal', 'action_type':'cash_merger'}
        return old,new

    def test_originals_unchanged_and_definitions_appended_only(self):
        old,new=self.fixture();saved=deepcopy(old)
        result=m.append_events(old,[new],sessions=['2021-11-01'])
        self.assertEqual(old,saved);self.assertEqual(result[0],saved[0]);self.assertEqual(result[1],new)

    def test_duplicate_event_or_predecessor_effective_date_rejected(self):
        old,new=self.fixture()
        for changed in [{**new,'event_id':'old'},{**new,'predecessor_id':'old_native','effective_date':'2020-01-03'}]:
            with self.assertRaises(m.VerificationError): m.append_events(old,[changed],sessions=['2020-01-03','2021-11-01'])

    def test_unsupported_zero_entitlement_or_unverified_event_rejected(self):
        old,new=self.fixture()
        for changed in [{**new,'cash_per_raw_share':0.},{**new,'verified':False},{**new,'stock_ratio':-1.},{**new,'action_type':'invented_exit'}]:
            with self.assertRaises(m.VerificationError):m.append_events(old,[changed],sessions=['2021-11-01'])

    def test_stock_entitlement_requires_distinct_successor(self):
        old,new=self.fixture()
        with self.assertRaises(m.VerificationError):m.append_events(old,[{**new,'stock_ratio':2.}],sessions=['2021-11-01'])
        result=m.append_events(old,[{**new,'stock_ratio':2.,'successor_id':'execution_cusip:123456789','action_type':'mixed_merger'}],sessions=['2021-11-01'])
        self.assertEqual(result[-1]['stock_ratio'],2.)


class PairedNativeUnitTests(unittest.TestCase):
    def test_exact_observed_copy_and_two_sided_controls_pass(self):
        proof,source,repairs,original=unit_fixture()
        result=m.verify_unit_rows(proof,source,repairs,original)
        self.assertEqual(result['repaired_rows'],1)
        self.assertFalse(result['factor_interpolation_or_price_fill'])
        self.assertFalse(result['historical_identity_or_units_certified'])

    def test_unbracketed_controls_rejected_by_default(self):
        proof,source,repairs,original=unit_fixture();proof['control_dates']=['2021-10-28','2021-10-29']
        with self.assertRaisesRegex(m.VerificationError,'surround'):
            m.verify_unit_rows(proof,source,repairs,original)

    def test_interpolation_or_changed_paired_replacement_rejected(self):
        proof,source,repairs,original=unit_fixture();repairs.loc[0,'adj_open']+=.001
        with self.assertRaisesRegex(m.VerificationError,'native-only remap'):
            m.verify_unit_rows(proof,source,repairs,original)

    def test_raw_adjusted_inconsistency_rejected(self):
        proof,source,repairs,original=unit_fixture();source.loc[1,'adj_open']*=1.01
        with self.assertRaisesRegex(m.VerificationError,'inconsistent'):
            m.verify_unit_rows(proof,source,repairs,original)

    def test_source_target_control_unit_mismatch_rejected(self):
        proof,source,repairs,original=unit_fixture();original.loc[0,['open','high','low','close']]*=1.01
        with self.assertRaisesRegex(m.VerificationError,'controls do not match'):
            m.verify_unit_rows(proof,source,repairs,original)

    def test_source_identity_or_extra_date_rejected(self):
        proof,source,repairs,original=unit_fixture()
        source.loc[1,'original_source_native']='another_native'
        with self.assertRaises(m.VerificationError):m.verify_unit_rows(proof,source,repairs,original)

    def test_existing_target_control_cannot_be_replaced_by_external_control(self):
        proof,source,repairs,original=unit_fixture();external=original.copy();external.loc[0,'factor']=.9
        with self.assertRaisesRegex(m.VerificationError,'existing original bar'):
            m.verify_unit_rows(proof,source,repairs,original,target_controls=external)

    def test_provisional_one_for_one_exception_cannot_be_used_for_another_stock(self):
        proof,source,repairs,original=unit_fixture();proof['control_coverage_policy']='two_pre_gap_controls_and_primary_dated_one_for_one_reorganization'
        with self.assertRaisesRegex(m.VerificationError,'exact dated RCM'):
            m.verify_unit_rows(proof,source,repairs,original)

    def test_exact_RCM_provisional_two_pre_gap_control_exception(self):
        proof,_,_,_=unit_fixture()
        proof.update(intrinio_security_id='sec_zMM1vd',source_native='sec_yKbEk4',
            control_dates=['2021-10-28','2021-10-29'],
            control_coverage_policy='two_pre_gap_controls_and_primary_dated_one_for_one_reorganization',
            after_control_unavailable_evidence_pin={'path':'metadata_only_absence.json'})
        dates=pd.bdate_range('2021-11-01','2021-12-01').strftime('%Y-%m-%d').tolist();dates.remove('2021-11-25')
        self.assertEqual(len(dates),22)
        source=pd.DataFrame([{**bar(day,native='sec_yKbEk4'),'original_source_native':'sec_yKbEk4'}
            for day in [*proof['control_dates'],*dates]])
        repairs=pd.DataFrame([bar(day,native='sec_zMM1vd') for day in dates])
        original=pd.DataFrame([bar(day,native='sec_zMM1vd') for day in proof['control_dates']])
        absence={'financial_columns':[],'common_dates':[],
            'provenance':{'columns':['trading_date','intrinio_security_id'],
                'request':{'start':'2021-12-02','end':'2022-06-21','date_column':'trading_date',
                    'filters':{'intrinio_security_id':['sec_yKbEk4','sec_zMM1vd']}}},
            'dates':{'sec_yKbEk4':[str(i) for i in range(138)]}}
        report=m.verify_unit_rows(proof,source,repairs,original,unavailable=absence)
        self.assertTrue(report['provisional_one_for_one_reorganization_exception'])
        self.assertEqual(report['repaired_rows'],22)
        broken=deepcopy(absence);broken['dates']['sec_zMM1vd']=['2021-12-02']
        with self.assertRaisesRegex(m.VerificationError,'post-control absence'):
            m.verify_unit_rows(proof,source,repairs,original,unavailable=broken)


def literal_fixture():
    days=['2022-04-01','2022-04-04','2022-04-11']
    proof={'intrinio_security_id':'target','source_native':'published_sip:CLASS_A',
        'method':m.PUBLISHED_METHOD,'factor_method':'per_row_observed_source_adj_over_raw',
        'published_source_product_id':'russell_1000_2000_3000','published_unified_security_id':'unified_example',
        'published_routes':[{'start':'2022-04-01','end':'2022-04-08','query_symbol':'OLD','selected_source_security_id':'old_context'},
            {'start':'2022-04-11','end':'2022-04-11','query_symbol':'NEW','selected_source_security_id':'new_context'}],
        'control_dates':[days[0],days[-1]],'max_paired_residual':2e-4,'max_control_relative_residual':2e-4}
    rows=[]
    for day in days:
        context='new_context' if day==days[-1] else 'old_context'
        for adjustment,factor in [('raw',1.),('all',.5)]:
            rows.append({'trading_date':day,'unified_security_id':'unified_example','adjustment':adjustment,
                'query_symbol':'NEW' if context=='new_context' else 'OLD','feed':'sip','timeframe':'1Day',
                'selected_source_product_id':'russell_1000_2000_3000','selected_source_security_id':context,
                'open':10.*factor,'high':11.*factor,'low':9.*factor,'close':10.2*factor})
    paired=pd.DataFrame(rows)
    source=m.published_rows(proof,paired,set(days))
    repairs=source.loc[source.trading_date.eq(days[1]),list(m.ACTION_FIELDS)].copy()
    repairs['intrinio_security_id']='target'
    original=source.loc[source.trading_date.isin(proof['control_dates']),list(m.ACTION_FIELDS)].copy()
    original['intrinio_security_id']='target'
    return proof,paired,source,repairs,original


class PublishedRawAllTests(unittest.TestCase):
    def test_observed_adjustment_factor_is_derived_instead_of_assumed_one(self):
        proof,paired,source,repairs,original=literal_fixture()
        self.assertTrue(source.factor.eq(.5).all())
        report=m.verify_published_unit_rows(proof,paired,source,repairs,original)
        self.assertTrue(report['raw_all_factor_observed_not_assumed'])
        self.assertFalse(report['source_context_identity_canonically_promoted'])

    def test_dated_publisher_route_identity_mismatch_rejected(self):
        proof,paired,source,repairs,original=literal_fixture();paired.loc[0,'selected_source_security_id']='wrong_context'
        with self.assertRaisesRegex(m.VerificationError,'missing, duplicated or ambiguous'):
            m.verify_published_unit_rows(proof,paired,source,repairs,original)

    def test_duplicate_or_missing_adjustment_pair_rejected(self):
        proof,paired,source,repairs,original=literal_fixture()
        for modified in [paired.iloc[1:].copy(),pd.concat([paired,paired.iloc[:1]],ignore_index=True)]:
            with self.assertRaises(m.VerificationError):
                m.verify_published_unit_rows(proof,modified,source,repairs,original)

    def test_normalized_row_cannot_claim_another_literal_context(self):
        proof,paired,source,repairs,original=literal_fixture();source.loc[0,'original_source_native']='fabricated'
        with self.assertRaisesRegex(m.VerificationError,'literal published'):
            m.verify_published_unit_rows(proof,paired,source,repairs,original)

    def test_normalized_factor_and_cash_fields_must_match_literal_derivation(self):
        for field,value in [('factor',1.),('dividend',0.),('split_ratio',1.)]:
            proof,paired,source,repairs,original=literal_fixture();source.loc[0,field]=value
            with self.subTest(field=field),self.assertRaisesRegex(m.VerificationError,'literal published'):
                m.verify_published_unit_rows(proof,paired,source,repairs,original)

    def test_future_or_other_product_metadata_never_enters_selected_pairs(self):
        proof,paired,source,repairs,original=literal_fixture()
        poison=paired.iloc[:1].copy();poison['trading_date']='2024-01-02';poison['open']=float('nan')
        other=paired.iloc[:1].copy();other['selected_source_product_id']='another_dataset';other['open']=float('nan')
        combined=pd.concat([paired,poison,other],ignore_index=True)
        result=m.published_rows(proof,combined,set(source.trading_date))
        self.assertEqual(len(result),3)
        self.assertFalse(result.trading_date.eq('2024-01-02').any())

    def test_predecessor_listing_control_uses_distinct_new_share_class_basis(self):
        days=pd.bdate_range('2021-06-14','2021-07-01').strftime('%Y-%m-%d').tolist()
        target='execution_cusip:48214T305'
        proof={'intrinio_security_id':target,'source_native':target,'method':m.PUBLISHED_METHOD,
            'factor_method':'per_row_observed_source_adj_over_raw','published_source_product_id':'russell_1000_2000_3000',
            'published_unified_security_id':'grub_unified','published_routes':[{'start':days[0],'end':days[-1],
                'query_symbol':'GRUB','selected_source_security_id':'grub_context'}],
            'control_dates':[],'control_coverage_policy':m.LISTING_POLICY,'predecessor_control_date':days[0],
            'stock_ratio':3.355,'max_paired_residual':2e-4,'max_control_relative_residual':2e-4}
        paired=[]
        for day in days:
            p=10. if day==days[0] else 3.
            for adjustment,factor in [('raw',1.),('all',.5 if day==days[0] else 2.)]:
                paired.append({'trading_date':day,'unified_security_id':'grub_unified','adjustment':adjustment,
                    'query_symbol':'GRUB','feed':'sip','timeframe':'1Day','selected_source_product_id':'russell_1000_2000_3000',
                    'selected_source_security_id':'grub_context','open':p*factor,'high':(p+1)*factor,
                    'low':(p-1)*factor,'close':(p+.2)*factor})
        paired=pd.DataFrame(paired);normalized=m.published_rows(proof,paired,set(days))
        source=normalized.loc[normalized.trading_date.ne(days[0])].reset_index(drop=True)
        repairs=source.loc[:,list(m.ACTION_FIELDS)].copy()
        original=pd.DataFrame([bar(days[0],native='sec_zEbnBy')])
        predecessor={'schema':'dmv2-ret-repair-predecessor-control/v1','row':original.iloc[0].to_dict()}
        event={'predecessor_id':'sec_zEbnBy','successor_id':target,'effective_date':'2021-06-15','stock_ratio':3.355,'cash_per_raw_share':0.}
        report=m.verify_published_unit_rows(proof,paired,source,repairs,original,predecessor=predecessor,events=[event])
        self.assertEqual(report['repaired_rows'],13)
        self.assertTrue(source.factor.eq(2.).all())
        with self.assertRaisesRegex(m.VerificationError,'3.355 stock entitlement'):
            m.verify_published_unit_rows(proof,paired,source,repairs,original,predecessor=predecessor,events=[{**event,'stock_ratio':1.}])


if __name__=='__main__':unittest.main()
