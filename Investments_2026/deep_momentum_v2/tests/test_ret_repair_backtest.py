"""Synthetic execution-only overlay and frozen-scope tests; no research data."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from dmv2 import ret_repair_backtest as m
from dmv2.universe import DataContractError


def row(native='native_a', day='2020-01-02', *, accepted=True, price=10.):
    return {'intrinio_security_id': native, 'trading_date': day, 'frequency': 'daily',
        'intraperiod': False, 'price_quality_status': 'accepted' if accepted else 'provider_ohlc_anomaly',
        'raw_ohlc_bounds_valid': accepted, 'adjusted_ohlc_bounds_valid': accepted,
        'open': price, 'high': price + 1, 'low': price - 1, 'close': price,
        'adj_open': price / 2, 'adj_high': (price + 1) / 2,
        'adj_low': (price - 1) / 2, 'adj_close': price / 2,
        'factor': None, 'split_ratio': 1., 'dividend': 0.}


def amendment(before, after, kind='replace', reject=False):
    result = {'intrinio_security_id': after['intrinio_security_id'], 'trading_date': after['trading_date'],
        'kind': kind, 'original_bar_sha256': m.canonical_sha256(m._bar(before)) if before else None,
        'replacement_bar_sha256': m.canonical_sha256(m._bar(after))}
    if reject: result['rejection_evidence_pin'] = {'path': 'independent_rejection.json', 'sha256': 'pinned', 'size_bytes': 10}
    return result


def config():
    pin = {'path': 'private.json', 'sha256': 'pinned', 'size_bytes': 10}
    return {'schema': 'dmv2-ret-repair-execution/v1', 'status': 'FROZEN_EXECUTION_AUTHORIZED',
        'execution_authorized': True, 'resource_limits': deepcopy(m.LIMITS), 'arm_ids': list(m.ARMS),
        'scenarios': 40, 'financial_2024_plus_authorized': False, 'cloud_operations': False,
        'new_estimator_fits': 0, 'new_preprocessor_fits': 0, 'new_inference_calls': 0,
        'scores_and_selections_unchanged': True, 'allow_withheld_paths': True,
        **{name: deepcopy(pin) for name in (*m.PARENT_FIELDS, 'repair_manifest_pin')}}


class QuoteOverlayTests(unittest.TestCase):
    def test_rejected_bar_repair_preserves_all_other_rows_and_original_frame(self):
        before, after, other = row(accepted=False), row(price=12.), row('native_b')
        original = pd.DataFrame([before, other]); snapshot = original.copy(deep=True)
        result = m.overlay_quotes(original, pd.DataFrame([after]), [amendment(before, after)])
        pd.testing.assert_frame_equal(original, snapshot, check_exact=True)
        self.assertEqual(m._bar(result.iloc[0]), after)
        self.assertEqual(m._bar(result.iloc[1]), other)

    def test_append_keeps_original_population_and_adds_only_missing_key(self):
        before, added = row(), row('native_b')
        result = m.overlay_quotes(pd.DataFrame([before]), pd.DataFrame([added]), [amendment(None, added, 'append')])
        self.assertEqual(len(result), 2)
        self.assertEqual(m._bar(result.iloc[0]), before)

    def test_append_cannot_replace_and_replace_cannot_fill_missing_key(self):
        before, after = row(accepted=False), row()
        for original, request in ((pd.DataFrame([before]), amendment(None, after, 'append')),
                                  (pd.DataFrame([row('other')]), amendment(before, after))):
            with self.assertRaises(DataContractError): m.overlay_quotes(original, pd.DataFrame([after]), [request])

    def test_existing_accepted_bar_requires_explicit_rejection_pin(self):
        before, after = row(), row(price=12.)
        with self.assertRaisesRegex(DataContractError, 'rejection evidence'):
            m.overlay_quotes(pd.DataFrame([before]), pd.DataFrame([after]), [amendment(before, after)])
        result = m.overlay_quotes(pd.DataFrame([before]), pd.DataFrame([after]), [amendment(before, after, reject=True)])
        self.assertEqual(result.iloc[0].close, 12.)

    def test_before_and_after_tampering_fail_closed(self):
        before, after = row(accepted=False), row()
        for field in ('original_bar_sha256', 'replacement_bar_sha256'):
            request = amendment(before, after); request[field] = 'tampered'
            with self.subTest(field=field), self.assertRaises(DataContractError):
                m.overlay_quotes(pd.DataFrame([before]), pd.DataFrame([after]), [request])

    def test_extra_or_duplicate_quote_keys_are_rejected(self):
        before, after = row(accepted=False), row()
        for reviewed in (pd.DataFrame([after, after]), pd.DataFrame([row('other')])):
            with self.assertRaises(DataContractError): m.overlay_quotes(pd.DataFrame([before]), reviewed, [amendment(before, after)])
        with self.assertRaises(DataContractError):
            m.overlay_quotes(pd.DataFrame([before, before]), pd.DataFrame([after]), [amendment(before, after)])

    def test_negative_or_inconsistent_raw_and_adjusted_ohlc_rejected(self):
        before, after = row(accepted=False), row()
        for field, value in (('open', -1.), ('adj_high', 1.), ('close', float('inf'))):
            changed = dict(after); changed[field] = value
            with self.subTest(field=field), self.assertRaises(DataContractError):
                m.overlay_quotes(pd.DataFrame([before]), pd.DataFrame([changed]), [amendment(before, after)])

    def test_metadata_only_overlay_does_not_require_prices_or_use_adjustment_units(self):
        before, after = row(accepted=False), row()
        columns = [name for name in m.QUOTE_METADATA_COLUMNS if name in before]
        result = m.overlay_quotes(pd.DataFrame([before])[columns], pd.DataFrame([after])[columns],
            [amendment(before, after)], metadata_only=True)
        self.assertTrue(m._accepted(result.iloc[0]))
        self.assertNotIn('open', result)

    def test_boolean_metadata_flags_are_strict(self):
        for field in ('intraperiod', 'raw_ohlc_bounds_valid', 'adjusted_ohlc_bounds_valid'):
            changed = row(); changed[field] = 0 if field == 'intraperiod' else 1
            with self.subTest(field=field): self.assertFalse(m._accepted(changed))

    def test_null_and_nan_bar_hashes_have_identical_canonical_representation(self):
        first, second = row(), row(); second['factor'] = np.nan
        self.assertEqual(m.canonical_sha256(m._bar(first)), m.canonical_sha256(m._bar(second)))


class FrozenRepairScopeTests(unittest.TestCase):
    def test_frozen_configuration_and_compute_budget(self):
        with patch.object(m, 'read_json', return_value=config()): self.assertEqual(m.configuration('unused')['scenarios'], 40)
        for field, value in (('new_estimator_fits', 1), ('new_inference_calls', 1),
                             ('scores_and_selections_unchanged', False), ('scenarios', 41),
                             ('financial_2024_plus_authorized', True)):
            changed = config(); changed[field] = value
            with patch.object(m, 'read_json', return_value=changed), self.assertRaises(DataContractError): m.configuration('unused')

    def test_read_reservations_cumulative_tamper_and_retry_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory); path = run / 'read_usage.json'
            path.write_text(json.dumps({'schema': 'dmv2-ret-repair-read-usage/v1', 'limit_bytes': m.LIMITS['cumulative_compressed_read_bytes'],
                'total_projected_compressed_bytes': 0, 'reservations': []}))
            m.reserve_reads(run, 'preflight', 10); m.reserve_reads(run, 'backtest', 20)
            self.assertEqual(json.loads(path.read_text())['total_projected_compressed_bytes'], 30)
            with self.assertRaises(DataContractError): m.reserve_reads(run, 'backtest', 20)
            with self.assertRaises(DataContractError): m.reserve_reads(run, 'verification', -1)
            state = json.loads(path.read_text()); state['total_projected_compressed_bytes'] = 0; path.write_text(json.dumps(state))
            with self.assertRaises(DataContractError): m.reserve_reads(run, 'verification', 1)


class IndependentPairedUnitTests(unittest.TestCase):
    def fixture(self):
        controls = [row('target', '2020-01-02'), row('target', '2020-01-06')]
        target = row('target', '2020-01-03')
        source = [row('source', day) for day in ('2020-01-02', '2020-01-03', '2020-01-06')]
        for item in source: item[m.SOURCE_NATIVE_COLUMN] = 'source'
        proof = {'intrinio_security_id': 'target', 'source_native': 'source',
            'control_dates': ['2020-01-02', '2020-01-06'], 'max_paired_residual': 2e-4,
            'max_control_relative_residual': 2e-4}
        observed = {'target': {'source': pd.DataFrame(source), 'target_controls': None}}
        return pd.DataFrame(controls), pd.DataFrame([target]), [proof], observed

    def test_observed_paired_fields_copy_with_native_only_remap(self):
        m.verify_quote_units(*self.fixture())

    def test_missing_control_and_wrong_scale_are_rejected(self):
        original, target, proofs, observed = self.fixture()
        with self.assertRaises(DataContractError): m.verify_quote_units(original.iloc[:1], target, proofs, observed)
        changed = deepcopy(observed); changed['target']['source'].loc[0, 'adj_close'] *= 2
        with self.assertRaises(DataContractError): m.verify_quote_units(original, target, proofs, changed)

    def test_mixed_units_and_edited_overlay_are_rejected(self):
        original, target, proofs, observed = self.fixture()
        changed = deepcopy(observed); changed['target']['source'].loc[1, 'adj_open'] *= 1.1
        with self.assertRaises(DataContractError): m.verify_quote_units(original, target, proofs, changed)
        target.loc[0, 'close'] += .01
        with self.assertRaises(DataContractError): m.verify_quote_units(original, target, proofs, observed)


class PublishedRawAllProofTests(unittest.TestCase):
    def fixture(self):
        original, target, proofs, observed = IndependentPairedUnitTests().fixture()
        proof = proofs[0]
        proof.update(method=m.PUBLISHED_METHOD, source_native='published_route',
            published_source_product_id='russell_1000_2000_3000', published_unified_security_id='unified_target',
            published_routes=[{'start': '2020-01-02', 'end': '2020-01-03', 'query_symbol': 'OLD',
                               'selected_source_security_id': 'source_old'},
                              {'start': '2020-01-06', 'end': '2020-01-06', 'query_symbol': 'NEW',
                               'selected_source_security_id': 'source_new'}])
        source = observed['target']['source'].copy(); literal = []
        source['intrinio_security_id'] = 'published_route'; source['factor'] = .5
        source['dividend'] = None; source['split_ratio'] = None
        for index, item in source.iterrows():
            current = item.trading_date == '2020-01-06'
            identity = 'source_new' if current else 'source_old'
            source.loc[index, m.SOURCE_NATIVE_COLUMN] = identity
            for adjustment, prefix in (('raw', ''), ('all', 'adj_')):
                literal.append({'unified_security_id': 'unified_target', 'trading_date': item.trading_date,
                    'adjustment': adjustment, 'query_symbol': 'NEW' if current else 'OLD', 'feed': 'sip', 'timeframe': '1Day',
                    'selected_source_product_id': 'russell_1000_2000_3000', 'selected_source_security_id': identity,
                    **{field: item[prefix + field] for field in ('open', 'high', 'low', 'close')}})
        observed['target']['source'] = source; observed['target']['published_pairs'] = pd.DataFrame(literal)
        target = source.loc[source.trading_date.eq('2020-01-03')].copy(); target['intrinio_security_id'] = 'target'
        return original, target, proofs, observed

    def test_literal_raw_all_pairs_support_observed_factor_other_than_one(self):
        m.verify_quote_units(*self.fixture())

    def test_normalized_pair_cannot_change_literal_price_or_identity(self):
        original, target, proofs, observations = self.fixture()
        for field, value in (('open', 10.001), (m.SOURCE_NATIVE_COLUMN, 'wrong_source')):
            changed = deepcopy(observations)
            changed['target']['source'].loc[1, field] = value
            with self.subTest(field=field), self.assertRaises(DataContractError):
                m.verify_quote_units(original, target, proofs, changed)

    def test_exact_publisher_route_excludes_aliases_and_requires_every_raw_all_cell(self):
        _, _, proofs, observations = self.fixture(); proof = proofs[0]
        metadata = observations['target']['published_pairs'][list(m.PUBLISHED_META)]
        alias = metadata.iloc[:1].copy(); alias['selected_source_product_id'] = 'another_product'
        combined = pd.concat([metadata, alias], ignore_index=True)
        dates = {'2020-01-02', '2020-01-03', '2020-01-06'}
        self.assertEqual(int(m.published_mask(combined, proof, dates).sum()), 6)
        with self.assertRaises(DataContractError): m.published_mask(metadata.iloc[1:], proof, dates)
        protected = metadata.copy(); protected.loc[0, 'trading_date'] = '2024-01-02'
        with self.assertRaises(DataContractError): m.published_mask(protected, proof, dates)

    def test_supplemental_original_target_controls_cannot_override_cache(self):
        original, target, proofs, observed = self.fixture()
        observed['target']['target_controls'] = original.copy()
        m.verify_quote_units(original.iloc[:1], target, proofs, observed)
        observed['target']['target_controls'].loc[0, 'factor'] = 2.
        with self.assertRaises(DataContractError): m.verify_quote_units(original, target, proofs, observed)


if __name__ == '__main__': unittest.main()
