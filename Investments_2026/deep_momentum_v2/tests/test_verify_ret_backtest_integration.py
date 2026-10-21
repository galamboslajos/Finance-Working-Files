"""Synthetic selector/verifier interface checks; no local research panels."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow.parquet as pq


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/verify_ret_backtest.py'
if not SCRIPT.exists():
    SCRIPT = ROOT.parent / 'verification/scripts/verify_ret_backtest.py'
FIXTURE = Path(__file__).with_name('test_ret_selection.py')
if not FIXTURE.exists():
    FIXTURE = ROOT.parent / 'implementation/tests/test_ret_selection.py'
v = load_module('ret_independent_interface_under_test', SCRIPT)
f = load_module('ret_selector_synthetic_interface_fixture', FIXTURE)


def exercise(count=25, *, restrict=False, ineligible=False, flat=False,
             withheld=False, shuffle=False):
    frame, payoffs = f.predictions_fixture(count=count, flat=flat)
    frame['identity_eligible'] = True
    frame['requested'] = True
    frame['target_available'] = False  # no forward availability can narrow ranks
    if restrict:
        frame.loc[frame.source_security_id.eq('id000'), 'intrinio_security_id'] = f.EXCLUDED_NATIVE_IDS[0]
        frame.loc[frame.source_security_id.eq('id001'), 'intrinio_security_id'] = f.EXCLUDED_NATIVE_IDS[1]
    if ineligible:
        mask = frame.source_security_id.eq(f'id{count - 1:03d}')
        frame.loc[mask, ['prediction_eligible', 'paper_prediction_eligible']] = False
        frame.loc[mask, list(f.PROBABILITIES)] = np.nan
    if flat:
        eligible = frame.prediction_eligible
        frame.loc[eligible, list(f.PROBABILITIES)] = .1
    if withheld:
        payoffs = dict(payoffs)
        payoffs['2019-12-31'] = f.payoff_fixture('2019-12-31', missing=True)
    if shuffle:
        frame = frame.sample(frac=1, random_state=518).reset_index(drop=True)
    actual = f.select(frame, payoffs)
    independent = {signal: {'mu': (np.asarray(panel.table['payoff'].to_pylist())
        if panel.provenance['vector_available'] else None), 'fingerprint': panel.fingerprint}
        for signal, panel in payoffs.items()}
    tables = {'original_ledger': actual.ledger, 'selected': actual.selections,
        'monthly': actual.monthly, 'rebalances': actual.rebalances}
    windows = f.clock_fixture()[1].iloc[:12].copy()
    report = v.verify_selection(frame, tables, windows, independent)
    return frame, actual, independent, windows, report


class IndependentSelectorInterfaceTests(unittest.TestCase):
    def test_saved_forecasts_and_selection_parquet_round_trip(self):
        frame, _ = f.predictions_fixture(count=25)
        frame['identity_eligible'] = True
        frame['requested'] = True
        frame['target_available'] = False
        # Old saved forecast metadata may have object-backed string columns;
        # retain those source dtypes before applying the new selector.
        for name in ('source_security_id', 'intrinio_security_id', 'signal_date',
                     'outcome_month', 'outcome_start', 'outcome_end'):
            frame[name] = frame[name].astype(object)
        with tempfile.TemporaryDirectory(prefix='dmv2-ret-interface-') as directory:
            folder = Path(directory)
            frame.to_parquet(folder / 'forecasts.parquet', index=False)
            forecasts = pq.ParquetFile(folder / 'forecasts.parquet').read().to_pandas()
            _, payoffs = f.predictions_fixture(count=25)
            actual = f.select(forecasts, payoffs)
            tables = {}
            for name, value in (('original_ledger', actual.ledger), ('selected', actual.selections),
                                ('monthly', actual.monthly), ('rebalances', actual.rebalances)):
                path = folder / (name + '.parquet')
                value.to_parquet(path, index=False)
                tables[name] = pq.ParquetFile(path).read().to_pandas()
            independent = {signal: {'mu': np.asarray(panel.table['payoff'].to_pylist()),
                'fingerprint': panel.fingerprint} for signal, panel in payoffs.items()}
            report = v.verify_selection(forecasts, tables, f.clock_fixture()[1].iloc[:12], independent)
            self.assertEqual(report['selected_slots'], 240)

    def test_noncontiguous_probability_projection_matches_canonical_scores(self):
        frame, actual, _, _, report = exercise()
        # Pandas commonly returns Fortran-contiguous projections. The verifier
        # must reproduce the production class-alignment C-copy before BLAS,
        # after separately validating the mathematical scalar dot product.
        self.assertEqual(report['signal_months'], 12)
        self.assertEqual(report['selected_slots'], 240)
        self.assertTrue(actual.ledger.predicted_class.eq(5).all())

    def test_large_book_with_fixed_restrictions_and_ineligible_rows(self):
        frame, actual, _, _, report = exercise(count=128, restrict=True, ineligible=True)
        self.assertEqual(report['original_rows'], 12 * 128)
        self.assertEqual(report['prediction_rows'], 12 * 127)
        self.assertEqual(report['selected_slots'], 240)
        self.assertFalse(actual.selections.intrinio_security_id.isin(f.EXCLUDED_NATIVE_IDS).any())
        self.assertTrue(actual.ledger.loc[actual.ledger.execution_restricted, 'selection_score'].notna().all())
        self.assertTrue(actual.ledger.loc[~actual.ledger.prediction_eligible, 'selection_score'].isna().all())

    def test_full_row_permutation_preserves_scores_and_exact_ties(self):
        _, actual, _, _, report = exercise(count=25, flat=True, shuffle=True)
        self.assertEqual(report['selected_slots'], 240)
        self.assertTrue(actual.monthly.ret_score_flat.all())

    def test_balanced_floor_half_and_zero_breadth_both_verify(self):
        for count, n in ((5, 2), (1, 0)):
            with self.subTest(count=count):
                _, actual, _, _, report = exercise(count=count)
                self.assertEqual(report['selected_slots'], 12 * 2 * n)
                self.assertTrue(actual.monthly.selected_names_per_side.eq(n).all())

    def test_one_unavailable_payoff_signal_keeps_full_original_ledger(self):
        _, actual, _, _, report = exercise(withheld=True)
        self.assertEqual(report['original_rows'], 12 * 25)
        self.assertEqual(report['selected_slots'], 11 * 20)
        self.assertFalse(report['months'][0]['vector_available'])
        self.assertEqual(report['months'][0]['names_per_side'], 0)
        self.assertTrue(actual.ledger.loc[actual.ledger.signal_date.eq('2019-12-31'), 'selection_score'].isna().all())

    def test_mutated_saved_score_cannot_pass_interface_verifier(self):
        frame, actual, payoffs, windows, _ = exercise()
        changed = actual.ledger.copy()
        changed.loc[0, 'selection_score'] += 1e-12
        with self.assertRaisesRegex(v.VerificationError, 'Stored RET score'):
            v.verify_selection(frame, {'original_ledger': changed,
                'selected': actual.selections, 'monthly': actual.monthly,
                'rebalances': actual.rebalances}, windows, payoffs)


if __name__ == '__main__':
    unittest.main()
