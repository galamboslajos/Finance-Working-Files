"""Verified cash-plus-stub signs, retirement ordering, and double-count guards."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from dmv2.action_accounting import run_action_accounting
from dmv2.action_loading import EVENT_TERMS
from dmv2.sni_action import SNI_EVENT
from dmv2.universe import DataContractError

try:
    from dmv2 import vvc_action as worker
except ImportError:
    spec = importlib.util.spec_from_file_location('dmv2.vvc_action', Path(__file__).with_name('vvc_action.py'))
    worker = importlib.util.module_from_spec(spec); spec.loader.exec_module(worker)

JANUARY = pd.bdate_range('2019-01-02', '2019-01-31').difference(pd.DatetimeIndex(['2019-01-21'])).strftime('%Y-%m-%d').tolist()


def controls():
    return pd.DataFrame([dict(trading_date=day, intrinio_security_id=worker.VVC_PREDECESSOR_ID,
        open=70., high=70., low=70., close=70., adj_open=70., adj_high=70., adj_low=70., adj_close=70.,
        frequency='daily', intraperiod=False, price_quality_status='accepted', dividend=0., split_ratio=1.) for day in JANUARY])


def evidence(directory):
    path = directory / 'predecessor_controls.parquet'; controls().to_parquet(path, index=False)
    receipt = {'schema': 'dmv2-vvc-action-evidence/v1', 'event': deepcopy(worker.VVC_EVENT),
        'no_new_quote_created': True, 'stub_dividend_not_previously_represented': True,
        'artifacts': {path.name: {'path': path.name, 'size_bytes': path.stat().st_size,
            'sha256': 'sha256:' + hashlib.sha256(path.read_bytes()).hexdigest()}}}
    (directory / 'vvc_action_evidence.json').write_text(json.dumps(receipt))


def old_nine():
    return [dict(event_id=ticker+'-'+terms[1], predecessor_id=terms[0], effective_date=terms[2],
        cash_per_raw_share=terms[3], stock_ratio=terms[4], verified=True,
        original_audit={'preserve': ticker}) for ticker, terms in EVENT_TERMS.items()] + [deepcopy(SNI_EVENT)]


class VvcActionTests(unittest.TestCase):
    def account(self, side='long', held=True):
        sessions = JANUARY + ['2019-02-01']
        rows = []
        for day in sessions:
            for native in ('other', 'early'):
                rows.append(dict(trading_date=day, intrinio_security_id=native, adj_open=100., adj_close=100.,
                    frequency='daily', intraperiod=False, price_quality_status='accepted'))
        prices = pd.concat([controls(), pd.DataFrame(rows)], ignore_index=True)
        selections = []
        for signal, execution, native in [('2018-12-31', '2019-01-02', worker.VVC_PREDECESSOR_ID if held else 'early'),
                                         ('2019-01-31', '2019-02-01', worker.VVC_PREDECESSOR_ID)]:
            selections.extend([dict(signal_date=signal, execution_date=execution,
                source_security_id='vvc' if native == worker.VVC_PREDECESSOR_ID else 'early', intrinio_security_id=native, side=side),
                dict(signal_date=signal, execution_date=execution, source_security_id='other', intrinio_security_id='other',
                    side='short' if side == 'long' else 'long')])
        return run_action_accounting(pd.DataFrame(selections), prices, sessions,
            rebalances=pd.DataFrame([dict(signal_date='2018-12-31', execution_date='2019-01-02'),
                dict(signal_date='2019-01-31', execution_date='2019-02-01')]),
            terminal_date='2019-02-01', events=[worker.VVC_EVENT])

    def test_held_long_receives_merger_cash_and_stub_once_before_rebalance(self):
        result = self.account()
        self.assertEqual(result.status, 'complete_proxy')
        self.assertEqual(len(result.action_events), 1)
        action = result.action_events.iloc[0]
        self.assertAlmostEqual(action.signed_raw_entitlement, .5 / 70.)
        self.assertAlmostEqual(action.signed_cash, (.5 / 70.) * (72. + .41145))
        self.assertAlmostEqual(result.summary['final_nav'], 1. + (.5 / 70.) * (72.41145 - 70.))
        self.assertFalse(result.trades.phase.eq('merger').any())
        retired = result.orders.loc[result.orders.status.eq('unfilled_retired_security')]
        self.assertEqual(len(retired), 1)
        self.assertEqual(retired.iloc[0].new_adjusted_units, 0.)

    def test_held_short_owes_both_signed_cash_components(self):
        result = self.account(side='short')
        self.assertEqual(result.status, 'complete_proxy')
        self.assertAlmostEqual(result.action_events.iloc[0].signed_cash, (-.5 / 70.) * 72.41145)
        self.assertAlmostEqual(result.summary['final_nav'], 1. - (.5 / 70.) * (72.41145 - 70.))

    def test_new_order_at_retired_open_receives_no_cash_or_stub(self):
        result = self.account(held=False)
        self.assertEqual(result.status, 'complete_proxy')
        self.assertTrue(result.action_events.empty)
        self.assertAlmostEqual(result.summary['final_nav'], 1.)
        self.assertEqual((result.orders.status == 'unfilled_retired_security').sum(), 1)

    def test_additive_event_preserves_all_nine_and_decomposes_total_cash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); evidence(directory)
            original = old_nine(); before = deepcopy(original)
            result = worker.append_vvc_action(original, directory, prices=controls())
            self.assertEqual(len(result), 10)
            self.assertEqual(result[:9], before)
            self.assertEqual(original, before)
            self.assertAlmostEqual(result[-1]['cash_per_raw_share'],
                result[-1]['merger_cash_per_raw_share'] + result[-1]['stub_dividend_per_raw_share'])

    def test_existing_dividend_or_adjustment_prevents_double_counting(self):
        for alteration in ('dividend', 'adjustment'):
            with self.subTest(alteration=alteration):
                prices = controls()
                if alteration == 'dividend': prices.loc[0, 'dividend'] = .41145
                else: prices.loc[:, 'adj_close'] *= .99
                with self.assertRaises(DataContractError): worker.validate_vvc_action_prices(prices)

    def test_missing_january31_or_post_suspension_quote_rejected(self):
        with self.assertRaises(DataContractError): worker.validate_vvc_action_prices(controls().iloc[:-1])
        prices = controls(); row = prices.iloc[-1].copy(); row.trading_date = '2019-02-01'
        with self.assertRaises(DataContractError):
            worker.validate_vvc_action_prices(pd.concat([prices, row.to_frame().T], ignore_index=True))

    def test_evidence_drift_or_changed_previous_sni_terms_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary); evidence(directory)
            events = old_nine(); events[-1]['cash_per_raw_share'] += 1.
            with self.assertRaises(DataContractError): worker.append_vvc_action(events, directory)
            with (directory / 'predecessor_controls.parquet').open('ab') as stream: stream.write(b'drift')
            with self.assertRaises(DataContractError): worker.load_vvc_action(directory)

    def test_later_financial_dates_rejected(self):
        prices = controls(); prices.loc[0, 'trading_date'] = '2020-01-02'
        with self.assertRaises(DataContractError): worker.validate_vvc_action_prices(prices)


if __name__ == '__main__': unittest.main()
