"""Strict producer hash contracts, plus all inherited independent math tests."""
from pathlib import Path
import importlib.util
import hashlib
import json
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('unstopped_verifier_v2', ROOT/'scripts/verify_unstopped_target_comparison_v2.py')
v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v2)


class ExactDigestContracts(unittest.TestCase):
    def test_backtest_plan_matches_independent_plain_hex_not_prefixed_model_contract(self):
        body={'schema':'invented','scenarios':72,'economic_months':47,'model_plan_sha256':'sha256:'+'a'*64}
        encoded=json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
        expected=hashlib.sha256(encoded).hexdigest()
        self.assertEqual(v2.canonical_hex(body),expected)
        self.assertEqual(v2.canonical(body),'sha256:'+expected)
        self.assertNotEqual(expected,v2.canonical(body))

    def test_payoff_body_uses_same_strict_plain_contract_including_unicode(self):
        body={'table':[{'class_id':1,'payoff':.02}], 'monthly':[], 'provenance':{'name':'matched–clock'}}
        expected=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
        self.assertEqual(v2.canonical_hex(body),expected)
        self.assertNotEqual(v2.canonical_hex(body),'sha256:'+expected)
        self.assertNotEqual(v2.canonical_hex(body),expected.upper())

    def test_actual_body_or_payoff_change_still_changes_digest(self):
        before={'scenarios':72,'payoff':.02}
        self.assertNotEqual(v2.canonical_hex(before),v2.canonical_hex({**before,'scenarios':71}))
        self.assertNotEqual(v2.canonical_hex(before),v2.canonical_hex({**before,'payoff':.0200000000001}))

    def test_nonfinite_hash_body_is_rejected(self):
        with self.assertRaises(ValueError):v2.canonical_hex({'payoff':float('nan')})

    def test_only_two_comparisons_switch_to_hex(self):
        source=(ROOT/'scripts/verify_unstopped_target_comparison_v2.py').read_text()
        self.assertIn("model['plan_sha256'] == canonical(",source)
        self.assertIn("bt['plan_sha256'] == canonical_hex(",source)
        self.assertIn("saved['fingerprint'] == canonical_hex(",source)


def load_tests(loader, tests, pattern):
    import test_verify_unstopped_target_comparison as inherited
    inherited.v=v2
    tests.addTests(loader.loadTestsFromModule(inherited))
    return tests


if __name__=='__main__':unittest.main()
