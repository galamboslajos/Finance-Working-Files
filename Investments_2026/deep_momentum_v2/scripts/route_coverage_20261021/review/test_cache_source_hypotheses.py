import unittest
from cache_source_hypotheses import triage

class HypothesisTests(unittest.TestCase):
 def test_no_request_is_request_evidence_not_data_absence(self):
  r=triage(requested=False,source_dates=[],cache_dates=[],same_pinned_source=True,complete_metadata_scope=True)
  self.assertEqual(r['state'],'request_or_admission_omission_candidate');self.assertFalse(r['dataset_absent_claimed'])
 def test_exact_source_keys_missing_from_cache_flag_only_candidate(self):
  r=triage(requested=True,source_dates=['2010-01-04'],cache_dates=[],same_pinned_source=True,complete_metadata_scope=True)
  self.assertEqual(r['state'],'cache_or_loader_loss_candidate');self.assertFalse(r['mapping_change_authorized'])
 def test_full_scope_empty_route_is_not_absent_company_history(self):
  r=triage(requested=True,source_dates=[],cache_dates=[],same_pinned_source=True,complete_metadata_scope=True)
  self.assertEqual(r['state'],'no_exact_native_keys_in_this_pinned_source_window');self.assertFalse(r['historical_identity_verified'])
 def test_partial_scope_empty_not_overclaimed(self):
  r=triage(requested=True,source_dates=[],cache_dates=[],same_pinned_source=True,complete_metadata_scope=False)
  self.assertEqual(r['state'],'partial_metadata_empty_result_not_source_absence')
 def test_release_mismatch_precedes_absence_conclusion(self):
  r=triage(requested=True,source_dates=[],cache_dates=['2010-01-04'],same_pinned_source=False,complete_metadata_scope=True)
  self.assertEqual(r['state'],'release_or_lineage_comparison_unresolved')
 def test_missing_metadata_is_unknown(self):
  r=triage(requested=True,source_dates=None,cache_dates=[],same_pinned_source=True,complete_metadata_scope=True)
  self.assertEqual(r['state'],'insufficient_metadata')
 def test_equal_key_sets_do_not_certify_prices_or_identity(self):
  r=triage(requested=True,source_dates=['2010-01-04'],cache_dates=['2010-01-04'],same_pinned_source=True,complete_metadata_scope=True)
  self.assertFalse(r['financial_value_validity_verified']);self.assertFalse(r['mapping_change_authorized'])

if __name__=='__main__':unittest.main()
