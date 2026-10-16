"""Synthetic metadata checks; no financial fixtures or provider calls."""
from copy import deepcopy
from dataclasses import replace
import hashlib,json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa

from dmv2.identity_amendment import (DEFINITION, SOURCE_ROUTES, WRONG_NATIVE, OLD_ISIN,
    load_identity_amendment, apply_identity_amendment, amend_universe_snapshot,
    require_corrected_price_route)
from dmv2.identity_amendment import read_saved_identity_review
from dmv2.identity_policy import qualify_identity_records, PROVISIONAL_IDENTITY_POLICY
from dmv2.universe import DataContractError, UniverseSnapshot


def record(sid=None,uid=None,native=WRONG_NATIVE,day="2018-01-31"):
    return {"source_security_id":sid or next(iter(SOURCE_ROUTES)),"signal_date":day,
        "unified_security_id":uid or next(iter(SOURCE_ROUTES.values())),
        "intrinio_security_id":native,"identity_status":"needs_review",
        "identity_review_required":True,"resolution_status":"resolved",
        "indices":["sp400"]}


class IdentityAmendmentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        shared={"intrinio_security_id":WRONG_NATIVE,"source_response_hash":"a"*64}
        self.proof={"conclusions":{"old_bed_bath_isin":{"value":OLD_ISIN,
            "verified_in_pinned_siblis_identity_metadata":True},
            "historical_overstock_isin":"US6903701018","conflicting_modern_native_id":WRONG_NATIVE,
            "correct_old_bed_bath_intrinio_native_id":None},
            "metadata_projection":{"intrinio_identity_links":[
                {**shared,"unified_security_id":SOURCE_ROUTES["siblis_isin_c297a5f81a23c9f90503"],
                 "query_identifier_type":"isin","query_identifier":"BBBY",
                 "source_relative_path":"raw_pages/query_identifier=BBBY-synthetic/security.json.gz"},
                {**shared,"unified_security_id":next(iter(SOURCE_ROUTES.values()))},
                {**shared,"unified_security_id":"equity_us_78d1c5654bcd67cc2e83df43",
                 "query_identifier":"US6903701018"}]}}
        self.package()

    def package(self):
        path=self.root/"lookup.json";path.write_text(json.dumps(self.proof))
        self.manifest={"definition":deepcopy(DEFINITION),"evidence":[{"path":"lookup.json",
            "size_bytes":path.stat().st_size,"sha256":"sha256:"+hashlib.sha256(path.read_bytes()).hexdigest()}]}
        (self.root/"amendment.json").write_text(json.dumps(self.manifest))
        self.amendment=load_identity_amendment(self.root)

    def test_wrong_route_disabled_all_original_rows_and_flags_retained(self):
        rows=[record(),record("genuine_overstock","modern_uid")];before=deepcopy(rows)
        result=apply_identity_amendment(rows,self.amendment)
        self.assertEqual(rows,before);self.assertEqual(len(result.records),2)
        fixed=result.records[0]
        self.assertIsNone(fixed["intrinio_security_id"])
        self.assertFalse(fixed["identity_eligible"])
        self.assertEqual(fixed["original_intrinio_security_id"],WRONG_NATIVE)
        self.assertEqual(fixed["source_identity_status"],"needs_review")
        self.assertTrue(fixed["identity_review_required"])
        self.assertEqual(fixed["reviewed_security_isin"],OLD_ISIN)
        self.assertEqual(result.records[1]["intrinio_security_id"],WRONG_NATIVE)
        self.assertTrue(result.records[1]["identity_eligible"])

    def test_common_provisional_qualifier_cannot_readmit_conflict(self):
        result=apply_identity_amendment([record()],self.amendment)
        again=qualify_identity_records(list(result.records),identity_policy=PROVISIONAL_IDENTITY_POLICY)
        self.assertFalse(again[0]["identity_eligible"])

    def test_isin_source_alias_also_blocked_without_reusing_overstock_figi(self):
        sid="siblis_isin_c297a5f81a23c9f90503"
        result=apply_identity_amendment([record(sid,SOURCE_ROUTES[sid])],self.amendment)
        self.assertFalse(result.records[0]["identity_eligible"])

    def test_reference_snapshot_and_source_schema_survive(self):
        # Put the unaffected row first: Arrow must retain later audit fields.
        rows=[record("control","control_uid","control_native"),record()]
        for row in rows:row["as_of"]=row.pop("signal_date")
        snapshot=UniverseSnapshot(pa.Table.from_pylist(rows),pa.table({}),"2018-01-31","combined",
                                  {"source_pins":["synthetic"]},{"constituents":2,"identity_status_counts":{"needs_review":2}})
        before=snapshot.table.to_pylist();fixed=amend_universe_snapshot(snapshot,self.amendment)
        self.assertEqual(snapshot.table.to_pylist(),before)
        self.assertEqual(fixed.table.num_rows,2)
        self.assertEqual(fixed.table.to_pylist()[1]["original_intrinio_security_id"],WRONG_NATIVE)
        self.assertEqual(fixed.provenance["source_pins"],["synthetic"])

    def test_missing_replacement_never_becomes_ready_for_rebuild(self):
        result=apply_identity_amendment([record()],self.amendment)
        with self.assertRaisesRegex(DataContractError,"remain unresolved"):
            require_corrected_price_route(result)

    def test_unknown_new_route_is_never_overwritten(self):
        for change in ({"intrinio_security_id":"possible_new_native"},{"unified_security_id":"new_uid"}):
            with self.subTest(change=change),self.assertRaises(DataContractError):
                apply_identity_amendment([{**record(),**change}],self.amendment)

    def test_financial_columns_rejected_before_any_route_use(self):
        for field in ("adj_close","target_class","momentum","provider_adjusted_return"):
            with self.subTest(field=field),self.assertRaisesRegex(DataContractError,"metadata columns only"):
                apply_identity_amendment([{**record(),field:object()}],self.amendment)

    def test_future_dates_need_explicit_metadata_scope_and_2024_always_rejected(self):
        with self.assertRaises(DataContractError):
            apply_identity_amendment([record(day="2020-01-31")],self.amendment)
        result=apply_identity_amendment([record(day="2020-01-31")],self.amendment,analysis_through="2023-12-31")
        self.assertFalse(result.summary["financial_values_read"])
        with self.assertRaises(DataContractError):
            apply_identity_amendment([record(day="2024-01-31")],self.amendment,analysis_through="2024-12-31")

    def test_conflicting_date_aliases_cannot_bypass_scope_or_collision_date(self):
        for alias in ("as_of","trading_date"):
            with self.subTest(alias=alias),self.assertRaisesRegex(DataContractError,"Conflicting cohort date aliases"):
                apply_identity_amendment([{**record(day="2023-01-31"),alias:"2018-01-31"}],
                    self.amendment,date_field=alias)

    def test_reference_feature_and_request_flags_are_audit_only(self):
        row={**record(),"paper_prediction_eligible":True,"prediction_eligible":True,"requested":True}
        result=apply_identity_amendment([row],self.amendment)
        for field in ("paper_prediction_eligible","prediction_eligible","requested"):
            self.assertNotIn(field,result.records[0])
            self.assertIs(result.records[0]["original_"+field],True)

    def test_changed_config_or_evidence_hash_refused(self):
        self.manifest["definition"]["replacement_native_id"]="invented"
        (self.root/"amendment.json").write_text(json.dumps(self.manifest))
        with self.assertRaises(DataContractError):load_identity_amendment(self.root)
        self.package();(self.root/"lookup.json").write_text("mutated")
        with self.assertRaises(DataContractError):load_identity_amendment(self.root)

    def test_query_type_tag_alone_is_not_old_isin_provider_evidence(self):
        self.proof["metadata_projection"]["intrinio_identity_links"][0]["source_response_hash"]="b"*64
        path=self.root/"lookup.json";path.write_text(json.dumps(self.proof))
        self.manifest["evidence"][0].update(size_bytes=path.stat().st_size,
            sha256="sha256:"+hashlib.sha256(path.read_bytes()).hexdigest())
        (self.root/"amendment.json").write_text(json.dumps(self.manifest))
        with self.assertRaisesRegex(DataContractError,"query-literal"):
            load_identity_amendment(self.root)

    def test_mutated_in_memory_definition_refused(self):
        changed=deepcopy(self.amendment.definition);changed["reviewed_isin"]="fake"
        with self.assertRaises(DataContractError):
            apply_identity_amendment([record()],replace(self.amendment,definition=changed))

    def test_duplicate_original_keys_remain_contract_failure(self):
        with self.assertRaisesRegex(DataContractError,"Duplicate"):
            apply_identity_amendment([record(),record()],self.amendment)

    def saved_audit(self):
        path=self.root/'amended_metadata.parquet';path.write_bytes(b'synthetic hash-only artifact')
        audit={'schema':'dmv2-bbby-identity-review/v1','amendment_sha256':self.amendment.fingerprint,
            'replacement_native_id_resolved':False,'corrected_features_rebuilt':False,'models_refitted':False,
            'summary':{'status':'replacement_pending'},'artifacts':[{'path':path.name,
                'size_bytes':path.stat().st_size,'sha256':'sha256:'+hashlib.sha256(path.read_bytes()).hexdigest()}]}
        (self.root/'audit.json').write_text(json.dumps(audit))

    def test_saved_review_reader_is_passive(self):
        self.saved_audit()
        with patch('pyarrow.parquet.ParquetFile',side_effect=AssertionError('panel decode')):
            result=read_saved_identity_review(self.root)
        self.assertEqual(result['summary']['status'],'replacement_pending')

    def test_saved_review_reader_rejects_changed_artifacts(self):
        self.saved_audit();(self.root/'amended_metadata.parquet').write_bytes(b'mutated')
        with self.assertRaisesRegex(DataContractError,'artifact hash/size'):
            read_saved_identity_review(self.root)


if __name__=="__main__":unittest.main()
