"""Synthetic metadata, old-price partition and immutable annual-reference guards."""
from pathlib import Path
import importlib.util,sys,json,tempfile,copy
from dataclasses import replace
import unittest
from unittest.mock import patch
import pyarrow as pa,pyarrow.parquet as pq
try:from dmv2 import walkforward_loading as m
except ImportError:
 path=Path(__file__).parents[1]/'src/dmv2/walkforward_loading.py';s=importlib.util.spec_from_file_location('dmv2.walkforward_loading',path);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m)
class Legacy:
 fingerprint='frozen_legacy';provenance={'only_through':'2019-12-31'}
 def __init__(self,kind='correct'):self.calls=[];self.kind=kind
 def verify(self,root):return None
 def apply(self,t,*,start,end):
  assert all(d<='2019-12-31' for d in t['trading_date'].to_pylist());self.calls.append((start,end));rows=t.to_pylist()
  if self.kind=='reorder':rows.reverse()
  else:
   for r in rows:r['open']+=10.
  if self.kind=='volume':rows[0]['volume']+=1.
  if self.kind=='append':rows.append(dict(rows[0],trading_date='2019-01-03'))
  if self.kind=='futureappend':rows.append(dict(rows[0],trading_date='2020-01-03'))
  return pa.Table.from_pylist(rows,schema=t.schema)
def quotes():return pa.Table.from_pylist([dict(intrinio_security_id='A',trading_date='2019-01-02',open=1.,volume=10.),dict(intrinio_security_id='A',trading_date='2020-01-02',open=2.,volume=20.),dict(intrinio_security_id='B',trading_date='2019-01-02',open=3.,volume=30.),dict(intrinio_security_id='B',trading_date='2020-01-02',open=4.,volume=40.)])
def save(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value))
def pin(root,path):return dict(path=str(path.relative_to(root)),size_bytes=path.stat().st_size,sha256=m._hash(path))
class HistoricalQuoteAdapterTests(unittest.TestCase):
 def test_mixed_native_order_and_post2019_values_preserved(self):
  old=Legacy();a=m.wrap_walkforward_quotes(old);t=quotes();after=a.apply(t,start='2019-01-02',end='2020-01-02');self.assertEqual(after['intrinio_security_id'].to_pylist(),t['intrinio_security_id'].to_pylist());self.assertEqual(after['open'].to_pylist(),[11.,2.,13.,4.]);self.assertEqual(after['volume'].to_pylist(),t['volume'].to_pylist());self.assertEqual(old.calls,[('2019-01-02','2019-12-31')])
 def test_future_only_never_calls_legacy(self):
  old=Legacy();a=m.wrap_walkforward_quotes(old);t=quotes().take(pa.array([1,3]));self.assertIs(a.apply(t),t);self.assertEqual(old.calls,[])
 def test_protected2024_refused_before_legacy(self):
  old=Legacy();t=quotes().to_pylist();t[0]['trading_date']='2024-01-02'
  with self.assertRaises(m.DataContractError):m.wrap_walkforward_quotes(old).apply(pa.Table.from_pylist(t))
  self.assertEqual(old.calls,[])
 def test_appended_actual_pre2020_row_retains_all_originals(self):
  t=quotes();after=m.wrap_walkforward_quotes(Legacy('append')).apply(t);self.assertEqual(len(after),len(t)+1);self.assertEqual(after['trading_date'].to_pylist()[:len(t)],t['trading_date'].to_pylist());self.assertEqual(after['trading_date'][-1].as_py(),'2019-01-03')
 def test_legacy_reordering_refused(self):
  with self.assertRaises(m.DataContractError):m.wrap_walkforward_quotes(Legacy('reorder')).apply(quotes())
 def test_legacy_volume_change_refused(self):
  with self.assertRaises(m.DataContractError):m.wrap_walkforward_quotes(Legacy('volume')).apply(quotes())
 def test_legacy_future_append_refused(self):
  with self.assertRaises(m.DataContractError):m.wrap_walkforward_quotes(Legacy('futureappend')).apply(quotes())
 def test_duplicate_original_keys_refused(self):
  t=quotes()
  with self.assertRaises(m.DataContractError):m.wrap_walkforward_quotes(Legacy()).apply(pa.concat_tables([t,t.slice(0,1)]))
 def test_explicit_scope_cannot_hide_old_row(self):
  with self.assertRaises(m.DataContractError):m.wrap_walkforward_quotes(Legacy()).apply(quotes(),start='2020-01-01')
 def test_financial_loader_requires_explicit_authority(self):
  with self.assertRaises(m.DataContractError):m.load_walkforward_year('/private/tmp',plan=None)
 def test_planner_does_not_accept_protected_year(self):
  with self.assertRaises(m.DataContractError):m.plan_walkforward_year('/private/tmp',2024,amendment=None,approved_missing_route_fallback=True)
 def test_wrap_is_idempotent_and_integrity_checked(self):
  a=m.wrap_walkforward_quotes(Legacy());self.assertIs(m.wrap_walkforward_quotes(a),a);a.verify('/private/tmp')
class AnnualReferenceTests(unittest.TestCase):
 def reused(self,root):
  run=root/'new';parent=root/'parent';y=parent/'years/2010';pidentity={'recipe':'old'};save(parent/'build_identity.json',pidentity);pid=m.canonical_sha256(pidentity);save(y/'audit.json',dict(status='COMPLETE',year=2010,build_identity_sha256=pid,artifacts={}));save(parent/'audit.json',dict(status='COMPLETE',build_identity_sha256=pid));identity={'annual_sources':{'2010':dict(directory=str(y.relative_to(root)),build_identity_sha256=pid,audit_pin=pin(root,y/'audit.json'),parent_identity_pin=pin(root,parent/'build_identity.json'),parent_audit_pin=pin(root,parent/'audit.json'))}};save(run/'build_identity.json',identity);return run,identity
 def test_resolves_reused_recipe_without_requiring_newparentidentity(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();run,identity=self.reused(root)
   with patch.object(m.pq,'read_table',side_effect=AssertionError('financial/table read')):x=m.resolve_walkforward_year(root,run,identity,2010)
   self.assertEqual(x['audit']['year'],2010);self.assertNotEqual(x['build_identity_sha256'],m.canonical_sha256(identity))
 def test_old_parent_pin_change_refused(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();run,identity=self.reused(root);save(root/'parent/build_identity.json',{'changed':True})
   with self.assertRaises(m.DataContractError):m.resolve_walkforward_year(root,run,identity,2010)
 def test_newyear_bound_by_root_completion_audit(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();run=root/'new';identity={'annual_sources':{'2020':dict(directory='years/2020',directory_scope='run',parent_reference='THIS_BUILD',audit_pin=None)}};save(run/'build_identity.json',identity);pid=m.canonical_sha256(identity);save(run/'years/2020/audit.json',dict(status='COMPLETE',year=2020,build_identity_sha256=pid));save(run/'audit.json',dict(status='COMPLETE',build_identity_sha256=pid,years={'2020':pin(run,run/'years/2020/audit.json')}));x=m.resolve_walkforward_year(root,run,identity,2020);self.assertEqual(x['directory'],run/'years/2020');self.assertEqual(x['build_identity_sha256'],pid)
 def test_missing_explicit_year_refuses(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d).resolve();run,identity=self.reused(root)
   with self.assertRaises(m.DataContractError):m.resolve_walkforward_year(root,run,identity,2020)
if __name__=='__main__':unittest.main()
