import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
from dmv2.paper15_baseline_reporting import read_paper15_baseline,checked
class Reader(unittest.TestCase):
 def test_running_is_metadata_only(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'ledger.json').write_text(json.dumps({'status':'RUNNING','phase':'fit-5-2020-rf','fits_completed':0,'active_seconds':1}))
   with patch('builtins.__import__',wraps=__import__) as imp:r=read_paper15_baseline(root)
   self.assertEqual(r['status'],'RUNNING');self.assertFalse(any(c.args[0] in ('joblib','sklearn','pyarrow.parquet','pandas') for c in imp.call_args_list))
 def test_false_completion_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);(root/'ledger.json').write_text(json.dumps({'status':'COMPLETE_VERIFIED','phase':None,'fits_completed':1,'fits_started':1,'completed':[],'active_seconds':1}))
   with self.assertRaises(ValueError):read_paper15_baseline(root)
 def test_oversize_before_bytes(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'x';p.write_text('x')
   with patch.object(Path,'read_bytes',side_effect=AssertionError('no read')):
    with self.assertRaises(ValueError):checked({'path':str(p),'size_bytes':50,'sha256':'x'},maximum=10)
if __name__=='__main__':unittest.main()
