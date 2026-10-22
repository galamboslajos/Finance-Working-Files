import unittest
from dmv2.importance_reporting import summarize_importance

class ImportanceAccounting(unittest.TestCase):
    def mapping(self):return [dict(feature_name=n,group_id=g,group_kind='ordinary_sessions',group_label=g,sort_key=[i]) for n,g,i in [('a','5days',0),('b','5days',0),('c','20days',1)]]
    def rows(self):return [dict(feature_name=n,year=y,retained=True,value_importance=v,missing_importance=m,total_importance=v+m) for y in [2020,2021] for n,v,m in [('a',.2,.1),('b',.3,.0),('c',.4,.0)]]
    def test_group_totals_and_per_feature_conserve_global_mass(self):
        s=summarize_importance(self.rows(),self.mapping());self.assertAlmostEqual(s['grouped'][0]['mean_total_importance'],.6)
        self.assertAlmostEqual(s['grouped'][0]['mean_importance_per_retained_feature'],.3)
        self.assertAlmostEqual(s['grouped'][0]['mean_missing_importance'],.1)
        self.assertAlmostEqual(sum(g['mean_total_importance'] for g in s['grouped']),1)
    def test_dropped_is_unestimated_not_zero(self):
        rows=self.rows()
        for r in rows:
            if r['feature_name']=='b':r.update(retained=False,value_importance=None,missing_importance=None,total_importance=None)
            elif r['feature_name']=='a':r.update(value_importance=.5,total_importance=.6)
        s=summarize_importance(rows,self.mapping());self.assertEqual(s['grouped'][0]['retained_count_by_year'],{'2020':1,'2021':1})
        self.assertAlmostEqual(s['grouped'][0]['mean_importance_per_retained_feature'],.6)
    def test_missing_duplicate_or_unknown_original_features_fail(self):
        for rows in [self.rows()[:-1],self.rows()+[self.rows()[0]],self.rows()+[dict(self.rows()[0],feature_name='unknown')]]:
            with self.assertRaises(ValueError):summarize_importance(rows,self.mapping())
    def test_negative_nonfinite_nonconserving_and_filled_dropped_fail(self):
        for change in [dict(value_importance=-.2),dict(value_importance=float('nan')),dict(total_importance=.5),dict(retained=False)]:
            rows=self.rows();rows[0].update(change)
            with self.assertRaises(ValueError):summarize_importance(rows,self.mapping())

if __name__=='__main__':unittest.main()

class SavedArtifactIntegrity(unittest.TestCase):
    def test_relative_report_rejects_traversal_and_tampering(self):
        import tempfile,hashlib
        from pathlib import Path
        from dmv2.importance_reporting import _report_artifact
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'a.json';p.write_text('{}')
            item={'size_bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}
            self.assertEqual(_report_artifact(root,'a.json',item),p.resolve())
            for name in ['../a.json',str(p),'.hidden']:
                with self.assertRaises(ValueError):_report_artifact(root,name,item)
            p.write_text('[]')
            with self.assertRaises(ValueError):_report_artifact(root,'a.json',item)
