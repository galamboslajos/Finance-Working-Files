"""Fabricated status JSON only; never reads actual running-job metadata."""
from pathlib import Path
import importlib.util
import json
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('status_under_test',Path(__file__).with_name('progress_status.py'))
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)


def save(root,name,value):
    p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return p


class Tests(unittest.TestCase):
    def test_running_counts_clocks_and_peaks_without_score_reads(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d)
            save(root,'run/models/pipeline/progress.json',{'status':'RUNNING','completed_estimator_fits':2,
                'completed_preprocessors':2,'completed_phases':5,'current_step':{'action':'fit','year':2020,'arm':'rf_extended'},
                'attempts':[{'phase':'pipeline_old_a001'},{'phase':'pipeline_new_a001'}]})
            save(root,'run/control/active_time.json',{'consumed_seconds':{'models':100,'verification':20}})
            save(root,'run/control/pipeline_old_a001.json',{'status':'COMPLETE','active_seconds':100,
                 'time_category':'models','peak_process_tree_rss_bytes':6*1024**3})
            save(root,'run/control/pipeline_new_a001.json',{'status':'RUNNING','active_seconds':7,
                 'time_category':'models','peak_process_tree_rss_bytes':4*1024**3,'macos_memory_pressure':1})
            (root/'scores.json').write_text('invalid JSON must never be opened')
            z=g.snapshot(root)
            self.assertEqual(z['completed_estimator_fits'],2)
            self.assertEqual(z['overall_guard_peak_rss_bytes'],6*1024**3)
            self.assertEqual(z['estimated_total_active_seconds_by_category']['models'],107)
            self.assertIn('sampled pressure normal',g.text_status(z))

    def test_completed_guard_is_not_double_counted(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d)
            save(root,'run/models/pipeline/progress.json',{'status':'PAUSED','attempts':[{'phase':'pipeline_done_a001'}]})
            save(root,'run/control/active_time.json',{'consumed_seconds':{'models':100}})
            save(root,'run/control/pipeline_done_a001.json',{'status':'COMPLETE','active_seconds':100,'time_category':'models'})
            self.assertEqual(g.snapshot(root)['estimated_total_active_seconds_by_category']['models'],100)

    def test_missing_status_is_explicit_not_success(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            z=g.snapshot(Path(d));self.assertEqual(z['pipeline_status'],'NOT_YET_RECORDED')
            self.assertEqual(z['current_guard_status'],'NOT_YET_RECORDED')

    def test_invalid_guard_reference_cannot_escape_status_directory(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as d:
            root=Path(d);save(root,'run/models/pipeline/progress.json',{'status':'RUNNING','attempts':[{'phase':'../../scores'}]})
            with self.assertRaisesRegex(ValueError,'referenced guard'):g.snapshot(root)


if __name__=='__main__':unittest.main()
