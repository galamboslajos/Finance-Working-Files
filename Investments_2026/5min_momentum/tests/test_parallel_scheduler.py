from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_parallel_learner_suite import select_variant

VARIANTS=('ret_tuned','ranker')
def task(variant):return dict(job=dict(variant=variant))

class Scheduling(unittest.TestCase):
    def test_unequal_duration_jobs_cannot_starve_ranker(self):
        queues={v:[True] for v in VARIANTS};active=[];turn=0
        for _ in range(10):
            v=select_variant(queues,active,VARIANTS,turn);active.append(task(v));turn=1-VARIANTS.index(v)
        for _ in range(100):
            active.remove(next(t for t in active if t['job']['variant']=='ranker'))
            v=select_variant(queues,active,VARIANTS,turn)
            self.assertEqual(v,'ranker');active.append(task(v));turn=1-VARIANTS.index(v)
            self.assertEqual(sum(t['job']['variant']=='ret_tuned' for t in active),5)

    def test_unbalanced_occupancy_prefers_waiting_stream(self):
        self.assertEqual(select_variant({v:[True] for v in VARIANTS},[task('ret_tuned')]*9,VARIANTS,0),'ranker')

    def test_all_slots_reusable_when_other_queue_done(self):
        queues={'ret_tuned':[True],'ranker':[]}
        self.assertEqual(select_variant(queues,[task('ret_tuned')]*9,VARIANTS,1),'ret_tuned')

    def test_ties_alternate_and_empty_queues_reject(self):
        for turn,v in enumerate(VARIANTS):
            self.assertEqual(select_variant({k:[True] for k in VARIANTS},[],VARIANTS,turn),v)
        with self.assertRaises(ValueError):select_variant({k:[] for k in VARIANTS},[],VARIANTS,0)

if __name__=='__main__':unittest.main()
