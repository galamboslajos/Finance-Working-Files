"""Probability averaging, exact ensemble size, seed diversity and checkpoint safety."""
from pathlib import Path
import json
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import train_xgb_ret_ensemble as ensemble
import ret_backtest as base


class EnsembleTests(unittest.TestCase):
    def test_100_independent_reproducible_member_seeds_per_fold(self):
        seeds=[ensemble.member_seeds('2017Q1',m) for m in range(100)]
        self.assertEqual(len(set(seeds)),100)
        self.assertEqual(seeds[17],ensemble.member_seeds('2017Q1',17))
        self.assertNotEqual(seeds[17],ensemble.member_seeds('2017Q2',17))
        with self.assertRaises(ValueError):
            ensemble.member_seeds('2017Q1',100)

    def test_fresh_80_20_partitions_are_disjoint_and_cover_only_supplied_history(self):
        train,valid=ensemble.split_indices(1000,777)
        self.assertEqual(len(train),800)
        self.assertEqual(len(valid),200)
        self.assertEqual(len(set(train)&set(valid)),0)
        self.assertEqual(set(train)|set(valid),set(range(1000)))
        other,_=ensemble.split_indices(1000,778)
        self.assertFalse(np.array_equal(train,other))

    def test_all_100_probabilities_are_averaged_before_ret(self):
        with tempfile.TemporaryDirectory() as folder:
            paths=[]
            for m in range(100):
                p=np.zeros((1,10),dtype=np.float32)
                p[0,0 if m<60 else 9]=1
                path=Path(folder)/f'{m}.npz'
                np.savez_compressed(path,probabilities=p)
                paths.append(path)
            average=ensemble.average_probabilities(paths)
            self.assertAlmostEqual(average[0,0],.6)
            self.assertAlmostEqual(average[0,9],.4)
            mu=np.zeros(10);mu[0]=.05;mu[9]=-.04
            self.assertAlmostEqual(float((average@mu)[0]),.014)
            with self.assertRaises(ValueError):
                ensemble.average_probabilities(paths[:-1])
            with self.assertRaises(ValueError):
                ensemble.average_probabilities(paths[:-1]+[paths[0]])

    def test_checkpoint_is_committed_only_after_both_verified_files_exist(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            prefix=ensemble.member_prefix(root,'2017Q1',0)
            prefix.parent.mkdir(parents=True)
            prefix.with_suffix('.ubj').write_bytes(b'owned-model-checkpoint')
            self.assertFalse(ensemble.checkpoint_complete(root,'2017Q1',0))
            np.savez_compressed(prefix.with_suffix('.npz'),probabilities=np.full((1,10),.1))
            metadata=dict(member=0,fold_id='2017Q1',
                model_sha256=base.fingerprint(prefix.with_suffix('.ubj')),
                probabilities_sha256=base.fingerprint(prefix.with_suffix('.npz')))
            prefix.with_suffix('.json').write_text(json.dumps(metadata))
            self.assertTrue(ensemble.checkpoint_complete(root,'2017Q1',0))
            prefix.with_suffix('.ubj').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                ensemble.checkpoint_complete(root,'2017Q1',0)


if __name__=='__main__':
    unittest.main()
