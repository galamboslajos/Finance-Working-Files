import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from daily_network_series import RollingDistances, align_seed, consecutive_ready, select_nodes


class DailyNetworkTests(unittest.TestCase):
    def test_distance_updates_and_node_changes_match_direct_history(self):
        rng = np.random.default_rng(113)
        cube = rng.normal(size=(14, 8, 8))
        cache = RollingDistances(cube, lookbacks=(3, 5), rebase_sessions=5)
        for t, ids in [(5, [0,1,2,3]), (6, [3,0,2,4]), (7, [0,2,4]), (10, [2,4,5,6])]:
            actual = cache.at(t, ids)
            for w in (3, 5):
                vectors = cube[t-w+1:t+1, ids].transpose(1,0,2).reshape(len(ids), -1)
                np.testing.assert_allclose(actual[w], squareform(pdist(vectors, 'sqeuclidean')), atol=1e-12)

    def test_future_values_cannot_change_distances_or_node_readiness(self):
        rng = np.random.default_rng(114)
        cube = rng.normal(size=(1270, 4, 8))
        original = RollingDistances(cube).at(1260, [0,1,2])
        before = consecutive_ready(cube)[:1261].copy()
        cube[1261:] = np.nan
        changed = RollingDistances(cube).at(1260, [0,1,2])
        for w in original:
            np.testing.assert_array_equal(original[w], changed[w])
        np.testing.assert_array_equal(before, consecutive_ready(cube)[:1261])

    def test_dated_membership_identity_and_feature_gaps_control_nodes(self):
        cohort = pd.DataFrame({'source_security_id':['b','a','c'], 'intrinio_security_id':['n2','n1','n3'],
                               'identity_eligible':[True,True,False]})
        runs = np.array([[1259,1260,1300]])
        ledger, ids = select_nodes(cohort, pd.Index(['n1','n2','n3']), runs, 0)
        self.assertEqual(ledger.source_security_id.tolist(), ['a','b','c'])
        self.assertEqual(ledger.eligible_1260.tolist(), [False,True,False])
        self.assertEqual(ids.tolist(), [1])

    def test_primal_seed_alignment_preserves_only_matching_native_ids(self):
        previous = np.array([[0.,2.,3.],[2.,0.,4.],[3.,4.,0.]])
        actual = align_seed(np.array([5,7,9]), previous, np.array([9,5,10]))
        np.testing.assert_array_equal(actual, [[0,3,0],[3,0,0],[0,0,0]])


if __name__ == '__main__':
    unittest.main()
