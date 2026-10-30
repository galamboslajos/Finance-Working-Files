import unittest

import numpy as np
from scipy.spatial.distance import pdist, squareform

from graph_grid import fit_grid_graph, fit_primal
from learn_momentum_network import fit_graph


class GraphGridTests(unittest.TestCase):
    def test_primal_matches_independent_dual_objective(self):
        rng = np.random.default_rng(618)
        z = squareform(pdist(rng.normal(size=(12, 8)), 'sqeuclidean'))
        reference, _, certificate, _ = fit_graph(z, beta=1., tolerance=1e-9)
        actual, _, check = fit_primal(z, beta=1., tolerance=1e-7)
        self.assertLess(abs(check['primal_objective'] - certificate['primal_objective']), 1e-6)
        np.testing.assert_allclose(actual, reference, atol=3e-6)

    def test_extreme_grid_product_is_finite_and_certified(self):
        rng = np.random.default_rng(610)
        z = 1000 * squareform(pdist(rng.normal(size=(14, 12)), 'sqeuclidean'))
        a, _, check = fit_grid_graph(z, 1e-8)
        self.assertTrue(np.isfinite(a).all())
        self.assertTrue((a.sum(axis=1) > 0).all())
        self.assertLess(check['relative_kkt_residual'], 1e-6)
        self.assertLess(check['gap_per_node'], 2e-6)


if __name__ == '__main__':
    unittest.main()
