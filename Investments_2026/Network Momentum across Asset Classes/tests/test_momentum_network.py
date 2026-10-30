import unittest

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

from learn_momentum_network import ensemble_graph, fit_graph, graph_objective


class MomentumNetworkTests(unittest.TestCase):
    def test_two_nodes_has_analytic_optimum(self):
        z, alpha, beta = 7., .3, 2.
        distances = np.array([[0., z], [z, 0.]])
        adjacency, _, certificate, _ = fit_graph(distances, alpha, beta, tolerance=1e-11)
        expected = 4 * alpha / (z + np.sqrt(z * z + 32 * alpha * beta))
        self.assertAlmostEqual(adjacency[0, 1], expected, places=10)
        self.assertLess(certificate["primal_dual_gap"], 1e-12)

    def test_dual_matches_independent_primal_solver(self):
        random = np.random.default_rng(352)
        for scale in (1., 40.):
            distances = squareform(pdist(random.normal(size=(7, 5)), "sqeuclidean")) * scale
            alpha, beta = .7, 1.3
            adjacency, _, certificate, _ = fit_graph(distances, alpha, beta, tolerance=1e-7)
            left, right = np.triu_indices(7, 1)
            def objective(w):
                a = np.zeros((7, 7)); a[left, right] = w; a[right, left] = w
                return graph_objective(a, distances, alpha, beta)
            def gradient(w):
                d = np.bincount(left, weights=w, minlength=7) + np.bincount(right, weights=w, minlength=7)
                return distances[left, right] + 4 * beta * w - alpha * (1 / d[left] + 1 / d[right])
            reference = minimize(objective, np.full(len(left), .02 / np.sqrt(scale)), jac=gradient,
                bounds=[(0., None)] * len(left), method="SLSQP", options={"ftol": 1e-10, "maxiter": 2000})
            self.assertTrue(reference.success, reference.message)
            self.assertLess(abs(reference.fun - certificate["primal_objective"]), 2e-7)
            np.testing.assert_allclose(adjacency[left, right], reference.x, atol=3e-6)

    def test_frobenius_penalty_and_trace_factor(self):
        adjacency = np.array([[0., .2, .7], [.2, 0., .5], [.7, .5, 0.]])
        vectors = np.arange(12).reshape(3, 4) * .13
        distances = squareform(pdist(vectors, "sqeuclidean"))
        laplacian = np.diag(adjacency.sum(axis=1)) - adjacency
        reference = np.trace(vectors.T @ laplacian @ vectors) - .3 * np.log(adjacency.sum(axis=1)).sum() + 2. * (adjacency ** 2).sum()
        self.assertAlmostEqual(graph_objective(adjacency, distances, .3, 2.), reference, places=12)

    def test_node_permutation_and_positive_degrees(self):
        random = np.random.default_rng(75)
        distances = squareform(pdist(random.normal(size=(10, 20)), "sqeuclidean"))
        adjacency, _, _, _ = fit_graph(distances)
        order = random.permutation(10)
        shuffled, _, _, _ = fit_graph(distances[np.ix_(order, order)])
        np.testing.assert_allclose(shuffled, adjacency[np.ix_(order, order)], atol=1e-8)
        self.assertTrue((adjacency.sum(axis=1) > 0).all())
        self.assertTrue(np.all(np.diag(adjacency) == 0))
        self.assertTrue(np.array_equal(adjacency, adjacency.T))

    def test_normalization_is_after_averaging(self):
        a = np.array([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])
        b = np.array([[0., 0., 4.], [0., 0., 2.], [4., 2., 0.]])
        average, normalized = ensemble_graph([a, b])
        expected = average / np.sqrt(np.outer(average.sum(axis=1), average.sum(axis=1)))
        np.testing.assert_allclose(normalized, expected)
        wrong = np.mean([ensemble_graph([g])[1] for g in [a, b]], axis=0)
        self.assertGreater(np.max(abs(normalized - wrong)), .01)

    def test_identical_features_and_invalid_inputs(self):
        adjacency, _, certificate, _ = fit_graph(np.zeros((5, 5)))
        self.assertTrue(np.isfinite(adjacency).all())
        self.assertLess(certificate["relative_degree_residual"], 1e-6)
        with self.assertRaises(ValueError):
            fit_graph(np.ones((2, 2)))

    def test_warm_start_preserves_solution_and_validates_alignment(self):
        random = np.random.default_rng(95)
        vectors = random.normal(size=(12, 15))
        distances = squareform(pdist(vectors, "sqeuclidean"))
        _, dual, _, _ = fit_graph(distances, tolerance=1e-9)
        vectors += .001 * random.normal(size=vectors.shape)
        changed = squareform(pdist(vectors, "sqeuclidean"))
        cold, _, cold_check, _ = fit_graph(changed, tolerance=1e-9)
        warm, _, warm_check, _ = fit_graph(changed, tolerance=1e-9, initial_dual=dual)
        np.testing.assert_allclose(warm, cold, rtol=1e-7, atol=1e-9)
        self.assertLess(warm_check['iterations'], cold_check['iterations'])
        with self.assertRaises(ValueError):
            fit_graph(changed, initial_dual=dual[:-1])


if __name__ == "__main__":
    unittest.main()
