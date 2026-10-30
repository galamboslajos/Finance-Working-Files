import unittest

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

from learn_momentum_network import ensemble_graph, fit_graph
from walk_forward import causal_rows, fit_ols, grid_groups, make_folds, predict_ols


class WalkForwardTests(unittest.TestCase):
    def test_annual_expansion_and_whole_date_validation(self):
        dates = pd.bdate_range('2016-01-04', '2026-01-02')
        folds = make_folds(dates)
        self.assertEqual(folds.test_year.tolist(), list(range(2019, 2026)))
        self.assertEqual(folds.training_start.nunique(), 1)
        self.assertTrue(folds.training_signal_dates.is_monotonic_increasing)
        self.assertTrue((folds.validation_signal_dates == np.ceil(folds.training_signal_dates * .1)).all())
        self.assertTrue((folds.fit_labeled_dates_max == folds.fit_signal_dates - 1).all())
        self.assertTrue((folds.validation_labeled_dates_max == folds.validation_signal_dates - 1).all())
        self.assertEqual(folds.iloc[0].test_start, '2018-12-31')
        self.assertEqual(folds.iloc[0].test_label_start, '2019-01-01')
        self.assertEqual(folds.iloc[-1].last_test_label_date, '2025-12-31')

    def test_future_targets_are_purged_only_from_fit_rows(self):
        frame = pd.DataFrame({'date': pd.to_datetime(['2018-12-28','2018-12-31']),
                              'label_date': pd.to_datetime(['2018-12-31','2019-01-02']), 'target': [1., -2.]})
        keep = causal_rows(frame, '2018-01-01', '2018-12-31', '2018-12-31')
        self.assertEqual(keep.tolist(), [True, False])
        self.assertEqual(len(frame), 2)

    def test_scaler_and_ols_use_training_rows_only(self):
        rng = np.random.default_rng(15)
        x = rng.normal(size=(200, 8)); beta = rng.normal(size=8); y = .3 + x @ beta
        model = fit_ols(x[:150], y[:150])
        np.testing.assert_allclose(predict_ols(model, x[150:]), y[150:], atol=1e-12)
        expected = predict_ols(model, x[:20])
        x[150:] *= 1000; y[150:] = -1000
        again = fit_ols(x[:150], y[:150])
        np.testing.assert_array_equal(predict_ols(again, x[:20]), expected)

    def test_hyperparameter_grouping_keeps_all_pairs(self):
        groups = grid_groups()
        self.assertEqual(len(groups), 30)
        self.assertEqual(sum(len(g['pairs']) for g in groups), 121)
        self.assertEqual(len({tuple(p) for g in groups for p in g['pairs']}), 121)

    def test_shared_parameter_product_preserves_normalized_ensemble(self):
        rng = np.random.default_rng(90)
        graphs_a, graphs_b = [], []
        for dimension in [8, 16]:
            z = squareform(pdist(rng.normal(size=(7, dimension)), 'sqeuclidean'))
            a, _, _, _ = fit_graph(z, alpha=.5, beta=2., tolerance=1e-9)
            b, _, _, _ = fit_graph(z, alpha=1., beta=1., tolerance=1e-9)
            np.testing.assert_allclose(a, .5 * b, rtol=1e-7, atol=1e-10)
            graphs_a.append(a); graphs_b.append(b)
        np.testing.assert_allclose(ensemble_graph(graphs_a)[1], ensemble_graph(graphs_b)[1], rtol=1e-7, atol=1e-9)


if __name__ == '__main__':
    unittest.main()
