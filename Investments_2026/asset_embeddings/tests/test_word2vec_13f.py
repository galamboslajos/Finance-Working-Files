import unittest

import numpy as np
from scipy import sparse

from src.word2vec_13f import (
    Word2VecConfig,
    Word2VecModel,
    _batch_loss_and_gradient,
    _training_examples,
    fit_word2vec,
    score_hidden,
)


class Word2Vec13FTests(unittest.TestCase):
    def test_hidden_rank_gap_is_not_closed_or_used_for_training(self) -> None:
        contexts, targets, skipped = _training_examples(
            [np.array([0, -1, 2, 3], dtype=np.int32)],
            vocabulary_size=5,
            radius=1,
        )
        np.testing.assert_array_equal(targets, [2, 3])
        self.assertEqual(skipped, 1)
        np.testing.assert_array_equal(
            contexts.toarray(),
            [[0, 0, 0, 1, 0], [0, 0, 1, 0, 0]],
        )
        self.assertTrue((contexts.indices >= 0).all())
        self.assertTrue((targets >= 0).all())

    def test_context_is_mean_of_visible_neighbors_only(self) -> None:
        contexts, targets, skipped = _training_examples(
            [np.array([0, 1, -1, 3, 4], dtype=np.int32)],
            vocabulary_size=5,
            radius=2,
        )
        np.testing.assert_array_equal(targets, [0, 1, 3, 4])
        self.assertEqual(skipped, 0)
        np.testing.assert_allclose(
            contexts.toarray(),
            [
                [0, 1, 0, 0, 0],
                [0.5, 0, 0, 0.5, 0],
                [0, 0.5, 0, 0, 0.5],
                [0, 0, 0, 1, 0],
            ],
        )

    def test_tied_softmax_loss_and_gradient_match_direct_calculation(self) -> None:
        embeddings = np.array(
            [[0.2, 0.1], [-0.3, 0.4], [0.5, -0.2]], dtype=np.float64
        )
        contexts = sparse.csr_matrix(
            [[0, 0.5, 0.5], [1, 0, 0]], dtype=np.float64
        )
        targets = np.array([0, 2], dtype=np.int32)
        loss, gradient = _batch_loss_and_gradient(embeddings, contexts, targets)
        means = contexts @ embeddings
        logits = means @ embeddings.T
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        expected_loss = -np.log(probabilities[np.arange(2), targets]).mean()
        self.assertAlmostEqual(loss, expected_loss)
        epsilon = 1e-6
        finite_difference = np.zeros_like(embeddings)
        for row in range(embeddings.shape[0]):
            for column in range(embeddings.shape[1]):
                plus = embeddings.copy()
                minus = embeddings.copy()
                plus[row, column] += epsilon
                minus[row, column] -= epsilon
                plus_loss, _ = _batch_loss_and_gradient(plus, contexts, targets)
                minus_loss, _ = _batch_loss_and_gradient(minus, contexts, targets)
                finite_difference[row, column] = (
                    plus_loss - minus_loss
                ) / (2 * epsilon)
        np.testing.assert_allclose(gradient, finite_difference, rtol=1e-5, atol=1e-6)

    def test_score_hidden_uses_original_rank_two_slot(self) -> None:
        embeddings = np.array(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
            dtype=np.float32,
        )
        config = Word2VecConfig(dimensions=2, radius=1, epochs=1)
        model = Word2VecModel(
            embeddings=embeddings,
            config=config,
            vocabulary_size=4,
            training_examples=1,
            skipped_no_context=0,
            epoch_losses=(0.0,),
        )
        logits = score_hidden(model, np.array([0, -1, 2, 3], dtype=np.int32))
        np.testing.assert_allclose(logits, embeddings @ ((embeddings[0] + embeddings[2]) / 2))
        self.assertEqual(logits.shape, (4,))

    def test_same_seed_and_inputs_produce_identical_model(self) -> None:
        sequences = [
            np.array([0, -1, 1, 2, 3], dtype=np.int32),
            np.array([3, 1, 0, 2], dtype=np.int32),
            np.array([2, 3, 1, 0], dtype=np.int32),
        ]
        config = Word2VecConfig(
            dimensions=2, radius=2, epochs=3, batch_size=3,
            learning_rate=0.01, seed=23,
        )
        first = fit_word2vec(sequences, vocabulary_size=4, config=config)
        second = fit_word2vec(sequences, vocabulary_size=4, config=config)
        np.testing.assert_array_equal(first.embeddings, second.embeddings)
        self.assertEqual(first.epoch_losses, second.epoch_losses)
        self.assertEqual(first.training_examples, second.training_examples)
        self.assertEqual(first.training_examples, 12)
        self.assertEqual(first.skipped_no_context, 0)
        self.assertTrue(np.isfinite(first.embeddings).all())

    def test_invalid_or_unscorable_sequences_fail_explicitly(self) -> None:
        config = Word2VecConfig(dimensions=2, radius=1, epochs=1)
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            fit_word2vec([np.array([0, 4])], vocabulary_size=4, config=config)
        with self.assertRaisesRegex(ValueError, "at most one hidden"):
            fit_word2vec([np.array([-1, 0, -1])], vocabulary_size=4, config=config)
        with self.assertRaisesRegex(ValueError, "repeat"):
            fit_word2vec([np.array([0, 1, 0])], vocabulary_size=4, config=config)
        with self.assertRaisesRegex(ValueError, "visible rank-window neighbor"):
            fit_word2vec([np.array([0, -1])], vocabulary_size=4, config=config)
        model = fit_word2vec([np.array([0, 1])], vocabulary_size=4, config=config)
        with self.assertRaisesRegex(ValueError, "Exactly one hidden"):
            score_hidden(model, np.array([0, 1]))
        with self.assertRaisesRegex(ValueError, "no visible rank-window context"):
            score_hidden(model, np.array([-1]))


if __name__ == "__main__":
    unittest.main()
