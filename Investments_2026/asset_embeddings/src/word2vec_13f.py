"""Tied-embedding CBOW for rank-ordered, point-in-time holding portfolios.

Each sequence contains vocabulary indices in descending holding-size order. A
``-1`` marks an already-hidden asset and retains its original rank position.
This module does not select a universe, split managers, or claim a sealed ASMP
benchmark; those decisions must precede construction of these sequences.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable

import numpy as np
from scipy import sparse


@dataclass(frozen=True)
class Word2VecConfig:
    """Declared training choices; the paper does not specify radius or optimizer."""

    dimensions: int
    radius: int
    epochs: int = 5
    batch_size: int = 128
    learning_rate: float = 0.01
    seed: int = 17
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in ("dimensions", "radius", "epochs", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, Integral):
            raise ValueError("seed must be an integer")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        for name in ("adam_beta1", "adam_beta2"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if not np.isfinite(self.adam_epsilon) or self.adam_epsilon <= 0:
            raise ValueError("adam_epsilon must be finite and positive")


@dataclass(frozen=True)
class Word2VecModel:
    """The trained vectors, example counts, and pre-update epoch losses."""

    embeddings: np.ndarray
    config: Word2VecConfig
    vocabulary_size: int
    training_examples: int
    skipped_no_context: int
    epoch_losses: tuple[float, ...]


def _checked_sequence(raw: np.ndarray, vocabulary_size: int) -> np.ndarray:
    sequence = np.asarray(raw)
    if sequence.ndim != 1 or sequence.dtype.kind not in "iu":
        raise ValueError("Each portfolio must be a one-dimensional integer array")
    if ((sequence < -1) | (sequence >= vocabulary_size)).any():
        raise ValueError("Portfolio token lies outside the vocabulary or hidden sentinel")
    if np.count_nonzero(sequence == -1) > 1:
        raise ValueError("A portfolio can contain at most one hidden slot")
    visible = sequence[sequence >= 0]
    if np.unique(visible).size != visible.size:
        raise ValueError("A portfolio cannot repeat a visible asset")
    return sequence.astype(np.int32, copy=False)


def _training_examples(
    sequences: Iterable[np.ndarray], *, vocabulary_size: int, radius: int
) -> tuple[sparse.csr_matrix, np.ndarray, int]:
    """Return contexts, visible targets, and explicit no-context exclusion count."""

    target_chunks: list[np.ndarray] = []
    row_chunks: list[np.ndarray] = []
    column_chunks: list[np.ndarray] = []
    target_count = 0
    for raw in sequences:
        sequence = _checked_sequence(raw, vocabulary_size)
        length = len(sequence)
        if not length:
            continue
        visible_positions = np.flatnonzero(sequence >= 0)
        target_chunks.append(sequence[visible_positions])
        row_for_position = np.full(length, -1, dtype=np.int32)
        row_for_position[visible_positions] = np.arange(
            target_count, target_count + len(visible_positions), dtype=np.int32
        )
        target_count += len(visible_positions)
        for offset in range(-radius, radius + 1):
            if offset == 0:
                continue
            start = max(0, -offset)
            stop = min(length, length - offset)
            if stop <= start:
                continue
            positions = np.arange(start, stop)
            neighbors = positions + offset
            both_visible = (sequence[positions] >= 0) & (sequence[neighbors] >= 0)
            if both_visible.any():
                row_chunks.append(row_for_position[positions[both_visible]])
                column_chunks.append(sequence[neighbors[both_visible]])
    if not target_count:
        raise ValueError("No visible training targets were provided")
    targets = np.concatenate(target_chunks)
    if not row_chunks:
        raise ValueError("No visible target has a visible rank-window neighbor")
    rows = np.concatenate(row_chunks)
    columns = np.concatenate(column_chunks)
    contexts = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, columns)),
        shape=(target_count, vocabulary_size),
    ).tocsr()
    counts = np.diff(contexts.indptr)
    keep = counts > 0
    contexts.data /= np.repeat(counts, counts)
    return contexts[keep], targets[keep], int((~keep).sum())


def _batch_loss_and_gradient(
    embeddings: np.ndarray, contexts: sparse.csr_matrix, targets: np.ndarray
) -> tuple[float, np.ndarray]:
    """Exact full-softmax loss and both terms of the tied-embedding gradient."""

    if contexts.shape[0] == 0:
        raise ValueError("A minibatch must contain at least one example")
    mean_context = contexts @ embeddings
    logits = mean_context @ embeddings.T
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    normalizers = exponentials.sum(axis=1, keepdims=True)
    row_indices = np.arange(len(targets))
    loss = np.log(normalizers[:, 0]) - shifted[row_indices, targets]
    probability_gradient = exponentials / normalizers
    probability_gradient[row_indices, targets] -= 1
    probability_gradient /= len(targets)
    output_gradient = probability_gradient.T @ mean_context
    context_gradient = probability_gradient @ embeddings
    input_gradient = contexts.T @ context_gradient
    return float(loss.mean()), np.asarray(output_gradient + input_gradient)


def fit_word2vec(
    sequences: Iterable[np.ndarray], *, vocabulary_size: int, config: Word2VecConfig
) -> Word2VecModel:
    """Fit Eq. 4-style CBOW with tied vectors, full softmax, and minibatch Adam.

    Every visible token with at least one visible neighbor becomes a target.
    Hidden slots are neither targets nor inputs. The complete context matrix is
    held in memory, so callers should train a bounded quarter at a time.
    """

    if (
        isinstance(vocabulary_size, bool)
        or not isinstance(vocabulary_size, Integral)
        or vocabulary_size < 2
        or vocabulary_size > np.iinfo(np.int32).max
    ):
        raise ValueError("vocabulary_size must be an integer from 2 through int32 max")
    if not isinstance(config, Word2VecConfig):
        raise TypeError("config must be a Word2VecConfig")
    contexts, targets, skipped_no_context = _training_examples(
        sequences, vocabulary_size=vocabulary_size, radius=config.radius
    )
    rng = np.random.default_rng(config.seed)
    embeddings = rng.normal(
        scale=1 / np.sqrt(config.dimensions),
        size=(vocabulary_size, config.dimensions),
    ).astype(np.float32)
    first_moment = np.zeros_like(embeddings)
    second_moment = np.zeros_like(embeddings)
    step = 0
    epoch_losses: list[float] = []
    for _ in range(config.epochs):
        order = rng.permutation(len(targets))
        loss_sum = 0.0
        for start in range(0, len(targets), config.batch_size):
            batch_indices = order[start : start + config.batch_size]
            loss, gradient = _batch_loss_and_gradient(
                embeddings, contexts[batch_indices], targets[batch_indices]
            )
            loss_sum += loss * len(batch_indices)
            step += 1
            first_moment *= config.adam_beta1
            first_moment += (1 - config.adam_beta1) * gradient
            second_moment *= config.adam_beta2
            second_moment += (1 - config.adam_beta2) * np.square(gradient)
            corrected_first = first_moment / (1 - config.adam_beta1**step)
            corrected_second = second_moment / (1 - config.adam_beta2**step)
            embeddings -= config.learning_rate * corrected_first / (
                np.sqrt(corrected_second) + config.adam_epsilon
            )
        epoch_losses.append(loss_sum / len(targets))
    if not np.isfinite(embeddings).all():
        raise FloatingPointError("Word2Vec embeddings became non-finite")
    return Word2VecModel(
        embeddings=embeddings,
        config=config,
        vocabulary_size=vocabulary_size,
        training_examples=len(targets),
        skipped_no_context=skipped_no_context,
        epoch_losses=tuple(epoch_losses),
    )


def score_hidden(model: Word2VecModel, sequence: np.ndarray) -> np.ndarray:
    """Return full-vocabulary logits for the one ``-1`` slot at its original rank.

    The caller must exclude observed assets and normalize over the candidate
    complement for an ASMP likelihood; these are deliberately raw logits.
    """

    if not isinstance(model, Word2VecModel):
        raise TypeError("model must be a Word2VecModel")
    sequence = _checked_sequence(sequence, model.vocabulary_size)
    hidden_positions = np.flatnonzero(sequence == -1)
    if len(hidden_positions) != 1:
        raise ValueError("Exactly one hidden slot is required for scoring")
    hidden = int(hidden_positions[0])
    start = max(0, hidden - model.config.radius)
    stop = min(len(sequence), hidden + model.config.radius + 1)
    context_tokens = sequence[start:stop]
    context_tokens = context_tokens[context_tokens >= 0]
    if not len(context_tokens):
        raise ValueError("The hidden slot has no visible rank-window context")
    mean_context = model.embeddings[context_tokens].mean(axis=0)
    return np.asarray(model.embeddings @ mean_context)
