"""Shared, provisional masked-holding likelihood benchmark for saved 13F pairs.

The saved quarterly universe was concentration- and degree-filtered before a
holding is hidden here. Its membership can therefore depend on the hidden
holding. These diagnostics are not a sealed paper replication or a trading test.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral, Real
from typing import Callable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import logsumexp


MASKED_ASSET = -1
PAIR_COLUMNS = ("manager_cik", "typed_security_id", "position_value_usd")


@dataclass(frozen=True)
class ASMPQueries:
    """Model-visible queries; row indexes are anonymous and contain no labels.

    ``ranked_columns`` uses decreasing position value and ascending security ID
    for ties. Index one is always the original rank-two slot, set to -1. The
    sparse matrix has one row per query and one column per vocabulary asset.
    ``row_indices`` lets batch scorers align precomputed manager factors without
    exposing manager identifiers.
    """

    vocabulary: tuple[str, ...]
    ranked_columns: tuple[tuple[int, ...], ...]
    visible: sparse.csr_matrix
    row_indices: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.ranked_columns)

    def batch(self, start: int, stop: int) -> ASMPQueries:
        """Give a scorer only the model-visible rows in one batch."""

        if not 0 <= start < stop <= len(self):
            raise ValueError("Invalid ASMP batch bounds")
        return ASMPQueries(
            vocabulary=self.vocabulary,
            ranked_columns=self.ranked_columns[start:stop],
            visible=self.visible[start:stop].tocsr(),
            row_indices=self.row_indices[start:stop],
        )


@dataclass(frozen=True)
class ASMPLabels:
    """Targets and aggregate eligibility counts, kept out of model queries."""

    target_columns: tuple[int, ...]
    input_portfolios: int
    excluded_short_portfolios: int
    excluded_out_of_vocabulary: int
    excluded_too_few_candidates: int


@dataclass(frozen=True)
class ASMPManagerSplit:
    """Anonymous train portfolios and rank-two test queries for one quarter.

    Training portfolios retain every asset, including rank two. The vocabulary
    is the union of training assets only. Test labels never enter training
    sequences or the vocabulary; test OOV exclusions remain explicit in
    ``test_labels``. The saved input universe is still prefiltered.
    """

    train_ranked_columns: tuple[tuple[int, ...], ...]
    train_visible: sparse.csr_matrix
    test_queries: ASMPQueries
    test_labels: ASMPLabels
    vocabulary: tuple[str, ...]
    train_manager_count: int
    test_manager_count: int


def _canonical_vocabulary(
    observed_assets: pd.Series, vocabulary: Sequence[str] | None
) -> tuple[str, ...]:
    if vocabulary is None:
        return tuple(sorted(observed_assets.unique()))
    supplied = tuple(vocabulary)
    if not supplied or any(not isinstance(asset, str) or not asset for asset in supplied):
        raise ValueError("The frozen vocabulary must contain nonempty security IDs")
    if len(supplied) != len(set(supplied)):
        raise ValueError("The frozen vocabulary contains duplicate security IDs")
    return tuple(sorted(supplied))


def _validated_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    """Validate the minimal saved-pair contract without changing the input."""

    missing = set(PAIR_COLUMNS) - set(pairs.columns)
    if missing:
        raise ValueError(f"Missing ASMP pair columns: {sorted(missing)}")
    if pairs.empty or pairs[list(PAIR_COLUMNS)].isna().any().any():
        raise ValueError("ASMP pairs must be nonempty and have no null inputs")
    if "binary_owned" in pairs and not pairs["binary_owned"].eq(1).all():
        raise ValueError("Stored ASMP pairs must have binary_owned=1")

    work = pairs.loc[:, PAIR_COLUMNS].copy()
    values = pd.to_numeric(work["position_value_usd"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("ASMP position values must be finite and positive")
    work["position_value_usd"] = values
    work["manager_cik"] = work["manager_cik"].astype(str)
    work["typed_security_id"] = work["typed_security_id"].astype(str)
    if (work["manager_cik"].eq("") | work["typed_security_id"].eq("")).any():
        raise ValueError("ASMP manager and security IDs must be nonempty")
    if work.duplicated(["manager_cik", "typed_security_id"]).any():
        raise ValueError("ASMP manager-security pairs must be unique")
    return work


def build_asmp_queries(
    pairs: pd.DataFrame, *, vocabulary: Sequence[str] | None = None
) -> tuple[ASMPQueries, ASMPLabels]:
    """Hide each manager's second-largest saved holding before model fitting.

    A supplied vocabulary may be frozen from earlier training data. Portfolios
    with an out-of-vocabulary visible or target asset are excluded as a whole,
    and the exclusion is reported. When omitted, the vocabulary comes from the
    already-filtered saved pair table and is necessarily provisional.
    """

    work = _validated_pairs(pairs)

    frozen_vocabulary = _canonical_vocabulary(work["typed_security_id"], vocabulary)
    asset_columns = {asset: column for column, asset in enumerate(frozen_vocabulary)}
    ordered = work.sort_values(
        ["manager_cik", "position_value_usd", "typed_security_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    ranked_columns: list[tuple[int, ...]] = []
    targets: list[int] = []
    row_indexes: list[int] = []
    column_indexes: list[int] = []
    excluded_short = excluded_oov = excluded_candidates = 0
    input_portfolios = 0
    for _, portfolio in ordered.groupby("manager_cik", sort=False):
        input_portfolios += 1
        assets = portfolio["typed_security_id"].tolist()
        if len(assets) < 2:
            excluded_short += 1
            continue
        if any(asset not in asset_columns for asset in assets):
            excluded_oov += 1
            continue
        # The target belongs to the complement, so at least two complement
        # choices are needed for a nondegenerate random-choice denominator.
        if len(frozen_vocabulary) - (len(assets) - 1) < 2:
            excluded_candidates += 1
            continue
        columns = [asset_columns[asset] for asset in assets]
        targets.append(columns[1])
        columns[1] = MASKED_ASSET
        row = len(ranked_columns)
        ranked_columns.append(tuple(columns))
        visible_columns = [column for column in columns if column != MASKED_ASSET]
        row_indexes.extend([row] * len(visible_columns))
        column_indexes.extend(visible_columns)

    if not ranked_columns:
        raise ValueError("No ASMP portfolio has a valid rank-two target and candidate set")
    visible = sparse.coo_matrix(
        (
            np.ones(len(row_indexes), dtype=np.int8),
            (row_indexes, column_indexes),
        ),
        shape=(len(ranked_columns), len(frozen_vocabulary)),
    ).tocsr()
    visible.sort_indices()
    queries = ASMPQueries(
        frozen_vocabulary,
        tuple(ranked_columns),
        visible,
        tuple(range(len(ranked_columns))),
    )
    labels = ASMPLabels(
        target_columns=tuple(targets),
        input_portfolios=input_portfolios,
        excluded_short_portfolios=excluded_short,
        excluded_out_of_vocabulary=excluded_oov,
        excluded_too_few_candidates=excluded_candidates,
    )
    return queries, labels


def build_manager_holdout(
    pairs: pd.DataFrame, *, seed: int = 17, test_fraction: float = 0.2
) -> ASMPManagerSplit:
    """Make a deterministic manager-level split before rank-two test masking.

    Managers are ordered by a SHA-256 digest of the seed and manager ID, then
    the nearest feasible test_fraction (at least one on each side) is held out.
    This is invariant to pair-row order and uses no target values for the split.
    A per-quarter split does not itself seal historical multi-quarter training:
    that requires a frozen cross-quarter manager partition.
    """

    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise ValueError("The manager split seed must be an integer")
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, Real)
        or not np.isfinite(test_fraction)
        or not 0 < test_fraction < 1
    ):
        raise ValueError("test_fraction must be strictly between zero and one")
    work = _validated_pairs(pairs)
    managers = tuple(sorted(work["manager_cik"].unique()))
    if len(managers) < 2:
        raise ValueError("Manager holdout requires at least two portfolios")

    def split_key(manager: str) -> tuple[bytes, str]:
        digest = sha256(f"{int(seed)}\0{manager}".encode("utf-8")).digest()
        return digest, manager

    shuffled = tuple(sorted(managers, key=split_key))
    test_count = min(
        len(managers) - 1,
        max(1, int(np.floor(len(managers) * test_fraction + 0.5))),
    )
    held_out = set(shuffled[:test_count])
    train = work.loc[~work["manager_cik"].isin(held_out)].copy()
    test = work.loc[work["manager_cik"].isin(held_out)].copy()
    vocabulary = _canonical_vocabulary(train["typed_security_id"], None)
    asset_columns = {asset: column for column, asset in enumerate(vocabulary)}
    ordered_train = train.sort_values(
        ["manager_cik", "position_value_usd", "typed_security_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    train_ranked_columns = tuple(
        tuple(asset_columns[asset] for asset in group["typed_security_id"])
        for _, group in ordered_train.groupby("manager_cik", sort=False)
    )
    row_indexes = [
        row
        for row, ranked in enumerate(train_ranked_columns)
        for _ in ranked
    ]
    column_indexes = [
        column for ranked in train_ranked_columns for column in ranked
    ]
    train_visible = sparse.coo_matrix(
        (
            np.ones(len(row_indexes), dtype=np.int8),
            (row_indexes, column_indexes),
        ),
        shape=(len(train_ranked_columns), len(vocabulary)),
    ).tocsr()
    train_visible.sort_indices()
    test_queries, test_labels = build_asmp_queries(test, vocabulary=vocabulary)
    return ASMPManagerSplit(
        train_ranked_columns=train_ranked_columns,
        train_visible=train_visible,
        test_queries=test_queries,
        test_labels=test_labels,
        vocabulary=vocabulary,
        train_manager_count=len(train_ranked_columns),
        test_manager_count=test_count,
    )


def evaluate_asmp(
    queries: ASMPQueries,
    labels: ASMPLabels,
    score_batch: Callable[[ASMPQueries], np.ndarray],
    *,
    batch_size: int = 128,
) -> dict[str, object]:
    """Score identical unheld candidate sets with any model's vocabulary logits.

    The scorer sees only ``ASMPQueries``. Its logits must be finite and have
    shape ``(batch_portfolios, vocabulary_size)``. Observed holdings are removed
    before normalization, including for full-vocabulary neural output heads.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if len(queries) < 1 or len(labels.target_columns) != len(queries):
        raise ValueError("ASMP queries and hidden labels must have matching rows")
    asset_count = len(queries.vocabulary)
    if queries.visible.shape != (len(queries), asset_count):
        raise ValueError("ASMP visible membership has the wrong shape")
    if len(queries.row_indices) != len(queries) or len(set(queries.row_indices)) != len(queries):
        raise ValueError("ASMP query row indexes must be unique and complete")
    if tuple(sorted(set(queries.vocabulary))) != queries.vocabulary:
        raise ValueError("ASMP vocabulary must be sorted and unique")
    if labels.input_portfolios < len(queries) or (
        len(queries)
        + labels.excluded_short_portfolios
        + labels.excluded_out_of_vocabulary
        + labels.excluded_too_few_candidates
        != labels.input_portfolios
    ):
        raise ValueError("ASMP eligibility counts do not reconcile")

    target_columns = np.asarray(labels.target_columns)
    if target_columns.dtype.kind not in "iu" or (
        (target_columns < 0) | (target_columns >= asset_count)
    ).any():
        raise ValueError("ASMP target columns must be valid vocabulary indexes")

    log_probabilities: list[float] = []
    random_log_choices: list[float] = []
    ranks: list[int] = []
    random_hit_at_100: list[float] = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        model_queries = queries.batch(start, stop)
        logits = np.asarray(score_batch(model_queries), dtype=np.float64)
        if logits.shape != (stop - start, asset_count):
            raise ValueError("ASMP scorer returned an incorrect logit shape")
        if not np.isfinite(logits).all():
            raise ValueError("ASMP logits must all be finite")
        for offset, row in enumerate(range(start, stop)):
            ranked = queries.ranked_columns[row]
            if len(ranked) < 2 or ranked[1] != MASKED_ASSET or (
                ranked.count(MASKED_ASSET) != 1
            ):
                raise ValueError("ASMP queries must preserve the original rank-two mask slot")
            observed = queries.visible.indices[
                queries.visible.indptr[row] : queries.visible.indptr[row + 1]
            ]
            actual_observed = tuple(column for column in ranked if column != MASKED_ASSET)
            if (
                len(set(actual_observed)) != len(actual_observed)
                or any(column < 0 or column >= asset_count for column in actual_observed)
                or set(observed) != set(actual_observed)
            ):
                raise ValueError("ASMP ranked holdings and sparse membership disagree")
            target = int(target_columns[row])
            if target in observed:
                raise ValueError("The hidden ASMP target appears in model-visible holdings")
            eligible = np.ones(asset_count, dtype=bool)
            eligible[observed] = False
            choices = np.flatnonzero(eligible)
            if len(choices) < 2:
                raise ValueError("ASMP requires at least two candidate assets per query")
            scores = logits[offset, choices]
            target_score = float(logits[offset, target])
            log_probability = target_score - float(logsumexp(scores))
            if not np.isfinite(log_probability):
                raise ValueError("ASMP log probability is not finite")
            log_probabilities.append(log_probability)
            random_log_choices.append(float(np.log(len(choices))))
            # Canonical vocabulary order is the deterministic tie-breaker.
            ranks.append(
                1
                + int(np.count_nonzero(scores > target_score))
                + int(np.count_nonzero((scores == target_score) & (choices < target)))
            )
            random_hit_at_100.append(min(100 / len(choices), 1.0))

    mean_log_probability = float(np.mean(log_probabilities))
    mean_log_choices = float(np.mean(random_log_choices))
    rank_array = np.asarray(ranks, dtype=np.int32)
    return {
        "benchmark": "provisional_13f_asmp_prefiltered_universe_not_sealed",
        "masked_rank": 2,
        "candidate_policy": "frozen_vocabulary_minus_visible_holdings",
        "vocabulary_size": asset_count,
        "input_portfolios": labels.input_portfolios,
        "scored_portfolios": len(queries),
        "coverage": len(queries) / labels.input_portfolios,
        "excluded_short_portfolios": labels.excluded_short_portfolios,
        "excluded_out_of_vocabulary": labels.excluded_out_of_vocabulary,
        "excluded_too_few_candidates": labels.excluded_too_few_candidates,
        "mean_log_target_probability": mean_log_probability,
        "mean_log_random_probability": -mean_log_choices,
        "asmp_normalized_log_likelihood": 1.0 + mean_log_probability / mean_log_choices,
        "hit_at_100": float(np.mean(rank_array <= 100)),
        "mean_reciprocal_rank": float(np.mean(1.0 / rank_array)),
        "random_expected_hit_at_100": float(np.mean(random_hit_at_100)),
        "limitations": [
            "The saved matrix universe was filtered before targets were hidden.",
            "Coverage is measured only against portfolios in the saved matrix.",
            "This is neither a sealed paper replication nor a trading backtest.",
        ],
    }
