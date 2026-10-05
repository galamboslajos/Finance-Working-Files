"""Small, explicit PCA diagnostics for a saved point-in-time 13F matrix.

These are representation tests, not return forecasts or trading backtests. The
matrix universe was filtered before the masked edge is selected, so the masking
exercise is a smoke test rather than a sealed paper benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from scipy.sparse.linalg import LinearOperator, svds


PAIR_COLUMNS = (
    "manager_cik",
    "typed_security_id",
    "position_value_usd",
    "binary_owned",
)


@dataclass(frozen=True)
class MaskedQuarter:
    """Train/test edges and indexes; identifiers must stay in local memory."""

    train: pd.DataFrame
    test: pd.DataFrame
    managers: pd.Index
    assets: pd.Index


def load_pairs(path: str | Path) -> pd.DataFrame:
    """Load only the columns needed for a representation smoke test."""

    pairs = pq.read_table(path, columns=list(PAIR_COLUMNS)).to_pandas()
    if pairs.empty:
        raise ValueError("The quarterly pair table is empty")
    if pairs[list(PAIR_COLUMNS)].isna().any().any():
        raise ValueError("The quarterly pair table has null model inputs")
    if pairs.duplicated(["manager_cik", "typed_security_id"]).any():
        raise ValueError("Manager-security pairs must be unique")
    values = pairs["position_value_usd"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Position values must be finite and positive")
    if not pairs["binary_owned"].eq(1).all():
        raise ValueError("Every stored pair must have binary_owned=1")
    return pairs


def mask_second_largest(pairs: pd.DataFrame) -> MaskedQuarter:
    """Hide one second-largest retained position per manager, with stable ties."""

    managers = pd.Index(sorted(pairs["manager_cik"].unique()))
    assets = pd.Index(sorted(pairs["typed_security_id"].unique()))
    ordered = pairs.sort_values(
        ["manager_cik", "position_value_usd", "typed_security_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    second = ordered.groupby("manager_cik", sort=False).cumcount().eq(1)
    test = ordered.loc[second].copy()
    train = pairs.drop(index=test.index).copy()
    if test.empty:
        raise ValueError("No manager has a second retained position")
    return MaskedQuarter(train=train, test=test, managers=managers, assets=assets)


def holdings_matrix(split: MaskedQuarter, model: str) -> sparse.csr_matrix:
    """Construct paper-style binary or within-manager percentile rank values."""

    if model not in {"binary", "ranks"}:
        raise ValueError(f"Unknown representation model: {model}")
    train = split.train
    rows = split.managers.get_indexer(train["manager_cik"])
    columns = split.assets.get_indexer(train["typed_security_id"])
    if (rows < 0).any() or (columns < 0).any():
        raise ValueError("A train edge is absent from the original matrix universe")
    if model == "binary":
        values = np.ones(len(train), dtype=np.float64)
    else:
        # Break equal-value ties by typed security ID, then assign exactly
        # 1/n to the smallest and 1 to the largest retained position.
        ranked = train.sort_values(
            ["manager_cik", "position_value_usd", "typed_security_id"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        ordinal = ranked.groupby("manager_cik", sort=False).cumcount() + 1
        sizes = ranked.groupby("manager_cik")["manager_cik"].transform("size")
        values = (ordinal / sizes).reindex(train.index).to_numpy(dtype=np.float64)
    result = sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(split.managers), len(split.assets)),
    ).tocsr()
    result.sort_indices()
    return result


def centered_pca(
    matrix: sparse.csr_matrix, *, dimensions: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute top components of column-centered sparse holdings via SVD."""

    if dimensions < 1 or dimensions >= min(matrix.shape):
        raise ValueError("dimensions must be between 1 and min(matrix.shape)-1")
    manager_count, asset_count = matrix.shape
    means = np.asarray(matrix.mean(axis=0)).ravel()

    def matvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ vector).ravel() - float(means @ vector)

    def rmatvec(vector: np.ndarray) -> np.ndarray:
        return np.asarray(matrix.T @ vector).ravel() - means * vector.sum()

    def matmat(vectors: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ vectors) - np.outer(
            np.ones(manager_count), means @ vectors
        )

    def rmatmat(vectors: np.ndarray) -> np.ndarray:
        return np.asarray(matrix.T @ vectors) - np.outer(
            means, vectors.sum(axis=0)
        )

    centered = LinearOperator(
        (manager_count, asset_count),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=np.float64,
    )
    left, singular, right = svds(
        centered,
        k=dimensions,
        which="LM",
        tol=1e-5,
        maxiter=1000,
        random_state=seed,
    )
    order = np.argsort(singular)[::-1]
    left, singular, right = left[:, order], singular[order], right[order, :]
    centered_sum_squares = float(
        np.square(matrix.data).sum() - manager_count * np.square(means).sum()
    )
    if centered_sum_squares <= 0:
        raise ValueError("The centered holdings matrix has no variation")
    return left, singular, right, means, centered_sum_squares


def _masked_ranks(
    matrix: sparse.csr_matrix,
    target_columns: np.ndarray,
    *,
    latent: np.ndarray | None = None,
    right: np.ndarray | None = None,
    means: np.ndarray | None = None,
    popularity: np.ndarray | None = None,
    batch_size: int = 128,
) -> tuple[np.ndarray, np.ndarray]:
    """Rank each hidden asset among assets absent from that train portfolio."""

    managers, assets = matrix.shape
    ranks = np.empty(managers, dtype=np.int32)
    choices = assets - np.diff(matrix.indptr)
    columns = np.arange(assets)
    for start in range(0, managers, batch_size):
        stop = min(start + batch_size, managers)
        if popularity is not None:
            scores = np.broadcast_to(popularity, (stop - start, assets)).copy()
        else:
            if latent is None or right is None or means is None:
                raise ValueError("PCA scores require latent factors, loadings, and means")
            scores = latent[start:stop] @ right + means
        for offset, manager in enumerate(range(start, stop)):
            observed = matrix.indices[matrix.indptr[manager] : matrix.indptr[manager + 1]]
            scores[offset, observed] = -np.inf
        targets = target_columns[start:stop]
        target_scores = scores[np.arange(stop - start), targets]
        better = scores > target_scores[:, None]
        earlier_tie = (scores == target_scores[:, None]) & (
            columns[None, :] < targets[:, None]
        )
        ranks[start:stop] = 1 + (better | earlier_tie).sum(axis=1)
    return ranks, choices


def _rank_metrics(ranks: np.ndarray, choices: np.ndarray) -> dict[str, float]:
    return {
        "hit_at_100": float(np.mean(ranks <= 100)),
        "mean_reciprocal_rank": float(np.mean(1.0 / ranks)),
        "mean_rank_percentile": float(
            np.mean(1.0 - (ranks - 1) / np.maximum(choices - 1, 1))
        ),
    }


def run_smoke_test(
    pairs: pd.DataFrame, *, dimensions: tuple[int, ...] = (4, 10), seed: int = 17
) -> dict[str, object]:
    """Run target-excluded diagnostics and a popularity comparison."""

    if not dimensions or any(d < 1 for d in dimensions):
        raise ValueError("dimensions must be positive and nonempty")
    started = perf_counter()
    split = mask_second_largest(pairs)
    targets = split.managers.get_indexer(split.test["manager_cik"])
    if len(targets) != len(split.managers) or not np.array_equal(
        np.sort(targets), np.arange(len(split.managers))
    ):
        raise ValueError("Every manager must have exactly one held-out position")
    test_sorted = split.test.iloc[np.argsort(targets)]
    target_columns = split.assets.get_indexer(test_sorted["typed_security_id"])
    binary = holdings_matrix(split, "binary")
    popularity = np.asarray(binary.getnnz(axis=0), dtype=np.float64)
    manager_degrees = np.diff(binary.indptr)
    popularity_ranks, choices = _masked_ranks(
        binary, target_columns, popularity=popularity
    )
    output: dict[str, object] = {
        "managers": len(split.managers),
        "assets": len(split.assets),
        "train_pairs": len(split.train),
        "masked_pairs": len(split.test),
        "minimum_train_assets_per_manager": int(manager_degrees.min()),
        "minimum_train_managers_per_asset": int(popularity.min()),
        "cold_start_masked_assets": int(np.sum(popularity[target_columns] == 0)),
        "popularity": _rank_metrics(popularity_ranks, choices),
        "random_expected_hit_at_100": float(np.mean(np.minimum(100 / choices, 1))),
        "models": {},
    }
    for model in ("binary", "ranks"):
        matrix = binary if model == "binary" else holdings_matrix(split, model)
        model_started = perf_counter()
        left, singular, right, means, total_squares = centered_pca(
            matrix, dimensions=max(dimensions), seed=seed
        )
        model_result = {}
        for k in sorted(set(dimensions)):
            latent = left[:, :k] * singular[:k]
            ranks, _ = _masked_ranks(
                binary,
                target_columns,
                latent=latent,
                right=right[:k, :],
                means=means,
            )
            model_result[str(k)] = {
                **_rank_metrics(ranks, choices),
                "centered_variance_share": float(
                    np.square(singular[:k]).sum() / total_squares
                ),
            }
        output["models"][model] = model_result
        output[f"{model}_seconds"] = round(perf_counter() - model_started, 3)
    output["total_seconds"] = round(perf_counter() - started, 3)
    return output
