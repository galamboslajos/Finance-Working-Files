"""Compare 13F representations on a provisional cross-manager ASMP test.

Within one saved point-in-time quarter, 80% of managers fit the vocabulary and
models using complete portfolios. Rank two is hidden only for the untouched
20% of managers. The saved matrix was universe-filtered before this split, so
its results remain diagnostic rather than the paper's sealed ASMP benchmark.
Run from the project directory with ``python -m scripts.run_13f_asmp_pilot``.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

from scripts.run_13f_pca_smoke import (
    DEFAULT_MATRIX_ROOT,
    MATRIX_CONTRACT,
    PROJECT_ROOT,
    file_sha256,
    validate_matrix_manifest,
)
from src.asmp_13f import ASMPLabels, ASMPQueries, build_manager_holdout, evaluate_asmp
from src.assetbert_13f import (
    AssetBERT,
    AssetBERTConfig,
    encode_masked_rank2,
    encode_ranked_portfolio,
    mask_chunk_for_training,
    masked_cross_entropy,
)
from src.representation_13f import centered_pca, load_pairs
from src.word2vec_13f import Word2VecConfig, fit_word2vec, score_hidden


DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "model_runs" / "13f_asmp_pilot"
SOURCE_FILES = (
    "src/asmp_13f.py",
    "src/assetbert_13f.py",
    "src/representation_13f.py",
    "src/word2vec_13f.py",
    "scripts/run_13f_pca_smoke.py",
    "scripts/run_13f_asmp_pilot.py",
)


def ranked_training_matrix(
    ranked_columns: tuple[tuple[int, ...], ...], vocabulary_size: int
) -> sparse.csr_matrix:
    """Construct complete rank-weighted fit rows without changing rank gaps."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, ranked in enumerate(ranked_columns):
        original_size = len(ranked)
        for rank, column in enumerate(ranked):
            if not 0 <= column < vocabulary_size:
                raise ValueError("Training rows must contain complete vocabulary assets")
            rows.append(row)
            columns.append(column)
            values.append((original_size - rank) / original_size)
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=(len(ranked_columns), vocabulary_size),
        dtype=np.float64,
    ).tocsr()


def ranked_matrix(queries: ASMPQueries) -> sparse.csr_matrix:
    """Preserve original holding-rank percentiles while omitting rank two."""

    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    for row, ranked in enumerate(queries.ranked_columns):
        original_size = len(ranked)
        for rank, column in enumerate(ranked):
            if column < 0:
                continue
            rows.append(row)
            columns.append(column)
            values.append((original_size - rank) / original_size)
    return sparse.coo_matrix(
        (values, (rows, columns)),
        shape=queries.visible.shape,
        dtype=np.float64,
    ).tocsr()


def run_pca_asmp(
    training_binary: sparse.csr_matrix,
    training_ranked_columns: tuple[tuple[int, ...], ...],
    queries: ASMPQueries,
    labels: ASMPLabels,
    *,
    dimensions: tuple[int, ...] = (4, 10),
    seed: int = 17,
) -> dict[str, object]:
    """Fit complete training portfolios and fold in masked test managers."""

    if not dimensions or min(dimensions) < 1:
        raise ValueError("PCA dimensions must be positive and nonempty")
    binary = training_binary.astype(np.float64)
    if binary.shape != (len(training_ranked_columns), len(queries.vocabulary)):
        raise ValueError("Training rows and the frozen vocabulary do not align")
    results: dict[str, object] = {
        "uniform": evaluate_asmp(
            queries,
            labels,
            lambda batch: np.zeros(
                (len(batch), len(batch.vocabulary)), dtype=np.float64
            ),
        )
    }
    popularity = np.log1p(np.asarray(binary.getnnz(axis=0), dtype=np.float64))
    results["popularity_log_count_plus_one"] = evaluate_asmp(
        queries,
        labels,
        lambda batch: np.broadcast_to(popularity, (len(batch), len(popularity))),
    )
    rank_train = ranked_training_matrix(
        training_ranked_columns, len(queries.vocabulary)
    )
    for name, matrix, test_matrix in (
        ("rs_binary", binary, queries.visible.astype(np.float64)),
        ("rs_ranks", rank_train, ranked_matrix(queries)),
    ):
        _, singular, right, means, variance = centered_pca(
            matrix, dimensions=max(dimensions), seed=seed
        )
        by_dimension: dict[str, object] = {}
        for dimension in sorted(set(dimensions)):
            loadings = right[:dimension, :]
            # The test manager is new: infer its factor solely from visible
            # holdings while the global training loadings and means stay fixed.
            latent = np.asarray(test_matrix @ loadings.T) - means @ loadings.T

            def score_batch(batch: ASMPQueries) -> np.ndarray:
                row_indexes = np.asarray(batch.row_indices, dtype=np.int64)
                return latent[row_indexes] @ loadings + means

            metrics = evaluate_asmp(queries, labels, score_batch)
            metrics["centered_variance_share"] = float(
                np.square(singular[:dimension]).sum() / variance
            )
            by_dimension[str(dimension)] = metrics
        results[name] = by_dimension
    return results


def run_word2vec_asmp(
    training_ranked_columns: tuple[tuple[int, ...], ...],
    queries: ASMPQueries,
    labels: ASMPLabels,
    *,
    dimensions: int,
    radius: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, object]:
    """Fit tied-embedding CBOW on complete fit managers and score test rank two."""

    started = perf_counter()
    config = Word2VecConfig(
        dimensions=dimensions,
        radius=radius,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    model = fit_word2vec(
        training_ranked_columns,
        vocabulary_size=len(queries.vocabulary),
        config=config,
    )

    def score_batch(batch: ASMPQueries) -> np.ndarray:
        return np.stack(
            [score_hidden(model, np.asarray(row)) for row in batch.ranked_columns]
        )

    metrics = evaluate_asmp(queries, labels, score_batch)
    metrics["fit"] = {
        "dimensions": dimensions,
        "rank_radius": radius,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_examples": model.training_examples,
        "training_managers": len(training_ranked_columns),
        "skipped_no_context": model.skipped_no_context,
        "epoch_losses": model.epoch_losses,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "implementation": "tied_embedding_full_softmax_cbow",
    }
    return metrics


def run_assetbert_asmp(
    training_ranked_columns: tuple[tuple[int, ...], ...],
    queries: ASMPQueries,
    labels: ASMPLabels,
    *,
    dimensions: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    requested_device: str,
) -> dict[str, object]:
    """Train random MLM on complete fit managers; infer on hidden test rank two."""

    import torch

    started = perf_counter()
    if requested_device == "auto":
        device_name = "mps" if torch.backends.mps.is_available() else "cpu"
    else:
        device_name = requested_device
    if device_name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The requested MPS device is unavailable")
    device = torch.device(device_name)
    torch.manual_seed(seed)
    vocabulary_size = len(queries.vocabulary)
    config = AssetBERTConfig(
        vocabulary_size=vocabulary_size, embedding_dim=dimensions, seed=seed
    )
    model = AssetBERT(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0)
    training_chunks = []
    first_chunk_indexes: set[int] = set()
    for ranked in training_ranked_columns:
        first_chunk_indexes.add(len(training_chunks))
        training_chunks.extend(encode_ranked_portfolio(ranked, vocabulary_size))
    order_rng = np.random.default_rng(seed)
    mask_rng = random.Random(seed)
    epoch_losses: list[float] = []
    selected_counts: list[int] = []
    supervised_rank_two_counts: list[int] = []
    for _ in range(epochs):
        model.train()
        order = order_rng.permutation(len(training_chunks))
        weighted_loss = 0.0
        selected_total = 0
        rank_two_supervised = 0
        for start in range(0, len(order), batch_size):
            chunk_indexes = order[start : start + batch_size]
            batch = [
                mask_chunk_for_training(
                    training_chunks[index], vocabulary_size, mask_rng
                )
                for index in chunk_indexes
            ]
            rank_two_supervised += sum(
                item.labels[1] != -100
                for index, item in zip(chunk_indexes, batch)
                if index in first_chunk_indexes
            )
            token_ids = torch.tensor(
                [item.input_ids for item in batch], dtype=torch.long, device=device
            )
            attention_mask = torch.tensor(
                [item.attention_mask for item in batch], dtype=torch.bool, device=device
            )
            target_ids = torch.tensor(
                [item.labels for item in batch], dtype=torch.long, device=device
            )
            optimizer.zero_grad(set_to_none=True)
            loss = masked_cross_entropy(model(token_ids, attention_mask), target_ids)
            loss.backward()
            optimizer.step()
            selected = int(target_ids.ne(-100).sum().item())
            weighted_loss += float(loss.item()) * selected
            selected_total += selected
        epoch_losses.append(weighted_loss / selected_total)
        selected_counts.append(selected_total)
        supervised_rank_two_counts.append(rank_two_supervised)

    def score_batch(batch: ASMPQueries) -> np.ndarray:
        first_chunks = [
            encode_masked_rank2(row, vocabulary_size)[0]
            for row in batch.ranked_columns
        ]
        token_ids = torch.tensor(
            [chunk.token_ids for chunk in first_chunks], dtype=torch.long, device=device
        )
        attention_mask = torch.tensor(
            [chunk.attention_mask for chunk in first_chunks],
            dtype=torch.bool,
            device=device,
        )
        model.eval()
        with torch.no_grad():
            return model(token_ids, attention_mask)[:, 1, :].cpu().numpy()

    metrics = evaluate_asmp(queries, labels, score_batch, batch_size=batch_size)
    metrics["fit"] = {
        "dimensions": dimensions,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "optimizer": "AdamW_weight_decay_zero",
        "device": device_name,
        "training_chunks": len(training_chunks),
        "training_managers": len(training_ranked_columns),
        "selected_training_tokens_per_epoch": selected_counts,
        "supervised_training_rank_two_tokens_per_epoch": supervised_rank_two_counts,
        "epoch_losses": epoch_losses,
        "elapsed_seconds": round(perf_counter() - started, 3),
        "architecture": "four_encoder_layers_two_heads_64_stock_chunks",
    }
    return metrics


def code_status() -> tuple[str, bool, dict[str, str]]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *SOURCE_FILES],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digests = {
        relative: file_sha256(PROJECT_ROOT / relative) for relative in SOURCE_FILES
    }
    return commit, not bool(status), digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarter", required=True, help="Report quarter in YYYYQ[1-4] form")
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--word2vec", action="store_true", help="Fit paper-style CBOW")
    parser.add_argument("--assetbert", action="store_true", help="Fit masked-token encoder")
    parser.add_argument("--neural-dimensions", type=int, default=4)
    parser.add_argument("--w2v-radius", type=int, default=5)
    parser.add_argument("--w2v-epochs", type=int, default=1)
    parser.add_argument("--w2v-batch-size", type=int, default=128)
    parser.add_argument("--w2v-learning-rate", type=float, default=0.01)
    parser.add_argument("--bert-epochs", type=int, default=1)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-learning-rate", type=float, default=0.001)
    parser.add_argument("--bert-device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--save", action="store_true", help="Save ignored local JSON")
    arguments = parser.parse_args()
    if not re.fullmatch(r"\d{4}Q[1-4]", arguments.quarter):
        parser.error("--quarter must have YYYYQ[1-4] form")
    pair_path = arguments.matrix_root / f"report_quarter={arguments.quarter}" / "pairs.parquet"
    if not pair_path.is_file():
        parser.error("The requested quarterly pair table is not available locally")
    pair_sha256 = file_sha256(pair_path)
    validate_matrix_manifest(arguments.matrix_root, arguments.quarter, pair_sha256)
    split = build_manager_holdout(load_pairs(pair_path), seed=arguments.seed)
    queries, labels = split.test_queries, split.test_labels
    vocabulary_sha256 = sha256(
        "\n".join(split.vocabulary).encode("utf-8")
    ).hexdigest()
    results = run_pca_asmp(
        split.train_visible,
        split.train_ranked_columns,
        queries,
        labels,
        seed=arguments.seed,
    )
    if arguments.word2vec:
        results["word2vec"] = run_word2vec_asmp(
            split.train_ranked_columns,
            queries,
            labels,
            dimensions=arguments.neural_dimensions,
            radius=arguments.w2v_radius,
            epochs=arguments.w2v_epochs,
            batch_size=arguments.w2v_batch_size,
            learning_rate=arguments.w2v_learning_rate,
            seed=arguments.seed,
        )
    if arguments.assetbert:
        results["assetbert"] = run_assetbert_asmp(
            split.train_ranked_columns,
            queries,
            labels,
            dimensions=arguments.neural_dimensions,
            epochs=arguments.bert_epochs,
            batch_size=arguments.bert_batch_size,
            learning_rate=arguments.bert_learning_rate,
            seed=arguments.seed,
            requested_device=arguments.bert_device,
        )
    commit, committed, digests = code_status()
    execution_environment = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": None,
        "resolved_bert_device": None,
    }
    if arguments.assetbert:
        import torch

        execution_environment["torch"] = torch.__version__
        execution_environment["resolved_bert_device"] = results["assetbert"]["fit"]["device"]
    configuration = {
        "seed": arguments.seed,
        "test_manager_fraction": 0.2,
        "pca_dimensions": [4, 10],
        "word2vec": arguments.word2vec,
        "assetbert": arguments.assetbert,
        "neural_dimensions": arguments.neural_dimensions,
        "w2v_radius": arguments.w2v_radius,
        "w2v_epochs": arguments.w2v_epochs,
        "w2v_batch_size": arguments.w2v_batch_size,
        "w2v_learning_rate": arguments.w2v_learning_rate,
        "bert_epochs": arguments.bert_epochs,
        "bert_batch_size": arguments.bert_batch_size,
        "bert_learning_rate": arguments.bert_learning_rate,
        "bert_device": arguments.bert_device,
    }
    payload = {
        "purpose": "provisional_13f_cross_manager_asmp_not_paper_replication",
        "report_quarter": arguments.quarter,
        "matrix_contract": MATRIX_CONTRACT,
        "source_pair_file_sha256": pair_sha256,
        "fitted_vocabulary_sha256": vocabulary_sha256,
        "manager_split_policy": "deterministic_80_percent_fit_20_percent_heldout_within_quarter",
        "fit_manager_count": split.train_manager_count,
        "test_manager_count": split.test_manager_count,
        "fit_eligible_manager_count": len(split.train_ranked_columns),
        "source_code_commit": commit,
        "model_code_committed_at_head": committed,
        "model_code_sha256": digests,
        "execution_environment": execution_environment,
        "configuration": configuration,
        "masked_edge_policy": "one_second_largest_retained_position_per_manager",
        "limitations": [
            "The saved universe was filtered using all managers before the split and targets were hidden.",
            "The paper's FactSet fund/company data differ from 13F manager/security data.",
            "This is cross-manager within-quarter generalization, not the paper's RV/RC asset split.",
            "These are representation diagnostics, not return forecasts.",
        ],
        "results": results,
    }
    if arguments.save:
        DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
        run_identity = {
            "report_quarter": arguments.quarter,
            "source_pair_file_sha256": pair_sha256,
            "configuration": configuration,
            "model_code_sha256": digests,
            "execution_environment": execution_environment,
        }
        run_id = sha256(json.dumps(run_identity, sort_keys=True).encode()).hexdigest()[:16]
        destination = DEFAULT_RUN_ROOT / f"{arguments.quarter}_{run_id}.json"
        # Exclusive creation preserves earlier local results when a configuration
        # or source changes and prevents an identical rerun from replacing them.
        with destination.open("x", encoding="utf-8") as output:
            output.write(json.dumps(payload, indent=2) + "\n")
        print(f"Saved local ignored run summary: {destination}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
