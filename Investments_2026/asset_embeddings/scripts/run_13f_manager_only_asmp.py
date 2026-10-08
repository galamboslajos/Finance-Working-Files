"""Run a fit-manager-only 13F ASMP benchmark from raw local PIT history.

The saved all-manager matrices are deliberately not inputs: their universe was
filtered before the manager split. This runner rebuilds one quarter at its
decision cutoff, splits eligible managers, and fits the universe on one side.
Its results are representation diagnostics, not forecasts or trading returns.

Run from the project directory, for example::

    python -m scripts.run_13f_manager_only_asmp --quarter 2026Q1 --word2vec --save
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy

from scripts.run_13f_asmp_pilot import (
    PROJECT_ROOT,
    SOURCE_FILES as PILOT_SOURCE_FILES,
    file_sha256,
    run_assetbert_asmp,
    run_pca_asmp,
    run_word2vec_asmp,
)
from src.manager_only_asmp_13f import build_manager_only_quarter
from src.matrix_13f import (
    DEFAULT_HOLDING_COLUMNS,
    MatrixParameters,
    decision_complete_quarters,
    product_input_fingerprint,
)
from src.point_in_time_13f import build_local_quarter_panel
from src.security_master_13f import classify_security_rows


DEFAULT_PRODUCT_ROOT = PROJECT_ROOT / "data" / "full_history" / "13f" / "v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "model_runs" / "13f_manager_only_asmp"
BENCHMARK = "13f_cross_manager_fit_only_universe_not_temporal_or_paper_replication"
SOURCE_FILES = tuple(
    dict.fromkeys(
        (*PILOT_SOURCE_FILES,
         "src/data_spine_13f.py",
         "src/full_history_13f.py",
         "src/holdings_exploration.py",
         "src/point_in_time_13f.py",
         "src/security_master_13f.py",
         "src/matrix_13f.py",
         "src/manager_only_asmp_13f.py",
         "scripts/run_13f_manager_only_asmp.py")
    )
)


def _code_status() -> tuple[str, bool, dict[str, str]]:
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
    hashes = {
        name: file_sha256(PROJECT_ROOT / name) for name in SOURCE_FILES
    }
    return commit, not bool(status), hashes


def _quarter_cutoff(product_root: Path, quarter: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    coverage = decision_complete_quarters(product_root)
    target = pd.Period(quarter, freq="Q")
    report_periods = (
        pd.to_datetime(coverage["report_date_utc"], utc=True)
        .dt.tz_localize(None)
        .dt.to_period("Q")
    )
    selected = coverage.loc[report_periods.eq(target)]
    if len(selected) != 1:
        raise ValueError("Requested quarter is not a unique decision-complete period")
    row = selected.iloc[0]
    return pd.Timestamp(row["report_date_utc"]), pd.Timestamp(row["information_cutoff_utc"])


def _assert_point_in_time(
    classified: pd.DataFrame,
    decision_at: pd.Timestamp,
    *,
    report_date: pd.Timestamp,
) -> None:
    eligible = classified.loc[classified["matrix_eligible"]]
    if eligible.empty:
        raise ValueError("The PIT snapshot has no eligible cash-share holdings")
    available = pd.to_datetime(
        eligible["source_event_available_at_utc"], errors="coerce", utc=True
    )
    snapshot = pd.to_datetime(
        eligible["snapshot_decision_at_utc"], errors="coerce", utc=True
    )
    last_event = pd.to_datetime(
        eligible["state_last_event_at_utc"], errors="coerce", utc=True
    )
    reported = pd.to_datetime(
        eligible["period_of_report"], errors="coerce", utc=True
    ).dt.normalize()
    expected_report = pd.Timestamp(report_date)
    expected_report = (
        expected_report.tz_localize("UTC")
        if expected_report.tzinfo is None
        else expected_report.tz_convert("UTC")
    ).normalize()
    if available.isna().any() or available.gt(decision_at).any():
        raise ValueError("A modeled holding was unavailable at the decision cutoff")
    if snapshot.isna().any() or snapshot.ne(decision_at).any():
        raise ValueError("A modeled holding belongs to a different PIT snapshot")
    if last_event.isna().any() or last_event.gt(decision_at).any():
        raise ValueError("A modeled manager state uses an event after the decision cutoff")
    if reported.isna().any() or reported.ne(expected_report).any():
        raise ValueError("A modeled holding belongs to a different report period")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarter", required=True, help="Decision-complete YYYYQ[1-4]")
    parser.add_argument("--product-root", type=Path, default=DEFAULT_PRODUCT_ROOT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--word2vec", action="store_true")
    parser.add_argument("--assetbert", action="store_true")
    parser.add_argument("--neural-dimensions", type=int, default=4)
    parser.add_argument("--w2v-radius", type=int, default=5)
    parser.add_argument("--w2v-epochs", type=int, default=1)
    parser.add_argument("--w2v-batch-size", type=int, default=128)
    parser.add_argument("--w2v-learning-rate", type=float, default=0.01)
    parser.add_argument("--bert-epochs", type=int, default=1)
    parser.add_argument("--bert-batch-size", type=int, default=32)
    parser.add_argument("--bert-learning-rate", type=float, default=0.001)
    parser.add_argument("--bert-device", choices=("auto", "cpu", "mps"), default="auto")
    parser.add_argument("--save", action="store_true", help="Save aggregate JSON below ignored data/")
    args = parser.parse_args()
    if not re.fullmatch(r"\d{4}Q[1-4]", args.quarter):
        parser.error("--quarter must have YYYYQ[1-4] form")
    report_date, decision_at = _quarter_cutoff(args.product_root, args.quarter)
    panel = build_local_quarter_panel(
        args.product_root,
        report_date=report_date,
        decision_at=decision_at,
        holdings_columns=DEFAULT_HOLDING_COLUMNS,
    )
    eligible_manager_ids = tuple(
        panel.manager_periods.loc[
            panel.manager_periods["model_eligible"].astype("boolean").fillna(False),
            "manager_cik",
        ].astype(str)
    )
    classified = classify_security_rows(panel.holdings_asof)
    _assert_point_in_time(classified, decision_at, report_date=report_date)
    parameters = MatrixParameters()
    quarter = build_manager_only_quarter(
        classified,
        eligible_manager_ids=eligible_manager_ids,
        parameters=parameters,
        seed=args.seed,
        test_fraction=args.test_fraction,
    )
    split = quarter.split
    vocabulary_sha256 = sha256("\n".join(split.vocabulary).encode()).hexdigest()
    results = run_pca_asmp(
        split.train_visible,
        split.train_ranked_columns,
        split.test_queries,
        split.test_labels,
        seed=args.seed,
        benchmark=BENCHMARK,
    )
    if args.word2vec:
        results["word2vec"] = run_word2vec_asmp(
            split.train_ranked_columns,
            split.test_queries,
            split.test_labels,
            dimensions=args.neural_dimensions,
            radius=args.w2v_radius,
            epochs=args.w2v_epochs,
            batch_size=args.w2v_batch_size,
            learning_rate=args.w2v_learning_rate,
            seed=args.seed,
            benchmark=BENCHMARK,
        )
    if args.assetbert:
        results["assetbert"] = run_assetbert_asmp(
            split.train_ranked_columns,
            split.test_queries,
            split.test_labels,
            dimensions=args.neural_dimensions,
            epochs=args.bert_epochs,
            batch_size=args.bert_batch_size,
            learning_rate=args.bert_learning_rate,
            seed=args.seed,
            requested_device=args.bert_device,
            benchmark=BENCHMARK,
        )
    commit, committed, source_hashes = _code_status()
    config = {
        "seed": args.seed,
        "test_manager_fraction": args.test_fraction,
        "fit_universe_filters": asdict(parameters),
        "pca_dimensions": [4, 10],
        "word2vec": args.word2vec,
        "assetbert": args.assetbert,
        "neural_dimensions": args.neural_dimensions,
        "w2v_radius": args.w2v_radius,
        "w2v_epochs": args.w2v_epochs,
        "w2v_batch_size": args.w2v_batch_size,
        "w2v_learning_rate": args.w2v_learning_rate,
        "bert_epochs": args.bert_epochs,
        "bert_batch_size": args.bert_batch_size,
        "bert_learning_rate": args.bert_learning_rate,
        "bert_device": args.bert_device,
    }
    payload = {
        "purpose": BENCHMARK,
        "report_quarter": args.quarter,
        "report_date_utc": report_date.isoformat(),
        "decision_at_utc": decision_at.isoformat(),
        "input_fingerprint": product_input_fingerprint(args.product_root),
        "input_fingerprint_method": "relative_path_size_mtime_cache_signal_not_content_checksum",
        "fit_pair_content_sha256": quarter.fit_pair_content_sha256,
        "test_pair_content_sha256": quarter.test_pair_content_sha256,
        "fitted_vocabulary_sha256": vocabulary_sha256,
        "manager_split_policy": "PIT-eligible manager IDs split before fit-side universe filters",
        "fit_manager_count": split.train_manager_count,
        "test_manager_count": split.test_manager_count,
        "eligibility": {
            "eligible_manager_count": quarter.eligible_manager_count,
            "assigned_fit_managers": quarter.assigned_fit_managers,
            "assigned_test_managers": quarter.assigned_test_managers,
            "fit_managers_after_filters": quarter.fit_managers_after_filters,
            "test_managers_with_positive_positions": quarter.test_managers_with_positive_positions,
            "fit_pairs_after_filters": quarter.fit_pairs_after_filters,
            "test_pairs_before_universe_projection": quarter.test_pairs_before_universe_projection,
            "dropped_oov_visible_positions": (
                split.test_labels.dropped_out_of_vocabulary_visible_positions
            ),
        },
        "source_code_commit": commit,
        "model_code_committed_at_head": committed,
        "model_code_sha256": source_hashes,
        "execution_environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": None,
        },
        "configuration": config,
        "masked_edge_policy": (
            "original_second_largest_positive_eligible_cash_share_position; "
            "ranks_one_and_two_require_fit_vocabulary; later_OOV_visible_positions_dropped"
        ),
        "limitations": [
            "The fit-only universe removes held-out-manager influence on vocabulary and degree filters.",
            "The test cohort is still conditioned on having a rank-two in-vocabulary target.",
            "Later OOV visible positions are removed, compressing model-visible rank gaps and rank weights.",
            "Manager partitions are recalculated within each quarter, not frozen across time.",
            "This is a within-quarter cross-manager diagnostic, not future-quarter prediction.",
            "The 13F manager/security data differ from the paper's FactSet fund/company data.",
            "PCA logits are not calibrated on a separate validation-manager split.",
            "These are representation scores, not investment or trading returns.",
        ],
        "results": results,
    }
    if args.assetbert:
        import torch

        payload["execution_environment"]["torch"] = torch.__version__
        payload["execution_environment"]["resolved_bert_device"] = results["assetbert"]["fit"]["device"]
    if args.save:
        DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
        run_identity = {
            "quarter": args.quarter,
            "input_fingerprint": payload["input_fingerprint"],
            "fit_pair_content_sha256": quarter.fit_pair_content_sha256,
            "test_pair_content_sha256": quarter.test_pair_content_sha256,
            "configuration": config,
            "source_code_hashes": source_hashes,
        }
        run_id = sha256(json.dumps(run_identity, sort_keys=True).encode()).hexdigest()[:16]
        path = DEFAULT_RUN_ROOT / f"{args.quarter}_{run_id}.json"
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2) + "\n")
        print(f"Saved ignored aggregate result: {path.relative_to(PROJECT_ROOT)}")
    print(json.dumps({
        "purpose": BENCHMARK,
        "report_quarter": args.quarter,
        "model_code_committed_at_head": committed,
        "fit_manager_count": split.train_manager_count,
        "test_manager_count": split.test_manager_count,
        "scored_test_managers": split.test_labels.input_portfolios
            - split.test_labels.excluded_short_portfolios
            - split.test_labels.excluded_out_of_vocabulary
            - split.test_labels.excluded_too_few_candidates,
        "asmp": {name: result["asmp_normalized_log_likelihood"]
                 for name, result in results.items()
                 if "asmp_normalized_log_likelihood" in result},
    }, indent=2))


if __name__ == "__main__":
    main()
