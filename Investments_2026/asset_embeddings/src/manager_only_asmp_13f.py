"""Build a within-quarter ASMP universe using fit managers only.

This is a cross-manager representation diagnostic, not a temporal forecast or
the paper's original fund/company benchmark. Call it on a point-in-time 13F
snapshot, before constructing the ordinary all-manager saved matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import struct
from typing import Sequence

import pandas as pd

from src.asmp_13f import (
    ASMPManagerSplit,
    build_partitioned_manager_holdout,
    split_manager_ids,
)
from src.holdings_exploration import require_columns
from src.matrix_13f import MatrixParameters
from src.security_master_13f import build_cash_share_matrix_candidate


@dataclass(frozen=True)
class ManagerOnlyQuarter:
    """Private model inputs and aggregate diagnostic counts; never serialize IDs."""

    split: ASMPManagerSplit
    eligible_manager_count: int
    assigned_fit_managers: int
    assigned_test_managers: int
    fit_managers_after_filters: int
    test_managers_with_positive_positions: int
    fit_pairs_after_filters: int
    test_pairs_before_universe_projection: int
    fit_filter_checks: pd.DataFrame
    fit_pair_content_sha256: str
    test_pair_content_sha256: str


def _pair_content_sha256(pairs: pd.DataFrame) -> str:
    """Hash exact aggregated model-input rows without serializing private IDs."""

    digest = sha256()
    ordered = pairs.sort_values(["manager_cik", "typed_security_id"])
    for row in ordered.itertuples(index=False):
        digest.update(str(row.manager_cik).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.typed_security_id).encode("utf-8"))
        digest.update(b"\0")
        digest.update(struct.pack(">d", float(row.position_value_usd)))
    return digest.hexdigest()


def build_manager_only_quarter(
    classified: pd.DataFrame,
    *,
    eligible_manager_ids: Sequence[str],
    parameters: MatrixParameters = MatrixParameters(),
    seed: int = 17,
    test_fraction: float = 0.2,
) -> ManagerOnlyQuarter:
    """Split PIT-eligible managers before concentration and degree filtering.

Only fit-manager holdings determine the 75% concentration rule, iterative
20-by-20 core, and vocabulary. Evaluation managers are aggregated from their
positive eligible cash-share positions without applying the fit-universe
degree or concentration rules to them. Rank two remains their original second
largest eligible position; later OOV visible holdings are omitted explicitly.
"""

    require_columns(
        classified,
        ("manager_cik", "matrix_eligible", "typed_security_id", "value_usd_numeric"),
        context="manager-only classified holdings",
    )
    fit_ids, test_ids = split_manager_ids(
        tuple(eligible_manager_ids), seed=seed, test_fraction=test_fraction
    )
    fit_set, test_set = set(fit_ids), set(test_ids)
    if fit_set & test_set:
        raise AssertionError("Manager partitions overlap")
    fit_rows = classified.loc[classified["manager_cik"].isin(fit_set)].copy()
    test_rows = classified.loc[classified["manager_cik"].isin(test_set)].copy()
    fit_candidate = build_cash_share_matrix_candidate(
        fit_rows,
        maximum_position_weight=parameters.maximum_position_weight,
        minimum_assets_per_manager=parameters.minimum_assets_per_manager,
        minimum_managers_per_asset=parameters.minimum_managers_per_asset,
    )
    if fit_candidate.checks["violations"].sum() != 0:
        raise ValueError("Fit-only matrix candidate failed its integrity checks")
    # The candidate's prefilter positions retain test portfolios' original
    # rank ordering. No test holdings participate in a fit-side rule.
    test_candidate = build_cash_share_matrix_candidate(
        test_rows,
        maximum_position_weight=1.0,
        minimum_assets_per_manager=1,
        minimum_managers_per_asset=1,
    )
    fit_pairs = fit_candidate.filtered_positions
    test_pairs = test_candidate.positions
    if fit_pairs.empty or test_pairs.empty:
        raise ValueError("Fit or evaluation partition has no eligible positive positions")
    split = build_partitioned_manager_holdout(
        fit_pairs,
        test_pairs,
        out_of_vocabulary_visible="drop_after_rank_two",
    )
    if split.train_manager_count != int(fit_pairs["manager_cik"].nunique()):
        raise AssertionError("Fit portfolio count does not reconcile")
    return ManagerOnlyQuarter(
        split=split,
        eligible_manager_count=len(eligible_manager_ids),
        assigned_fit_managers=len(fit_ids),
        assigned_test_managers=len(test_ids),
        fit_managers_after_filters=int(fit_pairs["manager_cik"].nunique()),
        test_managers_with_positive_positions=int(test_pairs["manager_cik"].nunique()),
        fit_pairs_after_filters=len(fit_pairs),
        test_pairs_before_universe_projection=len(test_pairs),
        fit_filter_checks=fit_candidate.checks,
        fit_pair_content_sha256=_pair_content_sha256(fit_pairs),
        test_pair_content_sha256=_pair_content_sha256(test_pairs),
    )
