"""Recompute frozen formulas and shared classes on the explicit fallback view."""
from dataclasses import dataclass

import pyarrow as pa

from .fallback_loading import (FALLBACK_POLICY, KEYS, LoadedFallbackYear,
                               _fingerprint, _need)
from .full_features import build_full_feature_panel
from .identity_amendment import SOURCE_ROUTES
from .identity_policy import PROVISIONAL_IDENTITY_POLICY
from .monthly import MonthlyReturnPanel, build_monthly_returns
from .paper_features import COMMON_MOMENTUM_POLICY
from .targets import MonthlyTargetPanel, build_monthly_targets


@dataclass(frozen=True)
class FallbackFeatureYear:
    features: object
    monthly: MonthlyReturnPanel
    targets: MonthlyTargetPanel
    loaded: LoadedFallbackYear
    provenance: dict


def build_fallback_year(loaded):
    """Keep every original key; recompute cross sections, benchmarks and ranks.

    The wrong BBBY route contributes no own price features, shared price
    statistics, label or RET payoff. Missing rows remain in all original
    denominators. Other valid histories, including DF/RDC, remain inputs.
    No model is fitted and no future-year source is opened by this worker.
    """
    _need(isinstance(loaded, LoadedFallbackYear)
          and _fingerprint(loaded.plan) == loaded.plan.fingerprint,
          "Unchanged explicitly approved fallback inputs required")
    plan = loaded.plan
    _need(loaded.provenance.get("policy") == FALLBACK_POLICY
          and loaded.provenance.get("approved_missing_route_fallback") is True
          and loaded.provenance.get("fully_identity_repaired") is False,
          "Fallback authority cannot become a complete identity repair claim")
    features = build_full_feature_panel(loaded.feature_prices, plan.calendar, plan.snapshots,
        requested_security_ids=plan.requested_ids, daily_cohorts=plan.daily_cohorts,
        mode="historical_reconstruction", identity_policy=PROVISIONAL_IDENTITY_POLICY,
        normalization_policy=COMMON_MOMENTUM_POLICY, context_policy="available_cohort_80pct")
    monthly = build_monthly_returns(loaded.outcome_prices, plan.calendar, plan.snapshots,
        start_month=f"{plan.year}-01", end_month=f"{plan.year}-12",
        requested_security_ids=plan.requested_ids, mode="historical_reconstruction",
        identity_policy=PROVISIONAL_IDENTITY_POLICY)
    targets = build_monthly_targets(features.coverage, monthly,
        label_observed_through=plan.windows[-1].outcome_end)
    original = plan.original_cohort.select(list(KEYS)).to_pylist()
    for table in (features.table, features.coverage, monthly.table, targets.table):
        _need(table.select(list(KEYS)).to_pylist() == original,
              "Fallback computation changed ordered original cohort denominators")
    coverage = features.coverage.to_pylist();labels = targets.table.to_pylist()
    for feature, target in zip(coverage, labels, strict=True):
        if feature["source_security_id"] in SOURCE_ROUTES:
            _need(feature["identity_eligible"] is False and feature["requested"] is False
                  and feature["paper_prediction_eligible"] is False
                  and target["target_available"] is False,
                  "Quarantined old BBBY route acquired a feature/label admission")
    provenance = {**loaded.provenance, "features_calculated": True,
        "monthly_returns_recomputed": True, "target_classes_recomputed": True,
        "all_original_rows_and_denominators_retained": True,
        "paper15_common_population_recomputed": True,
        "daily_historical_benchmarks_recomputed": True,
        "ret_payoff_source_is_new_shared_target_panel": True,
        "old_features_targets_or_model_predictions_reused": False,
        "model_fitted": False, "daily_backtest_run": False,
        "reconstruction_not_knowledge_time_certification": True}
    return FallbackFeatureYear(features, monthly, targets, loaded, provenance)
