"""Signal-only balanced breadth amendment to the fixed-ten class baskets.

The original selector remains frozen. Each contemporaneous mode pair supplies
N=min(10, long candidates, short candidates) names per side, then the original
class probability and stable-source-ID ranking chooses those names. No observed
outcome, quote coverage, subsequent membership or realized PnL enters N or ranks.
This is a documented execution-policy adaptation, not a quote repair.
"""
from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd

from .backtest_selection import BASKETS, BasketSelection, select_monthly_basket
from .fallback_backtest import select_restricted_monthly_basket
from .universe import DataContractError


BALANCED_BREADTH_POLICY = {
    "policy_id": "balanced_contemporaneous_mode_breadth_cap10_v1",
    "maximum_names_per_side": 10,
    "breadth_formula": "min(10, contemporaneous_trade_eligible_long_mode_count, contemporaneous_trade_eligible_short_mode_count)",
    "weights": "50_percent_long_50_percent_short_equal_weight_within_actual_side",
    "ranking": "corresponding_mode_probability_descending_then_stable_source_id_ascending",
    "class_pairs": {"class_1_10": [1, 10], "class_2_9": [2, 9]},
    "zero_breadth": "withhold_rebalance_and_continuous_performance_no_invented_cash_path",
    "trade_restrictions": "same_frozen_DF_RDC_only_before_ranking",
    "prediction_cohorts": "all_original_rows_and_admissions_preserved",
    "ret_gate": "unchanged_120_completed_months_use_original_selector",
    "future_quote_or_target_inputs": False,
    "retired_security_orders": "existing_action_engine_retains_unfilled_cash_slots_no_replacement",
    "required_names_per_side_in_accounting": None,
}


def _build(predictions, *, basket, sessions, configuration):
    if basket not in BASKETS:
        raise DataContractError("Balanced breadth applies only to the two separate mode-class baskets; RET gate is unchanged")
    # This preserved worker validates the original recipe, prediction/admission
    # state, every original probability vector and all timing keys. Its full
    # ledger also reports the only authorized trading restrictions (DF/RDC).
    original = predictions.reset_index(drop=True)
    reference = select_restricted_monthly_basket(original, basket=basket,
        sessions=sessions, configuration=configuration)
    ledger = reference.ledger.copy(deep=True)
    rebalances = reference.rebalances.copy(deep=True)
    monthly = reference.monthly.copy(deep=True)
    for month_index, row in monthly.iterrows():
        n = min(10, int(row.long_candidates), int(row.short_candidates))
        signal = row.signal_date
        positions = ledger.index[ledger.signal_date.eq(signal)]
        adaptation = ("unchanged_full_ten_per_side" if n == 10 else
            "balanced_to_available_contemporaneous_mode_candidates" if n else
            "withheld_zero_contemporaneous_mode_candidates")
        if 0 < n < 10:
            # Only the contemporaneous prediction mask is narrowed by the same
            # two fixed trade restrictions. No financial data are supplied.
            working = original.iloc[positions].copy(deep=True).reset_index(drop=True)
            working["prediction_eligible"] = ledger.loc[positions, "trade_eligible"].to_numpy(dtype=bool)
            adapted = select_monthly_basket(working, basket=basket,
                sessions=sessions, top_k=n)
            if len(adapted.monthly) != 1 or adapted.monthly.iloc[0].status != "available":
                raise DataContractError("Positive balanced breadth failed the preserved selector")
            # Replace selection decisions only. Original prediction eligibility,
            # flags, admissions, modes and all denominator rows remain intact.
            for field in ("selection_score", "side", "selected", "selection_status"):
                ledger.loc[positions, field] = adapted.ledger[field].to_numpy()
            restricted = ledger.index[ledger.signal_date.eq(signal)
                & ledger.prediction_eligible & ledger.execution_restricted]
            ledger.loc[restricted, "selection_status"] = "execution_excluded_retrospective_data_quality"
            for field in adapted.monthly:
                if field.startswith(("long_cutoff_", "short_cutoff_")):
                    monthly.loc[month_index, field] = adapted.monthly.iloc[0][field]
        status = "available" if n else "withheld_zero_mode_candidates"
        if n == 0:
            trading = ledger.index[ledger.signal_date.eq(signal) & ledger.trade_eligible]
            ledger.loc[trading, "selection_status"] = status
            ledger.loc[positions, "selected"] = False
            ledger.loc[positions, "side"] = None
        monthly.loc[month_index, "top_k_per_side"] = n
        monthly.loc[month_index, "long_selected"] = n
        monthly.loc[month_index, "short_selected"] = n
        monthly.loc[month_index, "status"] = status
        monthly.loc[month_index, "requested_top_k"] = 10
        monthly.loc[month_index, "selected_names_per_side"] = n
        monthly.loc[month_index, "effective_per_name_abs_weight"] = .5 / n if n else np.nan
        monthly.loc[month_index, "long_per_name_weight"] = .5 / n if n else np.nan
        monthly.loc[month_index, "short_per_name_weight"] = -.5 / n if n else np.nan
        monthly.loc[month_index, "breadth_adaptation_reason"] = adaptation
        monthly.loc[month_index, "breadth_policy"] = BALANCED_BREADTH_POLICY["policy_id"]
        rebalance_index = rebalances.index[rebalances.signal_date.eq(signal)]
        rebalances.loc[rebalance_index, "selection_status"] = status
        rebalances.loc[rebalance_index, "requested_top_k"] = 10
        rebalances.loc[rebalance_index, "selected_names_per_side"] = n
        rebalances.loc[rebalance_index, "breadth_adaptation_reason"] = adaptation
        ledger.loc[positions, "requested_top_k"] = 10
        ledger.loc[positions, "selected_names_per_side"] = n
        ledger.loc[positions, "effective_per_name_abs_weight"] = .5 / n if n else np.nan
        ledger.loc[positions, "breadth_adaptation_reason"] = adaptation
    for table in (ledger, rebalances, monthly):
        table["requested_top_k"] = table.requested_top_k.astype(int)
        table["selected_names_per_side"] = table.selected_names_per_side.astype(int)
    selected = ledger.loc[ledger.selected].merge(
        rebalances[["signal_date", "execution_date"]], on="signal_date", validate="many_to_one")
    selected = selected.sort_values(["signal_date", "side", "source_security_id"], kind="stable")
    selected["intended_weight"] = np.where(selected.side.eq("long"),
        selected.effective_per_name_abs_weight, -selected.effective_per_name_abs_weight)
    return BasketSelection(ledger.reset_index(drop=True), selected.reset_index(drop=True),
        rebalances.reset_index(drop=True), monthly.reset_index(drop=True))


def select_balanced_monthly_basket(predictions, *, basket, sessions, configuration):
    """Return original rows plus separate balanced-N mode-class selections.

    The passed configuration is the unchanged frozen data-limited contract;
    this module's separately pinned ``BALANCED_BREADTH_POLICY`` is the explicit
    amendment. The accounting call must set required_names_per_side=None after
    ``validate_balanced_monthly_basket`` verifies these actual equal counts.
    """
    return _build(predictions, basket=basket, sessions=tuple(sessions), configuration=configuration)


def validate_balanced_monthly_basket(result, predictions, *, basket, sessions, configuration):
    """Reconcile all admissions, ranks and N before accounting from signals only.

    Recomputing the permitted selection from original forecasts detects added
    route exclusions, target-driven ordering, unequal counts, altered weights,
    changed ties and dropped denominator rows. No quote or outcome fields are
    decoded. Call before adding unrelated arm/report columns to the tables.
    """
    if not isinstance(result, BasketSelection):
        raise DataContractError("Balanced selection must retain the BasketSelection contract")
    expected = _build(predictions, basket=basket, sessions=tuple(sessions), configuration=configuration)
    for name in ("ledger", "selections", "rebalances", "monthly"):
        try:
            pd.testing.assert_frame_equal(getattr(result, name), getattr(expected, name),
                check_exact=True, check_like=False)
        except (AssertionError, TypeError, ValueError) as exc:
            raise DataContractError("Balanced selection changed original admissions, ranks, denominators or weights: " + name) from exc
    selected = result.selections
    for row in result.monthly.itertuples():
        n = min(10, int(row.long_candidates), int(row.short_candidates))
        month = selected.loc[selected.signal_date.eq(row.signal_date)]
        counts = month.groupby("side").size().to_dict()
        if counts.get("long", 0) != n or counts.get("short", 0) != n:
            raise DataContractError("Balanced selection must contain exactly equal permitted N on both sides")
        if n and (not np.isclose(month.loc[month.side.eq("long"), "intended_weight"].sum(), .5,
                    rtol=0., atol=1e-14)
                or not np.isclose(month.loc[month.side.eq("short"), "intended_weight"].sum(), -.5,
                    rtol=0., atol=1e-14)):
            raise DataContractError("Balanced intended weights must remain 50 percent long / 50 percent short")
    return {"policy": deepcopy(BALANCED_BREADTH_POLICY), "monthly_rebalances": len(result.monthly),
        "original_rows": len(result.ledger), "selected_rows": len(selected),
        "adapted_months": int(result.monthly.selected_names_per_side.between(1, 9).sum()),
        "withheld_zero_breadth_months": int(result.monthly.selected_names_per_side.eq(0).sum()),
        "minimum_names_per_side": int(result.monthly.selected_names_per_side.min()),
        "maximum_names_per_side": int(result.monthly.selected_names_per_side.max()),
        "original_prediction_cohort_preserved": True, "only_original_DF_RDC_trade_restrictions": True,
        "outcomes_or_future_quote_coverage_used": False}
