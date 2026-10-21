"""Additive, signal-only RET selection on the preserved walk-forward ledger.

This adapter changes only the stock-ranking layer. It consumes explicit verified
120-calendar-month payoff objects, never target rows, quotes, model objects or
feature matrices. Original prediction admissions survive the two frozen DF/RDC
trade restrictions. RET uses every class probability and has no mode-class gate.
"""
from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy

import numpy as np
import pandas as pd

from .ret import CLASS_IDS, align_class_probabilities, score_ret
from .universe import DataContractError
from .walkforward_clock import YEARS, walkforward_clock
from .walkforward_selection import (
    BasketSelection, EXCLUDED_NATIVE_IDS, KEYS, PROBABILITIES, SAFE_COLUMNS,
    _dates, _select_fixed_monthly_basket, _selection_configuration,
)


RET_SELECTION_POLICY = {
    "policy_id": "balanced_full_distribution_RET_cap10_v1",
    "basket": "ret",
    "ranking_population": "all_original_prediction_eligible_minus_fixed_DF_RDC_trade_restrictions",
    "mode_class_filter": False,
    "payoff_policy": "realized_class_equal_month_120_v1",
    "breadth_formula": "min(10,floor(contemporaneous_trade_eligible_count/2))",
    "long_ranking": "RET_descending_then_stable_source_id_ascending",
    "short_ranking": "RET_ascending_then_stable_source_id_ascending_after_removing_selected_longs",
    "long_reservation_precedes_short_tie_resolution": True,
    "weights": "50_percent_long_50_percent_short_equal_weight_within_actual_side",
    "zero_breadth": "withhold_rebalance_and_continuous_performance_no_invented_cash_path",
    "incomplete_payoff_grid": "withhold_entire_signal_no_shortened_or_imputed_history",
    "positive_or_negative_score_filter": False,
    "flat_score_handling": "record_diagnostic_preserve_declared_stable_id_ties",
    "future_quote_or_target_inputs": False,
    "required_names_per_side_in_accounting": None,
}


def _safe_dates(values, name):
    try:
        return _dates(pd.DatetimeIndex(values), name)
    except (TypeError, ValueError, OverflowError) as error:
        raise DataContractError("Invalid RET " + name) from error


def _clock_scope(sessions, signal_dates, outcome_years):
    try:
        years = tuple(outcome_years)
    except TypeError as error:
        raise DataContractError("RET outcome_years must explicitly enumerate the bounded scope") from error
    if (any(type(year) is not int for year in years)
            or years not in (YEARS, *((year,) for year in YEARS))):
        raise DataContractError("RET selection requires exactly one canonical 12-month year or all 48 months")
    _, windows, clock = walkforward_clock()
    calendar = tuple(_safe_dates(sessions, "sessions"))
    if calendar != tuple(clock["selection_sessions"]):
        raise DataContractError("RET selection sessions differ from the complete independent walk-forward clock")
    expected = tuple(windows.loc[windows.outcome_month.str[:4].astype(int).isin(years), "signal_date"])
    if tuple(sorted(set(signal_dates))) != expected:
        raise DataContractError("RET original ledger must retain every canonical signal in the requested 12/48-month scope")
    return calendar, expected


def _build(predictions, *, payoffs, sessions, configuration, outcome_years):
    cfg = _selection_configuration(configuration)
    required = {*KEYS, "intrinio_security_id", "prediction_eligible", *PROBABILITIES}
    if (not isinstance(predictions, pd.DataFrame) or predictions.empty
            or not required.issubset(predictions.columns)
            or not predictions.columns.is_unique):
        raise DataContractError("RET requires unique original key/eligibility/probability columns")
    if not isinstance(payoffs, Mapping):
        raise DataContractError("RET payoffs must be an explicit signal-keyed mapping")
    # Project safe fields before copying: target availability/returns and future
    # quote evidence are deliberately never accessed, including for missingness.
    columns = [column for column in (*SAFE_COLUMNS, *PROBABILITIES) if column in predictions]
    original = predictions.loc[:, columns].reset_index(drop=True).copy(deep=True)
    if original[list(KEYS)].isna().any().any():
        raise DataContractError("RET original keys must be nonmissing")
    original["signal_date"] = _safe_dates(original.signal_date, "signal dates")
    if (any(not isinstance(value, str) or not value.strip() for value in original.source_security_id)
            or original.duplicated(list(KEYS)).any()):
        raise DataContractError("RET original stable source IDs must be nonempty and unique per signal")
    if not original.prediction_eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataContractError("RET original prediction eligibility must be explicit boolean")
    calendar, expected_signals = _clock_scope(sessions, original.signal_date, outcome_years)
    if any(not isinstance(key, str) or key not in expected_signals for key in payoffs):
        raise DataContractError("RET payoff mapping contains a key outside the explicit selection scope")
    eligible = original.prediction_eligible.to_numpy(dtype=bool)
    if (any(not isinstance(value, str) or not value.strip()
            for value in original.loc[eligible, "intrinio_security_id"])
            or original.loc[eligible].duplicated(["signal_date", "intrinio_security_id"]).any()):
        raise DataContractError("RET eligible native identities must be nonempty and unique per signal")
    # Validate even restricted original forecasts: exclusions cannot hide a bad
    # probability vector or silently alter the original model cohort/mode.
    aligned = align_class_probabilities(
        original.loc[eligible, list(PROBABILITIES)].to_numpy(), class_ids=CLASS_IDS)
    modes = pd.Series(pd.array([pd.NA] * len(original), dtype="Int64"))
    modes.loc[eligible] = np.argmax(aligned, axis=1) + 1
    restricted = original.intrinio_security_id.isin(EXCLUDED_NATIVE_IDS).to_numpy()
    trading = eligible & ~restricted
    ledger_parts, monthly_parts, rebalance_parts = [], [], []
    for signal in expected_signals:
        positions = np.flatnonzero(original.signal_date.eq(signal).to_numpy())
        group = original.iloc[positions].reset_index(drop=True)
        group_eligible, group_restricted, group_trading = eligible[positions], restricted[positions], trading[positions]
        count = int(group_trading.sum())
        available_n = min(10, count // 2)
        working = group.copy(deep=True)
        working["prediction_eligible"] = group_trading
        # The preserved RET branch supplies exact full-distribution arithmetic,
        # date/native checks, stable-ID ties and disjoint short ranking. A zero
        # breadth uses its legal minimum K for validation, then explicitly
        # withholds rather than inventing a cash-only portfolio.
        selected = _select_fixed_monthly_basket(working, basket="ret", sessions=calendar,
            top_k=max(1, available_n), payoffs=payoffs)
        ledger = selected.ledger.copy(deep=True)
        ledger["prediction_eligible"] = group_eligible
        ledger["predicted_class"] = modes.iloc[positions].reset_index(drop=True)
        ledger["trade_eligible"] = group_trading
        ledger["execution_restricted"] = group_restricted
        ledger["execution_restriction_reason"] = group.intrinio_security_id.map(cfg["execution_restrictions"]["native_ids"])
        ledger["retrospective_execution_restriction"] = group_restricted
        ledger["prediction_status"] = (group.prediction_status if "prediction_status" in group else
            np.where(group_eligible, "predicted", "withheld_signal_time_baseline"))
        ledger["ret_status"] = np.where(group_eligible,
            "withheld_incomplete_120_month_class_grid", "withheld_signal_time_baseline")
        payoff = payoffs.get(signal)
        fingerprint = None
        if payoff is not None:
            scored = score_ret(group.loc[group_eligible, list(PROBABILITIES)].to_numpy(), payoff,
                class_ids=CLASS_IDS, signal_date=signal, label_observed_through=signal)
            # Keep the exact floating-point scores that produced trading ranks.
            # Restricted forecasts still receive diagnostic RET scores, using
            # the same shared vector; they cannot change any selected holding.
            restricted_predictions = group_eligible & group_restricted
            all_scores = pd.Series(scored.ret_score.to_numpy(), index=group.index[group_eligible])
            ledger.loc[restricted_predictions, "selection_score"] = all_scores.reindex(group.index[restricted_predictions]).to_numpy()
            ledger.loc[group_eligible, "ret_status"] = scored.ret_status.to_numpy()
            fingerprint = payoff.fingerprint
        ledger["payoff_fingerprint"] = fingerprint
        status = str(selected.monthly.iloc[0].status)
        if not available_n and status == "withheld_insufficient_distinct_candidates":
            status = "withheld_zero_distinct_RET_breadth"
        actual_n = available_n if status == "available" else 0
        if not actual_n:
            ledger["selected"] = False
            ledger["side"] = None
            ledger.loc[group_trading, "selection_status"] = status
        ledger.loc[group_eligible & group_restricted, "selection_status"] = "execution_excluded_retrospective_data_quality"
        score = ledger.loc[group_trading, "selection_score"]
        finite_scores = score.dropna()
        flat = bool(len(finite_scores) and finite_scores.eq(finite_scores.iloc[0]).all())
        ledger["ret_long_rank"] = pd.array([pd.NA] * len(ledger), dtype="Int64")
        ledger["ret_short_rank_after_long_reservation"] = pd.array([pd.NA] * len(ledger), dtype="Int64")
        if len(finite_scores):
            ranked = ledger.loc[group_trading].sort_values(
                ["selection_score", "source_security_id"], ascending=[False, True], kind="stable")
            lows = ranked.drop(index=ranked.index[:available_n]).sort_values(
                ["selection_score", "source_security_id"], ascending=[True, True], kind="stable")
            ledger.loc[ranked.index, "ret_long_rank"] = np.arange(1, len(ranked) + 1)
            ledger.loc[lows.index, "ret_short_rank_after_long_reservation"] = np.arange(1, len(lows) + 1)
        adaptation = ("unchanged_full_ten_per_side" if actual_n == 10 else
            "balanced_to_available_distinct_RET_candidates" if actual_n else status)
        for name, value in (("requested_top_k", 10), ("selected_names_per_side", actual_n),
                ("effective_per_name_abs_weight", .5 / actual_n if actual_n else np.nan),
                ("breadth_adaptation_reason", adaptation), ("ret_score_flat", flat)):
            ledger[name] = value
        ledger.index = positions
        ledger_parts.append(ledger)
        monthly = selected.monthly.copy(deep=True)
        monthly["prediction_eligible"] = int(group_eligible.sum())
        monthly["trade_eligible"] = count
        monthly["long_candidates"] = count
        monthly["short_candidates"] = count - available_n
        monthly["execution_restricted_original_rows"] = int(group_restricted.sum())
        monthly["execution_restricted_prediction_rows"] = int((group_eligible & group_restricted).sum())
        monthly["retrospective_execution_restriction"] = True
        monthly["historical_universe_performance_claim"] = False
        monthly["top_k_per_side"] = actual_n
        monthly["long_selected"] = actual_n
        monthly["short_selected"] = actual_n
        monthly["requested_top_k"] = 10
        monthly["available_names_per_side"] = available_n
        monthly["selected_names_per_side"] = actual_n
        monthly["effective_per_name_abs_weight"] = .5 / actual_n if actual_n else np.nan
        monthly["long_per_name_weight"] = .5 / actual_n if actual_n else np.nan
        monthly["short_per_name_weight"] = -.5 / actual_n if actual_n else np.nan
        monthly["breadth_adaptation_reason"] = adaptation
        monthly["breadth_policy"] = RET_SELECTION_POLICY["policy_id"]
        monthly["status"] = status
        monthly["ret_score_flat"] = flat
        monthly["ret_score_min"] = float(finite_scores.min()) if len(finite_scores) else np.nan
        monthly["ret_score_max"] = float(finite_scores.max()) if len(finite_scores) else np.nan
        monthly["payoff_fingerprint"] = fingerprint
        monthly_parts.append(monthly)
        rebalances = selected.rebalances.copy(deep=True)
        rebalances["selection_status"] = status
        rebalances["requested_top_k"] = 10
        rebalances["selected_names_per_side"] = actual_n
        rebalances["breadth_adaptation_reason"] = adaptation
        rebalance_parts.append(rebalances)
    ledger = pd.concat(ledger_parts).sort_index().reset_index(drop=True)
    rebalances = pd.concat(rebalance_parts, ignore_index=True)
    monthly = pd.concat(monthly_parts, ignore_index=True)
    selected = ledger.loc[ledger.selected].merge(rebalances[["signal_date", "execution_date"]],
        on="signal_date", validate="many_to_one").sort_values(
            ["signal_date", "side", "source_security_id"], kind="stable").reset_index(drop=True)
    selected["intended_weight"] = np.where(selected.side.eq("long"),
        selected.effective_per_name_abs_weight, -selected.effective_per_name_abs_weight)
    if selected.intrinio_security_id.isin(EXCLUDED_NATIVE_IDS).any():
        raise DataContractError("A frozen trade-restricted native ID reached RET holdings")
    return BasketSelection(ledger, selected, rebalances, monthly)


def select_ret_walkforward_basket(predictions, *, payoffs, sessions, configuration,
                                 outcome_years=YEARS):
    """Select full-distribution RET extremes without changing the label/model.

    ``configuration`` is the preserved walk-forward contract, used only to
    authenticate its two existing trade restrictions and 50/50 cap-ten rules.
    ``payoffs`` maps each requested signal to a checked ``RetPayoffPanel``.
    Missing vectors withhold that signal. Supply one canonical annual forecast
    (``outcome_years=(year,)``), or all 48 months (the explicit default).
    """
    return _build(predictions, payoffs=payoffs, sessions=tuple(sessions),
        configuration=configuration, outcome_years=outcome_years)


def validate_ret_walkforward_basket(result, predictions, *, payoffs, sessions,
                                   configuration, outcome_years=YEARS):
    """Recompute all signal-only admissions, ranks, ties and weights exactly.

    This validation shares the preserved RET selection implementation; the
    execution audit must independently reconstruct the arithmetic as well.
    """
    if not isinstance(result, BasketSelection):
        raise DataContractError("RET selection must preserve the BasketSelection contract")
    expected = _build(predictions, payoffs=payoffs, sessions=tuple(sessions),
        configuration=configuration, outcome_years=outcome_years)
    for name in ("ledger", "selections", "rebalances", "monthly"):
        try:
            pd.testing.assert_frame_equal(getattr(result, name), getattr(expected, name),
                check_exact=True, check_like=False)
        except (AssertionError, TypeError, ValueError) as error:
            raise DataContractError("RET selection altered original admissions, ranks, denominators or weights: " + name) from error
    for month in result.monthly.itertuples():
        group = result.selections.loc[result.selections.signal_date.eq(month.signal_date)]
        n = int(month.selected_names_per_side)
        counts = group.groupby("side").size().to_dict()
        if (counts.get("long", 0) != n or counts.get("short", 0) != n
                or group.source_security_id.duplicated().any()):
            raise DataContractError("RET sides must be balanced and disjoint")
        if n and (not np.isclose(group.loc[group.side.eq("long"), "intended_weight"].sum(), .5, rtol=0, atol=1e-14)
                or not np.isclose(group.loc[group.side.eq("short"), "intended_weight"].sum(), -.5, rtol=0, atol=1e-14)):
            raise DataContractError("RET weights must remain 50 percent long / 50 percent short")
    return {"policy": deepcopy(RET_SELECTION_POLICY), "monthly_rebalances": len(result.monthly),
        "original_rows": len(result.ledger), "selected_rows": len(result.selections),
        "adapted_months": int(result.monthly.selected_names_per_side.between(1, 9).sum()),
        "withheld_months": int(result.monthly.status.ne("available").sum()),
        "flat_score_months": int(result.monthly.ret_score_flat.sum()),
        "original_prediction_cohort_preserved": True,
        "only_original_DF_RDC_trade_restrictions": True,
        "outcomes_or_future_quote_coverage_used": False}
