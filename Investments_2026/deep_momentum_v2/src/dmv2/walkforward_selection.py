"""Signal-only, fixed-breadth monthly portfolio selection.

V2 adaptation: ten names per direction within each corresponding mode class,
ranked by that class probability. No outcomes or price availability are inputs.
RET accepts only the existing verified 120-month payoff object; absent history
withholds RET rather than substituting a shorter window or class numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .ret import CLASS_IDS, align_class_probabilities, score_ret
from .universe import DataContractError

PROBABILITIES = tuple(f"probability_{k}" for k in CLASS_IDS)
KEYS = ("source_security_id", "signal_date")
BASKETS = {"class_1_10": (1, 10), "class_2_9": (2, 9)}
SAFE_COLUMNS = (
    *KEYS, "intrinio_security_id", "outcome_month", "outcome_start", "outcome_end",
    "prediction_eligible", "paper_prediction_eligible", "prediction_status",
    "identity_status", "source_identity_status", "identity_eligible",
    "identity_eligibility_policy", "identity_eligibility_reason", "needs_review",
    "identity_needs_review", "identity_review_required", "resolution_status",
)


@dataclass(frozen=True)
class BasketSelection:
    ledger: pd.DataFrame
    selections: pd.DataFrame
    rebalances: pd.DataFrame
    monthly: pd.DataFrame


def _dates(values, name):
    try:
        dates = pd.to_datetime(values, errors="raise")
    except (TypeError, ValueError) as error:
        raise DataContractError(f"Invalid {name}") from error
    if dates.tz is not None or dates.isna().any() or (dates != dates.normalize()).any():
        raise DataContractError(f"Missing or intraday {name}")
    return dates.strftime("%Y-%m-%d")


def _select_fixed_monthly_basket(predictions, *, basket, sessions, top_k=10, payoffs=None):
    """Retain original rows and select exactly K per side using signal data only.

    ``sessions`` is an independently constructed XNYS calendar, including the
    prior December. A shortage on either side withholds that entire rebalance.
    The engine must retain the withheld rebalance to fail the continuous path.
    Equal-probability/score ties use ascending stable source ID, never prices.
    """
    if basket not in (*BASKETS, "ret"):
        raise DataContractError("Unknown separate portfolio basket")
    if type(top_k) is not int or top_k < 1:
        raise DataContractError("top_k must be a positive integer")
    required = {*KEYS, "intrinio_security_id", "prediction_eligible", *PROBABILITIES}
    if not required.issubset(predictions.columns):
        raise DataContractError("Predictions lack original keys, eligibility or probabilities")
    if predictions.empty or predictions[list(KEYS)].isna().any().any():
        raise DataContractError("Prediction ledger cannot be empty or have missing keys")
    if any(not isinstance(x, str) or not x.strip() for x in predictions.source_security_id):
        raise DataContractError("Source security IDs must be nonempty stable strings")
    original = predictions.reset_index(drop=True).copy(deep=False)
    ledger = original[[c for c in SAFE_COLUMNS if c in original]].copy(deep=True)
    ledger["source_security_id"] = ledger.source_security_id.astype(str)
    ledger["signal_date"] = _dates(pd.DatetimeIndex(ledger.signal_date), "signal dates")
    if ledger.duplicated(list(KEYS)).any():
        raise DataContractError("Duplicate original signal keys")
    if not all(isinstance(x, (bool, np.bool_)) for x in ledger.prediction_eligible):
        raise DataContractError("Prediction eligibility must be explicit boolean signal state")
    calendar = tuple(_dates(pd.DatetimeIndex(sessions), "sessions"))
    if not calendar or tuple(sorted(set(calendar))) != calendar or calendar[-1] > "2023-12-31":
        raise DataContractError("Independent sessions must be unique, ordered and end by 2023")
    eligible = ledger.prediction_eligible.to_numpy(dtype=bool)
    if ledger.loc[eligible, "intrinio_security_id"].isna().any():
        raise DataContractError("Eligible prediction lacks native security identity")
    if any(not isinstance(x, str) or not x.strip()
           for x in ledger.loc[eligible, "intrinio_security_id"]):
        raise DataContractError("Eligible prediction has a blank/non-string native identity")
    if ledger.loc[eligible].duplicated(["signal_date", "intrinio_security_id"]).any():
        raise DataContractError("Eligible native identity collision at a signal")
    # Validate every timing key before numerical probability conversion.
    for signal in ledger.signal_date.unique():
        same_month = [s for s in calendar if s[:7] == signal[:7]]
        if (not "2019-12-01" <= signal <= "2023-11-30"
                or not same_month or signal != same_month[-1]):
            raise DataContractError("Signal is outside scope or not independent XNYS month end")
        outcome = (pd.Timestamp(signal).to_period("M") + 1).strftime("%Y-%m")
        next_month = [s for s in calendar if s[:7] == outcome]
        if not next_month:
            raise DataContractError("Missing independent next-month session evidence")
        group = ledger.loc[ledger.signal_date == signal]
        expected = {"outcome_month": outcome, "outcome_start": next_month[0], "outcome_end": next_month[-1]}
        for field, value in expected.items():
            if field in group and any(str(x) != value for x in group[field]):
                raise DataContractError("Prediction outcome metadata differs from independent calendar")
    probabilities = align_class_probabilities(
        original.loc[eligible, list(PROBABILITIES)].to_numpy(), class_ids=CLASS_IDS)
    ledger["predicted_class"] = pd.array([pd.NA] * len(ledger), dtype="Int64")
    ledger.loc[eligible, "predicted_class"] = np.argmax(probabilities, axis=1) + 1
    ledger["selection_score"] = np.nan
    ledger["side"] = pd.Series([None] * len(ledger), index=ledger.index, dtype=object)
    ledger["selected"] = False
    ledger["selection_status"] = np.where(eligible, "not_selected", "withheld_signal_time_baseline")
    rebalances, monthly = [], []
    # All probability values are indexed by original row positions; no target
    # field or future quote is accessed while ordering these candidates.
    pframe = pd.DataFrame(probabilities, index=ledger.index[eligible], columns=CLASS_IDS)
    for signal, group in ledger.groupby("signal_date", sort=True):
        if not "2019-12-01" <= signal <= "2023-11-30":
            raise DataContractError("Backtest signal outside Dec2019–Nov2023 walk-forward scope")
        same_month = [s for s in calendar if s[:7] == signal[:7]]
        if not same_month or signal != same_month[-1]:
            raise DataContractError("Signal is not independent XNYS month end")
        next_sessions = [s for s in calendar if s > signal]
        if not next_sessions:
            raise DataContractError("No independently evidenced next-open execution date")
        execution = next_sessions[0]
        outcome = (pd.Timestamp(signal).to_period("M") + 1).strftime("%Y-%m")
        if execution[:7] != outcome or not "2020-01" <= outcome <= "2023-12":
            raise DataContractError("Execution must start the next development outcome month")
        if "outcome_month" in group and any(str(x) != outcome for x in group.outcome_month):
            raise DataContractError("Prediction outcome month differs from next calendar month")
        idx = group.index[group.prediction_eligible]
        status = "available"
        candidate_counts, cutoffs = {}, {}
        if basket == "ret":
            payoff = None if payoffs is None else payoffs.get(signal)
            if payoff is None:
                status = "withheld_incomplete_120_month_class_grid"
                candidates = {"long": group.iloc[0:0], "short": group.iloc[0:0]}
            else:
                scored = score_ret(pframe.loc[idx].to_numpy(), payoff,
                    class_ids=CLASS_IDS, signal_date=signal, label_observed_through=signal)
                # score_ret preserves positional rows; its canonical verified
                # payoff provenance determines all-or-nothing availability.
                if not payoff.provenance["vector_available"]:
                    status = "withheld_incomplete_120_month_class_grid"
                    candidates = {"long": group.iloc[0:0], "short": group.iloc[0:0]}
                else:
                    values = np.asarray(scored["ret_score"], dtype=float)
                    ledger.loc[idx, "selection_score"] = values
                    ranked = ledger.loc[idx].sort_values(
                        ["selection_score", "source_security_id"], ascending=[False, True], kind="stable")
                    # A fixed-breadth tie is resolved by stable source ID.
                    # Reserve the long names first, then choose disjoint lows.
                    lows = ledger.loc[idx].drop(index=ranked.index[:top_k]).sort_values(
                        ["selection_score", "source_security_id"], ascending=[True, True], kind="stable")
                    candidates = {"long": ranked, "short": lows}
                    if len(ranked) < 2 * top_k:
                        status = "withheld_insufficient_distinct_candidates"
        else:
            best, worst = BASKETS[basket]
            candidates = {}
            for side, class_id in (("long", best), ("short", worst)):
                chosen = idx[ledger.loc[idx, "predicted_class"].to_numpy() == class_id]
                ledger.loc[chosen, "selection_score"] = pframe.loc[chosen, class_id]
                candidates[side] = ledger.loc[chosen].sort_values(
                    ["selection_score", "source_security_id"], ascending=[False, True], kind="stable")
                if len(chosen) < top_k:
                    status = "withheld_insufficient_mode_candidates"
        for side in ("long", "short"):
            candidate_counts[side] = len(candidates[side])
            ranked_side = candidates[side]
            cutoff = None if len(ranked_side) < top_k else float(ranked_side.iloc[top_k - 1].selection_score)
            cutoffs[side] = {"cutoff_probability_or_ret": cutoff,
                "cutoff_equal_candidates": 0 if cutoff is None else int((ranked_side.selection_score == cutoff).sum())}
        if status == "available":
            for side in ("long", "short"):
                selected_idx = candidates[side].index[:top_k]
                ledger.loc[selected_idx, "side"] = side
                ledger.loc[selected_idx, "selected"] = True
                ledger.loc[selected_idx, "selection_status"] = "selected"
        else:
            ledger.loc[idx, "selection_status"] = status
        rebalances.append({"signal_date": signal, "execution_date": execution,
                           "selection_status": status})
        monthly.append({"signal_date": signal, "execution_date": execution,
            "outcome_month": outcome, "basket": basket, "top_k_per_side": top_k,
            "original_rows": len(group), "prediction_eligible": len(idx),
            "long_candidates": candidate_counts["long"], "short_candidates": candidate_counts["short"],
            "long_selected": top_k if status == "available" else 0,
            "short_selected": top_k if status == "available" else 0, "status": status,
            **{f"{side}_{name}": value for side, row in cutoffs.items() for name, value in row.items()}})
    rebalance = pd.DataFrame(rebalances)
    selected = ledger.loc[ledger.selected].merge(
        rebalance[["signal_date", "execution_date"]], on="signal_date", validate="many_to_one")
    selected = selected.sort_values(["signal_date", "side", "source_security_id"], kind="stable")
    return BasketSelection(ledger.reset_index(drop=True), selected.reset_index(drop=True),
                           rebalance, pd.DataFrame(monthly))

from copy import deepcopy


EXCLUDED_NATIVE_IDS = ("sec_gjP4xM", "sec_zmYNnz")


def _selection_configuration(cfg):
    restrictions, bt = cfg.get("execution_restrictions", {}), cfg.get("backtest", {})
    if (cfg.get("protocol_id") != "dmv2-walkforward-2020-2023-v1"
            or tuple(sorted(restrictions.get("native_ids", {}))) != EXCLUDED_NATIVE_IDS
            or restrictions.get("period") != "all_2020_2023_signals_before_selection"
            or restrictions.get("valid_historical_training_rows_retained") is not True
            or restrictions.get("prediction_rows_and_probabilities_retained") is not True
            or bt.get("start") != "2020-01-01" or bt.get("end") != "2023-12-31"
            or bt.get("top_k_per_side") != 10
            or bt.get("baskets") != ["class_1_10", "class_2_9"]
            or bt.get("long_weight") != .5 or bt.get("short_weight") != -.5):
        raise DataContractError("Walk-forward selection requires the declared scope and unchanged two trade-only restrictions")
    return cfg


def _restricted_selection(predictions, *, basket, sessions, configuration):
    """Keep prediction eligibility intact; restrict execution before ordering.

    No actual quote, outcome or future membership is inspected. The two
    excluded instruments are a declared retrospective availability restriction,
    not historically available evidence or a claim to remove selection bias.
    """
    cfg = _selection_configuration(configuration)
    required = {"prediction_eligible", "intrinio_security_id", "source_security_id", "signal_date", *PROBABILITIES}
    if not isinstance(predictions,pd.DataFrame) or not required <= set(predictions):
        raise DataContractError("Original prediction keys/eligibility/probabilities required")
    if not predictions.prediction_eligible.map(lambda value:isinstance(value,(bool,np.bool_))).all():
        raise DataContractError("Original prediction eligibility must be explicit boolean")
    original = predictions.reset_index(drop=True)
    eligible = original.prediction_eligible.to_numpy(dtype=bool)
    restricted = original.intrinio_security_id.isin(cfg["execution_restrictions"]["native_ids"]).to_numpy()
    trading = eligible & ~restricted
    working = original.copy(deep=True)
    working["prediction_eligible"] = trading
    selected = _select_fixed_monthly_basket(working,basket=basket,sessions=sessions,top_k=cfg["backtest"]["top_k_per_side"])
    ledger = selected.ledger.copy(deep=True)
    if not ledger[["source_security_id","signal_date"]].equals(original[["source_security_id","signal_date"]]):
        raise DataContractError("Frozen selector changed original prediction keys/order")
    ledger["prediction_eligible"] = eligible
    ledger["trade_eligible"] = trading
    ledger["execution_restricted"] = restricted
    ledger["execution_restriction_reason"] = original.intrinio_security_id.map(cfg["execution_restrictions"]["native_ids"])
    ledger["retrospective_execution_restriction"] = restricted
    ledger.loc[eligible & restricted,"selection_status"] = "execution_excluded_retrospective_data_quality"
    # Preserve the actual classifier mode even for a prediction excluded from
    # trading. Validate all original forecast vectors, never silently hide them.
    probabilities = align_class_probabilities(original.loc[eligible,list(PROBABILITIES)].to_numpy(),class_ids=CLASS_IDS)
    ledger.loc[eligible,"predicted_class"] = np.argmax(probabilities,axis=1)+1
    ledger["prediction_status"] = original.prediction_status if "prediction_status" in original else np.where(eligible,"predicted","withheld_signal_time_baseline")
    monthly = selected.monthly.copy(deep=True)
    grouped = ledger.groupby("signal_date",sort=True)
    eligible_counts = grouped.prediction_eligible.sum()
    restricted_counts = grouped.apply(lambda group:int((group.prediction_eligible & group.execution_restricted).sum()))
    monthly["trade_eligible"] = monthly["prediction_eligible"]
    monthly["prediction_eligible"] = monthly.signal_date.map(eligible_counts).astype(int)
    monthly["execution_restricted_original_rows"] = monthly.signal_date.map(grouped.execution_restricted.sum()).astype(int)
    monthly["execution_restricted_prediction_rows"] = monthly.signal_date.map(restricted_counts).astype(int)
    monthly["retrospective_execution_restriction"] = True
    monthly["historical_universe_performance_claim"] = False
    actual = ledger.loc[ledger.selected].merge(selected.rebalances[["signal_date","execution_date"]],
        on="signal_date",validate="many_to_one").sort_values(["signal_date","side","source_security_id"],kind="stable")
    if actual.intrinio_security_id.isin(EXCLUDED_NATIVE_IDS).any():
        raise DataContractError("Execution-excluded instrument reached selected holdings")
    return BasketSelection(ledger,actual.reset_index(drop=True),selected.rebalances.copy(deep=True),monthly)


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
    "ret_gate": "not_evaluated_in_this_class_basket_experiment",
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
    reference = _restricted_selection(original, basket=basket,
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
            adapted = _select_fixed_monthly_basket(working, basket=basket,
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


def select_walkforward_basket(predictions, *, basket, sessions, configuration):
    """Return original rows plus separate balanced-N mode-class selections.

    The passed configuration is the frozen 2020–2023 walk-forward contract.
    The accounting call must set required_names_per_side=None after
    ``validate_walkforward_basket`` verifies these actual equal counts.
    """
    return _build(predictions, basket=basket, sessions=tuple(sessions), configuration=configuration)


def validate_walkforward_basket(result, predictions, *, basket, sessions, configuration):
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
