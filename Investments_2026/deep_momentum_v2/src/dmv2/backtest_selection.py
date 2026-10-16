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


def select_monthly_basket(predictions, *, basket, sessions, top_k=10, payoffs=None):
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
    if not calendar or tuple(sorted(set(calendar))) != calendar or calendar[-1] > "2019-12-31":
        raise DataContractError("Independent sessions must be unique, ordered and end by 2019")
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
        if (not "2017-12-01" <= signal <= "2019-11-30"
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
        if not "2017-12-01" <= signal <= "2019-11-30":
            raise DataContractError("Backtest signal outside Dec2017–Nov2019 development scope")
        same_month = [s for s in calendar if s[:7] == signal[:7]]
        if not same_month or signal != same_month[-1]:
            raise DataContractError("Signal is not independent XNYS month end")
        next_sessions = [s for s in calendar if s > signal]
        if not next_sessions:
            raise DataContractError("No independently evidenced next-open execution date")
        execution = next_sessions[0]
        outcome = (pd.Timestamp(signal).to_period("M") + 1).strftime("%Y-%m")
        if execution[:7] != outcome or not "2018-01" <= outcome <= "2019-12":
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
