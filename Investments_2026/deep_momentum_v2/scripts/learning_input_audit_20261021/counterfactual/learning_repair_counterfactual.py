"""Pure conditional data-repair audit under the unchanged monthly label policy.

All inputs must be authenticated and financially projected by the separately
frozen caller. This module performs no I/O, fitting, inference or backtesting.
It never promotes a shadow label/payoff state to the production research panel.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
from collections.abc import Mapping

import numpy as np
import pandas as pd


CLASS_IDS = tuple(range(1, 11))
KEYS = ("source_security_id", "signal_date")
MONTH_FIELDS = ("outcome_month", "outcome_end", "class_id", "original_member_count",
    "prediction_eligible_count", "member_count", "class_mean_return", "status")
TARGET_FIELDS = (*KEYS, "outcome_month", "outcome_start", "outcome_end", "identity_eligible",
    "requested", "paper_prediction_eligible", "prediction_eligible", "outcome_requested",
    "target_class", "label_return", "return_available", "observed_return_available",
    "target_available", "target_status", "return_observation_status", "source_return_status")


class CounterfactualError(ValueError):
    pass


def need(condition, message):
    if not bool(condition): raise CounterfactualError(message)


def month_step(month, offset):
    need(isinstance(month, str) and len(month) == 7 and month[4] == "-", "Explicit calendar month required")
    year, number = map(int, month.split("-")); need(1 <= number <= 12, "Invalid calendar month")
    value = year * 12 + number - 1 + offset
    return f"{value // 12:04d}-{value % 12 + 1:02d}"


def day(value):
    need(isinstance(value, str), "Date metadata must be canonical strings")
    try: parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as error: raise CounterfactualError("Invalid date metadata") from error
    need(parsed.tz is None and parsed == parsed.normalize() and parsed.strftime("%Y-%m-%d") == value
         and "1962-01-01" <= value <= "2023-12-31", "Protected, missing or noncanonical date")
    return value


def strict_bool(value):
    return isinstance(value, (bool, np.bool_))


def missing(value):
    return value is None or value is pd.NA or bool(pd.isna(value))


def finite_return(value):
    need(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
         and math.isfinite(value) and value >= -1, "Observed return must be finite real >= -1")
    return float(value)


def same_number(a, b):
    return missing(a) and missing(b) or not missing(a) and not missing(b) and float(a) == float(b)


def classify_repair_label_domain(repairs, windows):
    """Metadata-only endpoint dependencies, with no inferred economic return.

    Each repair must explicitly state ``same_security_price_only``. New share
    classes and legal cash/stock entitlements require another return policy;
    a successor quote is never copied into an old issuer's monthly close.
    """
    output = []
    for repair in repairs:
        date = day(repair["date"])
        price_only = repair.get("same_security_price_only") is True
        for window in windows.to_dict("records"):
            for name in ("signal_date", "outcome_start", "outcome_end"): day(window[name])
            role = ("prior_month_end_anchor" if date == window["signal_date"] else
                "outcome_month_end" if date == window["outcome_end"] else
                "interior_daily_bar" if window["outcome_start"] <= date <= window["outcome_end"] else None)
            if role is None: continue
            output.append({"repair_id": repair["repair_id"], "repair_date": date,
                "outcome_month": window["outcome_month"], "signal_date": window["signal_date"],
                "role": role, "same_security_price_only": price_only,
                "classification": ("requires_separate_economic_return_policy" if not price_only else
                    "endpoint_return_candidate_requires_both_observed_endpoints" if role != "interior_daily_bar" else
                    "no_endpoint_label_change"), "automatic_label_change": False})
    return pd.DataFrame(output)


def _metadata(frame, observed_through):
    need(isinstance(frame, pd.DataFrame) and not frame.empty and frame.columns.is_unique
         and set(TARGET_FIELDS) <= set(frame), "Complete original target-month rows required")
    day(observed_through)
    need(not frame[list(KEYS)].isna().any().any() and not frame.duplicated(list(KEYS)).any(), "Unique original cohort keys required")
    need(frame.source_security_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all(), "Stable source IDs required")
    windows = frame[["signal_date", "outcome_month", "outcome_start", "outcome_end"]].drop_duplicates()
    need(len(windows) == 1, "Exactly one exhaustive original outcome-month cross section required")
    window = windows.iloc[0].to_dict()
    for field in ("signal_date", "outcome_start", "outcome_end"): day(window[field])
    need("2010-01" <= window["outcome_month"] <= "2023-12"
         and window["outcome_month"] == month_step(window["signal_date"][:7], 1)
         and window["outcome_start"][:7] == window["outcome_end"][:7] == window["outcome_month"]
         and window["signal_date"] < window["outcome_start"] <= window["outcome_end"] <= observed_through,
         "Completed next-calendar-month metadata must pass before return conversion")
    for field in ("identity_eligible", "requested", "paper_prediction_eligible", "prediction_eligible",
                  "return_available", "observed_return_available", "target_available"):
        need(frame[field].map(strict_bool).all(), "Original availability/admission flags must be Boolean")
    need(frame.outcome_requested.map(lambda value: missing(value) or strict_bool(value)).all(), "Forward request state must be Boolean or unknown")
    need(frame.prediction_eligible.eq(frame.identity_eligible & frame.paper_prediction_eligible).all()
         and frame.return_available.eq(frame.observed_return_available).all()
         and (~frame.return_available | frame.prediction_eligible).all()
         and (~frame.target_available | frame.return_available).all(), "Original cohort/availability states are inconsistent")
    need(frame.target_class.map(lambda value: missing(value) or
         isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
         and math.isfinite(value) and value == int(value) and 1 <= value <= 10).all(),
         "Original target classes must be missing or integer class IDs 1..10")
    return window


def _reconstruct_classes(frame, *, observed_through):
    window = _metadata(frame, observed_through)
    result = frame.copy(deep=True)
    observed = result.prediction_eligible & result.return_available
    values = {index: finite_return(value) for index, value in result.loc[observed, "label_return"].items()}
    for index in result.index[observed]:
        need(result.at[index, "source_return_status"] == "ok"
             and result.at[index, "return_observation_status"] == "ok"
             and not missing(result.at[index, "outcome_requested"])
             and bool(result.at[index, "outcome_requested"]), "Observed return lacks original exhaustive request/source evidence")
    exhaustive = all(not bool(row.identity_eligible) or (bool(row.requested)
        and not missing(row.outcome_requested) and bool(row.outcome_requested)
        and row.source_return_status != "not_requested") for row in result.itertuples())
    status = "non_exhaustive_original_requests" if not exhaustive else "insufficient_observed_population" if len(values) < 10 else "ok"
    result["target_class"] = pd.array([pd.NA] * len(result), dtype="Int64")
    result["target_available"] = False
    for index in values: result.at[index, "target_status"] = status
    if status == "ok":
        ranked = sorted(values, key=lambda index: values[index], reverse=True)
        lower = 0
        while lower < len(ranked):
            upper = lower + 1
            while upper < len(ranked) and values[ranked[upper]] == values[ranked[lower]]: upper += 1
            cls = min(10, max(1, 1 + (10 * (lower + upper)) // (2 * len(ranked))))
            for index in ranked[lower:upper]:
                result.at[index, "target_class"] = cls; result.at[index, "target_available"] = True
            lower = upper
    return result, {**window, "rank_status": status, "original_members": len(result),
        "prediction_eligible_members": int(result.prediction_eligible.sum()), "observed_return_members": len(values),
        "labelled_members": int(result.target_available.sum()), "exhaustive_requests": exhaustive}


def monthly_class_means(frame):
    window = frame[["outcome_month", "outcome_end"]].drop_duplicates()
    need(len(window) == 1, "Single target month required for realized class means")
    info = window.iloc[0].to_dict(); result = []
    for cls in CLASS_IDS:
        mask = frame.target_available & frame.target_class.eq(cls).fillna(False)
        values = [finite_return(value) for value in frame.loc[mask, "label_return"]]
        result.append({**info, "class_id": cls, "original_member_count": len(frame),
            "prediction_eligible_count": int(frame.prediction_eligible.sum()), "member_count": len(values),
            "class_mean_return": math.fsum(value / len(values) for value in values) if values else None,
            "status": "available" if values else "empty_realized_class"})
    return pd.DataFrame(result, columns=list(MONTH_FIELDS))


@dataclass(frozen=True)
class MonthlyCounterfactual:
    baseline: pd.DataFrame
    shadow: pd.DataFrame
    changes: pd.DataFrame
    monthly_before: pd.DataFrame
    monthly_after: pd.DataFrame
    summary: dict


def rerank_month_after_endpoint_repair(target_rows, change, *, observed_through):
    """One explicit, conditional endpoint fix; reproduce the saved labels first.

    ``change`` names the original source key, dates, expected old return and
    both observed adjusted closes. It cannot supply a replacement return or
    change requests, identities, the cohort, targets' definition or model state.
    """
    baseline = target_rows.reset_index(drop=True).copy(deep=True)
    rebuilt, old_summary = _reconstruct_classes(baseline, observed_through=observed_through)
    need(baseline.target_available.equals(rebuilt.target_available)
         and all(same_number(a, b) for a, b in zip(baseline.target_class, rebuilt.target_class))
         and baseline.target_status.equals(rebuilt.target_status), "Baseline exact original monthly classes/status could not be reproduced")
    need(change.get("scope") == "conditional_shadow_only" and change.get("same_security_price_only") is True
         and change.get("uses_cash_or_successor_equity") is False, "Economic cash/successor labels require a separate policy")
    for field in ("signal_date", "start_date", "end_date", "repaired_date"): day(change[field])
    need(change["signal_date"] == change["start_date"] == old_summary["signal_date"]
         and change["end_date"] == old_summary["outcome_end"]
         and change["repaired_date"] in (change["start_date"], change["end_date"]),
         "Interior bars or a changed holding window cannot alter an endpoint label")
    indices = baseline.index[baseline.source_security_id.eq(change["source_security_id"])
        & baseline.signal_date.eq(change["signal_date"])]
    need(len(indices) == 1, "Endpoint repair must name one retained original source key")
    index = indices[0]; original = baseline.loc[index]
    need(bool(original.prediction_eligible) and bool(original.requested)
         and not missing(original.outcome_requested) and bool(original.outcome_requested),
         "Shadow endpoint cannot newly admit a member or manufacture a forward request")
    need("expected_old_label_return" in change and same_number(original.label_return, change["expected_old_label_return"]),
         "Endpoint repair old-return precondition differs from the pinned baseline")
    prices = [change["start_adj_close"], change["end_adj_close"]]
    need(all(isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
             and math.isfinite(value) and value > 0 for value in prices), "Both actual positive adjusted endpoints are required")
    new_return = finite_return(float(prices[1]) / float(prices[0]) - 1.)
    shadow = baseline.copy(deep=True)
    for field, value in {"label_return": new_return, "return_available": True,
            "observed_return_available": True, "source_return_status": "ok", "return_observation_status": "ok",
            "target_available": False, "target_status": "ok"}.items(): shadow.at[index, field] = value
    shadow, new_summary = _reconstruct_classes(shadow, observed_through=observed_through)
    preserved = [*KEYS, "outcome_month", "outcome_start", "outcome_end", "identity_eligible", "requested",
        "paper_prediction_eligible", "prediction_eligible", "outcome_requested"]
    need(baseline[preserved].equals(shadow[preserved]), "Counterfactual changed the frozen cohort or request/identity metadata")
    changes = []
    for position, old in baseline.iterrows():
        new = shadow.loc[position]
        changed_return = not same_number(old.label_return, new.label_return)
        changed_class = not same_number(old.target_class, new.target_class)
        changed_status = old.target_status != new.target_status or bool(old.target_available) != bool(new.target_available)
        if changed_return or changed_class or changed_status:
            changes.append({**{key: old[key] for key in KEYS}, "outcome_month": old.outcome_month,
                "baseline_return": None if missing(old.label_return) else float(old.label_return),
                "shadow_return": None if missing(new.label_return) else float(new.label_return),
                "baseline_class": None if missing(old.target_class) else int(old.target_class),
                "shadow_class": None if missing(new.target_class) else int(new.target_class),
                "baseline_status": old.target_status, "shadow_status": new.target_status,
                "return_changed": changed_return, "class_changed": changed_class,
                "availability_or_status_changed": changed_status})
    return MonthlyCounterfactual(baseline, shadow, pd.DataFrame(changes), monthly_class_means(rebuilt),
        monthly_class_means(shadow), {"scope": "conditional_shadow_only", "production_promoted": False,
            "baseline_numeric_classes_reproduced": True, "baseline": old_summary, "shadow": new_summary,
            "direct_endpoint_row": change["source_security_id"], "new_endpoint_return": new_return,
            "return_rows_changed": sum(row["return_changed"] for row in changes),
            "class_rows_changed": sum(row["class_changed"] for row in changes),
            "models_fitted": 0, "preprocessors_fitted": 0, "new_inference_calls": 0})


def propagate_ret_payoff_impact(base_monthly_means, month_changes, signal_windows, *, baseline_vector_fingerprints):
    """Conditional shared-payoff changes; no frozen model/probability recompute."""
    need(isinstance(base_monthly_means, pd.DataFrame) and set(MONTH_FIELDS) <= set(base_monthly_means)
         and not base_monthly_means.duplicated(["outcome_month", "class_id"]).any(), "Original 167-month true-class means required")
    expected = [(month_step("2010-01", i), cls) for i in range(167) for cls in CLASS_IDS]
    need(sorted(map(tuple, base_monthly_means[["outcome_month", "class_id"]].to_numpy())) == expected,
         "Exact January2010–November2023 ten-class calendar grid required")
    for row in base_monthly_means.itertuples():
        day(row.outcome_end); need(row.outcome_end[:7] == row.outcome_month, "Original mean completion month changed")
    need(isinstance(baseline_vector_fingerprints, Mapping) and len(signal_windows) == 48
         and signal_windows.signal_date.tolist() == sorted(set(signal_windows.signal_date))
         and [value[:7] for value in signal_windows.signal_date] == [month_step("2019-12", i) for i in range(48)]
         and set(baseline_vector_fingerprints) == set(signal_windows.signal_date), "Exactly 48 original vector fingerprints/signals required")
    base = {(row["outcome_month"], int(row["class_id"])): row for row in base_monthly_means.to_dict("records")}
    shadow = {key: dict(value) for key, value in base.items()}; patched_months = set()
    for change in month_changes:
        need(isinstance(change, MonthlyCounterfactual) and change.summary.get("production_promoted") is False,
             "Only explicit unpromoted conditional monthly changes admitted")
        month = change.summary["baseline"]["outcome_month"]
        need(month not in patched_months, "Duplicate monthly shadow patch")
        patched_months.add(month)
        for before, after in zip(change.monthly_before.to_dict("records"), change.monthly_after.to_dict("records"), strict=True):
            key = month, before["class_id"]
            need(key in base and all(same_number(base[key][field], before[field]) if field == "class_mean_return"
                 else base[key][field] == before[field] for field in MONTH_FIELDS), "Counterfactual baseline monthly means differ from verified old RET means")
            shadow[key] = after
    output = []
    for signal in signal_windows.signal_date:
        day(signal); need(isinstance(baseline_vector_fingerprints[signal], str) and baseline_vector_fingerprints[signal], "Preserved baseline vector fingerprint required")
        months = [month_step(signal[:7], offset) for offset in range(-119, 1)]
        vectors = []
        for states in (base, shadow):
            grid = [states[(month, cls)] for month in months for cls in CLASS_IDS]
            need(all(row["outcome_end"] <= signal for row in grid), "Uncompleted outcome entered a past RET window")
            available = all(row["status"] == "available" for row in grid)
            if available:
                need(all(isinstance(row["member_count"], Integral) and row["member_count"] > 0 for row in grid), "Available payoff class is empty")
                for row in grid: finite_return(row["class_mean_return"])
            vectors.append(([math.fsum(states[(month, cls)]["class_mean_return"] / 120 for month in months)
                              for cls in CLASS_IDS] if available else [None] * 10, available))
        before, after = vectors
        for cls in CLASS_IDS:
            a, b = before[0][cls - 1], after[0][cls - 1]
            output.append({"signal_date": signal, "class_id": cls,
                "first_outcome_month": months[0], "last_outcome_month": months[-1],
                "baseline_vector_fingerprint": baseline_vector_fingerprints[signal],
                "baseline_payoff": a, "shadow_payoff": b,
                "payoff_delta": b - a if a is not None and b is not None else None,
                "baseline_vector_available": before[1], "shadow_vector_available": after[1],
                "patched_month_in_window": any(month in months for month in patched_months),
                "conditional_shadow_only": True, "production_promoted": False})
    return pd.DataFrame(output)


def verify_preserved_ret_states(base_monthly_means, saved_states):
    """Reproduce all old logical vector fingerprints before conditional changes.

    The caller separately authenticates the original three raw byte pins per
    signal. Reconstructed table/monthly values must reproduce the exact saved
    JSON fingerprint; no saved vector financial Parquet decode is necessary.
    """
    fingerprints = {signal: state['fingerprint'] for signal, state in saved_states.items()}
    signals = pd.DataFrame({'signal_date': sorted(saved_states)})
    vectors = propagate_ret_payoff_impact(base_monthly_means, [], signals,
        baseline_vector_fingerprints=fingerprints)
    means = {(row['outcome_month'], row['class_id']): row
             for row in base_monthly_means[list(MONTH_FIELDS)].to_dict('records')}
    for signal, state in saved_states.items():
        rows = vectors[vectors.signal_date.eq(signal)]
        months = [month_step(signal[:7], offset) for offset in range(-119, 1)]
        monthly = [means[(month, cls)] for month in months for cls in CLASS_IDS]
        available = bool(rows.baseline_vector_available.all())
        table = [{'class_id': row.class_id, 'payoff': None if missing(row.baseline_payoff) else row.baseline_payoff,
            'status': 'available' if available else 'withheld_incomplete_120_month_class_grid'}
            for row in rows.itertuples()]
        provenance = state['provenance']
        need(provenance['signal_date'] == provenance['label_observed_through'] == signal
             and provenance['first_outcome_month'] == months[0]
             and provenance['last_outcome_month'] == months[-1]
             and provenance['vector_available'] is available,
             'Saved baseline RET provenance clock/availability changed')
        payload = {'table': table, 'monthly': monthly, 'provenance': provenance}
        digest = 'sha256:' + hashlib.sha256(json.dumps(payload, sort_keys=True,
            separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()
        need(digest == state['fingerprint'], 'Saved baseline RET vector logical fingerprint mismatch')
    return fingerprints


def annual_dependency_table(changed_outcome_months, tuning_protocol):
    """Date-only dependencies: potential inputs/metrics, never a refit decision."""
    months = sorted(set(changed_outcome_months))
    need(all("2010-01" <= month <= "2023-12" and month_step(month, 0) == month for month in months), "Bounded changed outcome months required")
    result = []
    for fold in tuning_protocol["chronology"]["folds"]:
        outer = fold["outer_outcome_year"]
        intervals = [(f"{outer}_outer_train", "supervised_training_labels", fold["train_outcome_start"], fold["train_outcome_end"]),
                     (f"{outer}_outer_evaluation", "evaluation_labels_only", f"{outer}-01-01", f"{outer}-12-31")]
        for split in fold["inner_splits"]:
            intervals.extend([(split["split_id"], "inner_training_labels", split["train_outcome_start"], split["train_outcome_end"]),
                (split["split_id"], "inner_validation_labels_only", split["validation_outcome_start"], split["validation_outcome_end"])])
        for name, role, first, last in intervals:
            for month in months:
                if first[:7] <= month <= last[:7]:
                    result.append({"outer_year": outer, "split_id": name, "dependency_role": role,
                        "changed_outcome_month": month, "interval_start": first, "interval_end": last,
                        "interpretation": "conditional_date_dependency_not_refit_instruction",
                        "fits_or_predictions_performed": False, "all_models_need_refit_claim": False})
    return pd.DataFrame(result)
