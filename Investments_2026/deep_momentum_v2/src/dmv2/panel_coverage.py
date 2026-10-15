"""Streaming descriptive coverage for a fixed, signal-only feature dictionary.

This module receives completed feature panels, never prices or outcomes.  It
keeps original historical denominators, reports baseline-qualified populations
separately, and fits no preprocessing, threshold or feature-selection rule.
"""
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow as pa

from .catalog import canonical_sha256
from .features import KEYS, registered_feature_specs
from .paper_features import COMMON_MOMENTUM_POLICY, PAPER_FEATURE_NAMES
from .universe import DataContractError, iso_date


MISSING_REASONS = (
    "identity_blocked", "not_requested", "insufficient_calendar_warmup",
    "missing_exact_session", "invalid_quote_metadata", "missing_or_invalid_input",
    "undefined_formula", "paper_population_or_inputs", "context_incomplete_or_undefined",
    "unclassified",
)
_DAILY_REASONS = {
    "insufficient_calendar_warmup": "insufficient_calendar_warmup",
    "missing_session": "missing_exact_session",
    "invalid_quote_metadata": "invalid_quote_metadata",
    "missing_or_invalid_input": "missing_or_invalid_input",
    "undefined_formula": "undefined_formula",
}


def _frame(value, label):
    if isinstance(value, pd.DataFrame):
        result = value.copy(deep=False)
    elif isinstance(value, pa.Table):
        result = value.to_pandas()
    else:
        raise DataContractError(f"{label} requires an Arrow table or DataFrame")
    if result.columns.duplicated().any():
        raise DataContractError(f"{label} contains duplicate columns")
    return result


def _keys(frame, label):
    if not set(KEYS) <= set(frame) or frame[list(KEYS)].isna().any().any():
        raise DataContractError(f"{label} requires nonmissing original source/signal keys")
    if frame.duplicated(list(KEYS)).any():
        raise DataContractError(f"{label} contains duplicate original source/signal keys")
    if not frame.source_security_id.map(lambda v: isinstance(v, str) and bool(v.strip())).all():
        raise DataContractError(f"{label} contains invalid original source IDs")
    for signal in frame.signal_date.unique():
        if iso_date(signal, "signal_date") >= "2024-01-01":
            raise DataContractError("Protected 2024+ signals cannot enter panel coverage")
    return set(frame[list(KEYS)].itertuples(index=False, name=None))


def _booleans(frame, name):
    if name not in frame or not frame[name].map(lambda v: isinstance(v, (bool, np.bool_))).all():
        raise DataContractError(f"Feature coverage requires explicit {name} booleans")
    return frame[name].to_numpy(dtype=bool)


def _outcome_year(signal):
    # The preceding December signal belongs to January's training/test outcome.
    return (pd.Period(signal[:7], freq="M") + 1).year


class StreamingPanelCoverage:
    """Accumulate bounded panels without keeping their feature values in memory.

    ``original_membership`` must contain every panel key and an explicit
    ``indices`` list; it must come from original dated membership before prices.
    A year is the NEXT-month outcome year, never the signal's calendar year.
    ``finish`` requires every declared signal unless explicitly told to produce
    a partial checkpoint.  No report admits, drops, fills or selects a predictor.
    """

    def __init__(self, specs, expected_signals, *, start_year=2010, end_year=2023):
        self.specs = tuple(specs)
        frozen = registered_feature_specs()
        expected_names = {s["name"] for s in frozen}
        names = [s["name"] for s in self.specs]
        if len(names) != 728 or len(set(names)) != 728 or set(names) != expected_names:
            raise DataContractError("Coverage requires the fixed 728-definition feature dictionary")
        frozen_roles = {s["name"]: (s["role"], s["family"], bool(s.get("model_eligible", True))) for s in frozen}
        for spec in self.specs:
            if (spec["role"], spec["family"], bool(spec.get("model_eligible", True))) != frozen_roles[spec["name"]]:
                raise DataContractError("Coverage dictionary changed a frozen role/family/predictor allowance")
            original = next(s for s in frozen if s["name"] == spec["name"])
            declared, locked = dict(spec), dict(original)
            if spec["role"] == "paper":
                if declared.get("normalization_policy") not in ("complete_cohort", COMMON_MOMENTUM_POLICY):
                    raise DataContractError("Coverage dictionary has an unsupported paper population policy")
                declared.pop("normalization_policy", None)
                locked.pop("normalization_policy", None)
            if canonical_sha256(declared) != canonical_sha256(locked):
                raise DataContractError("Coverage dictionary changed a frozen feature definition")
        self.names = names
        self.dictionary_sha256 = canonical_sha256(self.specs)
        if not isinstance(start_year, int) or not isinstance(end_year, int) or not 2010 <= start_year <= end_year <= 2023:
            raise DataContractError("Full-panel coverage is confined to 2010–2023 outcome years")
        self.start_year, self.end_year = start_year, end_year
        if isinstance(expected_signals, (str, bytes)):
            raise DataContractError("Expected signals require an explicit sequence")
        signals = tuple(expected_signals)
        if not signals or tuple(sorted(set(signals))) != signals:
            raise DataContractError("Expected signals must be unique, nonempty and chronological")
        for signal in signals:
            iso_date(signal, "expected signal")
            if not start_year <= _outcome_year(signal) <= end_year:
                raise DataContractError("Expected signal is outside the declared outcome years")
        self.expected_signals = signals
        self._seen = {}
        self._totals = {}
        self._cohorts = []
        self._worker_states = Counter()

    def add_signal(self, panel, *, original_membership):
        """Compatibility wrapper for a one-signal panel; use add_panel for years."""
        return self.add_panel(panel, original_membership=original_membership)

    def add_panel(self, panel, *, original_membership):
        if canonical_sha256(tuple(panel.specs)) != self.dictionary_sha256:
            raise DataContractError("Feature dictionary changed across panel batches")
        if panel.provenance.get("outcomes_consumed") is not False:
            raise DataContractError("Coverage requires explicit outcome-free feature provenance")
        table = _frame(panel.table, "Feature table")
        if set(table) != set((*KEYS, *self.names)):
            raise DataContractError("Feature table contains missing/unregistered or target/audit columns")
        keys = _keys(table, "Feature table")
        ledger = _frame(panel.coverage, "Feature coverage")
        if _keys(ledger, "Feature coverage") != keys:
            raise DataContractError("Feature coverage changed the original historical denominator")
        members = _frame(original_membership, "Original membership")
        if any(str(name).lower().startswith(("target", "outcome", "future", "next_", "forward", "prediction", "class_label"))
               or name in ("provider_adjusted_return", "return_status", "label", "y") for name in members):
            raise DataContractError("Original membership contains prohibited outcome/target fields")
        if _keys(members, "Original membership") != keys:
            raise DataContractError("Feature table changed the original historical denominator")
        if "indices" not in members:
            raise DataContractError("Original membership needs explicit historical index sleeves")
        for indices in members.indices:
            if not isinstance(indices, (list, tuple, np.ndarray)) or not len(indices) or not set(indices) <= {"sp500", "sp400"}:
                raise DataContractError("Original membership has invalid historical index sleeves")
        signals = sorted(table.signal_date.unique())
        if not set(signals) <= set(self.expected_signals) or set(signals) & set(self._seen):
            raise DataContractError("Unexpected or repeated signal in streaming coverage")
        # Check all batches before mutating accumulated counts.
        for name in self.names:
            if not pd.api.types.is_numeric_dtype(table[name]) or pd.api.types.is_bool_dtype(table[name]):
                raise DataContractError(f"Feature {name} must contain numeric observations or nulls")
        ledger = ledger.set_index(list(KEYS)).reindex(table.set_index(list(KEYS)).index).reset_index()
        members = members.set_index(list(KEYS)).reindex(table.set_index(list(KEYS)).index).reset_index()
        identity = _booleans(ledger, "identity_eligible")
        requested = _booleans(ledger, "requested")
        values = table[self.names].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise DataContractError("Feature table must convert nonfinite infinities to explicit nulls")
        finite = np.isfinite(values)
        paper_positions = [self.names.index(n) for n in PAPER_FEATURE_NAMES]
        baseline = finite[:, paper_positions].all(axis=1)
        if any(s["role"] == "paper" and s.get("normalization_policy") == COMMON_MOMENTUM_POLICY for s in self.specs):
            paper_finite = finite[:, paper_positions].sum(axis=1)
            if ((paper_finite != 0) & (paper_finite != len(PAPER_FEATURE_NAMES))).any():
                raise DataContractError("Common-five paper policy must supply all fifteen inputs or withhold all fifteen")
        if "paper_prediction_eligible" in ledger and not np.array_equal(
                _booleans(ledger, "paper_prediction_eligible"), baseline):
            raise DataContractError("Paper eligibility differs from all-fifteen finite baseline inputs")
        if (baseline & (~identity | ~requested)).any():
            raise DataContractError("Paper-qualified rows lack admitted requested identities")
        # Validate clocks before using optional diagnostics; aggregated context
        # states cannot establish an exact missing cause for an individual stock.
        diagnostics = getattr(panel, "diagnostics", {})
        worker_counts = Counter()
        daily_status = {}
        wide_status = None
        status_legend = diagnostics.get("daily_status_legend", {})
        if diagnostics.get("daily_status_codes") is not None:
            wide = _frame(diagnostics["daily_status_codes"], "Compact daily diagnostics")
            technical = [s["name"] for s in self.specs if s["role"] == "technical"]
            wide_keys = ["intrinio_security_id", "signal_date"]
            if set(wide) != set((*wide_keys, *technical)) or wide.duplicated(wide_keys).any():
                raise DataContractError("Compact daily diagnostics require unique native-ID/signal keys and all technical definitions")
            if not set(wide.signal_date) <= set(signals):
                raise DataContractError("Compact daily diagnostics contain undeclared signals")
            if not isinstance(status_legend, dict) or not status_legend:
                raise DataContractError("Compact daily diagnostic codes require a nonempty legend")
            try:
                legend = {int(k): v for k, v in status_legend.items()}
            except (TypeError, ValueError) as error:
                raise DataContractError("Invalid compact daily diagnostic legend") from error
            if len(legend) != len(status_legend) or any(not isinstance(v, str) or not v for v in legend.values()):
                raise DataContractError("Invalid compact daily diagnostic legend")
            code_values = wide[technical].to_numpy()
            if not np.issubdtype(code_values.dtype, np.integer) or not set(np.unique(code_values)) <= set(legend):
                raise DataContractError("Unknown or nonintegral compact daily diagnostic code")
            if "intrinio_security_id" not in ledger:
                raise DataContractError("Compact daily diagnostic attachment needs the separate provider route")
            wide_status = ledger[wide_keys].merge(wide, on=wide_keys, how="left", validate="many_to_one")
            attached_status = wide_status[technical]
            admitted_requested = identity & requested
            if attached_status.loc[admitted_requested].isna().any().any():
                raise DataContractError("Admitted requested rows lack compact daily feature diagnostic codes")
            ok_codes = [code for code, state in legend.items() if state == "ok"]
            expected_finite = attached_status.isin(ok_codes).to_numpy()
            actual_finite = finite[:, [self.names.index(n) for n in technical]]
            if not np.array_equal(expected_finite[admitted_requested], actual_finite[admitted_requested]):
                raise DataContractError("Technical values disagree with compact daily diagnostic status")
            status_legend = legend
        for worker in ("daily", "paper", "context"):
            raw = diagnostics.get(worker)
            if raw is None:
                continue
            frame = _frame(raw, f"{worker} diagnostics")
            if frame.empty:
                continue
            if "signal_date" not in frame or not set(frame.signal_date) <= set(signals):
                raise DataContractError("Worker diagnostics contain undeclared signals")
            for clock in ("latest_input_date",):
                if clock in frame:
                    selected = frame[clock].notna()
                    if (frame.loc[selected, clock] > frame.loc[selected, "signal_date"]).any():
                        raise DataContractError("Feature diagnostic input clock exceeds its signal")
            status_column = "normalization_status" if worker == "paper" else "status"
            if status_column in frame:
                for row in frame[["signal_date", status_column]].itertuples(index=False, name=None):
                    worker_counts[(worker, *row)] += 1
            if worker == "daily" and {"intrinio_security_id", "feature_name", "status"} <= set(frame):
                key_names = ["intrinio_security_id", "signal_date", "feature_name"]
                if frame.duplicated(key_names).any():
                    raise DataContractError("Duplicate daily feature diagnostic key")
                daily_status = dict(zip(frame[key_names].itertuples(index=False, name=None), frame.status))
        staged = []
        for signal in signals:
            loc = np.flatnonzero(table.signal_date.to_numpy() == signal)
            day_members = members.iloc[loc]
            day_ledger = ledger.iloc[loc]
            day_finite, day_baseline = finite[loc], baseline[loc]
            day_identity, day_requested = identity[loc], requested[loc]
            year = _outcome_year(signal)
            membership_masks = {"combined": np.ones(len(loc), dtype=bool),
                **{scope: day_members.indices.map(lambda ids: scope in ids).to_numpy()
                   for scope in ("sp500", "sp400")}}
            reasons = np.zeros(day_finite.shape, dtype=np.uint8)
            # Codes are valid only at missing locations. A worker-reported null
            # is never relabeled as absent underlying company history.
            for j, spec in enumerate(self.specs):
                why = np.full(len(loc), MISSING_REASONS.index("unclassified"), dtype=np.uint8)
                if spec["role"] == "technical" and wide_status is not None:
                    states = wide_status.iloc[loc][spec["name"]].map(status_legend)
                    why = np.asarray([MISSING_REASONS.index(_DAILY_REASONS.get(state, "unclassified")) for state in states], dtype=np.uint8)
                elif spec["role"] == "technical" and daily_status:
                    if "intrinio_security_id" not in day_ledger:
                        raise DataContractError("Daily diagnostic attachment needs the separate provider route")
                    states = [daily_status.get((pid, signal, spec["name"])) for pid in day_ledger.intrinio_security_id]
                    why = np.asarray([MISSING_REASONS.index(_DAILY_REASONS.get(state, "unclassified")) for state in states], dtype=np.uint8)
                elif spec["role"] == "paper":
                    why[:] = MISSING_REASONS.index("paper_population_or_inputs")
                elif spec["role"] in ("cross_sectional", "market_state", "relative_context"):
                    why[:] = MISSING_REASONS.index("context_incomplete_or_undefined")
                why[day_identity & ~day_requested] = MISSING_REASONS.index("not_requested")
                why[~day_identity] = MISSING_REASONS.index("identity_blocked")
                reasons[:, j] = why
            for scope, scope_mask in membership_masks.items():
                cohort = {"signal_date": signal, "outcome_year": year, "universe": scope,
                    "original_rows": int(scope_mask.sum()),
                    "identity_eligible_rows": int((scope_mask & day_identity).sum()),
                    "requested_admitted_rows": int((scope_mask & day_identity & day_requested).sum()),
                    "paper_qualified_rows": int((scope_mask & day_baseline).sum()),
                    "paper_unqualified_rows": int((scope_mask & ~day_baseline).sum())}
                staged.append(("cohort", cohort))
                for population, mask in (("original", scope_mask), ("paper_qualified", scope_mask & day_baseline)):
                    count = int(mask.sum())
                    valid = day_finite[mask].sum(axis=0).astype(np.int64)
                    missing = ~day_finite[mask]
                    counts = np.stack([((reasons[mask] == i) & missing).sum(axis=0)
                                       for i in range(len(MISSING_REASONS))], axis=1)
                    if not np.array_equal(valid + counts.sum(axis=1), np.repeat(count, len(self.names))):
                        raise DataContractError("Missing reasons do not partition the original feature-value denominator")
                    staged.append(("counts", (year, scope, population, count, valid, counts)))
            original_pairs = sorted(table.iloc[loc][list(KEYS)].itertuples(index=False, name=None))
            staged.append(("signal", (signal, canonical_sha256(original_pairs))))
        # Commit only after every shape/clock/denominator check succeeds.
        for kind, value in staged:
            if kind == "cohort":
                self._cohorts.append(value)
            elif kind == "signal":
                self._seen[value[0]] = value[1]
            else:
                year, scope, population, count, valid, counts = value
                key = (year, scope, population)
                total = self._totals.setdefault(key, [0, np.zeros(len(self.names), dtype=np.int64),
                    np.zeros((len(self.names), len(MISSING_REASONS)), dtype=np.int64)])
                total[0] += count
                total[1] += valid
                total[2] += counts
        self._worker_states.update(worker_counts)
        return self

    def tables(self):
        rows = []
        for (year, scope, population), (count, finite, missing) in sorted(self._totals.items()):
            for j, spec in enumerate(self.specs):
                rows.append({"outcome_year": year, "universe": scope, "population": population,
                    "feature_name": spec["name"], "role": spec["role"], "family": spec["family"],
                    "model_eligible": bool(spec.get("model_eligible", True)),
                    "rows": int(count), "finite_rows": int(finite[j]),
                    "missing_rows": int(count - finite[j]),
                    **{"missing_" + name: int(missing[j, k]) for k, name in enumerate(MISSING_REASONS)}})
        yearly = pd.DataFrame(rows)
        count_columns = ["rows", "finite_rows", "missing_rows", *("missing_" + n for n in MISSING_REASONS)]
        groups = ["universe", "population", "feature_name", "role", "family", "model_eligible"]
        periods = []
        for period in ("full_history", "initial_train_2010_2019", "nominal_test_2020_2023"):
            selected = yearly if period == "full_history" else yearly[yearly.outcome_year.between(
                2010 if period.startswith("initial") else 2020, 2019 if period.startswith("initial") else 2023)] if not yearly.empty else yearly
            if not selected.empty:
                part = selected.groupby(groups, as_index=False, sort=True)[count_columns].sum()
                part.insert(0, "period", period)
                periods.append(part)
        by_period = pd.concat(periods, ignore_index=True) if periods else pd.DataFrame()
        for frame in (yearly, by_period):
            if not frame.empty:
                frame["finite_fraction"] = frame.finite_rows / frame.rows.replace(0, np.nan)
                frame["missing_fraction"] = frame.missing_rows / frame.rows.replace(0, np.nan)
        family = pd.DataFrame()
        if not yearly.empty:
            family = yearly.groupby(["outcome_year", "universe", "population", "role", "family", "model_eligible"],
                as_index=False, sort=True)[count_columns].sum().rename(columns={"rows": "feature_value_slots",
                    "finite_rows": "finite_values", "missing_rows": "missing_values"})
            family["finite_fraction"] = family.finite_values / family.feature_value_slots.replace(0, np.nan)
        worker = pd.DataFrame([{"worker": w, "signal_date": s, "status": status, "diagnostic_records": count}
            for (w, s, status), count in sorted(self._worker_states.items())])
        cohorts = pd.DataFrame(self._cohorts)
        by_cohort_year = pd.DataFrame()
        if not cohorts.empty:
            cohort_counts = ["original_rows", "identity_eligible_rows", "requested_admitted_rows", "paper_qualified_rows", "paper_unqualified_rows"]
            by_cohort_year = cohorts.groupby(["outcome_year", "universe"], as_index=False, sort=True)[cohort_counts].sum()
            by_cohort_year["paper_qualified_fraction"] = by_cohort_year.paper_qualified_rows / by_cohort_year.original_rows.replace(0, np.nan)
        return {"by_signal": cohorts, "by_cohort_year": by_cohort_year, "by_feature_year": yearly,
                "by_feature_period": by_period, "by_family_year": family, "worker_status_counts": worker}

    def finish(self, *, require_complete=True):
        missing = sorted(set(self.expected_signals) - set(self._seen))
        if require_complete and missing:
            raise DataContractError(f"Feature coverage is incomplete: {len(missing)} declared signals absent")
        tables = self.tables()
        signals = tables["by_signal"]
        combined = signals[signals.universe == "combined"] if not signals.empty else signals
        original = int(combined.original_rows.sum()) if not combined.empty else 0
        baseline = int(combined.paper_qualified_rows.sum()) if not combined.empty else 0
        summary = {"schema_version": 1, "status": "FULL_FEATURE_COVERAGE" if not missing else "PARTIAL_FEATURE_COVERAGE_CHECKPOINT",
            "outcome_start_year": self.start_year, "outcome_end_year": self.end_year,
            "expected_signals": len(self.expected_signals), "completed_signals": len(self._seen),
            "missing_signals": missing, "original_stock_months": original,
            "paper_qualified_stock_months": baseline,
            "paper_qualified_fraction": baseline/original if original else None,
            "registered_features": len(self.names),
            "candidate_predictors": sum(bool(s.get("model_eligible", True)) for s in self.specs),
            "feature_dictionary_sha256": self.dictionary_sha256,
            "original_signal_key_sha256": canonical_sha256(dict(sorted(self._seen.items()))),
            "denominator": "Every original historical signal member; no future-outcome or surviving-ID selection",
            "period_basis": "Next-month outcome year; December signals belong to the following year's outcomes",
            "missing_reason_semantics": "Technical causes use exact per-security diagnostics when supplied. Context and paper causes are broad worker categories; aggregate statuses cannot establish per-stock causal attribution. Identity/request reasons have precedence only for missing values.",
            "worker_status_semantics": "Counts of diagnostic records, not stocks or missing feature values",
            "coverage_role": "Descriptive audit only; no full-history feature selection, preprocessing fit, imputation, label read or model",
            "fits_performed": 0, "outcomes_consumed": False, "daily_data_consumed": False,
            "knowledge_time_verified": False, "accounting_certified": False,
            "daily_backtest_ready": False, "model_ready": False}
        return {"summary": summary, "tables": tables}
