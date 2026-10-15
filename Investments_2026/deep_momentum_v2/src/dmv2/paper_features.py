"""Causal monthly Han–Qin momentum/state inputs, omitting capitalization.

This worker receives daily quotes and historical SIGNAL cohorts, never the
monthly outcome ledger. Exact exchange-month-end adjusted-price ratios are a
declared data adaptation to CRSP. Current source revisions/identity routes do
not certify knowledge-time PIT. All fifteen outputs remain withheld when the
original cohort cannot support that horizon's normalization.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
from collections.abc import Mapping
import numpy as np

import pandas as pd
import pyarrow as pa

from .catalog import canonical_sha256
from .data import session_dates as calendar_session_dates
from .identity_policy import (STRICT_IDENTITY_POLICY, identity_policy_provenance,
                              qualify_identity_records, validate_identity_policy)
from .monthly import ACCEPTED_PRICE_QUALITY_STATES, HOLDOUT_START, _step_month
from .universe import DataContractError, UniverseSnapshot, iso_date


HORIZONS = (1, 3, 6, 9, 12)
COMMON_MOMENTUM_POLICY = "signal_qualified_common_momentum_v1"
PAPER_FEATURE_NAMES = tuple(f"{prefix}{h}" for prefix in ("zMOM", "meanMOM", "stdMOM") for h in HORIZONS)
PRICE_FIELDS = ("intrinio_security_id", "trading_date", "frequency",
                "intraperiod", "price_quality_status", "adj_close")
COHORT_FIELDS = ("source_security_id", "as_of", "intrinio_security_id", "identity_status")
OPTIONAL_COHORT_FIELDS = ("unified_security_id", "indices", "membership_tickers",
                         "membership_records_json", "identity_review_required",
                         "resolution_status", "bridge_method", "bridge_confidence",
                         "source_identity_status")


@dataclass(frozen=True)
class PaperFeaturesResult:
    table: pd.DataFrame
    coverage: pd.DataFrame
    specs: tuple[dict, ...]
    provenance: dict



@dataclass(frozen=True)
class CommonMomentumNormalizationResult:
    table: pd.DataFrame
    coverage: pd.DataFrame
    provenance: dict


def normalize_common_momentum(raw_frame, *, exhaustive_requested_by_signal):
    """Normalize one common, past-input-qualified population at each signal.

    The caller supplies the ORIGINAL membership ledger, not only successful
    quotes, and attests that all admitted native IDs were requested per signal.
    False attestations withhold statistical outputs, allowing bounded engineering
    samples. This worker cannot independently prove the caller's source query.
    Outcome/target information is prohibited, even when claimed to be unused.
    """
    if not isinstance(raw_frame, pd.DataFrame) or raw_frame.columns.duplicated().any():
        raise DataContractError("Raw momentum input must be a DataFrame with unique columns")
    required = {"source_security_id", "signal_date", "identity_eligible",
                *(f"MOM{h}_raw" for h in HORIZONS)}
    missing = required - set(raw_frame)
    if missing:
        raise DataContractError(f"Raw momentum input missing columns: {sorted(missing)}")
    forbidden = [name for name in raw_frame if str(name).lower().startswith(
        ('target', 'label', 'outcome', 'future', 'next_', 'forward', 'realized', 'prediction'))
        or name in ('provider_adjusted_return', 'return_status', 'y', 'last_price_date', 'delisted')]
    if forbidden:
        raise DataContractError(f"Raw normalization input contains prohibited target/outcome fields: {forbidden}")
    if not raw_frame.index.is_unique or raw_frame[["source_security_id", "signal_date"]].isna().any().any() or raw_frame.duplicated(
            ["source_security_id", "signal_date"]).any():
        raise DataContractError("Original momentum ledger requires unique nonmissing constituent/signal keys")
    for row in raw_frame[["source_security_id", "signal_date", "identity_eligible"]].to_dict('records'):
        _id(row['source_security_id'], 'source_security_id')
        if iso_date(row['signal_date'], 'signal_date') >= HOLDOUT_START:
            raise DataContractError("Protected 2024+ signal cannot enter normalization")
        if not isinstance(row['identity_eligible'], (bool, np.bool_)):
            raise DataContractError("identity_eligible must contain explicit booleans")
    signals = set(raw_frame.signal_date)
    if not isinstance(exhaustive_requested_by_signal, Mapping) or set(exhaustive_requested_by_signal) != signals:
        raise DataContractError("Exhaustive-request attestations must match every signal exactly")
    if any(not isinstance(v, (bool, np.bool_)) for v in exhaustive_requested_by_signal.values()):
        raise DataContractError("Exhaustive-request attestations must be explicit booleans")
    frame = raw_frame.copy(deep=True)
    for name in PAPER_FEATURE_NAMES:
        frame[name] = np.nan
    frame['paper_prediction_eligible'] = False
    frame['paper_normalization_status'] = pd.Series(index=frame.index, dtype='str')
    coverage = []
    for signal, group in frame.groupby('signal_date', sort=True):
        numeric = pd.DataFrame(index=group.index)
        for h in HORIZONS:
            original = group[f'MOM{h}_raw']
            numeric[h] = pd.to_numeric(original, errors='coerce').astype(float)
            # bool is not a measured return; non-ok statuses cannot be overruled
            # by a finite stale value left in a diagnostic column.
            good = np.isfinite(numeric[h]) & ~original.map(lambda v: isinstance(v, (bool, np.bool_)))
            if f'MOM{h}_raw_status' in group:
                good &= group[f'MOM{h}_raw_status'].eq('ok')
            numeric.loc[~good, h] = np.nan
        admitted = group.identity_eligible.astype(bool)
        common = admitted & numeric.notna().all(axis=1)
        count = int(common.sum())
        means, sds = {}, {}
        if not bool(exhaustive_requested_by_signal[signal]):
            state = 'incomplete_request'
        elif count < 2:
            state = 'insufficient_common_population'
        else:
            state = 'qualified_common_population'
            try:
                for h in HORIZONS:
                    vals = numeric.loc[common, h].tolist()
                    means[h] = math.fsum(vals) / count
                    sds[h] = math.sqrt(math.fsum((v - means[h]) ** 2 for v in vals) / (count - 1))
                if not all(math.isfinite(v) for v in (*means.values(), *sds.values())):
                    state = 'nonfinite_dispersion'
                elif any(v <= 0 for v in sds.values()):
                    state = 'degenerate_dispersion'
            except (OverflowError, ValueError):
                state = 'nonfinite_dispersion'
        frame.loc[group.index, 'paper_normalization_status'] = state
        if state == 'qualified_common_population':
            selected = group.index[common]
            for h in HORIZONS:
                frame.loc[selected, f'zMOM{h}'] = (numeric.loc[selected, h] - means[h]) / sds[h]
                frame.loc[selected, f'meanMOM{h}'] = means[h]
                frame.loc[selected, f'stdMOM{h}'] = sds[h]
            # A nonfinite standardized value invalidates ALL five baselines.
            if not np.isfinite(frame.loc[selected, list(PAPER_FEATURE_NAMES)].to_numpy(dtype=float)).all():
                frame.loc[group.index, list(PAPER_FEATURE_NAMES)] = np.nan
                frame.loc[group.index, 'paper_normalization_status'] = 'nonfinite_standardization'
                state = 'nonfinite_standardization'
            else:
                frame.loc[selected, 'paper_prediction_eligible'] = True
        for h in HORIZONS:
            frame.loc[group.index, f'MOM{h}_normalization_status'] = state
            coverage.append({'signal_date': signal, 'horizon_months': h,
                'original_cohort_count': len(group), 'identity_eligible_count': int(admitted.sum()),
                'raw_momentum_available_count': int((admitted & numeric[h].notna()).sum()),
                'common_momentum_population_count': count,
                'normalized_member_count': count if state == 'qualified_common_population' else 0,
                'exhaustive_request': bool(exhaustive_requested_by_signal[signal]),
                'normalization_status': state, 'normalization_policy': COMMON_MOMENTUM_POLICY,
                'standard_deviation_ddof': 1, 'knowledge_time_verified': False,
                'universe': group.universe.iloc[0] if 'universe' in group and len(group) else None})
    provenance = {'normalization_policy': COMMON_MOMENTUM_POLICY,
        'standard_deviation_ddof': 1, 'common_five_momentum_population': True,
        'exhaustive_requested_by_signal': {s: bool(exhaustive_requested_by_signal[s]) for s in sorted(signals)},
        'outcomes_consumed': False, 'original_rows_preserved': True,
        'shared_state_levels_preserved_through_time': True,
        'cohort_selection': 'original dated membership, admitted identity, five finite exact past-input momenta; no future survivor intersection',
        'minimum_population_count': 2, 'minimum_coverage_fraction': None,
        'adequacy_claim': 'mathematical computability only; not representativeness or complete economic coverage',
        'knowledge_time_verified': False, 'accounting_certified': False,
        'model_ready': False, 'identity_uncertainty_accepted_for_engineering': True}
    return CommonMomentumNormalizationResult(frame, pd.DataFrame(coverage), provenance)


def paper_feature_specs(normalization_policy="complete_cohort") -> tuple[dict, ...]:
    """The exact fifteen registered model inputs; raw momenta are diagnostics.

    ``ddof=1`` and complete-cohort withholding are proposed V2 conventions;
    neither is attributed to the paper, which leaves them unspecified.
    """
    specs = []
    for prefix in ("zMOM", "meanMOM", "stdMOM"):
        for horizon in HORIZONS:
            momentum = ("adj_close[t]/adj_close[t-1]-1" if horizon == 1 else
                        f"product(1+r[j], j=t-{horizon}+1..t-1)-1; "
                        f"{horizon - 1} exact calendar-month returns")
            formula = {
                "zMOM": f"(MOM{horizon}-cohort_mean(MOM{horizon}))/cohort_SD(MOM{horizon}, ddof=1)",
                "meanMOM": f"cohort_mean(MOM{horizon}) shared at signal t",
                "stdMOM": f"cohort_SD(MOM{horizon}, ddof=1) shared at signal t",
            }[prefix]
            specs.append({
                "name": f"{prefix}{horizon}", "family": "paper_momentum_state", "role": "paper",
                "required_inputs": ("adj_close", "historical_signal_cohort", "exchange_month_ends"),
                "lookback_months": horizon, "formula": formula,
                "raw_momentum_formula": momentum,
                "normalization_policy": normalization_policy, "standard_deviation_ddof": 1,
                "signal_cutoff": "close of the exact signal exchange session; no t+1 inputs",
                "evidence": "paper equations 6–8; ddof and declared population gates are V2 adaptations",
            })
    return tuple(specs)


def _id(value, label):
    if not isinstance(value, str) or not value.strip():
        raise DataContractError(f"Missing or invalid {label}")
    return value


def _columns(table, required, label):
    names = list(table.column_names) if hasattr(table, "column_names") else list(table.columns)
    if len(names) != len(set(names)):
        raise DataContractError(f"{label} has duplicate column names")
    if label == 'daily_prices':
        forbidden = [name for name in names if str(name).lower().startswith(
            ('target', 'label', 'outcome', 'future', 'next_', 'forward', 'realized', 'prediction'))
            or name in ('provider_adjusted_return', 'return_status', 'daily_backtest_ready', 'delisted', 'last_price_date')]
        if forbidden:
            raise DataContractError(f'Daily prices contain prohibited target/outcome fields: {forbidden}')
    missing = set(required) - set(names)
    if missing:
        raise DataContractError(f"{label} missing columns: {', '.join(sorted(missing))}")
    return names


def _records(table, columns):
    if hasattr(table, "select") and hasattr(table, "column_names"):
        return table.select(list(columns)).to_pylist()
    if isinstance(table, pd.DataFrame):
        return table.loc[:, list(columns)].to_dict("records")
    raise DataContractError("Prices/cohorts must be an Arrow table or pandas DataFrame")


def _sessions(value):
    if isinstance(value, pa.Table):
        _columns(value, ("trading_date",), "session calendar")
        raw = value.select(["trading_date"]).to_pylist()
        if len(raw) != len({row["trading_date"] for row in raw}):
            raise DataContractError("Duplicate calendar dates")
        dates = tuple(calendar_session_dates(value))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        dates = tuple(iso_date(day, "session_date") for day in value)
    else:
        raise DataContractError("session_dates must be an explicit sequence of ISO exchange sessions")
    if not dates or tuple(sorted(set(dates))) != dates:
        raise DataContractError("Exchange sessions must be nonempty, strictly ordered and unique")
    return dates


def _cohorts(snapshots, month_ends, identity_policy):
    if not isinstance(snapshots, Sequence) or isinstance(snapshots, (str, bytes)) or not snapshots:
        raise DataContractError("signal_cohorts must be a nonempty explicit sequence of UniverseSnapshot records")
    cohorts, provenance, universes = {}, {}, set()
    for snapshot in snapshots:
        if not isinstance(snapshot, UniverseSnapshot):
            raise DataContractError("signal_cohorts must contain UniverseSnapshot records")
        cutoff = iso_date(snapshot.as_of, "snapshot as_of")
        if cutoff >= HOLDOUT_START:
            raise DataContractError("Protected 2024+ signal cohorts cannot enter feature arithmetic")
        if cutoff in cohorts:
            raise DataContractError("Duplicate signal cohort date")
        if month_ends.get(cutoff[:7]) != cutoff:
            raise DataContractError("Signal date must be an exact exchange month end with adjacent-month calendar evidence")
        if snapshot.universe not in ("sp500", "sp400", "combined"):
            raise DataContractError("Unknown signal-cohort universe")
        if snapshot.provenance.get("mode", "historical_reconstruction") != "historical_reconstruction":
            raise DataContractError("Only disclosed historical_reconstruction cohorts are supported")
        names = _columns(snapshot.table, COHORT_FIELDS, "signal cohort")
        columns = (*COHORT_FIELDS, *(name for name in OPTIONAL_COHORT_FIELDS if name in names))
        records, seen = _records(snapshot.table, columns), set()
        for row in records:
            sid = _id(row["source_security_id"], "source_security_id")
            if sid in seen:
                raise DataContractError("Duplicate signal-cohort source_security_id")
            seen.add(sid)
            if iso_date(row["as_of"], "cohort row as_of") != cutoff:
                raise DataContractError("Cohort row as_of differs from signal cutoff")
            status = _id(row["identity_status"], "identity_status")
            target = row["intrinio_security_id"]
            if target not in (None, ""):
                _id(target, "intrinio_security_id")
            elif status == "resolved":
                raise DataContractError("Resolved identity has no provider ID")
        # Recompute research admission over the full original cohort. A
        # provisional flag exception does not relabel the producer's evidence.
        records = qualify_identity_records(records, identity_policy=identity_policy)
        cohorts[cutoff] = sorted(records, key=lambda row: row["source_security_id"])
        provenance[cutoff] = snapshot.provenance
        universes.add(snapshot.universe)
    if len(universes) != 1:
        raise DataContractError("Signal cohorts must share one explicit universe")
    return cohorts, provenance, next(iter(universes))


def _quote_value(row):
    if row is None:
        return None, "missing_exact_endpoint"
    if row["frequency"] != "daily" or row["intraperiod"] is not False or row["price_quality_status"] not in ACCEPTED_PRICE_QUALITY_STATES:
        return None, "source_quality_flag"
    value = row["adj_close"]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None, "invalid_endpoint"
    return float(value), "ok"


def _raw_momentum(row, cutoff, horizon, month_ends, quotes, requested):
    if not row["identity_eligible"]:
        return None, "identity_unresolved"
    target = row["intrinio_security_id"]
    if target not in requested:
        return None, "not_requested"
    signal_month = cutoff[:7]
    months = ([_step_month(signal_month, -1), signal_month] if horizon == 1 else
              [_step_month(signal_month, offset) for offset in range(-horizon, 0)])
    if any(month not in month_ends for month in months):
        return None, "insufficient_calendar_warmup"
    values = []
    for month in months:
        value, status = _quote_value(quotes.get((target, month_ends[month])))
        if status != "ok":
            return None, status
        values.append(value)
    # Check EVERY monthly endpoint. Telescoping alone would hide a missing
    # intermediate month and silently bridge an invalid monthly return.
    gross = []
    for start, end in zip(values, values[1:]):
        ratio = end / start
        if not math.isfinite(ratio) or ratio <= 0:
            return None, "nonfinite_momentum"
        gross.append(ratio)
    product = math.prod(gross)
    result = product - 1
    if not math.isfinite(product) or product <= 0 or not math.isfinite(result):
        return None, "nonfinite_momentum"
    return float(result), "ok"


def compute_paper_features(daily_prices, session_dates, *, signal_cohorts,
                           requested_security_ids, normalization_policy="complete_cohort",
                           identity_policy=STRICT_IDENTITY_POLICY) -> PaperFeaturesResult:
    """Build signal-only features, preserving every original constituent.

    Complete-cohort normalization is horizon-specific and fail-closed. A
    partial requested-ID engineering sample can show raw momentum diagnostics,
    but cannot masquerade as the complete market-state cross-section. Neither
    next-month membership nor outcomes are consulted. Missing month-end quotes
    are never filled, stale quotes substituted, or missing months compressed.
    """
    validate_identity_policy(identity_policy)
    if normalization_policy not in ("complete_cohort", COMMON_MOMENTUM_POLICY):
        raise DataContractError("Unknown paper normalization policy")
    sessions = _sessions(session_dates)
    grouped = {}
    for day in sessions:
        grouped.setdefault(day[:7], []).append(day)
    month_ends = {month: days[-1] for month, days in grouped.items()
                  if _step_month(month, 1) in grouped}
    cohorts, cohort_provenance, universe = _cohorts(signal_cohorts, month_ends, identity_policy)
    max_signal = max(cohorts)
    if not isinstance(requested_security_ids, Sequence) or isinstance(requested_security_ids, (str, bytes)):
        raise DataContractError("requested_security_ids must be an explicit sequence")
    requested = tuple(_id(value, "requested_security_id") for value in requested_security_ids)
    if len(requested) != len(set(requested)):
        raise DataContractError("Duplicate requested provider security ID")
    requested_set = set(requested)
    _columns(daily_prices, PRICE_FIELDS, "daily_prices")
    # Metadata is checked for the ENTIRE request before financial conversion.
    metadata = _records(daily_prices, PRICE_FIELDS[:-1])
    seen, session_set = set(), set(sessions)
    for row in metadata:
        target = _id(row["intrinio_security_id"], "price intrinio_security_id")
        day = iso_date(row["trading_date"], "price trading_date")
        if day >= HOLDOUT_START:
            raise DataContractError("Protected 2024+ financial rows cannot enter feature arithmetic")
        if day > max_signal:
            raise DataContractError("Price date exceeds latest signal cutoff; pass separate signal-only daily history")
        if day not in session_set:
            raise DataContractError("Price date is not a supplied exchange session")
        if target not in requested_set:
            raise DataContractError("Price ID falls outside the declared request")
        if (target, day) in seen:
            raise DataContractError("Duplicate daily price security/date key")
        seen.add((target, day))
    values = _records(daily_prices, ("adj_close",))
    quotes = {(row["intrinio_security_id"], row["trading_date"]): {**row, **value}
              for row, value in zip(metadata, values, strict=True)}
    output, coverage = [], []
    for cutoff, records in sorted(cohorts.items()):
        rows = [{**row, "signal_date": cutoff, "universe": universe,
                 "pit_mode": "historical_reconstruction", "knowledge_time_verified": False,
                 "accounting_certified": False} for row in records]
        for horizon in HORIZONS:
            raw = []
            statuses = []
            for row in rows:
                value, status = _raw_momentum(row, cutoff, horizon, month_ends, quotes, requested_set)
                row[f"MOM{horizon}_raw"], row[f"MOM{horizon}_raw_status"] = value, status
                statuses.append(status)
                if status == "ok":
                    raw.append(value)
            mean, sd = None, None
            if not records:
                state = "empty_cohort"
            elif len(raw) != len(records):
                state = "incomplete_cohort"
            elif len(raw) < 2:
                state = "insufficient_cohort_size"
            else:
                try:
                    mean = math.fsum(raw) / len(raw)
                    sd = math.sqrt(math.fsum((value - mean) ** 2 for value in raw) / (len(raw) - 1))
                except (OverflowError, ValueError):
                    mean, sd = None, None
                if mean is None or sd is None or not math.isfinite(mean) or not math.isfinite(sd):
                    state = "nonfinite_dispersion"
                    mean, sd = None, None
                elif sd <= 0:
                    state = "degenerate_dispersion"
                    mean, sd = None, None
                else:
                    state = "complete_cohort"
            for row in rows:
                row[f"MOM{horizon}_normalization_status"] = state
                row[f"zMOM{horizon}"] = ((row[f"MOM{horizon}_raw"] - mean) / sd
                                          if state == "complete_cohort" else None)
                row[f"meanMOM{horizon}"], row[f"stdMOM{horizon}"] = mean, sd
            coverage.append({
                "signal_date": cutoff, "universe": universe, "horizon_months": horizon,
                "original_cohort_count": len(records), "raw_momentum_available_count": len(raw),
                "identity_unresolved_count": statuses.count("identity_unresolved"),
                "identity_eligible_count": sum(row["identity_eligible"] for row in rows),
                "identity_eligibility_policy": identity_policy,
                "provisional_review_flags_accepted": sum(
                    row["identity_eligibility_reason"] == "provisional_review_flag_accepted" for row in rows),
                "not_requested_count": statuses.count("not_requested"),
                "normalization_status": state, "normalization_policy": normalization_policy,
                "standard_deviation_ddof": 1, "knowledge_time_verified": False,
            })
        output.extend(rows)
    specs = paper_feature_specs(normalization_policy)
    result_columns = ["source_security_id", "as_of", "intrinio_security_id", "identity_status",
                      "source_identity_status", "signal_date", "universe", "pit_mode",
                      "identity_review_required", "resolution_status", "identity_eligible",
                      "identity_eligibility_policy", "identity_eligibility_reason",
                      "knowledge_time_verified", "accounting_certified",
                      *[spec["name"] for spec in specs],
                      *[name for horizon in HORIZONS for name in
                        (f"MOM{horizon}_raw", f"MOM{horizon}_raw_status", f"MOM{horizon}_normalization_status")]]
    table = pd.DataFrame(output) if output else pd.DataFrame(columns=result_columns)
    common = None
    if normalization_policy == COMMON_MOMENTUM_POLICY:
        exhaustive = {cutoff: all(row['intrinio_security_id'] in requested_set
                                 for row in records if row['identity_eligible'])
                      for cutoff, records in cohorts.items()}
        # Empty original cohorts preserve their declared signal for the legacy
        # empty-ledger case; there is no cross-section to normalize.
        if not table.empty:
            common = normalize_common_momentum(table, exhaustive_requested_by_signal=exhaustive)
            table, coverage = common.table, common.coverage.to_dict('records')
        else:
            coverage = [{**item, 'normalization_policy': COMMON_MOMENTUM_POLICY,
                         'normalization_status': 'insufficient_common_population',
                         'common_momentum_population_count': 0, 'normalized_member_count': 0,
                         'exhaustive_request': True} for item in coverage]
            table['paper_prediction_eligible'] = pd.Series(dtype=bool)
            table['paper_normalization_status'] = pd.Series(dtype='str')
    provenance = {
        "mode": "historical_reconstruction", "knowledge_time_verified": False,
        **identity_policy_provenance(identity_policy),
        "identity_admission": "Declared research eligibility; producer flags/statuses retained; no historical identity certification",
        "accounting_certified": False, "max_signal_date": max_signal,
        "feature_columns": [spec["name"] for spec in specs], "feature_count": 15,
        "raw_momentum_columns_are_diagnostics": True,
        "normalization_policy": normalization_policy, "standard_deviation_ddof": 1,
        "cohort_policy": "original signal-date historical cohort; no outcome availability or future membership admission",
        "monthly_prices": "exact supplied exchange month ends; no stale endpoints, missing-month compression or fill",
        "adjustment_semantics": "provider-adjusted monthly ratios; no extra dividend/split application; not CRSP total-return certification",
        "incomplete_or_degenerate_cross_section": "withhold all three horizon-specific normalized/mean/SD inputs",
        "source_warmup": "pre-entry past prices permitted; calendar months retain missing endpoints",
        "proposed_v2_conventions": ["sample SD ddof=1", "complete original signal-cohort normalization", "zero dispersion withheld"],
        "daily_financial_last_date": max((row["trading_date"] for row in metadata), default=None),
        "calendar_sessions_sha256": canonical_sha256(list(sessions)),
        "cohort_sha256": canonical_sha256(cohorts), "snapshot_provenance": cohort_provenance,
        "requested_security_ids": sorted(requested), "feature_specs_sha256": canonical_sha256(specs),
    }
    if common is not None:
        provenance.update(common.provenance)
        provenance["incomplete_or_degenerate_cross_section"] = "withhold all15; only common five-momentum rows receive values"
        provenance["proposed_v2_conventions"] = ["sample SD ddof=1", "signal-specific common five-momentum population", "all-five positive dispersion", "exhaustive admitted-ID request"]
    return PaperFeaturesResult(table, pd.DataFrame(coverage), specs, provenance)
