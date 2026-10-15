"""Descriptive, outcome-free comparison of pinned 90% and 80% context runs.

The caller verifies run manifests. This worker independently checks invariant
arrays, ledgers and preserved artifacts and never selects or fits predictors.
Annual files are streamed; raw daily financial values are only byte-hashed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .context_features import balanced_cross_sectional_bases
from .features import KEYS, registered_feature_specs
from .paper_features import PAPER_FEATURE_NAMES
from .universe import DataContractError


PERIODS = {
    "full_history_2010_2023": (2010, 2023),
    "before_2016_2010_2015": (2010, 2015),
    "later_2016_2023": (2016, 2023),
    "initial_train_2010_2019": (2010, 2019),
    "nominal_test_2020_2023": (2020, 2023),
}
PRESERVED_ARTIFACTS = (
    "daily_prices_raw", "daily_history_raw", "feature_history_daily",
    "daily_cohorts", "exchange_calendar", "original_cohort", "daily_diagnostics",
    "daily_status_codes_diagnostics", "paper_diagnostics",
    "paper_raw_momentum_diagnostics",
)
COMPOSITION_METRICS = (
    "px_simple_return_1s", "risk_realized_vol_20s", "liq_log_dollar_volume_20s",
)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _aligned(frame, label):
    if not set(KEYS) <= set(frame) or frame.duplicated(list(KEYS)).any():
        raise DataContractError(f"{label} requires unique original source/signal keys")
    if frame[list(KEYS)].isna().any().any():
        raise DataContractError(f"{label} contains missing original keys")
    if not frame.signal_date.map(lambda s: isinstance(s, str) and s < "2024-01-01").all():
        raise DataContractError(f"{label} contains protected or malformed signal dates")
    return frame.set_index(list(KEYS)).sort_index()


def _assert_equal(left, right, label):
    try:
        pd.testing.assert_frame_equal(left, right, check_exact=True)
    except AssertionError as error:
        raise DataContractError(f"{label} changed between context policies: {error}") from error


def compare_feature_frames(reference, sensitivity, ledger, specs, *, outcome_year):
    """Return per-feature original/qualified counts and exact shared-value checks.

    A gain means a formerly null context cell is now finite. It does not mean
    that an individual missing stock has been recovered or admitted for models.
    """
    reference = _aligned(reference, "Reference feature panel")
    sensitivity = _aligned(sensitivity, "Sensitivity feature panel")
    ledger = _aligned(ledger, "Reference coverage ledger")
    names = [s["name"] for s in specs]
    if set(reference) != set(names) or set(sensitivity) != set(names):
        raise DataContractError("Compared panels changed the registered feature columns")
    if not reference.index.equals(sensitivity.index) or not reference.index.equals(ledger.index):
        raise DataContractError("Compared panels changed original historical keys")
    expected_year = pd.PeriodIndex(reference.index.get_level_values("signal_date").str[:7], freq="M") + 1
    if not (expected_year.year == outcome_year).all() or not 2010 <= outcome_year <= 2023:
        raise DataContractError("Comparison year must be the next-month outcome year in 2010–2023")
    invariant = [s["name"] for s in specs if s["role"] in ("paper", "technical")]
    _assert_equal(reference[invariant], sensitivity[invariant], "Paper and individual technical features")
    if not ledger.paper_prediction_eligible.map(lambda x: isinstance(x, (bool, np.bool_))).all():
        raise DataContractError("Comparison needs explicit paper eligibility flags")
    qualified = ledger.paper_prediction_eligible.to_numpy(dtype=bool)
    if set(PAPER_FEATURE_NAMES) <= set(reference) and not np.array_equal(
            qualified, np.isfinite(reference[list(PAPER_FEATURE_NAMES)].to_numpy(dtype=float)).all(axis=1)):
        raise DataContractError("Paper eligibility changed from the complete paper15 population")
    records = []
    for spec in specs:
        if spec["role"] in ("paper", "technical"):
            continue
        name = spec["name"]
        a, b = reference[name].to_numpy(dtype=float), sensitivity[name].to_numpy(dtype=float)
        if np.isinf(a).any() or np.isinf(b).any():
            raise DataContractError("Context comparison requires nulls instead of infinite values")
        fa, fb = np.isfinite(a), np.isfinite(b)
        if (fa & ~fb).any():
            raise DataContractError(f"Lower context threshold lost finite values for {name}")
        shared = fa & fb
        if not np.array_equal(a[shared], b[shared]):
            raise DataContractError(f"Shared finite context values changed for {name}")
        if not spec.get("model_eligible", True):
            continue
        for population, mask in (("original", np.ones(len(a), dtype=bool)), ("paper_qualified", qualified)):
            shared_mask, gained = shared & mask, ~fa & fb & mask
            differences = np.abs(b[shared_mask] - a[shared_mask])
            new_values = b[gained]
            records.append({
                "outcome_year": outcome_year, "population": population,
                "feature_name": name, "role": spec["role"], "family": spec["family"],
                "rows": int(mask.sum()), "reference_finite_rows": int((fa & mask).sum()),
                "sensitivity_finite_rows": int((fb & mask).sum()),
                "shared_finite_rows": int(shared_mask.sum()), "gained_finite_rows": int(gained.sum()),
                "lost_finite_rows": 0, "changed_shared_values": int((differences != 0).sum()),
                "shared_abs_difference_sum": float(differences.sum()),
                "shared_abs_difference_max": float(differences.max()) if len(differences) else np.nan,
                "gained_value_sum": float(new_values.sum()),
                "gained_value_sum_squared": float(np.square(new_values).sum()),
                "gained_value_min": float(new_values.min()) if len(new_values) else np.nan,
                "gained_value_max": float(new_values.max()) if len(new_values) else np.nan,
            })
    return _derived_value_columns(pd.DataFrame(records))


def _derived_value_columns(frame):
    result = frame.copy()
    for policy in ("reference", "sensitivity"):
        result[policy + "_finite_fraction"] = result[policy + "_finite_rows"] / result.rows.replace(0, np.nan)
    result["coverage_gain_percentage_points"] = 100 * (
        result.sensitivity_finite_fraction - result.reference_finite_fraction)
    result["shared_abs_difference_mean"] = result.shared_abs_difference_sum / result.shared_finite_rows.replace(0, np.nan)
    n = result.gained_finite_rows.replace(0, np.nan)
    result["gained_value_mean"] = result.gained_value_sum / n
    result["gained_value_std_population"] = np.sqrt(np.maximum(
        result.gained_value_sum_squared / n - result.gained_value_mean ** 2, 0))
    return result


def value_periods(yearly):
    """Pool counts by declared chronology; never average annual percentages."""
    grouping = ["population", "feature_name", "role", "family"]
    sums = ["rows", "reference_finite_rows", "sensitivity_finite_rows", "shared_finite_rows",
            "gained_finite_rows", "lost_finite_rows", "changed_shared_values",
            "shared_abs_difference_sum", "gained_value_sum", "gained_value_sum_squared"]
    parts = []
    for period, (start, end) in PERIODS.items():
        selected = yearly[yearly.outcome_year.between(start, end)]
        if selected.empty:
            continue
        values = selected.groupby(grouping, as_index=False)[sums].sum()
        extrema = selected.groupby(grouping, as_index=False).agg(
            shared_abs_difference_max=("shared_abs_difference_max", "max"),
            gained_value_min=("gained_value_min", "min"), gained_value_max=("gained_value_max", "max"))
        values = values.merge(extrema, on=grouping, validate="one_to_one")
        values.insert(0, "period", period)
        parts.append(_derived_value_columns(values))
    return pd.concat(parts, ignore_index=True)


def compare_diagnostics(reference, sensitivity, *, outcome_year, daily=False):
    """Compare status/count evidence, keeping overlapping annual warmups explicit."""
    keys = ["signal_date", "feature_name", "context_group"]
    for frame in (reference, sensitivity):
        if not set(keys) <= set(frame) or frame.duplicated(keys).any():
            raise DataContractError("Context diagnostics require unique date/feature/group keys")
    a = reference.set_index(keys).sort_index()
    b = sensitivity.set_index(keys).sort_index()
    if not a.index.equals(b.index):
        raise DataContractError("Context diagnostics changed original date/feature/group keys")
    invariants = ["cohort_size", "latest_input_date", "identity_eligible_members",
                  "requested_identity_eligible_members", "request_exhaustive"]
    _assert_equal(a[invariants], b[invariants], "Context diagnostic membership and requests")
    independent = ~a.index.get_level_values("feature_name").str.startswith("relative_")
    _assert_equal(a.loc[independent, ["observed_members", "coverage_fraction"]],
                  b.loc[independent, ["observed_members", "coverage_fraction"]],
                  "Context valid contributor populations")
    result = a[["cohort_size", "observed_members", "coverage_fraction", "request_exhaustive", "status"]].rename(
        columns={"observed_members": "reference_observed_members", "coverage_fraction": "reference_coverage_fraction",
                 "status": "reference_status"}).reset_index()
    for name in ("observed_members", "coverage_fraction", "status"):
        result["sensitivity_" + name] = b[name].to_numpy()
    result.insert(0, "partition_outcome_year", outcome_year)
    result["diagnostic_family"] = np.select(
        [result.feature_name.str.startswith("xs_"), result.feature_name.str.startswith("market_"),
         result.feature_name.str.startswith("relative_")],
        ["cross_sectional", "market_state", "relative_context"], default="daily_benchmark")
    result["reference_status_ok"] = result.reference_status.eq("ok")
    result["sensitivity_status_ok"] = result.sensitivity_status.eq("ok")
    if (result.reference_status_ok & ~result.sensitivity_status_ok).any():
        # Relative diagnostics summarize mixed member states, rather than one
        # cross-sectional gate; complete success still cannot be lost.
        raise DataContractError("Lower threshold lost an all-ok context diagnostic")
    result["newly_status_ok"] = ~result.reference_status_ok & result.sensitivity_status_ok
    result["date_role"] = "monthly_signal"
    if daily:
        owning_year = pd.PeriodIndex(result.signal_date.str[:7], freq="M") + 1
        result["date_role"] = np.where(owning_year.year == outcome_year,
            "primary_signal_year", "overlapping_warmup")
    return result


def cohort_composition(features, original, ledger, diagnostics, *, outcome_year):
    """Describe valid base populations, including omitted original constituents.

    Group records are feature/date/cohort memberships, not unique stocks. The
    optional profile metrics are unchanged past-only technical snapshots, and
    cannot describe stocks with missing inputs or establish economic bias.
    """
    values = _aligned(features, "Composition features")
    original = _aligned(original, "Composition original members")
    ledger = _aligned(ledger, "Composition ledger")
    if not values.index.equals(original.index) or not values.index.equals(ledger.index):
        raise DataContractError("Cohort composition changed original historical keys")
    members = original.copy()
    for name in ("identity_eligible", "requested", "identity_review_required"):
        members[name] = ledger[name]
    members["provisional_review"] = (members.identity_eligible & members.identity_review_required)
    members["paper_qualified"] = ledger.paper_prediction_eligible
    for sleeve in ("sp500", "sp400"):
        members[sleeve] = members.indices.map(lambda items: sleeve in items)
    gates = diagnostics[diagnostics.feature_name.str.startswith(("xs_universe_rank_pct__", "xs_index_sleeve_rank_pct__"))]
    lookup = gates.set_index(["signal_date", "feature_name", "context_group"])
    if not lookup.index.is_unique:
        raise DataContractError("Duplicate cross-sectional rank diagnostic")
    records = []
    for day, day_members in members.groupby(level="signal_date", sort=True):
        day_values = values.loc[day_members.index]
        for group in ("universe", "sp500", "sp400"):
            group_mask = np.ones(len(day_members), dtype=bool) if group == "universe" else day_members[group].to_numpy(dtype=bool)
            selected = day_members.loc[group_mask]
            inputs = day_values.loc[selected.index]
            scope = "universe" if group == "universe" else "index_sleeve"
            for base in balanced_cross_sectional_bases():
                observed = (np.isfinite(inputs[base].to_numpy(dtype=float)) & selected.identity_eligible.to_numpy(dtype=bool)
                            & selected.requested.to_numpy(dtype=bool))
                gate = lookup.loc[(day, f"xs_{scope}_rank_pct__{base}", group)]
                if int(gate.cohort_size) != len(selected) or int(gate.reference_observed_members) != int(observed.sum()) \
                        or int(gate.sensitivity_observed_members) != int(observed.sum()):
                    raise DataContractError("Composition valid population disagrees with saved context diagnostics")
                record = {"outcome_year": outcome_year, "signal_date": day, "context_group": group,
                    "base_feature": base, "cohort_size": len(selected), "observed_members": int(observed.sum()),
                    "omitted_members": int((~observed).sum()), "observed_fraction": observed.mean() if len(observed) else np.nan,
                    "reference_status_ok": bool(gate.reference_status_ok),
                    "sensitivity_status_ok": bool(gate.sensitivity_status_ok),
                    "passing_category": "both" if gate.reference_status_ok else "80_only" if gate.sensitivity_status_ok else "neither"}
                for label, mask in (("observed", observed), ("omitted", ~observed)):
                    for attribute in ("sp500", "sp400", "provisional_review", "paper_qualified"):
                        record[f"{label}_{attribute}_members"] = int(selected.loc[mask, attribute].sum())
                    for metric in COMPOSITION_METRICS:
                        profile = inputs.loc[mask, metric].to_numpy(dtype=float)
                        profile = profile[np.isfinite(profile)]
                        record[f"{label}_{metric}_finite_members"] = len(profile)
                        record[f"{label}_{metric}_sum"] = float(profile.sum())
                        record[f"{label}_{metric}_mean"] = float(profile.mean()) if len(profile) else np.nan
                admitted = selected.identity_eligible.to_numpy(dtype=bool)
                requested = selected.requested.to_numpy(dtype=bool)
                record["omitted_identity_blocked_members"] = int((~observed & ~admitted).sum())
                record["omitted_unrequested_admitted_members"] = int((~observed & admitted & ~requested).sum())
                record["omitted_base_input_missing_members"] = int((~observed & admitted & requested).sum())
                records.append(record)
    return pd.DataFrame(records)


def _composition_periods(frame):
    counts = [name for name in frame if name.endswith("_members") or name.endswith("_sum")]
    counts += ["cohort_size"]
    parts = []
    for period, (start, end) in PERIODS.items():
        selected = frame[frame.outcome_year.between(start, end)]
        group = ["context_group", "passing_category"]
        result = selected.groupby(group, as_index=False)[counts].sum()
        result["feature_date_cohort_groups"] = selected.groupby(group).size().reindex(
            pd.MultiIndex.from_frame(result[group])).to_numpy()
        result["observed_fraction"] = result.observed_members / result.cohort_size.replace(0, np.nan)
        for label in ("observed", "omitted"):
            for attribute in ("sp500", "sp400", "provisional_review", "paper_qualified"):
                result[f"{label}_{attribute}_fraction"] = result[f"{label}_{attribute}_members"] / result[f"{label}_members"].replace(0, np.nan)
            for metric in COMPOSITION_METRICS:
                result[f"{label}_{metric}_mean"] = result[f"{label}_{metric}_sum"] / result[f"{label}_{metric}_finite_members"].replace(0, np.nan)
        result.insert(0, "period", period)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def _gate_periods(frame):
    parts = []
    for period, (start, end) in PERIODS.items():
        selected = frame[frame.partition_outcome_year.between(start, end)]
        groups = ["diagnostic_family", "context_group", "date_role"]
        result = selected.groupby(groups, as_index=False).agg(
            diagnostic_records=("reference_status_ok", "size"),
            reference_status_ok_records=("reference_status_ok", "sum"),
            sensitivity_status_ok_records=("sensitivity_status_ok", "sum"),
            newly_status_ok_records=("newly_status_ok", "sum"))
        result.insert(0, "period", period)
        parts.append(result)
    return pd.concat(parts, ignore_index=True)


def compare_context_runs(reference_directory, sensitivity_directory, output_directory):
    """Verify and compare fourteen completed annual panels, writing audit reports."""
    reference_directory, sensitivity_directory, output_directory = map(Path,
        (reference_directory, sensitivity_directory, output_directory))
    if reference_directory.resolve() == sensitivity_directory.resolve():
        raise DataContractError("Context sensitivity requires separate reference and sensitivity runs")
    output_directory.mkdir(parents=True, exist_ok=True)
    specs = registered_feature_specs()
    context = [s for s in specs if s["role"] not in ("paper", "technical") and s.get("model_eligible", True)]
    if len(context) != 402:
        raise DataContractError("Context sensitivity requires the unchanged 402 eligible definitions")
    values, monthly, daily, composition, hashes = [], [], [], [], []
    original_rows = qualified_rows = 0
    for year in range(2010, 2024):
        a, b = (directory / "years" / str(year) for directory in (reference_directory, sensitivity_directory))
        for name in PRESERVED_ARTIFACTS:
            first, second = a / (name + ".parquet"), b / (name + ".parquet")
            digest_a, digest_b = _sha256(first), _sha256(second)
            if digest_a != digest_b:
                raise DataContractError(f"Preserved artifact changed: {year}/{name}")
            hashes.append({"outcome_year": year, "artifact": name, "bytes": first.stat().st_size,
                           "reference_sha256": digest_a, "sensitivity_sha256": digest_b, "byte_identical": True})
        fa, fb = (pq.read_table(path / "features.parquet").to_pandas() for path in (a, b))
        ca, cb = (pq.read_table(path / "feature_coverage.parquet").to_pandas() for path in (a, b))
        immutable = [name for name in ca if name not in ("available_feature_count", "feature_status")]
        if set(ca) != set(cb):
            raise DataContractError("Feature coverage changed ledger schema")
        _assert_equal(_aligned(ca[immutable], "Reference coverage"), _aligned(cb[immutable], "Sensitivity coverage"),
                      "Identity, original keys and paper eligibility ledgers")
        values.append(compare_feature_frames(fa, fb, ca, specs, outcome_year=year))
        original_rows += len(ca)
        qualified_rows += int(ca.paper_prediction_eligible.sum())
        dm = compare_diagnostics(*(pq.read_table(path / "context_diagnostics.parquet").to_pandas() for path in (a, b)), outcome_year=year)
        dd = compare_diagnostics(*(pq.read_table(path / "daily_benchmark_diagnostics.parquet").to_pandas() for path in (a, b)), outcome_year=year, daily=True)
        monthly.append(dm)
        daily.append(dd)
        composition.append(cohort_composition(fa, pq.read_table(a / "original_cohort.parquet").to_pandas(), ca, dm, outcome_year=year))
    yearly = pd.concat(values, ignore_index=True)
    periods = value_periods(yearly)
    monthly, daily, composition = (pd.concat(parts, ignore_index=True) for parts in (monthly, daily, composition))
    families = periods.groupby(["period", "population", "role"], as_index=False)[[
        "rows", "reference_finite_rows", "sensitivity_finite_rows", "shared_finite_rows", "gained_finite_rows",
        "lost_finite_rows", "changed_shared_values"]].sum()
    families["reference_finite_fraction"] = families.reference_finite_rows / families.rows.replace(0, np.nan)
    families["sensitivity_finite_fraction"] = families.sensitivity_finite_rows / families.rows.replace(0, np.nan)
    families["coverage_gain_percentage_points"] = 100 * (families.sensitivity_finite_fraction - families.reference_finite_fraction)
    reports = {
        "feature_value_comparison_year": yearly, "feature_value_comparison_period": periods,
        "family_coverage_comparison": families, "monthly_gate_comparison": monthly,
        "daily_gate_comparison": daily, "monthly_gate_period_comparison": _gate_periods(monthly),
        "daily_gate_period_comparison": _gate_periods(daily), "xs_cohort_composition": composition,
        "cohort_composition_period": _composition_periods(composition),
        "preserved_artifact_hashes": pd.DataFrame(hashes),
    }
    for name, frame in reports.items():
        frame.to_csv(output_directory / (name + ".csv"), index=False)
    all_period = periods[(periods.period == "full_history_2010_2023") & (periods.population == "original")]
    summary = {
        "schema_version": 1, "status": "CONTEXT_80_VS_90_COMPARISON_VERIFIED",
        "reference_directory": str(reference_directory.resolve()),
        "sensitivity_directory": str(sensitivity_directory.resolve()),
        "reference_policy": "available_cohort_90pct", "sensitivity_policy": "available_cohort_80pct",
        "outcome_start_year": 2010, "outcome_end_year": 2023,
        "original_stock_months": original_rows, "paper_qualified_stock_months": qualified_rows,
        "eligible_context_predictors": len(context),
        "reference_ever_finite_predictors": int((all_period.reference_finite_rows > 0).sum()),
        "sensitivity_ever_finite_predictors": int((all_period.sensitivity_finite_rows > 0).sum()),
        "gained_context_values_original": int(all_period.gained_finite_rows.sum()),
        "lost_context_values_original": int(all_period.lost_finite_rows.sum()),
        "changed_shared_context_values_original": int(all_period.changed_shared_values.sum()),
        "all_shared_finite_context_values_exact": True,
        "paper15_exact_unchanged": True, "technical298_exact_unchanged": True,
        "original_keys_identity_flags_paper_eligibility_exact_unchanged": True,
        "preserved_artifact_count": len(hashes), "preserved_artifacts_byte_identical": True,
        "coverage": families.to_dict("records"),
        "period_basis": "Next-month outcome year; before 2016 means outcome years 2010–2015.",
        "composition_basis": "Per-variable/date/cohort membership slots; valid populations are identical between policies. 80-only releases previously withheld statistics; missing individual stocks stay missing.",
        "composition_profile_limits": "Past-only return, volatility and log-dollar-volume metrics describe only finite available snapshots. Missing/unmapped stocks have no inferred sector, capitalization, distress or listing profile. These descriptive comparisons do not establish absence of selection bias.",
        "daily_gate_basis": "primary_signal_year covers December of the preceding year through November of the partition year; overlapping annual warmups are reported separately and must not be pooled as unique days.",
        "diagnostic_status_semantics": "Status-ok records include formula validity. Relative-context status summarizes all members, not an independent 80/90 cohort gate; inspect value counts for individual availability.",
        "fits_performed": 0, "outcomes_consumed": False, "raw_daily_financial_values_decoded": False,
        "knowledge_time_verified": False, "accounting_certified": False, "daily_backtest_ready": False,
        "reports": {name: name + ".csv" for name in reports},
    }
    # JSON is strict even when a zero-denominator report contains NaN.
    summary = json.loads(json.dumps(summary, default=lambda v: v.item() if isinstance(v, np.generic) else str(v)))
    (output_directory / "comparison_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    report_paths = [*(output_directory / (name + ".csv") for name in reports),
                    output_directory / "comparison_summary.json"]
    manifest = {
        "schema_version": 1, "status": "CONTEXT_COMPARISON_ARTIFACTS_PINNED",
        "reference_audit": {"path": str((reference_directory / "audit.json").resolve()),
                            "sha256": _sha256(reference_directory / "audit.json")},
        "sensitivity_audit": {"path": str((sensitivity_directory / "audit.json").resolve()),
                              "sha256": _sha256(sensitivity_directory / "audit.json")},
        "reports": {path.name: {"sha256": _sha256(path), "bytes": path.stat().st_size} for path in report_paths},
    }
    (output_directory / "comparison_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return summary
