"""New restricted-universe backtest; original probabilities/runs stay frozen.

DF/RDC remain in model training and prediction ledgers. Their explicit,
retrospective execution restriction applies BEFORE the existing top-ten selector.
All other missing held prices retain exposure and withhold full-period metrics.
This phase never fits a model, fetches data or reads post-2019 financial values.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .action_accounting import run_action_accounting
from .action_loading import ACTION_PRICE_COLUMNS, EVENT_TERMS, KNOWN_SUCCESSORS, RDC_UNQUOTED_ASSET, _source_url
from .backtest_accounting import DailyAccountingPolicy
from .backtest_loading import (ARMS, PREDICTION_METADATA, PRICE_KEYS, _hash, _independent_clock,
    _json, _pin, _projected_bytes, _validate_prediction_metadata, _verify_pin, _within)
from .backtest_selection import BasketSelection, PROBABILITIES, SAFE_COLUMNS, select_monthly_basket
from .catalog import canonical_sha256
from .daily_backtest import _save_table, _write_json, summarize_daily, _rss
from .ret import CLASS_IDS, align_class_probabilities
from .universe import DataContractError

CONFIGURATION_SHA256 = "sha256:c7ed9513a9739f9b926661c06d287dbab1f1475309d06108e94f423e18b97df6"
EXCLUDED_NATIVE_IDS = ("sec_gjP4xM", "sec_zmYNnz")
WORKERS = ("fallback_backtest.py", "fallback_quotes.py", "backtest_selection.py",
           "action_accounting.py", "backtest_accounting.py", "action_loading.py", "daily_backtest.py",
           "backtest_loading.py", "ret.py", "catalog.py", "universe.py", "exchange_sessions.py",
           "identity_policy.py", "fallback_models.py", "full_panel.py")
REPORTS = ("comparison", "selection_coverage", "ret_readiness", "failures", "action_coverage", "order_coverage")


def _configuration(configuration):
    if canonical_sha256(configuration) != CONFIGURATION_SHA256:
        raise DataContractError("Frozen data-limited fallback configuration changed")
    restrictions = configuration["execution_restrictions"]
    backtest = configuration["backtest"]
    if (tuple(sorted(restrictions["native_ids"])) != EXCLUDED_NATIVE_IDS
            or restrictions["valid_historical_training_rows_retained"] is not True
            or restrictions["prediction_rows_and_probabilities_retained"] is not True
            or restrictions["period"] != "all_2018_2019_signals_before_selection"
            or backtest["arms"] != list(ARMS)
            or backtest["baskets"] != ["class_1_10", "class_2_9"]
            or backtest["cost_bps"] != [0,5,10,25]
            or backtest["top_k_per_side"] != 10
            or backtest["start"] != "2018-01-01" or backtest["end"] != "2019-12-31"):
        raise DataContractError("Restricted execution must retain its exact separately declared recipe")
    return configuration


def select_restricted_monthly_basket(predictions, *, basket, sessions, configuration):
    """Keep prediction eligibility intact; restrict execution before ordering.

    No actual quote, outcome or future membership is inspected. The two
    excluded instruments are a declared retrospective availability restriction,
    not historically available evidence or a claim to remove selection bias.
    """
    cfg = _configuration(configuration)
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
    selected = select_monthly_basket(working,basket=basket,sessions=sessions,top_k=cfg["backtest"]["top_k_per_side"])
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


def _predictions(models, identity, windows, pins, root, *, read_stats=None):
    from .fallback_models import read_fallback_models
    saved = read_fallback_models(models,identity_sha256=identity)
    audit = saved["audit"]
    pins.extend([_pin(root,models/"audit.json"),_pin(root,models/"build_identity.json")])
    paths, metadata, key_order, common_cohort = {}, {}, None, None
    projected=0
    for arm in ARMS:
        arm_audit_path = _verify_pin(models,audit["arms"][arm])
        arm_audit = _json(arm_audit_path)
        if arm_audit.get("build_identity_sha256") != identity:
            raise DataContractError("New model arm belongs to a different parent reconstruction")
        path = _verify_pin(arm_audit_path.parent,arm_audit["artifacts"]["predictions"])
        pins.extend([_pin(root,arm_audit_path),_pin(root,path)])
        fields = set(pq.ParquetFile(path).schema_arrow.names)
        columns = [field for field in PREDICTION_METADATA if field in fields]
        meta = pq.ParquetFile(path).read(columns=columns)
        projected+=_projected_bytes(path,columns)
        keys = _validate_prediction_metadata(meta,windows)
        cohort=meta.select(["source_security_id","signal_date","intrinio_security_id",
            "identity_eligible","requested","paper_prediction_eligible","prediction_eligible"])
        if key_order is None:
            key_order = keys
        elif keys != key_order:
            raise DataContractError("New model arms have different original validation key order")
        if common_cohort is None:
            common_cohort=cohort
        elif not cohort.equals(common_cohort):
            raise DataContractError("New model arms have different signal-time prediction cohorts")
        if len(meta)>25000:
            raise DataContractError("New validation prediction ledger exceeds fixed row bound")
        metadata[arm],paths[arm] = meta,path
        financial_columns=list(dict.fromkeys([*[name for name in SAFE_COLUMNS if name in meta.column_names],*PROBABILITIES]))
        projected+=_projected_bytes(path,financial_columns)
    if projected>128*1024**2:
        raise DataContractError("Prediction projections exceed the explicit backtest read budget")
    if read_stats is not None:
        read_stats.update(candidate_rows=sum(len(table) for table in metadata.values()),
            projected_compressed_bytes=projected,metadata_and_probability_projections_counted=True,
            future_labels_decoded=False)
    # All timing, cohort and hash guards passed before any probability decode.
    result = {}
    for arm,path in paths.items():
        columns = list(dict.fromkeys([*[name for name in SAFE_COLUMNS if name in metadata[arm].column_names],*PROBABILITIES]))
        result[arm] = pq.ParquetFile(path).read(columns=columns).to_pandas()
    return result


def _annual_prices(root, run, identity, quote_fingerprint, pins, *, max_projected_bytes=128*1024**2,max_rows=1000000):
    metadata, paths, projected, rows = {}, {}, 0, 0
    for year in (2018,2019):
        directory=run/"years"/str(year)
        audit_path=directory/"audit.json"
        audit=_json(audit_path)
        if (audit.get("status")!="COMPLETE" or audit.get("year")!=year
                or audit.get("build_identity_sha256") != identity or audit.get("quote_amendments_fingerprint") != quote_fingerprint):
            raise DataContractError("Annual execution cache belongs to a different corrected quote recipe")
        item=audit["artifacts"].get("daily_prices_raw")
        if item is None:
            raise DataContractError("New annual audit omits disjoint outcome-year raw daily cache")
        path=_verify_pin(directory,item)
        keys=pq.ParquetFile(path).read(columns=list(PRICE_KEYS))
        frame=keys.to_pandas()
        if (frame.isna().any().any() or frame.duplicated(list(PRICE_KEYS)).any()
                or not frame.trading_date.map(lambda day:isinstance(day,str) and f"{year}-01-01"<=day<=f"{year}-12-31").all()):
            raise DataContractError("Annual execution cache contains invalid or out-of-year date/identity keys")
        metadata[year],paths[year]=keys,path
        rows+=len(keys)
        projected+=_projected_bytes(path,PRICE_KEYS)+_projected_bytes(path,ACTION_PRICE_COLUMNS)
        pins.extend([_pin(root,audit_path),_pin(root,path)])
    if rows>max_rows or projected>max_projected_bytes:
        raise DataContractError("New full-cohort execution projections exceed bounded read budget")
    # Read ALL new cached identities, never the old selected-ID inventory.
    prices=[]
    for year,path in paths.items():
        table=pq.ParquetFile(path).read(columns=list(ACTION_PRICE_COLUMNS))
        if not table.select(list(PRICE_KEYS)).equals(metadata[year]):
            raise DataContractError("Execution cache keys changed after metadata guard")
        prices.append(table.to_pandas())
    frame=pd.concat(prices,ignore_index=True)
    if frame.duplicated(list(PRICE_KEYS)).any():
        raise DataContractError("Disjoint execution years overlap")
    return frame,{"candidate_rows":rows,"projected_compressed_bytes":projected,
        "all_new_annual_cached_ids_loaded":True,"old_selected_id_filter_used":False,
        "financial_2020_plus_read":False,"metadata_guards_before_financial_decode":True}


def _reviewed_events(root,sessions):
    """Read the same pinned eight terms without following old model pointers."""
    path=Path(root)/"data/derived/daily_action_inputs/action_evidence.json"
    source=_json(path)
    records=source.get("events")
    if (not isinstance(records,list) or len(records)!=8
            or {row.get("old_ticker") for row in records if isinstance(row,dict)}!=set(EVENT_TERMS)):
        raise DataContractError("Exactly the original eight reviewed corporate actions are required")
    events=[]
    for item in records:
        ticker=item["old_ticker"]
        terms=(item.get("old_provider_id"),item.get("effective_date"),item.get("application_session"),
               item.get("cash_per_old_raw_share"),item.get("successor_raw_shares"))
        if (terms!=EVENT_TERMS[ticker] or item["application_session"] not in sessions
                or any(type(value) is bool for value in terms[-2:])):
            raise DataContractError("Reviewed dated corporate-action terms changed")
        successor=RDC_UNQUOTED_ASSET if ticker=="RDC" else KNOWN_SUCCESSORS.get(ticker)
        events.append({"event_id":ticker+"-"+item["effective_date"],
            "predecessor_id":item["old_provider_id"],"effective_date":item["application_session"],
            "legal_effective_date":item["effective_date"],"successor_id":successor,
            "cash_per_raw_share":item["cash_per_old_raw_share"],"stock_ratio":item["successor_raw_shares"],
            "verified":True,"evidence_reference":_source_url(item.get("source_url")),
            "timing_evidence_reference":_source_url(item.get("timing_source_url",item["source_url"])),
            "settlement_verified":False,"knowledge_time_verified":False})
    return events,path


def _phase_rss():
    # The orchestrator launches this phase in its own fresh process.
    return _rss()


@dataclass(frozen=True)
class SavedFallbackBacktest:
    path: Path
    audit: dict
    comparison: pd.DataFrame
    selection_coverage: pd.DataFrame
    ret_readiness: pd.DataFrame
    failures: pd.DataFrame
    action_coverage: pd.DataFrame
    order_coverage: pd.DataFrame


def read_fallback_backtest(directory, *, identity_sha256=None, phase_identity_sha256=None):
    directory=Path(directory).resolve()
    audit=_json(directory/"audit.json")
    if (audit.get("status")!="COMPLETE" or audit.get("scenarios")!=64
            or audit.get("research_status")!="DATA_LIMITED_RESTRICTED_UNIVERSE"
            or identity_sha256 is not None and audit.get("build_identity_sha256")!=identity_sha256
            or phase_identity_sha256 is not None and audit.get("phase_identity_sha256")!=phase_identity_sha256):
        raise DataContractError("Saved restricted-universe backtest is stale or incomplete")
    required={"build_identity.json",*[name+".parquet" for name in REPORTS]}
    if not required<=set(audit.get("artifacts",{})):
        raise DataContractError("Saved restricted backtest omits required pinned reports")
    for item in audit["artifacts"].values():
        _verify_pin(directory,item)
    phase=_json(directory/"build_identity.json")
    if (canonical_sha256(phase)!=audit.get("phase_identity_sha256")
            or phase.get("parent_identity_sha256")!=audit.get("build_identity_sha256")):
        raise DataContractError("Saved restricted backtest phase identity changed")
    reports={name:pd.read_parquet(directory/(name+".parquet")) for name in REPORTS}
    comparison=reports["comparison"]
    if (len(comparison)!=64 or comparison.duplicated(["arm","basket","variant","cost_bps"]).any()
            or set(comparison.arm)!=set(ARMS) or set(comparison.basket)!={"class_1_10","class_2_9"}
            or set(comparison.variant)!={"unstopped","asset_stop_10pct"} or set(comparison.cost_bps)!={0,5,10,25}):
        raise DataContractError("Saved restricted backtest lost its complete 64-scenario grid")
    return SavedFallbackBacktest(directory,audit,**reports)


def run_fallback_backtest(root,run_directory,configuration,identity_sha256,*,quote_amendments=None,models_directory=None):
    """Explicit 64-scenario evaluation on NEW models and NEW annual caches."""
    from .fallback_quotes import compile_fallback_quote_amendments
    root=Path(root).resolve()
    run=_within(root,Path(run_directory).resolve())
    cfg=_configuration(configuration)
    started=time.monotonic()
    precompiled_quotes=quote_amendments is not None
    quote_amendments=quote_amendments or compile_fallback_quote_amendments(root)
    quote_amendments.verify(root)
    main_identity=_json(run/"build_identity.json")
    if canonical_sha256(main_identity)!=identity_sha256 or main_identity.get("quote_amendments_fingerprint")!=quote_amendments.fingerprint:
        raise DataContractError("New backtest does not match the parent build identity/amended quote bundle")
    models=_within(run,Path(models_directory).resolve()) if models_directory is not None else run/"models"
    sessions,windows,clock=_independent_clock()
    pins=[_pin(root,run/"build_identity.json")]
    peak=_phase_rss()
    prediction_read={}
    predictions=_predictions(models,identity_sha256,windows,pins,root,read_stats=prediction_read)
    prices,read_plan=_annual_prices(root,run,identity_sha256,quote_amendments.fingerprint,pins,
        max_projected_bytes=128*1024**2-prediction_read.get("projected_compressed_bytes",0),
        max_rows=1000000-prediction_read.get("candidate_rows",0))
    read_plan={**read_plan,"prediction_projections":prediction_read,
        "total_projected_compressed_bytes":read_plan.get("projected_compressed_bytes",0)+prediction_read.get("projected_compressed_bytes",0),
        "total_candidate_rows":read_plan.get("candidate_rows",0)+prediction_read.get("candidate_rows",0),
        "quote_amendments_loaded_from_precompiled_bundle":precompiled_quotes}
    if not prices.trading_date.isin(sessions).all():
        raise DataContractError("New execution prices include non-XNYS sessions")
    event_records,event_path=_reviewed_events(root,sessions)
    pins.append(_pin(root,event_path))
    events=pd.DataFrame(event_records)
    phase_identity={"schema_version":1,"phase":"data_limited_restricted_backtest_v1",
        "parent_identity_sha256":identity_sha256,"configuration_sha256":canonical_sha256(cfg),
        "quote_amendments_fingerprint":quote_amendments.fingerprint,"source_pins":pins,
        "clock":clock,"read_plan":read_plan,"events":event_records,
        "workers":{name:_hash(Path(__file__).parent/name) for name in WORKERS},
        "runtime":{"python":platform.python_version(),**{name:version(name) for name in ("numpy","pandas","pyarrow","exchange-calendars")}}}
    phase_hash=canonical_sha256(phase_identity)
    directory=run/"backtest"
    if directory.exists():
        read_fallback_backtest(directory,identity_sha256=identity_sha256,phase_identity_sha256=phase_hash)
        return directory
    directory.mkdir(parents=True)
    artifacts={}
    _write_json(directory/"build_identity.json",phase_identity)
    artifacts["build_identity.json"]=_pin(directory,directory/"build_identity.json")
    _write_json(directory/"audit.json",{"status":"STARTED","build_identity_sha256":identity_sha256,"phase_identity_sha256":phase_hash})
    comparison,failures,coverage,ret_ready,action_coverage,order_coverage=[],[],[],[],[],[]

    def check_resources():
        nonlocal peak
        peak=max(peak,_phase_rss())
        limits=cfg["execution"]
        if (time.monotonic()-started>limits["backtest_maximum_wall_seconds"]
                or peak>limits["backtest_maximum_peak_rss_bytes"]
                or sum(path.stat().st_size for path in run.rglob("*") if path.is_file())>limits["maximum_output_bytes"]):
            raise DataContractError("Restricted-universe backtest exceeds its explicit local resource ceiling")
    try:
        for arm in ARMS:
            prediction=predictions[arm]
            ret=select_restricted_monthly_basket(prediction,basket="ret",sessions=clock["selection_sessions"],configuration=cfg)
            ret_ready.append(ret.monthly.assign(arm=arm))
            if not ret.selections.empty or not ret.monthly.status.eq("withheld_incomplete_120_month_class_grid").all():
                raise DataContractError("Restricted backtest must preserve the strict unavailable RET gate")
            for basket in cfg["backtest"]["baskets"]:
                selection=select_restricted_monthly_basket(prediction,basket=basket,sessions=clock["selection_sessions"],configuration=cfg)
                coverage.append(selection.monthly.assign(arm=arm))
                for name,frame in (("original_ledger",selection.ledger),("selected",selection.selections),("rebalances",selection.rebalances)):
                    _save_table(directory,f"selections/{arm}/{basket}/{name}.parquet",frame,artifacts)
                for variant,stop in cfg["backtest"]["variants"].items():
                    for fee in cfg["backtest"]["cost_bps"]:
                        check_resources()
                        result=run_action_accounting(selection.selections,prices,sessions,
                            rebalances=selection.rebalances,terminal_date="2019-12-31",events=events,
                            policy=DailyAccountingPolicy(cost_bps=fee,borrow_annual_bps=0,
                                required_names_per_side=10,asset_stop_fraction=stop))
                        relative=f"scenarios/{arm}/{basket}/{variant}/{fee}bps"
                        for name,frame in result.tables.items():
                            _save_table(directory,relative+"/"+name+".parquet",frame,artifacts)
                        summary,monthly=summarize_daily(result,arm=arm,basket=basket,variant=variant,
                            cost_bps=fee,expected_sessions=len(sessions))
                        summary.update(research_status="DATA_LIMITED_RESTRICTED_UNIVERSE",
                            retrospective_execution_restriction=True,full_historical_universe_performance_claim=False,
                            all_original_prediction_rows_retained=True,valid_DF_RDC_training_history_retained=True,
                            action_count=result.summary["action_count"],unfilled_retired_order_count=result.summary["unfilled_retired_order_count"])
                        if result.status=="complete_proxy":
                            summary["trading_cost_paid"]=float(result.market_trades.cost.sum())
                        comparison.append(summary)
                        _save_table(directory,relative+"/monthly.parquet",monthly,artifacts)
                        _write_json(directory/relative/"summary.json",{**result.summary,"research_status":"DATA_LIMITED_RESTRICTED_UNIVERSE"})
                        artifacts[relative+"/summary.json"]=_pin(directory,directory/relative/"summary.json")
                        if not result.failures.empty:
                            failures.append(result.failures.assign(arm=arm,basket=basket,variant=variant,cost_bps=fee))
                        if not result.action_events.empty:
                            action_coverage.extend(dict(arm=arm,basket=basket,variant=variant,cost_bps=fee,
                                event_id=event,side=side,applied_lot_count=int(count)) for (event,side),count
                                in result.action_events.groupby(["event_id","side"],dropna=False).size().items())
                        if not result.orders.empty:
                            order_coverage.extend(dict(arm=arm,basket=basket,variant=variant,cost_bps=fee,
                                status=status,original_order_count=int(count)) for status,count
                                in result.orders.groupby("status",dropna=False).size().items())
                        print(json.dumps({"phase":"restricted_backtest","arm":arm,"basket":basket,
                            "variant":variant,"cost_bps":fee,"status":result.status,
                            "first_failure_date":summary["first_failure_date"]}),flush=True)
        reports={"comparison":pd.DataFrame(comparison),"selection_coverage":pd.concat(coverage,ignore_index=True),
            "ret_readiness":pd.concat(ret_ready,ignore_index=True),
            "failures":pd.concat(failures,ignore_index=True) if failures else pd.DataFrame(columns=["date","mark","reason","source_security_id","intrinio_security_id","price_field","role","arm","basket","variant","cost_bps"]),
            "action_coverage":pd.DataFrame(action_coverage,columns=["arm","basket","variant","cost_bps","event_id","side","applied_lot_count"]),
            "order_coverage":pd.DataFrame(order_coverage,columns=["arm","basket","variant","cost_bps","status","original_order_count"])}
        for name,frame in reports.items():
            _save_table(directory,name+".parquet",frame,artifacts)
        for pin in pins:
            _verify_pin(root,pin)
        quote_amendments.verify(root)
        check_resources()
        audit={"status":"COMPLETE","research_status":"DATA_LIMITED_RESTRICTED_UNIVERSE",
            "build_identity_sha256":identity_sha256,"phase_identity_sha256":phase_hash,"artifacts":artifacts,
            "scenarios":len(comparison),"complete_proxy_paths":sum(row["complete_proxy_path"] for row in comparison),
            "failed_closed_paths":sum(not row["complete_proxy_path"] for row in comparison),
            "execution_seconds":time.monotonic()-started,"peak_rss_bytes":peak,
            "read_plan":read_plan,"source_pins_unchanged":True,"models_fitted":False,
            "financial_2020_plus_read":False,"original_pointers_changed":False,
            "RET":"withheld_all_24_signals_per_arm","execution_excluded_native_ids":list(EXCLUDED_NATIVE_IDS),
            "excluded_instruments_retained_in_training_and_predictions":True,
            "retrospective_data_quality_restriction":True,"historical_universe_performance_claim":False,
            "selection_and_survivorship_bias_removed":False,"knowledge_time_certified":False,
            "complete_economic_accounting_certified":False}
        _write_json(directory/"audit.json",audit)
    except Exception as error:
        _write_json(directory/"failure.json",{"exception":type(error).__name__,"message":str(error)})
        raise
    return directory
