"""Shared RET labels, past-only folds, payoffs and portfolio accounting.

Legacy fitting controllers are frozen in the private maintenance archive. This
module has no training entry point and does not mutate feature specifications.
"""
from __future__ import annotations
from pathlib import Path
import json, os, resource, sys, signal, time, hashlib, subprocess
import duckdb
import numpy as np
import pandas as pd
import psutil
from research_checks import require_past_training
EVALUATION_START="2017-01-01"
EVALUATION_END="2026-10-31"
PARAMETERS=dict(objective="multi:softprob",num_class=10,eval_metric="mlogloss",tree_method="hist",max_depth=3,max_bin=64,eta=.05,min_child_weight=20,subsample=.8,colsample_bytree=1.,reg_lambda=1.,seed=20261105,nthread=1)

def literal(value):
    return "'"+str(value).replace("'","''")+"'"

def save_json(path,data):
    path=Path(path)
    tmp=path.with_suffix(path.suffix+".tmp")
    def convert(value):
        if hasattr(value,"item"):
            return value.item()
        return str(value)
    tmp.write_text(json.dumps(data,indent=2,default=convert,allow_nan=False)+"\n")
    os.replace(tmp,path)

def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def peak_rss_gib():
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(1024**3 if sys.platform=="darwin" else 1024**2)

def folds(start_date=EVALUATION_START,end_date=EVALUATION_END):
    result=[]
    for start in pd.date_range(start_date,end_date,freq="QS",inclusive="left"):
        result.append(dict(fold_id=f"{start.year}Q{start.quarter}",
            train_start=(start-pd.DateOffset(months=12)).date().isoformat(),
            inner_start=(start-pd.DateOffset(months=2)).date().isoformat(),
            test_start=start.date().isoformat(),
            test_end=min(start+pd.DateOffset(months=3),pd.Timestamp(end_date)).date().isoformat(),
            planned_test_end=(start+pd.DateOffset(months=3)).date().isoformat(),
            partial_block=start+pd.DateOffset(months=3)>pd.Timestamp(end_date)))
    return result

def build_labels(con):
    """Attach exact-tie return deciles to the observed forecast cohort."""
    con.execute("""CREATE TEMP TABLE labels AS
      WITH ranked AS (
        SELECT security_id,signal_date,trade_date,
          rank() OVER (PARTITION BY trade_date ORDER BY future_oc_return DESC) rank_low,
          count(*) OVER (PARTITION BY trade_date,future_oc_return) tie_count,
          count(*) OVER (PARTITION BY trade_date) n
        FROM examples WHERE outcome_status='ok' AND isfinite(future_oc_return)
      )
      SELECT security_id,signal_date,
        CASE WHEN n>=10 THEN least(10,1+(10*(2*rank_low+tie_count-2))//(2*n)) END
          realized_class
      FROM ranked
    """)

def read_fold(run,fold):
    con=duckdb.connect()
    con.execute("SET threads=2")
    frame=con.execute("""SELECT * FROM read_parquet(?)
      WHERE trade_date>=CAST(? AS DATE) AND trade_date<CAST(? AS DATE)
      ORDER BY trade_date,security_id""",
      [str(run/"supervised_panel.parquet"),fold["train_start"],fold["test_end"]]).df()
    con.close()
    training=frame[frame.trade_date.lt(pd.Timestamp(fold["test_start"]))
                   & frame.realized_class.notna()].copy()
    testing=frame[frame.trade_date.ge(pd.Timestamp(fold["test_start"]))].copy()
    if training.trade_date.nunique()<100:
        raise ValueError("Fewer than 100 supervised training sessions.")
    if not training.trade_date.lt(pd.Timestamp(fold["test_start"])).all():
        raise ValueError("Future target admitted to training.")
    require_past_training(training,testing,fold["test_start"])
    return training,testing

def class_payoffs(training, expected_dates=None):
    """Equal-weight stocks within true class/day, then equal-weight complete days."""
    daily=training.pivot_table(index="trade_date",columns="realized_class",
                               values="future_oc_return",aggfunc="mean")
    if expected_dates is not None:
        daily=daily.reindex(pd.DatetimeIndex(expected_dates, name="trade_date"))
    daily=daily.reindex(columns=range(1,11))
    complete=daily.notna().all(axis=1)
    if complete.sum()<100:
        raise ValueError("Fewer than 100 training dates with all ten class payoffs.")
    mu=daily.loc[complete].mean(axis=0).to_numpy(dtype=np.float64)
    if not np.isfinite(mu).all():
        raise ValueError("Undefined class payoffs.")
    return mu,daily,complete

def select_portfolio_indices(scores,security_ids,n):
    """One outcome-blind selector for fitting diagnostics and final accounting."""
    if n not in (2,3):
        raise ValueError("The approved portfolios use 2 or 3 stocks per side.")
    scores=np.asarray(scores,dtype=float)
    ids=np.asarray(security_ids)
    if scores.ndim!=1 or ids.ndim!=1 or len(scores)!=len(ids) or len(scores)<2*n:
        raise ValueError("Invalid portfolio candidates or insufficient forecasts.")
    if not np.isfinite(scores).all() or pd.isna(ids).any():
        raise ValueError("Finite scores and security IDs are required.")
    ids=ids.astype(str)
    if len(np.unique(ids))!=len(ids):
        raise ValueError("Duplicate security IDs within a trading date.")
    order=np.lexsort((ids,-scores))
    return order[:n],order[-n:]


def portfolio_days(predictions,calendar,n):
    """Select using scores only, then attach outcomes; never replace missing exits."""
    if n not in (2,3):
        raise ValueError("The approved portfolios use 2 or 3 stocks per side.")
    if calendar.trade_date.duplicated().any() or predictions.duplicated(["security_id","trade_date"]).any():
        raise ValueError("Duplicate dates or stock/date forecasts.")
    if not np.isfinite(predictions.ret_score.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite ranking score; never silently drop forecasts.")
    daily,positions=[],[]
    grouped={pd.Timestamp(d):f for d,f in predictions.groupby("trade_date",sort=False)}
    for date in pd.to_datetime(calendar.trade_date):
        frame=grouped.get(date)
        if frame is None or len(frame)<2*n:
            daily.append(dict(trade_date=date,n_per_side=n,pnl_status="cash_insufficient_forecasts",
                gross_return=0.0,trade_notional=0.0,selected_positions=0,missing_positions=0))
            continue
        li,si=select_portfolio_indices(frame.ret_score,frame.security_id,n)
        long=frame.iloc[li].copy()
        short=frame.iloc[si].copy()
        long["weight"]=0.5/n
        short["weight"]=-0.5/n
        chosen=pd.concat([long,short],ignore_index=True)
        chosen["n_per_side"]=n
        chosen["side"]=np.where(chosen.weight>0,"long","short")
        chosen["contribution"]=chosen.weight*chosen.future_oc_return
        positions.append(chosen)
        known=chosen.future_oc_return.notna() & np.isfinite(chosen.future_oc_return)
        missing=int((~known).sum())
        daily.append(dict(trade_date=date,n_per_side=n,
            pnl_status="observed" if not missing else "unresolved_selected_outcome",
            gross_return=float(chosen.contribution.sum()) if not missing else np.nan,
            trade_notional=float((chosen.weight.abs()*(2+chosen.future_oc_return)).sum())
                if not missing else np.nan,
            selected_positions=2*n,missing_positions=missing))
    return pd.DataFrame(daily),pd.concat(positions,ignore_index=True) if positions else pd.DataFrame()

def performance(returns):
    values=np.asarray(returns,dtype=float)
    if len(values)<2 or not np.isfinite(values).all():
        raise ValueError("Metrics require an explicitly complete observation sample.")
    mean=float(values.mean())
    sd=float(values.std(ddof=1))
    equity=np.cumprod(1+values)
    peak=np.maximum.accumulate(np.r_[1.0,equity])[1:]
    return dict(sessions=len(values),annual_mean=mean*252,
        annual_vol=sd*np.sqrt(252),sharpe=mean/sd*np.sqrt(252) if sd>0 else None,
        total_return=float(equity[-1]-1),
        annual_compound=float(equity[-1]**(252/len(values))-1) if equity[-1]>0 else None,
        max_drawdown=float(np.max(1-equity/peak)),win_fraction=float((values>0).mean()))

def summarize(run, *, score_type="ret"):
    if score_type not in ("ret","ranking"):
        raise ValueError("Declare RET return probabilities or uncalibrated ranking scores.")
    paths=[run/"folds"/f["fold_id"]/"predictions.parquet" for f in folds()]
    predictions=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    if predictions.duplicated(["security_id","trade_date"]).any():
        raise ValueError("Overlapping outer WFO predictions.")
    calendar=pd.read_parquet(run/"wfo_calendar.parquet")
    books,positions=[],[]
    for n in (2,3):
        days,holdings=portfolio_days(predictions,calendar,n)
        books.append(days)
        positions.append(holdings)
    daily=pd.concat(books,ignore_index=True)
    positions=pd.concat(positions,ignore_index=True)
    wide=daily.pivot(index="trade_date",columns="n_per_side",values="gross_return")
    common=wide.notna().all(axis=1)
    metrics=[]
    for n in (2,3):
        book=daily[daily.n_per_side.eq(n)].set_index("trade_date").sort_index()
        usable=book.loc[common[common].index]
        unknown=int(book.gross_return.isna().sum())
        for cost in (0,1,5,10):
            net=usable.gross_return-usable.trade_notional*cost/10000
            record=dict(n_per_side=n,cost_bps_per_side=cost,
                metric_scope="full_walk_forward" if not unknown and common.all()
                    else "paired_complete_days_conditional",
                calendar_sessions=len(book),unresolved_days=unknown,
                common_omitted_days=int((~common).sum()),
                cash_days=int(book.pnl_status.eq("cash_insufficient_forecasts").sum()),
                **performance(net))
            metrics.append(record)
            daily.loc[daily.n_per_side.eq(n),f"net_return_{cost}bps"]= (
                book.gross_return-book.trade_notional*cost/10000).to_numpy()
    metrics=pd.DataFrame(metrics)
    # A complete equity curve is available only if every selected outcome is known.
    daily=daily.sort_values(["n_per_side","trade_date"])
    for n in (2,3):
        idx=daily.index[daily.n_per_side.eq(n)]
        for cost in (0,5):
            r=daily.loc[idx,f"net_return_{cost}bps"]
            cumulative=(1+r).cumprod(skipna=False)
            daily.loc[idx,f"equity_{cost}bps_until_first_unresolved"]=cumulative
    fold_metrics=[]
    for (n,quarter),book in daily.groupby(["n_per_side",daily.trade_date.dt.to_period("Q")]):
        known=book.gross_return.notna()
        rec=dict(n_per_side=n,quarter=str(quarter),calendar_sessions=len(book),
            unresolved_days=int((~known).sum()),cash_days=int(book.selected_positions.eq(0).sum()))
        if known.sum()>1:
            rec.update(performance(book.loc[known,"net_return_5bps"]))
        fold_metrics.append(rec)
    observed=predictions[predictions.realized_class.notna()].copy()
    if score_type=="ret":
        y=observed.realized_class.astype(int).to_numpy()-1
        probs=observed[[f"p_class_{k}" for k in range(1,11)]].to_numpy()
        logloss=float(-np.log(np.maximum(probs[np.arange(len(y)),y],1e-12)).mean())
        brier=float(np.mean(np.sum((probs-np.eye(10)[y])**2,axis=1)))
        accuracy=float((probs.argmax(axis=1)==y).mean())
    else:
        if "ranking_score" not in predictions or not np.array_equal(predictions.ret_score,predictions.ranking_score):
            raise ValueError("Ranking scores must be explicitly marked; legacy ret_score is an accounting alias, not a return forecast.")
        logloss=brier=accuracy=None
    daily_ic=[]
    for date,frame in predictions[predictions.future_oc_return.notna()].groupby("trade_date"):
        if frame.ret_score.nunique()>1 and frame.future_oc_return.nunique()>1:
            daily_ic.append(dict(trade_date=date,rank_ic=frame.ret_score.corr(frame.future_oc_return,method="spearman")))
    ic_frame=pd.DataFrame(daily_ic,columns=["trade_date","rank_ic"])
    classification=dict(predictions=len(predictions),labelled_predictions=len(observed),
        logloss=logloss,uniform_probability_logloss=float(np.log(10)) if score_type=="ret" else None,brier=brier,
        accuracy=accuracy,
        mean_daily_rank_ic=float(ic_frame.rank_ic.mean()) if len(ic_frame) else None)
    predictions.to_parquet(run/"predictions.parquet",index=False)
    positions.to_parquet(run/"positions.parquet",index=False)
    daily.to_parquet(run/"portfolio_daily.parquet",index=False)
    metrics.to_parquet(run/"portfolio_comparison.parquet",index=False)
    pd.DataFrame(fold_metrics).to_parquet(run/"portfolio_by_quarter.parquet",index=False)
    ic_frame.to_parquet(run/"daily_rank_ic.parquet",index=False)
    save_json(run/"RESULTS.json",dict(status="complete",classification=classification,score_type=score_type,
        evaluation_start=EVALUATION_START,evaluation_end_exclusive=EVALUATION_END,
        comparison=metrics.to_dict("records"),
        limitations=[
            "Research P&L uses five-minute endpoint proxies, not auction fills or demonstrated tradability.",
            "No quotes, short locates/borrow availability or actual fees supplied. Cost rates are sensitivity assumptions, not measured execution costs.",
            "Historical membership timing and retrospective adjusted-price vintages remain provisional.",
            "Missing selected outcomes remain unresolved; any complete-day metrics are conditional and do not certify full-period portfolio returns.",
            "Fixed-parameter 100-model experiment; previously viewed history is development research, not a separately reserved final holdout.",
        ]))

def terminate_owned(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid,signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL)
            process.wait(timeout=5)
        except ProcessLookupError:
            pass
