"""
Deep Momentum — Metrics & diagnostics.

Paper reference: Sections 3.2, 4.1, 4.2.1.

Functions:
  compute_confusion_matrix(predictions)
        10x10 matrix of (predicted class, actual class) counts over the full
        OOS cross-section. Replaces the per-strategy P&L view: this measures
        the model's classification quality before any 15L/15S selection is
        applied — i.e., the paper's Section 4.2.1 numbers.

  compute_classification_accuracy(predictions)
        Overall accuracy, per-class precision and recall, and the share of
        predictions falling in each class. Same paper convention.

  compute_bimodality(...)
        Section 3.2 / Table 2: HH, HL, LL, LH, BM measure of how much past
        winners/losers oscillate into the OPPOSITE future class.

  compute_crash_rate(portfolio_df, dd_threshold=0.20)
        Section 4.1: fraction of years where MDD > 20%, averaged over 12
        starting-month shifts.

All functions assume the new schema:
  predictions: assetid, date_mt, xgb_class, fwd_return_mt, prob_1..prob_10
  features:    assetid, date_mt, fwd_return_mt, LABEL_mt
  portfolio:   date_mt, ls_ret
"""

import numpy as np
import pandas as pd

N_CLASSES = 10  # paper's number of return-class deciles


# ─── Confusion matrix (paper Section 4.2.1) ──────────────────────────────────

def compute_confusion_matrix(predictions: pd.DataFrame,
                              pred_col: str = "xgb_class",
                              actual_col: str = "LABEL_mt") -> pd.DataFrame:
    """
    10x10 confusion matrix of (predicted class, actual class) counts.

    NOTE: this is computed across ALL stock-months in the OOS predictions
    panel, NOT just the 15L/15S strategy picks. It tells you how well the
    XGBoost classifier separates the 10 future-return deciles overall.

    The diagonal entries (predicted == actual) are correct calls. Random
    baseline = 1/10 = 10% per cell. The strategy's L/S P&L only depends on
    the corner cells (1,1) and (10,10), but the full matrix tells you whether
    the classifier is genuinely discriminating across the whole distribution
    or just lucky on the extremes.
    """
    df = predictions.dropna(subset=[pred_col, actual_col]).copy()
    if df.empty:
        return pd.DataFrame()

    df[pred_col]   = df[pred_col].astype(int)
    df[actual_col] = df[actual_col].astype(int)

    cm = pd.crosstab(df[pred_col], df[actual_col],
                      rownames=["predicted"], colnames=["actual"],
                      dropna=False)
    # Force the 10x10 grid even if some classes have no observations
    cm = cm.reindex(index=range(1, N_CLASSES + 1),
                     columns=range(1, N_CLASSES + 1),
                     fill_value=0)
    return cm


def compute_classification_accuracy(predictions: pd.DataFrame,
                                     pred_col: str = "xgb_class",
                                     actual_col: str = "LABEL_mt") -> dict:
    """
    Per-class precision and recall + overall accuracy. Paper Section 4.2.1.

    Precision_k = P(actual == k | predicted == k)   — out of stocks the model
                                                       called class k, what
                                                       fraction were really k?
    Recall_k    = P(predicted == k | actual == k)   — out of stocks that ARE
                                                       class k, what fraction
                                                       did the model catch?
    Random baseline = 0.10 for both (1/10 classes).
    """
    cm = compute_confusion_matrix(predictions, pred_col, actual_col)
    if cm.empty:
        return {}

    total = cm.values.sum()
    correct = np.trace(cm.values)
    accuracy = correct / total if total > 0 else 0.0

    precision = {}
    recall = {}
    pred_share = {}
    for k in range(1, N_CLASSES + 1):
        col_pred_k = cm.loc[k, :].sum()        # how many we predicted as k
        row_actual_k = cm.loc[:, k].sum()       # how many were actually k
        tp = cm.loc[k, k]
        precision[k] = tp / col_pred_k if col_pred_k > 0 else 0.0
        recall[k]    = tp / row_actual_k if row_actual_k > 0 else 0.0
        pred_share[k] = col_pred_k / total if total > 0 else 0.0

    return {
        "accuracy":   accuracy,
        "precision":  precision,           # dict {k: prec_k}
        "recall":     recall,              # dict {k: recall_k}
        "pred_share": pred_share,          # dict {k: share predicted as k}
        "n_obs":      int(total),
        # Paper-style summary aliases for H = class N_CLASSES, L = class 1
        "precision_H": precision[N_CLASSES],
        "precision_L": precision[1],
        "recall_H":    recall[N_CLASSES],
        "recall_L":    recall[1],
        "pred_ratio_H": pred_share[N_CLASSES],
        "pred_ratio_L": pred_share[1],
    }


def print_confusion_matrix(cm: pd.DataFrame, normalize: bool = False) -> None:
    """Pretty-print the 10x10 matrix. If normalize=True, divide each row by its sum."""
    if cm.empty:
        print("(empty)")
        return
    if normalize:
        m = cm.div(cm.sum(axis=1).replace(0, 1), axis=0)
        fmt = lambda v: f"{v:>6.1%}"
    else:
        m = cm
        fmt = lambda v: f"{v:>6d}"

    header = "pred\\actual"
    print(f"{header:<13s}" + "".join(f"{c:>7d}" for c in m.columns))
    for r in m.index:
        print(f"{r:>13d}" + "".join(fmt(m.loc[r, c]) for c in m.columns))


# ─── Bimodality (paper Section 3.2 / Eq. 1-5) ────────────────────────────────

def compute_bimodality(panel: pd.DataFrame,
                        pred_col: str = "xgb_class",
                        actual_col: str = "LABEL_mt") -> dict:
    """
    Cross-sectional bimodality of past winners / losers.

    HH = TP / (TP + FP) - 0.1     precision of predicted winners
    HL = FP / (TP + FP) - 0.1     fraction of predicted winners who were losers
    LL = TN / (TN + FN) - 0.1     precision of predicted losers
    LH = FN / (TN + FN) - 0.1     fraction of predicted losers who were winners
    BM = -(HH - HL + LL - LH) / 2

    Higher BM (less negative) = more bimodality (winners reverse, losers reverse).
    """
    df = panel.dropna(subset=[pred_col, actual_col]).copy()
    if df.empty:
        return {}

    rows = []
    for ym, grp in df.groupby(df["date_mt"].dt.to_period("M")):
        if len(grp) < N_CLASSES:
            continue
        pred_w = grp[grp[pred_col] == N_CLASSES]
        pred_l = grp[grp[pred_col] == 1]
        if pred_w.empty or pred_l.empty:
            continue
        tp = (pred_w[actual_col] == N_CLASSES).sum()
        fp = (pred_w[actual_col] == 1).sum()
        tn = (pred_l[actual_col] == 1).sum()
        fn = (pred_l[actual_col] == N_CLASSES).sum()
        n_w, n_l = len(pred_w), len(pred_l)
        if n_w == 0 or n_l == 0:
            continue
        hh, hl = tp / n_w - 0.1, fp / n_w - 0.1
        ll, lh = tn / n_l - 0.1, fn / n_l - 0.1
        bm = -((hh - hl) + (ll - lh)) / 2
        rows.append({"HH": hh, "HL": hl, "LL": ll, "LH": lh, "BM": bm})

    if not rows:
        return {}
    res = pd.DataFrame(rows).mean()
    return res.to_dict()


# ─── Crash rate (paper Section 4.1) ──────────────────────────────────────────

def compute_crash_rate(portfolio_df: pd.DataFrame,
                       dd_threshold: float = 0.20,
                       date_col: str = "date_mt",
                       ret_col: str = "ls_ret") -> float:
    """Fraction of 12-month windows with MDD > dd_threshold, averaged over 12 starting-month shifts."""
    if portfolio_df.empty or len(portfolio_df) < 12:
        return float("nan")

    ret = portfolio_df[ret_col].values
    crash_rates = []
    for shift in range(12):
        shifted = ret[shift:]
        if len(shifted) < 12:
            continue
        n_years = len(shifted) // 12
        crash_count = 0
        for y in range(n_years):
            year_ret = shifted[y * 12:(y + 1) * 12]
            cum = np.cumprod(1 + year_ret)
            peak = np.maximum.accumulate(cum)
            mdd = ((cum - peak) / peak).min()
            if mdd < -dd_threshold:
                crash_count += 1
        if n_years > 0:
            crash_rates.append(crash_count / n_years)
    return float(np.mean(crash_rates)) if crash_rates else float("nan")
