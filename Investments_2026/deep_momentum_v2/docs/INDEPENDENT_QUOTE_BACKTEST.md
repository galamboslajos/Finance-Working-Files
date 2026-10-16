# Independent quote repair and balanced baskets — October 16, 2026

USER DIRECTION: repair the remaining quote gaps using independent sources and
finish the full January 2018–December 2019 development backtest. Preserve the
original runs. This experiment uses the exact four already fitted 2010–2017
models; it does not train on the validation period or evaluate 2020–2023.

## Frozen scope and preservation

`configs/independent_backtest.json` declares the execution rule before the new
strategy results. The parent reconstruction is
`outputs/data_limited_backtest/2010-2019-50aa6e09f11b687c`, identity
`sha256:50aa6e09f11b687c36cd3cf52ed8eb51fdde591fb2391cd58162d02edbb29b10`.
New results use `outputs/independent_backtest/2018-2019-<identity>/`.
The parent source files, outputs, configs and pointers are hash checked and
preserved. No source prices or old outputs are overwritten.

Only 2018 and 2019 feature/target annual panels are rebuilt with the new quote
overlay. Daily indicators and historical cross-sectional context are recomputed
with the unchanged exact lookbacks, original membership denominators, 80%
context rule, Paper15 formulas and provisional identity admissions. The daily
tables remain separate. BBBY's demonstrated wrong route remains quarantined;
its replacement history remains unresolved. No additional stocks are excluded.

All 2010–2017 fit inputs, training matrices, medians, missing indicators, schemas
and four saved estimators are authenticated without numerically rebuilding or
refitting them. New validation matrices use the original training-only states.
Predictions retain eligible rows even when their future label is missing.
Classification classes are rebuilt using the original shared target worker;
labels never enter forecasts or basket selection. A corrected interior bar can
change daily indicators and the shared context later, so merely patching the
execution table would be insufficient.

## Independent quote evidence

The preflight reviews the entire cached validation input, rather than choosing
repairs from profitable paths: 69 existing quality-flagged OHLC rows and 34
interior missing native-ID/session keys across 2018–2019. An interior key lies
between that native security's first and last observed cached dates; it is a
review candidate, not proof that trading occurred. Suspensions and terminal
events require their own evidence. No future completeness filter changes
signal-time membership or prediction eligibility.

The independent raw SIP publication and its exact bounded projections are
pinned in the private evidence package. Source ID, historical symbol, dates,
feed and adjustment metadata are checked before exact requested financial rows
are returned. Parquet column chunks can transiently decode mixed dates in
Arrow; excluded values are not returned, converted to Python or analyzed.
The recorded read receipt distinguishes actual range bytes from a whole-object
hash, which is not claimed when the source is range read.

A replacement must pass the predeclared source and neighboring-control checks.
Rejected original values are retained in amendment hashes/audit evidence;
original source volume is not silently replaced. Missing actual quotes, identity
conflicts or incompatible adjustment units remain explicit refusals, with no
stale/zero/interpolated prices and no invented terminal payout. The same
compiled overlay is applied to feature history, monthly endpoints and execution.
The original twelve reviewed amendments remain intact beneath the additive
new bundle. Exact accepted/refused counts belong in the saved evidence ledger.

## SNI is a corporate conversion

VERIFIED PRIMARY EVIDENCE: the transaction legally closed March 6, 2018; Nasdaq
suspended SNI on March 7. The default consideration was **$65.82 cash plus
1.0584 Discovery Series C shares (DISCK)** per old SNI share. DISCA Series A is
the wrong class for this entitlement. See the
[Discovery closing 8-K](https://www.sec.gov/Archives/edgar/data/1437107/000143710718000042/form8-kxclosing20180306.htm),
[SNI closing 8-K](https://www.sec.gov/Archives/edgar/data/1430602/000119312518072063/d545990d8k.htm)
and [Nasdaq suspension notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2018-37).

VERIFIED LOCAL: DISCK native ID `sec_gA9OJg` has an accepted March 7 opening
quote in the already requested 2018 cache. The additive ninth action recognizes
signed cash and successor shares for existing long/short holdings. It preserves
the original eight action terms, actual paired raw/adjusted units and continuing
stop basis. It does not invent post-merger SNI quotes or alter the source monthly
endpoint label definition. Actual broker settlement and fractional cash prices
are unverified; the unchanged engine retains theoretical fractional shares.

## Why nine plus nine can be necessary

Actual target classes are approximately decile balanced. Predicted argmax
classes have unconstrained counts. In the original RF/Paper15 class 2/9
September 28, 2018 signal (October 1 execution), the permitted trading cohort
had nine class-2 long candidates and 37 class-9 short candidates. A strict
ten-per-side rule therefore withheld that path, regardless of quote coverage.

V2 ADAPTATION, declared before this rerun:

```
N = min(10, eligible_long_mode_candidates, eligible_short_mode_candidates)
long weight per selected name  = +0.5 / N
short weight per selected name = -0.5 / N
```

The two sides have the same number of names. Normally this is 10+10 at 5% per
name; the known shortage becomes 9+9 at about 5.56% per name. This preserves
50% long, 50% short and 100% gross exposure at rebalance, with slightly greater
concentration. It does not pool classes 1/2 or 9/10, relax class membership, add
unpredicted securities or select by future returns/quotes. Within each separate
mode class, rank by its probability descending and source ID ascending for ties.
If either side has zero candidates, the continuous path is withheld.

The original ledger and denominators remain complete. Dean Foods and Rowan
retain the prior fixed, retrospective execution restriction before ranking;
valid historical training and prediction admissions stay unchanged. Every
monthly basket is validated before accounting. Retired new orders can remain
unfilled cash slots under the existing event rule, and stops can reduce exposure
within a month; neither is replaced by another stock or silently reweighted.

## Interpretation and repeatability

The same 64 scenarios cover four arms × two distinct class pairs × unstopped or
10% per-asset stop × 0/5/10/25 bp one-way trading cost. There are 24 prediction
months and 503 XNYS sessions from January 2, 2018 through December 31, 2019.
RET's original 120-month payoff gate remains unavailable for these signals.
An unexplained held valuation gap retains the exposure and withholds complete
period return, volatility, drawdown and Sharpe, rather than displaying a prefix.

These are conditional development simulations. The fixed retrospective DF/RDC
restriction and unresolved identities mean that full-universe survivorship bias
has not been removed. Adjusted-price transport, fractional settlement, borrow,
funding and daily-bar stop fills remain provisional. The 10% asset stop is not
a guaranteed portfolio loss cap. Do not select a model solely by the completed
subset or use these results as an untouched outer test.

`independent_rebuild.py` provides explicit `plan`, `build`, `predict` and
`backtest` commands. Run each numerical phase in a fresh, single-thread process
with the verified locally compiled quote bundle. The master notebook's new saved
report section performs bounded receipt/report reads only; Run All never builds,
fetches, deserializes a model, refits or backtests.

Result-specific counts, repair refusals and full-period metrics are recorded in
the new run's private audit and final report after execution.

## Additive terminal-event completion

The first quote-repair rerun completed 56 of 64 paths. Its eight XGBoost expanded
class 2/9 paths first encountered VVC's February 1 retirement. A data-only check
of every selected predecessor throughout the full holding horizon also found
TSS's September 18 retirement. These require legal entitlements, rather than
post-retirement quotes. The quote bundle remains exactly the same.

VERIFIED PRIMARY EVIDENCE: VVC holders immediately before the February 1, 2019
merger receive $72 cash plus a $0.41145 closing dividend per raw share. VVC was
suspended before that day's NYSE open. All 21 January cached observations have
raw OHLC equal to adjusted OHLC and zero recorded dividends; the closing
dividend is therefore recognized once in the $72.41145 combined entitlement.
The dividend's documented payment deadline was February 8; actual broker
settlement is unverified. Recognition at legal entitlement follows the existing
zero-funding-cost accounting convention. Sources: [VVC closing 8-K](https://www.sec.gov/Archives/edgar/data/1096385/000119312519025987/d699426d8k.htm),
[CenterPoint closing 8-K](https://www.sec.gov/Archives/edgar/data/1130310/000119312519025980/d682591d8k.htm).

VERIFIED PRIMARY EVIDENCE: TSS converts to 0.8101 Global Payments common shares
per old share, with no ordinary cash consideration. The TSS closing filing
reports legal consummation September 17 and suspension before the September 18,
2019 open. Announcement dates differ; retain the legal and application dates
separately. Use the cached correctly identified GPN opening quote and its actual
paired raw/adjusted units. Actual fractional-share cash settlement remains
unverified; the existing theoretical fractional-share convention is retained.
Source: [TSS closing 8-K](https://www.sec.gov/Archives/edgar/data/721683/000119312519250774/d803104d8k.htm).

`configs/independent_backtest_completion.json` freezes this separate execution
phase. `completed_backtest.py` writes `backtest_completed/` beneath the same
repaired feature/inference run, preserving the initial 56/8 backtest and every
forecast. All selected/rebalance/original-ledger artifacts must match the prior
run byte for byte. The original nine terms are unchanged, and exactly these two
verified events are appended. No features, labels, predictors, models, ranking,
fees or stop rules change. Retired new orders retain unfilled cash slots.
The same 64 scenarios are rerun once with the eleven-action ledger.

The final passive notebook uses `completed_reporting.read_completed_run` and
four explicit audit receipts, including the preserved prior backtest receipt.
