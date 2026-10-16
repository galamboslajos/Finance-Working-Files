# Daily development backtester: 2018–2019

USER DIRECTION, 2026-10-15: build the daily development backtester using the
saved four-arm pilot probabilities. Keep RET as the intended primary score,
its complete 120-calendar-month gate intact, and 2020–2023 unevaluated. Run
separate class 1/10 and class 2/9 baskets first. The user confirmed 50% long /
50% short, rejected holding the full mode cohorts, suggested ten names per
direction, and selected a separate **10% stop per asset**.

The concrete fixed-ten convention and execution assumptions below are frozen
in [daily_backtest.json](../configs/daily_backtest.json), before portfolio P&L.
They are V2 development conventions, separate from the paper's all-mode-member
portfolios and unit-long minus unit-short capital convention.

## Selection and chronology

Each of RF/Paper15, XGBoost/Paper15, RF/expanded715 and XGBoost/expanded715
reuses its existing probabilities. No preprocessing, calibration or model is
refitted. These are the frozen **2010–2017 fit / 2018–2019 validation** models;
this backtest does not turn that eight-year pilot fit into a ten-year fit.
The later rolling ten-year annual workflow remains unchanged.

- Class 1/10: among argmax-class-1 stocks, take the ten largest p1; among
  argmax-class-10 stocks, take the ten largest p10.
- Class 2/9: apply that rule independently to argmax classes 2 and 9 using p2/p9.
- Argmax ties retain the existing lowest-canonical-class-ID convention.
  Boundary confidence ties resolve by ascending stable source ID. Keep the
  candidate counts, cutoff probability and equal-cutoff counts.
- Exactly ten names per side, each allocated 5% of pre-cost portfolio NAV.
  Fewer than ten candidates on either side withholds the rebalance and the
  continuous path. Do not supplement candidates from another class.
- All original cohort keys, provisional identity flags and missing target
  states remain in coverage. Selection accesses no future targets or prices.
  Quote failure never leads to a replacement name.

RET remains withheld for all 24 development signals: the existing class-return
history starts in January 2010 and cannot supply the required prior 120 months.
No earlier payoff extension, shortened window, artificial class-number payoff
or 2020 score is substituted. The generic RET selector accepts only canonical
verified `RetPayoffPanel` objects; its future fixed-ten score selection would
also be a V2 breadth adaptation, rather than the paper's score deciles.

## Daily cash and units

Signals are independent XNYS month ends, Dec 2017–Nov 2019. Execution is the next
month's first XNYS adjusted open. Keep existing units through the overnight
before each rebalance, value them at that open, then trade to new equal weights
using **pre-cost, pre-trade NAV**. Retained names can have small balancing trades;
monthly rebalancing does not imply replacing every stock.

The ledger begins with cash/NAV 1.0. Long purchases and short-sale proceeds
enter cash with their signs; NAV equals cash plus signed position values.
Units drift between trades. Record open and close marks separately and compute
daily performance from **one close per session**, never annualize the two
event marks as two trading days. Final liquidation is December 31, 2019 adjusted close,
an explicit finite-window exception that needs no January 2020 quote.

These are provider-adjusted **proxy units**, not executable raw-price shares.
Do not add source dividends/splits again. Cash interest, funding and borrow
charges are zero; borrowing is assumed. Cost scenarios charge 0/5/10/25 basis
points on absolute one-way actual traded notional at entry, net rebalances,
stops and final liquidation. They are sensitivities, not measured implementation
costs. A zero-cost return is gross; positive-cost returns still omit actual
borrowing, impact and execution slippage.

## Separate asset-stop alternative

The unstopped book is retained. The alternative places a nontrailing stop at
90% of monthly entry reference for longs, and 110% for shorts. Reference resets
at each monthly rebalance open, including retained names, to define a new
monthly holding episode. Old stops are cancelled at that scheduled open.

On later sessions, a gap beyond the stop fills at the actual adjusted open.
Otherwise an adverse adjusted daily low/high crossing fills at the threshold
under an **idealized resting-stop assumption**. This does not infer an intraday
path or guarantee a real fill. Charge the same notional cost. After stopping,
leave the proceeds in cash until the next scheduled monthly rebalance; no
replacement, reweighting of survivors or within-month reentry occurs.

Ten percent per asset is not a ten percent portfolio loss guarantee. At entry,
one 5%-weight holding losing 10% contributes 0.5% of portfolio NAV before costs;
gaps, short-price jumps and execution can produce larger losses. The
[FINRA stop-order notice](https://www.finra.org/sites/default/files/notice_other_file_ref/Regulatory-Notice-16-19.pdf)
explains why a stop price does not guarantee execution at that price.

## Coverage and economic limits

Missing/invalid selected entry or held valuation fails the continuous path
closed from that event onward. Preserve intended selections, positions, missing
keys and expected daily marks. Never drop a holding, substitute stale/zero
returns, renormalize surviving names or restart after a gap. Full-period
performance is null for failed paths; valid-prefix diagnostics are coverage,
not a complete Sharpe or full-period return. A stop genuinely filled earlier
can remove an exposure; future missingness cannot retroactively create a stop.

Historical membership and accepted provisional identity rules are unchanged.
Knowledge-time identities and provider adjustments remain unverified. The
release lacks a complete terminal-payout/delisting/merger ledger; even complete
proxy quotes do not certify economic continuity, available borrowing,
survivorship freedom or investability.

## Bounded execution and use

The metadata-first plan pins the saved pilot/root/arm audits, each prediction
file, the two annual outcome-only daily caches, and original membership/coverage
metadata. It validates all date/ID metadata before numerical probabilities or
prices. It never loads feature matrices, serialized estimators, raw source
fallbacks or later-year financial values. No cloud operation is involved.

The measured projection is about 21 MiB across ten Parquet files, including
454,580 selected daily records and all 21,827 original predictions per arm.
Limits: one million candidate rows, 128 MiB projected compressed bytes, 2 GiB peak
RSS, 15 minutes, 512 MiB output. Integrity hashing is additional I/O, not numerical
decoding. There are 64 fixed scenarios: four arms × two baskets × two stop
variants × four cost assumptions. No search or performance-based reselection.

From V2:

```sh
PYTHONPATH=src .venv/bin/python -B -m dmv2.daily_backtest plan
PYTHONPATH=src .venv/bin/python -B -m dmv2.daily_backtest run
PYTHONPATH=src .venv/bin/python -B -m dmv2.daily_backtest report
```

An identical complete run is verified and reused; an incomplete same-identity
run is refused for review. Private immutable output includes original selection
ledgers, selected names, rebalance schedules, daily open/close NAV, trades,
holdings, stops, failures, monthly returns, cost comparisons, read plan and
source hashes under ignored `outputs/daily_backtest/`. The saved report reader
checks hashes and only loads small reports.

Section 18 of the master notebook displays these reports. Run All does not run
the backtest or fit anything. Its pre-existing daily/monthly/feature tables
remain independent. Generated research outputs stay outside Git.


## First recorded execution, 2026-10-15

Execution completed under the frozen protocol in 34.47 seconds, peak RSS 663.52 MiB,
with all 19 pinned input/manifest files unchanged. All 64 scenarios encountered
a missing or rejected held valuation and correctly withheld continuous
full-period returns, volatility, Sharpe and drawdown. This is a completed
engineering run with incomplete economic paths, not a portfolio success claim.

The distinct first failures across scenarios involve eight provider/date keys:
five missing quotes and three incomplete/anomalous OHLC bars. Some stop paths
genuinely exited before an earlier baseline failure and encountered a later
failure; those stops never use future missingness to select names. These are
first failures, not an exhaustive inventory of all later data issues. Exact
IDs, dates, selected holdings and original bars remain in private failure and
source ledgers. Resolve those economic/source cases before presenting a complete
2018–2019 performance comparison; do not relax checks or drop the positions
using future coverage.

All 24 × 8 basket selections have ten names on each side. All 21,827 original rows
and 20,755 eligible predictions per model arm remain represented. RET is withheld
in all 24 months. No 2020–2023 financial values, raw source fallback or new model
fit entered the run. The source initialization, identity policy, 715 features and
original daily caches remain unchanged.

All 755 worker tests passed. Independent synthetic accounting replay reconciled
64 cases, 512 marks and 448 trades. Saved actual-run verification and fresh
notebook report-cell checks are recorded separately below once complete.

Independent actual-run verification passed: all 64 saved scenarios, 24,888 valid
prefix marks and 21,736 trades reconciled; all 3,840 selected names matched an
independent probability ranking. All 2,164 recorded stop reference/fill cases
matched monthly entry opens and daily OHLC. Every failed path remained null
after its first failure, without further trades. All 24 RET states and all
full-period performance withholdings matched the protocol. No fitting or 2020+
financial reads occurred. The new section 18 report cell completed in a fresh
kernel with implicit accounting/data preparation disabled, no model imports,
and unchanged separate sample frames. Only that new section was executed;
pre-existing full-period coverage report cells were not rerun. The master
remains output-free. Private receipts and the first-failure evidence queue are
under ignored `outputs/daily_backtest/verification/`.
