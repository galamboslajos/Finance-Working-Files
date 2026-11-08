# Research contract

## Joint experiment authorized 2026-11-06

`joint30` combines all fifteen original momentum/state inputs with the eight
network-paper individual inputs, six relative-volume inputs and current overnight
gap. Duplicate names and full columns identical at float32 model precision are
removed; different normalizations and correlated inputs are retained. The actual
joint panel has thirty distinct columns. It copies all source values, rows,
labels and outcomes unchanged, with no correlation/target-based selection.

The first fit and runtime benchmark cannot start before **15:40 Europe/Vienna
(14:40 UTC) on 2026-11-06**. After that there is no fixed RAM cap, memory-headroom
launch threshold, boosting-round budget or runtime deadline. Ten one-thread CPU
workers remain; validation stops after thirty non-improving rounds. Other learner,
seed, 100-member, 12/3-month, portfolio and missing-data settings are unchanged.
The old two-GiB available-memory restriction described below applies to the
completed historical experiments, not this joint run.

The new experiment remains exploratory. Existing clipping, warm-up, volume
reference, membership-vintage, eligibility and optimistic opening assumptions
are held fixed for comparison. Output lives in
`data/models/09_xgb_ret_joint100_v1`, with a new immutable feature release and
notebook 09. The comparison baseline is completed combined15, on matched dates.

This specifies the completed experiment, not execution readiness or literal
paper replication. Existing outputs are immutable. Changes to transforms,
cohorts, labels, validation or execution need a new version and declared comparison.

## Features and information clocks

Use the pinned Alpaca five-minute member-day release and the dense XNYS regular
session calendar, including early closes and daylight-saving changes. Bars label
their start; modeled availability is start + five minutes. Never compress missing
slots/days, fill prices or substitute later anchors. Join by security ID.

Fourteen inputs use the previous completed close:

| Names | Definition |
| --- | --- |
| `nm_scaled_1h`, `nm_scaled_1d`, `nm_scaled_3d`, `nm_scaled_5d`, `nm_scaled_10d` | End-to-end return / (five-minute exponentially weighted volatility × square root of scheduled horizon slots) |
| `nm_macd_1`, `nm_macd_2`, `nm_macd_3` | Three EMA differences divided by price dispersion and then normalized-MACD dispersion |
| `vol_relative_activity_10d` | Raw dollar activity per scheduled minute / median of available prior-ten-session rates |
| `vol_relative_last_hour_10d` | Last sixty regular minutes' dollar volume / median of available prior-ten closing hours |
| `vol_activity_trend_3v7` | Mean activity per minute in latest three sessions / mean in preceding seven; full ten days required |
| `vol_signed_balance_1d`, `vol_signed_balance_5d` | Signed dollar-volume sum / total dollar volume over one/five sessions |
| `vol_close_vwap_gap` | Previous session close / session VWAP − 1 |

The fifteenth input is `overnight_gap`: adjusted current open / adjusted previous
close − 1. Only the opening field and its source tag enter it; no first-bar high,
low, close, volume or later-day source information. Incomparable provenance stays
NaN, never a zero gap or a dropped forecast.

All volume inputs are dimensionless. Dollar volume is sum(raw bar VWAP × raw
shares); signing uses candle direction, not trade aggressors. There is no extra
volume z-score, cross-sectional ranking, clipping or target winsorization.
Ratios centered on one and balances centered on zero are distinct input types.

Return horizons are sixty trading minutes and 1/3/5/10 exchange sessions.
Multi-session endpoint returns include overnight moves. The two-week limit
concerns return horizons; older causal smoothing memory is allowed.

Momentum mapping is 780/252 paper periods to five-minute periods: volatility span
185.714 bars; EMA scales approximately 24.762/74.286, 49.524/148.571 and
99.048/297.143 bars, alpha = 1/scale. Price/MACD dispersion windows are 195/780
scheduled slots, population dispersion, minimum two observed values. Source
changes reset state. Current-inclusive five-sigma clipping with ten-session daily
EWM half-life is nonbinding; it is not effective tail control.

## Universe and outcomes

Use signal-close S&P 500 membership and resolved nonconflicting IDs. Membership
is retrospective, not certified historical as-known PIT. Never use future exits,
missing labels or realized returns to select forecasts. Member-day-only prices
omit pre-entry/post-exit histories.

Frozen eligibility requires all expected slots in the latest ten sessions, a
valid ten-session-old close, homogeneous source tags and finite original inputs.
Present buckets with 1–4 source minutes are admitted. This gate excludes some
legitimate halts. Feature comparisons kept it fixed. Future strictness should
follow the observations each feature needs and retain row-level exclusions;
blanket tightening/relaxation is a separately versioned cohort change.

The target is next-session first regular bar open to last regular bar close.
Class 1 is highest, class 10 lowest, with exact ties kept together using
average-rank midpoint percentiles. Labels are conditional on observed forecast-
cohort outcomes. Unknown labels cannot train; unknown test outcomes remain
forecasts and selected positions.

## Fitting and portfolio contract

- Rolling 12 calendar months of completed outcomes, then 3 forward months.
  Initial training 2016; forty outer blocks 2017Q1–partial 2026Q4, ending Oct 30.
- Exactly 100 independent random historical 80/20 stock-row splits per quarter.
  Use each 80% fit's validation-best iteration; no full-pool refit. Random inner
  validation is authorized but is not chronological validation.
- Ten-class XGBoost histogram trees, depth 3, eta .05, min_child_weight 20,
  subsample .8, max_bin 64, colsample 1, lambda 1, one thread per fit. These are
  retained custom settings, not the papers' defaults.
- Stop after 30 non-improving validation rounds. No boosting-round ceiling,
  fixed aggregate training-RAM ceiling or default time cutoff. Ten workers;
  2 GiB available-memory headroom controls new batch launches.
- Average all 100 aligned class probabilities, then RET = probabilities dot
  frozen class payoffs. Payoffs are equal-stock true-class means per day, then
  equal-day means over complete training days in the preceding 12 months.
  Coverage accounting includes exchange dates with no supervised rows.
- Long top 2/3 and short bottom 2/3; each leg 50% exposure, equal stock weights.
  Flat overnight. Never replace or zero-fill a selected unknown outcome.

Use matched observation sets for comparisons, retain cash dates, and report
omitted dates. Conditional observed-date curves are not complete tradable paths.
Cost sensitivities apply per side to entry and exit dollar notional; they are not
measured execution/borrow costs.

## Interpretation

The gap benchmark observes an opening print, ranks and fills at that same print.
Its stored bar normally arrives five minutes later. This deliberate research
assumption must fail an execution-ready timing check. One-minute/delayed-entry
work remains deferred. Quotes, borrow and auction access are absent.

Gain is normalized within each saved model and averaged over all 4,000 fits. It
measures training classification splits, not causal impact or explained return
variance. Held-out EV = 1 − Var(y − RET)/Var(y); R² uses MSE instead of residual
variance. Within-day demeaning is diagnostic only, never used to adjust a trade.

2017–2026 has been viewed repeatedly. Walk-forward development results are not a
fresh final holdout. Do not select features, N or filters on that history and
then call it untouched final evaluation. Numerical invariants, design choices,
source limitations and execution assumptions must be reported separately.
