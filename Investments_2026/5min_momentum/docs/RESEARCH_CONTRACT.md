# Research contract

## Corrected ranker authorized 2026-11-10

Run `data/models/12_ranker_portfolio100_v1` changes only validation selection and stopping relative to ranker v1. Keep all 42 inputs, labeled fitting pools, pairwise parameters and both LambdaRank normalizations, exact member seeds and whole-date split fingerprints, 100 members × 40 quarters, 12/3 WFO, and 30-round patience. No round, RAM, headroom or wall-clock ceilings. No one-minute work or ranker tuning. Existing RET remains uninterrupted under its pinned source/configuration.

Validation and final portfolios share `select_portfolio_indices`: descending score, ascending security ID, first/last N, disjoint equal-weight legs with 50% exposure each. For each authorized validation date, predict the full forecast candidate cohort, including unknown-outcome candidates. Before fitting, fix the primary metric date set to dates where every candidate outcome is observed and finite. Exclude incomplete dates only from the conditional validation statistic; preserve all predictions, unknown selected outcomes, fixed date exclusions and exact candidate ledgers. Unknown labels remain excluded from fitting, as before. The metric never changes its date mask with the model score, selected names or boosting round.

For N=2/3, R_N = 0.5 × mean(long returns) − 0.5 × mean(short returns). Maximize the equal-day mean of (R_2 + R_3)/2 in decimal gross-return units. Neither N is preferred. Costs at 1 bp/side and unknown-exit counts are secondary diagnostics. Keep random whole-date validation, not a silently substituted chronological split. Save every metric round, including the full patience interval, per-date returns and coverage.

Two lower-priority one-thread workers run independently alongside RET’s existing ten. This is CPU scheduling only; there are no memory/time ceilings. Completed member and trace checksums are committed last and resume without refitting. Source notebook 12 is output-free and inspection cannot train. Completion automatically generates common-observed-date comparisons against original ranker and historical RET42, feature gain and stopping diagnostics. Return EV/R² is undefined for uncalibrated rank scores. All prior execution, membership and reused-history qualifications remain.

## Ranker v1 audit — 2026-11-10 (no experiment change)

The completed run-11 ranker remains an immutable generic baseline. Its stopping
loop is correct, but its short-score tie selection differs from shared portfolio
accounting, and discounted ordinal NDCG@3 does not match equal-weight N=2/3
portfolio utility. Full evidence is in
`data/reports/11_ranker_objective_audit_20261110/AUDIT_REPORT.html`.

Before ranker tuning, use a new explicitly versioned protocol with one shared
security-ID-aware selector and declare a symmetric N=2/3 stopping metric. A
return-based metric must score the full validation forecast cohort before
attaching outcomes and explicitly handle unknown selected exits and consistent
observation sets. The old labeled-pool validation frame cannot silently serve
as a complete economic evaluation universe. Keep original split seeds, 100
members, 12/3 WFO and 30-round patience unless explicitly authorized otherwise.
Record per-round paths, including the full patience interval. Isolate metric
corrections from objective changes and hyperparameter search; do not infer from
these diagnostics that increasing training length improves outer returns.
The RET experiment continues under its pinned sources and configuration.

## Parallel RET tuning and ranking baseline authorized 2026-11-09

Two streams only, on the immutable completed 42-feature panel:
`data/models/11_ret_ranker_parallel_v1/ret_tuned` and `ranker`. No new network
inputs, feature construction, price downloads or one-minute work. Each final
experiment has 100 members × 40 quarters, unchanged 12/3-month WFO, N=2/3,
50% long/50% short exposure, target outcomes, forecast cohort and missing exits.

The RET search is the complete 24-setting grid: depth {2,3,4}, min_child_weight
{10,20}, lambda {1,10}, colsample_bytree {0.8,1}. Eta .05, subsample .8, max_bin
64 and the ten-class objective stay fixed. The original setting is included.
Each setting uses three fixed independent 80/20 whole-date inner splits within
the past 12-month training window; choose the lowest mean best validation
mlogloss, with deterministic first-candidate ties. All outer-quarter outcomes
are excluded from selection. This gives **2,880 exploratory fits**, separately
reported from the **4,000 final ensemble members**. Each fit has 30-round
validation patience and no boosting-round ceiling. Search validation scores
are selection statistics, not an unbiased holdout assessment.

The final tuned RET ensembles retain the completed RET42 run's exact random
80/20 stock-row member splits and booster seeds. No full-pool refit follows
early stopping. All 100 probability vectors are averaged before the original
full-training quarterly class-payoff mapping. The report checks all 4,000
seed/split/forecast-key fingerprints and forty payoff vectors against RET42.

The untuned ranker uses `rank:pairwise`, the baseline tree settings, mean pair
sampling with eight pairs per sample, and both native lambda normalizations.
Training labels are dense ascending ranks of observed future open-to-close
returns within each training date, keeping exact ties together. Queries are
complete dates; every member has a fresh random 80/20 **whole-date** split.
Unknown target outcomes cannot train and never remove forecast candidates.
Validation maximizes the equal-date average of linear-gain top-three NDCG and
reversed-score bottom-three NDCG, stopping after 30 non-improving rounds.
All-tied outcome queries score zero. No ranker hyperparameters are tuned here.

Ranker members output margins. Convert each member's margins to within-date
percentiles with average ties, then average all 100 members and select N=2/3.
The saved `ranking_score` is also stored as legacy `ret_score` solely for shared
portfolio sorting. It is **not a return forecast**. Do not calculate return
EV/R² or fake class probabilities/log loss/accuracy from uncalibrated scores.
Observed-date rank IC, selected P&L and loss-based training gain remain valid
diagnostics. Gain shares from ranking and classification use different losses.
If a ranker member has no splits, it uses no features and contributes zero gain
shares; the number of constant members is reported explicitly, without retuning.

The ranker changes both objective and inner validation protocol relative to RET.
No extra untuned grouped-RET control was requested. The existing RET42 run is a
historical reference; this is not a fully isolated objective-only comparison.

One shared ten-worker CPU scheduler alternates between both streams; completed
tuning for a quarter releases its 100 final RET fits. Free slots go to remaining
work. Ten workers match this machine's ten CPU cores. There are no fixed RAM,
available-memory, boosting-round or wall-clock ceilings. Every completed member
and tuning trial is checkpointed; stopped work resumes without changing config.
Tuning settings/scores are committed per quarter. Sources, inputs and configs
are pinned before workers start; completed releases remain untouched.

Output-free notebook 11 inspects only. On completion it compares all three
models × both N values on the same observed dates, with omitted/cash dates
recorded. RET EV uses matched observed stock-days; ranker return EV is undefined.
Same-opening knowledge/fill, retrospective membership/vintages, unknown exits
and repeatedly viewed development history remain unresolved limitations.

## Rich five-minute path experiment authorized 2026-11-09

`path42` adds twelve dimensionless prior-close inputs to all thirty frozen
`joint30` inputs. Use the same five-minute member-day download, raw adjustment
for intraday ratios and dollar activity. No one-minute data, new downloads,
cohort changes, duplicate-valued columns or target-based feature selection.
All original columns, labels, outcomes and clocks are copied exactly.

Let r be log(first-bar close / first-bar open), then log(close / preceding
scheduled close). This excludes overnight moves. Let V = sum(r²), N be the
actual session's scheduled slot count, and L be its last twelve slots.

| Added feature | Definition |
| --- | --- |
| `path_rv_relative_10d` | sqrt(V/N) / median sqrt(V/N) in preceding ten scheduled sessions |
| `path_downside_variance_share` | sum(min(r,0)²) / V |
| `path_signed_efficiency` | sum(r) / sum(abs(r)) |
| `path_late_variance_concentration` | sum over L of r² / V × N/12 |
| `path_late_vs_early_return` | (last twelve increments − first twelve increments) / sqrt(V) |
| `path_max_drawdown_scaled` | Maximum log drawdown in initial-open/observed-close path / sqrt(V); no inferred high/low ordering inside a candle |
| `path_vwap_above_fraction` | Fraction of closes above cumulative raw share-weighted VWAP through that slot |
| `path_return_autocorr_1` | Pearson correlation of adjacent scheduled intraday return pairs |
| `path_extrema_time_balance` | (Mean index of maximum-high slots − mean index of minimum-low slots) / (N−1); average ties, no intrabar sequencing |
| `path_return_volume_alignment` | Pearson correlation of r and log(1+q), where q is stock/slot-relative dollar activity |
| `path_late_relative_activity` | sum over L of q / sum all q × N/12 |
| `path_signed_late_relative_activity` | sum over L of sign(r)q / sum over L of q |

Dollar activity is raw VWAP × shares. Its q reference is the median for the
same elapsed-from-open five-minute slot in the preceding ten **calendar
exchange sessions**, excluding current. Every expected observation must be
valid and source-matched. Afternoon slots structurally absent on early closes
do not count as missing observations. The RMS reference requires ten valid
source-matched sessions. These stricter references apply only to the additions;
the old volume inputs retain their frozen historical values and policy.

Missing scheduled slots, duplicate/invalid bars, incomparable provenance,
zero denominators and undefined correlations produce native NaNs in affected
new inputs. Missing whole days never pull older observations into the window.
No new forecast eligibility gate, filling, epsilon denominator or tail clipping
is introduced. Exact per-row flags and feature NaN masks remain in
`daily_path_features.parquet`. Present five-minute buckets with 1–4 source
minutes remain admitted, consistent with the fixed baseline cohort.

Use `data/features/10_rich_5min_path_v1`, model run
`data/models/10_xgb_ret_rich_path100_v1`, and output-free notebook 10. New release
fingerprints pin the builder, calendar code, baseline, and every raw input file.
The same learner parameters, seed/split schedule, 100 fresh members per quarter,
40 quarters, 12/3 WFO, 30-round validation patience and N=2/3 equal-weight legs
remain. No round, RAM, memory-headroom or wall-clock ceilings. No new start gate.

Compare against completed joint30 on exact matched stock-days for return EV/R²
and common observed dates across all four model/N portfolios, with omissions
exported. Check all 4,000 seed/split/forecast-row fingerprints and forty payoff
vectors. Gain is training importance; correlated features can share it. This
experiment does not remedy retrospective membership, reused development
history, unknown selected exits or optimistic same-opening information/fills.

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


## Three fixed protective-stop overlays authorized 2026-11-10

New immutable private release `data/backtests/13_stop_overlays_v1` replays all ten completed position ledgers, without ML fits, new signals, replacement positions or one-minute data. Initial saved weights remain 50% long/50% short. Independent stops leave cash and can create net exposure.

Three scales: prior ten-session ATR (simple mean true range), absolute opening gap, and max(ATR, absolute gap). Distance is k times the scale; long barrier is entry minus distance, short barrier entry plus distance. All prices use the same published `adjustment=all` basis as historical target anchors. This preserves adjustment-vintage and retrospective PIT qualifications; gap is an adjusted-return gap, not a certified contemporaneous unadjusted price gap. Dense scheduled XNYS sessions, full valid price references and homogeneous historical source tags are required; no fills or compressed dates. Today's H/L/C cannot enter today's distance.

At each quarter start, calibrate k using only the preceding twelve calendar months of saved out-of-sample trades. Maximize equal-day mean net return at 1 bp/side, averaging N=2 and N=3 equally. Search all positive empirical first-touch adverse-extreme/bar-open breakpoints, with touch inclusive; no zero-width or machine-epsilon beyond-extreme candidates. No-stop is included and wins numerical ties; wider distances win other numerical ties. One k per family/model/quarter is shared by both N and both directions. Freeze for the next three months. First twelve months are explicitly calibration warm-up: 2017 for full-history releases, 2022 for the bounded validation-only release. This overlay calibration is a new exit policy, not a change to ML training/splits or a new untouched holdout.

Fix each model's evaluation/calibration date cohort across all three scales, no-stop and both N before selecting k: every selected original exit observed, entire scheduled five-minute path valid, ATR positive, adjusted gap finite/nonzero, and no-stop return consistent with saved outcome. Original cash dates remain cash. Export all selected-row reasons and omitted comparison dates; never remove forecasts or replace selected stocks. This stronger execution cohort is a conditional comparison, not a full investable history or a fair cross-model common-date ranking.

Protective stop is active from the optimistic opening entry, including the first five-minute bar. A touch fills at barrier; a bar opening beyond it fills at worse open. No take-profit or trailing order, re-entry, or extra slippage assumption. Costs use actual simulated exit notional. Missing/duplicate/invalid bars are unresolved for the primary cohort even if a tighter stop could appear to exit earlier. Partial 1–4 underlying-minute aggregations are admitted under the existing five-minute policy and flagged; path completeness refers to scheduled five-minute bars and cannot prove complete underlying trades. Opening delivery follows the assumed fill; no execution certification. Exposure duration is a bar-end approximation. Completed model/feature releases are never overwritten. Notebook 13 only reads completed results.


## Standard performance costs authorized 2026-11-10

New performance reporting uses `STANDARD_COST_BPS = 10` in `scripts/trading_costs.py`: 10 basis points per executed entry/exit dollar notional, for longs and shorts. Net return = gross return − 0.001 × total traded notional, where position notional = |initial weight| × (2 + raw entry-to-exit return). A flat round trip at 100% initial gross costs 20 basis points of equity. Costs scale with actual simulated exit prices, including stopped exits; cash is uncharged and unknown selected outcomes stay unresolved. This is an aggregate research allowance, not measured execution components or borrow.

`scripts/report_stop_costs.py` writes a new immutable report `data/reports/13_stop_overlays_10bps_v1` from completed release 13, verifies its costs against saved 10bp sensitivities and position exits, and keeps the exact original observed-date masks. Notebook 13 now displays that report and explicit 10bp/side labels. Historical 1bp reports and source-calibrated stop policies remain unchanged; the stops were calibrated at 1bp/side and this update only reprices their saved fills. There are zero ML fits or stop recalibrations. Legacy pinned experiment modules retain their historical cost defaults; use the central 10bp standard explicitly in new reports. No new price data or one-minute access.


## Cost-specific stop recalibration authorized 2026-11-10

New release `data/backtests/14_stop_cost_recalibration_v1` recalibrates ATR, adjusted opening-gap and hybrid stops separately at 5, 10 and 20bp per entry/exit side. Each model/family/cost quarter maximizes the prior twelve calendar months' equal-day mean of N=2 and N=3 net returns, then freezes one multiplier shared by both N and both directions for the next three months. Positive empirical first-touch candidates, wider-distance ties, no-stop ties, first-year warm-up, five-minute fills, original cash dates and the fixed parent observation masks stay unchanged. No-stop means retain the original entry and hold to close, not skip trading. There is no new entry filter, replacement or re-entry.

The completed 100-member ML models, predictions, selections, initial weights, random inner splits, 12/3 WFO and original source/price/eligibility rules remain untouched. Parent release 13's exact selected-row coverage and five-minute paths are fingerprinted; no new price construction or one-minute work. Affine cost conversion of gross first-touch responses is algebraically exact: net = gross × (1 − fee × direction) − 2 × fee × absolute weight. Dedicated tests compare it with direct cost-specific replay for both sides, test future perturbations and preserve original cash-date weighting. One-bp calibration controls must reproduce the parent policies before accepting new scenarios.

Compare cost-specific stops against original 1bp-calibrated stops at the same evaluation cost and dates; report every approach and N without choosing a policy on outer-quarter outcomes. Preserve all original row/date exclusions and separately export the nine-model common-date comparison; the bounded validation-only run is separate. Recalibration changes exits through past calibration only, with zero ML fits. It cannot remove the entry and exit cost of each preserved position. Notebook 14 reads completed outputs only and never constructs missing releases.


## Strict session-window features authorized 2026-11-11

Release `data/features/15_session_window_10_v1` replaces the older-history feature representation for this new experiment only. The user's strict boundary overrides the earlier allowance for longer smoothing history: exactly ten completed XNYS sessions ending at the signal close, nine gaps between adjacent sessions inside that block, no pre-window closing price, no current trade-day opening gap/OHLC, no EMA/MACD state and no outside normalization references. The next session's open-to-close investment target and 12-month/3-month ML WFO are unchanged; this task builds and audits features only, with zero model fits. Completed earlier releases keep their original definitions.

Each session supplies four cumulative opening log returns (5/10/15/30 minutes), four signed post-opening returns, four signed balances of subsequent five-minute return directions relative to the opening direction, intraday log return, signed path efficiency, four opening high/low log ranges, realized intraday volatility, downside variance share, and four relative opening dollar-activity values. Nine internal session pairs also supply gap return, gap-aligned intraday return and gap-fill indicator. The oldest session has none of those three gap-dependent columns: 24×10 + 3×9 = 267 candidates. Gap response and opening follow-through are continuous sign interactions, not overlapping full-day-return sign tests. A flat opening leaves persistence undefined; a zero gap leaves the fill event undefined.

For a prediction, returns and ranges share a risk unit equal to the RMS of realized volatilities of the ten completed sessions. Each realized volatility is sqrt(sum of squared intraday log increments), starting with first close/first open and then adjacent scheduled closes, excluding the overnight jump. Individual volatility columns retain decimal log-return units. Each opening activity uses raw VWAP×shares divided by the median for the same opening window across the other nine sessions. These references are evaluated at window end and are not claimed to have existed at the older historical day's open. All reference observations must be complete/source-comparable; zero or missing denominators stay NaN. No epsilon floors, price filling, compressed dates, prior-history substitute or implicit clipping.

Adjusted OHLC uses `adjustment=all`; activity uses the aligned raw files. Exact scheduled timestamps and start+5-minute modeled availability, unique slots, valid OHLC/quantity, provenance and early-close lengths are checked. Present 1–4 underlying-minute buckets remain admitted and flagged consistently with the published five-minute data policy. Exact feature missing-reason bytes and full row coverage are retained.

The label-free feature panel includes every resolved, nonconflicting signal-day S&P 500 member, without requiring prices, future outcomes or the legacy older-history eligibility gate. Conflict and unresolved membership ledgers are separate. No training population is selected in this task. A keys-only join reports coverage of the former 42-feature cohort without importing old features or labels. Membership and adjusted-price vintages remain retrospective/provisional. Full-release duplicate/constant diagnostics never select features using future data; candidates remain declared by construction.

Notebook 15 offers explicit opt-in building for manual use, defaults to completed-release inspection, and reads the independent construction audit. The shared inspection runner disables the build flag and cannot rebuild an absent release or start fits. Tests include pre-window/future perturbations, internal gap boundaries, timestamp/duplicate errors, raw-adjusted source mismatches, units, zero states and early-close anchors. Independent audits validate every row's window/missing mask and rebuild actual normal/early-close windows. Passing these checks does not certify historical as-known PIT, complete underlying minutes or executable trading.


## Strict-window learner comparison authorized 2026-11-11

Release `data/models/16_session_window_ret_ranker100_v1` trains unchanged baseline RET and the corrected portfolio-stopped ranker on the 267 audited release-15 features. Each has 100 models × 40 quarters, 12/3 WFO, unchanged seed schedule and learner parameters, and 30-round validation patience. There is no tuning or arbitrary round, RAM, memory-headroom or wall-clock ceiling. One shared ten-worker scheduler alternates learners. No one-minute data or current trade-day input is used.

The declared historical-quality gate requires complete, source-consistent ten-session price and raw-activity histories (missing reason codes 1/2/7 excluded). Legitimate undefined states remain native NaNs; all-267-finite is not eligibility. Exact exclusions are retained before labels are attached. Unknown target outcomes cannot fit but remain forecasts and selected positions. Endpoint/status rules are unchanged and must agree with historical outcomes on overlapping keys. Ten-class tie-preserving labels and RET payoffs are recomputed within this new cohort. This changes historical eligibility/cohort, normalization, feature representation and current-opening information relative to legacy 42; it is not a controlled feature-only attribution.

RET retains random 80/20 stock-row inner splits and probability averaging then past-only class payoffs. The corrected ranker retains random 80/20 whole-date splits, pairwise settings, full validation candidates, fixed complete-outcome date exclusions and mean N2/N3 gross-return stopping. The same seed/method schedule is retained; changed cohort means old row-index hashes/payoffs are not expected to match. Ranker percentiles are not return forecasts. All inputs, configs, sources and completed members are pinned and resumable. Inspection notebook 16 cannot fit or rebuild.

Completion reports both N variants at gross and 5/10/20bp per executed side, with 10bp standard, eight-way common-date comparisons, per-model coverage, matched RET return EV/R² and separately labeled training gain by feature/descriptor/lag. No stop overlay is recalibrated in this run. Prior-close inputs remove current-opening knowledge from the signal; auction fills, retrospective PIT/adjustments, partial underlying minutes, borrow and reused development-history limitations remain.


## Trade-day opening-gap clarification — 2026-11-12

The completed 267-feature experiment excludes the current trade-day opening gap. The assistant interpreted the strict ten-completed-session/nine-internal-gap history as a previous-close-only decision cutoff. The user had not explicitly requested removing the existing entry-day gap; this should have been presented as a separate material design decision. Consequently, comparison against the 42-feature models changes information availability in addition to history, normalization, eligibility and cohort-conditioned labels/payoffs. The results cannot isolate the effect of feature count or attribute the performance difference to any single change.

A proposed follow-up keeps the 267 historical inputs and restores the trade-day opening gap as a separate entry-time observation, using only the prior close and current open. This would have 268 inputs and retain the optimistic same-opening execution qualification. It has not been implemented or fitted. Completed feature/model releases and their pinned sources remain unchanged.
