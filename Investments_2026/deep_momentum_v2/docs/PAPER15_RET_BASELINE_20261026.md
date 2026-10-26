# Paper15 RET baseline: matching five/ten-year histories

USER DIRECTION,26October2026: create a separate baseline using U.S. data,
historical S&P500/400 membership, only the agreed paper inputs, RF/XGBoost,
rolling5/10year training, matching complete daily history and matching RET
payoff history. This side experiment is isolated on
`codex/paper15-ret-5y-10y`; all prior runs/work remain unchanged.
The frozen recipe is `configs/paper15_ret_baseline_20261026.json`.

## Four arms and chronology

RF/Paper15/5years, XGBoost/Paper15/5years, RF/Paper15/10years and
XGBoost/Paper15/10years. Each fits once annually for2020–2023:16empirical fits,
no tuning, early stopping or calibration. Data/outcome initialization stays2010;
both arms share the2020–2023 forecast period. For2020, the5year fit uses
2015–2019 completed outcomes; the10year fit uses2010–2019. Later years roll
forward. Training rows are qualified at their own historical signal dates,
not selected using the2020or2023 universe. Actual next-rebalance-open outcome completion
must precede the annual cutoff; the latest December outcome is purged without
reaching further back to replace it. Monthly forecasts retain unavailable later
labels and every original member in the eligibility ledger.

The paper uses16inputs, expanding classifier history, ten-year class payoffs
and100fits per annual ensemble. This deliberately smaller V2 baseline uses the
previously agreed15inputs without capitalization, rolling training and one
fixed-seed fit per annual arm. Five-year RET is a declared adaptation. It is
not the paper's full empirical replication or a comparison against untouched
holdout data:2020–2023 has already been examined in development. Financial2024+
values remain protected.

## Complete daily history means backward-looking coverage

At each monthly close signal, require every independently generated XNYS
session in `(signal date minus N calendar years, signal date]`, for N5or10
matching the arm. Every raw and adjusted OHLC field must be positive and finite,
with accepted daily/completion metadata and valid high/low bounds. A missing
row, field or rejected bar fails history completeness. No price imputation,
forward filling, tolerance for missing sessions or future-quote filter. Weekends
and exchange holidays are not expected observations; Nyears is not approximated
as252*Ndays. Volume is not required by this OHLC rule.

Membership is required at the signal, not throughout the preceding Nyears.
The source build therefore checks full pinned provider history from2000–2023
for the original admitted identities, including pre-index history. Verified
corrected2010–2023 cached observations override their raw keys. Duplicate raw
or corrected keys fail; these explicit reviewed overrides are recorded as a
source-priority rule, not silently deduplicated observations. Existing source
identity quarantines/flags and BK/XOM reviewed computational routes survive;
identities and historical knowledge time are still provisional. The execution
40quote overlays do not enter learning histories. Prior caches alone cannot
establish missingness for a name that was not requested before index entry.

A complete-history screen deliberately selects seasoned, historically observed
stocks and excludes IPOs/gapped histories. It does not establish survivorship
freedom. Future removals and future data availability never screen a signal;
future missing held quotes still require economic resolution. Original cohort,
identity-admitted, history-qualified, forecast and label denominators are reported.
Five/ten-year arms have different universes, normalization and return classes;
any performance difference changes more than training length alone.

## Exactly15features, shared classes and RET

Use the source-locked Han–Qin indexing: MOM1 includes the signal month's return;
MOM3/6/9/12 compound2/5/8/11returns ending in the preceding month. All intermediate
exact monthly anchors are required. Normalize the five momenta across the
common admitted/history-qualified/momentum-complete signal population, using
sample SD(ddof1). Keep its five cross-sectional means and five SDs as shared
time-varying state inputs. Recompute these statistics separately for each arm;
do not filter previously normalized expanded panels. No expanded technicals,
market cap, extra scalers, missing indicators or feature imputation. The actual
model matrix has15columns.

The target is the following month's **executable daily-path cumulative cash
return**: enter at the first open after the signal, value the held unit at every
daily open/close and exit at the following monthly first open. Use signed raw
share cash/stock entitlements for31reviewed actions(the prior29plusTIF/LHCG);
ordinary dividends/splits already represented in adjusted prices are not added
again. A missing required daily mark makes the label unavailable. The unstopped
long-unit return is then ranked in ten descending exact-midrank bins within
the signal-eligible cohort. This is an explicit executable-target adaptation
to the paper, not a month-end closing-price label. December2023 needs the
protected2January2024open and remains unlabelled; do not replace it with the
backtest's terminal December close.
Class1is best,10worst. Exact ties remain together; no jitter or missing-target
imputation. Fewer than10observed eligible outcomes withholds classes. Provider
adjustments/action coverage remain provisional research evidence; they are not a claim
of complete delisting cash-flow accounting.

For every signal, estimate RET as all ten probabilities times historical
realized-class mean returns. Average monthly class means equally over exactly
60months for5years or120for10years. Use only actually completed outcomes through the month before the
signal month, with their scheduled exits already observed; no probability-estimated class means, pooled stock
rows, shortened missing windows or test-month information. Every month/class
cell is required; missing cells withhold RET without erasing forecasts.
Both estimators in a window share precisely the same X/y/cohorts/payoffs.

RF uses500trees,depth12,minleaf20,sqrtfeatures,bootstrap,seed1729. XGBoost uses
300rounds,depth3,learningrate0.05,subsample/columnsample0.8,minchildweight20,
lambda10,histograms and seed1729. These are the earlier fixed baseline settings,
not settings selected on this experiment. One fit thread/worker. XGBoost's
native0–9classes map explicitly to1–10; float64 softmax of native margins is
the declared probability extraction. RF uses native predict_proba.

## Daily backtest and reporting

All four arms predict48months. The common next-open execution comparison covers
3February2020 through29December2023:985sessions/47months. Both arms retain
January2020 forecasts; its10year RET vector lacks120completed outcome months
at the December2019signal. Five-year RET can have60months there, but January is
outside the paired economic comparison for both arms. This clock is fixed
before fitting; it is not selected based on favorable coverage/performance.
Prediction diagnostics can observe47outcome months January2020–November2023;
December2023labels require protectedJanuary2024prices. Those47months differ
from the February2020–December2023economic period. Portfolio NAV uses its
previously agreed terminal29Decemberclose without turning that shortened
terminal period into a completed open-to-open learning label.

Rank RET highest for longs, lowest for shorts, stable source-ID ties. Use up to
10disjoint names per side with matching available counts,+50%/−50%equal weights.
Keep the previously fixed DF/RDC execution restrictions and exact PENN handover,
20reviewed current actions(includingTIF/LHCG),11historical retirements and40
execution-only quote amendments. Stopsnone/10%perasset cross costs0/5/10/25bp
for32scenarios.10%-stop/10bp is primary. Borrow/funding remain zero; stop fills
are idealized daily-OHLC assumptions. Retired new orders remain unfilled cash
slots. An unexplained required quote retains exposure and withholds the entire
continuous path; no hindsight selection/removal/restart. Known deferred cases
are not automatically repaired or silently assumed away.

Report equal-month and pooled log loss/accuracy/Brier, all ten separate class
precision/recall/support and confusion counts, uniform and annual training-prior
references, eligibility/label coverage and all32economic scenarios. Annual NAV
is continuous. Withheld rows remain explicit; a complete favorable subset is
not a complete four-arm ranking. No target/settings change is selected from
these results.

## Guarded local workflow

The source plan pins manifests, corrected source proofs, exact projections and
Parquet row groups before numeric reads. Budgets per source pass2GiB, independent
derived reads2GiB, peak RSS8GiB,newoutputs4GiB,16fits,one thread,600seconds perfit,
1800seconds perphase and3activehours total. Two full history passes are allowed:
preparation and independent reconstruction. Execution has separate bounded
projections of the existing four annual tables. Mixed source Parquet blocks
may transiently decode excluded Arrow values; metadata date/ID masks apply
before any excluded financial values reach Python, analysis or outputs.
No cloud operations or protected2024+value use.

From this branch's V2 directory, use the existing model environment or install
the pinned `requirements-models.lock.txt` into your own isolated environment:

```sh
PYTHONPATH=src /path/to/venv/bin/python -B scripts/run_paper15_baseline_current.py plan \
  --data-root /path/to/configured/deep_momentum_v2 \
  --protocol configs/paper15_ret_baseline_20261026.json \
  --output /path/to/private/outputs/paper15_ret_baseline/fresh-run \
  --plan /path/to/private/outputs/paper15_ret_baseline/plans/fresh-run.json
PYTHONPATH=src /path/to/venv/bin/python -B scripts/run_paper15_baseline_current.py run \
  --plan /path/to/private/outputs/paper15_ret_baseline/plans/fresh-run.json
```

The controller consumes each estimator attempt once, records immutable phase
receipts/scenario checkpoints and supports `pause`/`resume` only at clean phase
boundaries. Failed/interrupted attempts are preserved, never automatically
retried. The independent checker reconstructs source history, eligibility,
features, labels and RET before fitting; separately reloads all16saved classifiers
and checks predictions/metrics; then reconstructs source overlays, RET selections,
maximum holding preflight and signed cash/share/stop/cost equations. A successful
source or arithmetic audit does not certify source completeness/investability.

`notebooks/10_paper15_ret_baseline.ipynb` is a separate passive entry point.
Its saved-report reader must authenticate all independent proofs before reading
small results; it never starts preparation, model loads, fitting or backtesting.
Source data, models, forecasts, holdings and numerical receipts remain ignored.

## Planning correction before any run

The initial metadata-only plan20261026-v1 described calendar-close targets. The
user clarified executable daily returns before preparation or fitting began:
zero financial source projections decoded and zero empirical fits occurred
under that draft. Its plan and pinned implementation/config bytes are archived
privately as superseded planning evidence. The first executable plan was20261026-v2;
all sources, formulas, actual completion dates and budgets are resealed before
any numeric work. No original V2 empirical run was changed.

## Checker recovery and active execution

The first executable preparation completed, but its independent checker reused
the variable name `positions` for a missing-date index and lost the session
lookup. The controller stopped with zero empirical fits. Its original run,
workers, logs and prepared arrays/panels remain immutable under20261026-v2.
The explicit20261026-v3recovery reuses those prepared bytes and changes only
the independent lookup name. Its sealed addendum reserves one additional
source-reconstruction pass under the same2GiBper-pass/RSS/time ceilings;
previous active time is carried into the controller budget. No data producer
or consumed model fit is repeated. `verify_paper15_baseline_v3.py` authenticates
the independent source/feature/target/RET and inference/accounting phases.

`run_paper15_baseline_current.py` is the maintained entry point for fresh
plan/run/pause/resume commands, with the corrected checker. The earlier
controller/checker files are frozen provenance snapshots for the preserved
metadata/zero-fit attempts. The passive notebook reads the actual completed
run chosen in ignored `configs/paper15_baseline.local.json`; a fresh clone can
point that file to its own private run. No existing master notebook or original
V2results are replaced by this side experiment.

## Completed result

The independently verified v3 run completed16fits and32scenarios, with24complete
proxy paths and8withheld RF/5year variants. See [the full results and exact
new-order gap](PAPER15_RET_BASELINE_RESULTS_20261026.md). The source, protocol
and empirical worker bytes remain sealed; this completion note changes no run.
