# U.S. adaptation and replication decision ledger

Status: PAPER REVIEW COMPLETE; EXPERIMENT DESIGN OPEN. Updated 2026-10-14.
No feature build, model fit, portfolio simulation or holdout inspection was
performed during this paper review. This document is a design ledger, not an
execution authorization or a frozen research protocol.

Read [the locked paper reference](PAPER_REFERENCE_HAN_QIN_2026.md) first. Its
`PAPER`, `INTERPRETATION`, `VERIFIED LOCAL`, `PROPOSED` and `OPEN` distinctions
apply here. Machine-readable fixed facts and unresolved decisions are in
[`configs/paper_replication_draft.json`](../configs/paper_replication_draft.json).

## 1. Research question and boundaries

Can next-month return-class probabilities, converted to expected-return scores,
improve implementable momentum selection for historical U.S. S&P 500, S&P 400,
or their combined universe? After establishing the paper mechanism, do added
features and forest ensembles improve performance under identical chronology,
coverage and accounting?

The supplied Han-Qin March 26, 2026 paper uses monthly multiclass XGBoost,
16 inputs, probability averaging and reclassification. It already combines
boosted trees with an outer 100-fit probability ensemble. Our proposed forest
work changes the probability estimator; it need not change the monthly target
or RET payoff layer. No neural architecture is required by this PDF.

Our Intrinio/Siblis data and index universe differ from the paper's broad CRSP
sample. Even the closest baseline is a **U.S. data/universe adaptation**, not an
exact replication of its tables. Results from the broad U.S. sample, international
pooled model or all-period appendix are contextual comparisons, not acceptance
thresholds for S&P constituents.

The old `../deep_momentum/` folder remains a reference. Sadig's pinned recovery
snapshot is another reference: its scalar daily PRICE5 / 20-session next-open
experiment is not this paper's monthly distributional specification. The legacy
monthly Canadian implementation may inform code review; Canada is outside this
project's universe. Do not copy either implementation without checking formulas,
timing, identities and economic failure cases against the locked reference.

## 2. Local inputs: verified availability and gaps

The three previously downloaded source releases were rechecked on 2026-10-14:
149 objects, 1,792,717,079 bytes; all local SHA-256 values matched the manifests,
and Parquet footer row counts reconciled. Only metadata/footer reads were used
for this paper task. These facts establish file integrity, not a completed
research panel or economic validity. Exact private paths and release pins remain
in ignored local configuration and provenance files.

| Input | Verified local availability | What remains unverified or missing |
|---|---|---|
| Intrinio daily prices | Raw/adjusted OHLCV, dividend/split fields, adjustment factor, stable IDs and quality metadata; price table 26,330,528 rows | Provider adjustment semantics, historically available total-return reconstruction, usable monthly coverage by security, delisting payouts and executable quotes |
| Siblis historical constituents | Dated S&P 500/400/600 membership and change/identifier history; membership table 14,747,642 rows | Signal-time join rules, publication/availability versus effective dates, resolved ID ambiguity and coverage for the chosen universe |
| Unified index/identity release | Identity bridge with 20,283 rows plus historical index membership | Whether every proposed constituent maps uniquely at every required date; identity joins need temporal validation |
| Historical market capitalization | **No capitalization or shares-outstanding field found across the three release table schemas** | An approved historical source, its publication timing and identity linkage |
| Sadig feature code | PRICE5 and earlier 740-feature construction code in the pinned ignored snapshot | Neither is a model-ready V2 panel; the precomputed 740-feature warehouse was not downloaded |
| Sadig derived strategy panels | A prior read of the pinned calendar returned `403 / storage.objects.get`; derived panel downloads were incomplete | Access was not retested on 2026-10-14; use the ignored owner-request record if reuse is needed |

Product-wide stated coverage is Intrinio 1962-01-02 to 2026-07-24, Siblis
1970-03-05 to 2026-03-31 and unified membership 1970-03-31 to 2026-04-30.
These are release properties, not verified completeness for individual securities
or a license to inspect protected 2024+ financial values.

Market capitalization is required for the paper's size-decile input, bottom-5%
screen and value weighting. Price, volume, liquidity or S&P index labels are
not replacements. A 15-feature price/state-only prototype is possible, but must
be named an ablation and cannot be reported as the 16-feature baseline.

## 3. Candidate universe: one primary choice before results

| Candidate | Signal-time rule | Implication |
|---|---|---|
| S&P 500 | Constituents known/eligible as of the signal cutoff | More liquid large-cap sample; excludes the small-stock channel emphasized by the paper |
| S&P 400 | Historical mid-cap constituents at the cutoff | Different size/liquidity distribution; no assumption of the same return effect |
| S&P 500 + 400 | Dated union, deduplicated by stable security identity | Larger domestic cross-section; one combined normalization/ranking cohort is a proposed default, still OPEN |

Never substitute today's constituents or require a future price to admit a
security. Decide whether membership needs an effective date, an availability
date, or both; an after-cutoff announcement cannot become an earlier signal.
Resolve overlaps, share classes, exchange/type eligibility, entries/exits and
identifier conflicts explicitly. Decide whether features and target ranks use
only the signal-time cohort or a broader reference market; this changes the
meaning of the ten shared state inputs. No universe has yet been selected.

## 4. Causal monthly workflow and audit artifacts

1. **Construct identity and eligibility.** Produce a dated constituent-to-price
   bridge with exclusions and ambiguous/unmatched counts. Stable security ID
   is the research key; ticker is a dated display attribute. Preserve the
   signal-time eligible cohort independently of future quote availability.
2. **Define monthly returns.** Build a complete calendar grid. Specify last
   trading observation, adjustment/total-return semantics, dividends, splits,
   stale/missing quotes and terminal events. Missing months cannot collapse
   the calendar. Audit these rules on bounded approved pre-holdout records
   before a full build; an adjusted-close ratio is not automatically CRSP-equivalent.
3. **Build the paper feature set.** At end of month `t`, use `MOM1=r[t]`,
   and compound `t-m+1` through `t-1` for `m=3,6,9,12`. Normalize within the
   declared signal cohort; append five means, five SDs and historical size
   decile. Freeze window completeness, SD divisor, size orientation, ties,
   zero dispersion and feature availability. Record per-feature missingness.
4. **Build supervised targets.** For eligible signal cohort `U[t]`, rank realized
   month `t+1` returns into ten groups only once outcomes complete. Persist
   best-to-worst class mapping and rank/tie rules. Keep missing/terminal names
   in coverage and economic ledgers, even if their supervised label is unavailable.
5. **Fit at annual cutoffs.** Only completed labels available by the cutoff
   enter training. The paper expands classifier history, randomly splits
   historical rows 80:20 for early stopping and repeats 100 fits. Pin software,
   parameters and seeds; decide a separately labeled chronological-validation
   adaptation. Do not silently move retraining to every month or truncate
   classifier training to ten years.
6. **Estimate payoffs and predict.** Persist monthly realized class means;
   calculate class payoffs over the declared trailing ten-year window without
   future outcomes. Align and average classifier probabilities, then calculate
   RET. Persist probability vectors, class moments, training cutoffs and model
   versions so every score can be reconstructed. SRP is secondary until its
   class-variance convention is frozen; retain the printed DPR weights.
7. **Form and account for portfolios.** Under identical cohorts/timing, compare
   MOM, mode-selected XGB and top/bottom-decile RET. Decide close/next-open
   execution, gross exposure, position drift, cash/financing, borrow, event
   handling and cost model. Missing valuation is a failure/coverage state;
   never treat it as zero return or drop the holding from the denominator.
8. **Evaluate once under the recorded protocol.** Report prediction calibration,
   log loss, tail precision and the paper's bimodality diagnostic alongside
   long/short attribution, annualized arithmetic mean, CAGR, volatility, Sharpe,
   drawdown, turnover and net cost sensitivity. Record cohort/valuation coverage
   and all trial counts; model accuracy is not sufficient evidence of economics.

Each step needs a small inspectable artifact before scale-up: join/exclusion
ledger; return-rule audit; feature dictionary; label map; split/model manifest;
probability and class-payoff ledger; economic ledger; final evaluation report.
No full dataset build is authorized by the paper alone.

## 5. Chronology and dates: do not inherit the previous attempt silently

The paper requires at least ten years before initial model fitting and uses a
rolling ten-year class-payoff estimate. A source product reaching back decades
does not establish ten usable years for our chosen universe and return rules.
The exact first admissible forecast is determined by verified feature, label,
identity, membership and payoff coverage.

The prior recovery used 1998 warmup, 1999-2023 accepted signal-panel coverage,
2009-2023 primary inputs and 2015-2023 test folds. These are different roles,
not interchangeable sample dates. January 1998 data alone cannot justify a
paper-style initial fit in 1999. The available releases may support a longer
history, but that has not been audited. New start/end dates remain OPEN.

Protected 2024+ financial values remain uninspected. Define the development
period, annual outer forecasts, inner validation, tuning budget, embargo/purge
rules, intended holdout and acceptance criteria before evaluation. Overlapping
momentum histories and common monthly state inputs matter for validation;
group by time when defining the proposed temporal variant. The paper's random
past-row inner split can be preserved in a clearly labeled reference arm;
it is not permission to mix future test rows into model development.

## 6. Model and feature extensions: isolate the source of any gain

**PROPOSED comparison sequence, not a frozen search:**

1. Paper-mechanism reference: the complete 16 inputs, ten classes,
   multiclass XGBoost probability ensemble, mode and RET portfolios; MOM benchmark.
2. Probability-estimator change: Random Forest, optionally Extra Trees,
   on the same inputs/labels/cutoffs, feeding the unchanged RET layer.
3. Feature change: approved additional feature groups, holding the estimator
   and evaluation protocol fixed. Candidate groups include volatility/downside
   risk, liquidity/volume, shorter return dynamics and market-relative measures;
   they are proposals, not automatically Sadig's 740 columns.
4. Combined ensemble: align class maps and combine probabilities using weights
   fixed or learned solely from chronological development data. If calibration
   is used, fit it without outer-test data. Any learned meta-model needs its
   own out-of-fold time structure. Hard votes discard information needed by RET.

Compare `16 inputs × estimator change` separately from `expanded inputs ×
same estimator` before testing their combination. XGBoost is already an
ensemble; a forest substitution is not automatically more diversified or
better calibrated. Averaging many correlated models does not create independent
evidence. Class weighting/oversampling can alter probability interpretation;
record any correction/calibration rather than treating weighted outputs as
unconditional class probabilities by default.

Freeze feature formulas, lag/availability dates, scaling cohorts, missingness,
selection and hyperparameter bounds before results. Do not tune RET class payoffs,
ensemble weights or added features against the final portfolio test. Neural
models, direct return regressors, long-only books, alternate class counts and
CVaR/CVR may be separate extensions; they are not needed to satisfy this paper.

## 7. Decisions still OPEN

| Decision | Why it matters | Current state |
|---|---|---|
| Primary S&P universe and normalization/rank cohort | Defines the prediction distribution and eligible trades | OPEN |
| Return/terminal-event accounting and identity rules | Determines whether labels and portfolio cash flows are credible | OPEN; bounded audits needed |
| Historical capitalization source and timing | Required for size, paper screen and VW | MISSING locally |
| Coverage and experiment chronology | Ten-year startup, development/outer tests, holdout | OPEN; 2024+ protected |
| Paper underspecification conventions | Ties, SD, winsorization, size mapping, moments, seed/early-stop mechanics | OPEN; documented in paper reference |
| Execution/cost/exposure/borrow rules | Separates predictive results from investable returns | OPEN |
| Models, feature groups and trial bounds | Prevents results-driven expansion of the search | OPEN; forests/features proposed |
| Calibration/ensemble weighting and acceptance criteria | RET relies on probabilities and payoff estimation | OPEN |
| Resources, full-build/backtest plan | Controls compute and reproducibility | OPEN; no compute launched |

The next research step is a bounded pre-holdout identity/return coverage audit
and identification of an approved historical-capitalization source. Choosing
the primary S&P cohort can proceed alongside that work. These are design tasks,
not permission to grant IAM, mutate GCS, launch compute or start an unbounded scan.

## 8. Reuse and provenance

The repository-local reusable skill is
[`han-qin-deep-momentum`](../.agents/skills/han-qin-deep-momentum/SKILL.md).
It points to the paper account and this ledger rather than maintaining a second
formula specification. The original PDF stays local and ignored. Public Git
contains original notes, design decisions and code; credentials, downloaded
datasets, source snapshots and research output files remain excluded.

For an eventual run, record source-manifest hashes, identities, universe rules,
date bounds, feature/label versions, model/software versions and seeds,
validation folds, payoff windows, costs, exposure conventions and resource
budget. Preserve unsuccessful and incomplete economics as evidence. A new
paper version requires a separately identified review and diff; never silently
overwrite the March 26 source lock.
