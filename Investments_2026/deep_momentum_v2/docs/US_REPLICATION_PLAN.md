# U.S. adaptation and replication decision ledger

Status: FULL FEATURES AND TARGET COVERAGE AUDITED; FIRST FOUR-ARM CLASSIFIER
PILOT COMPLETE AND VERIFIED. Updated 2026-10-15.
No feature build, model fit, portfolio simulation or holdout inspection was
performed during the original paper review. Subsequent stages have their own
user authorization and contracts; this ledger alone is not execution authority.

**ACTIVE USER DIRECTION:** fit RF and XGBoost once each on Paper15 and once each
on expanded candidates, using completed 2010–2017 outcomes for fitting and
2018–2019 validation. The [pilot record](CLASSIFIER_PILOT.md) and
[`classifier_pilot.json`](../configs/classifier_pilot.json) freeze this limited
execution. Primary 80% context, seed 1729, training-only preprocessing and a
common prediction/evaluation cohort apply. No 90% sensitivity fits, parameter
search, early stopping, outer refit, 2020–2023 outcome evaluation, RET or backtest
is included. The one-fit pilot is a V2 adaptation, separate from the paper's
100-fit ensemble. `model_comparison.proposed.json` remains the earlier design
snapshot; the separately authorized pilot is now complete and verified.

**VERIFIED LOCAL PILOT:** all four arms fit the same 77,690 training rows and
predict all 20,755 eligible validation rows, retaining 46 missing labels and
scoring the same 20,709 observed classes. Mean monthly log losses are RF/Paper15
2.277891, XGBoost/Paper15 2.276638, RF/expanded 2.263056 and XGBoost/expanded
2.264372; uniform is 2.302585. Expanded RF has the lowest observed loss and
13.60% accuracy in this one development split. This descriptive comparison does
not establish significance, certified PIT identities or investable performance.
The run took 911.08 seconds at 5.01 GiB peak RSS, preserving 726 recorded source
files. Independent metric/matrix/source verification and exact model-probability
reloads passed, as did the 690-test suite, seven later focused verifier tests and
all 24 master cells with implicit-fit/build guards. Detailed results and audit
scope are in [CLASSIFIER_PILOT.md](CLASSIFIER_PILOT.md).

Subsequent user-authorized loading/universe engineering is implemented in
[data foundations](DATA_FOUNDATIONS.md). The 2010-01-29 combined audit retains
902 source constituents. Its original strict-policy audit counted 130 clear
routes, 755 review-flagged routes, ten collisions and seven unlinked names;
two exact-date quote reads worked. The [active provisional policy](PROVISIONAL_IDENTITY_POLICY.md)
admits unique resolved review-flagged routes with their flags retained. Missing/ambiguous
routes and collisions remain blocked. Source knowledge-time evidence and economic
completeness remain unverified. No model-ready complete panel is claimed.

Read [the locked paper reference](PAPER_REFERENCE_HAN_QIN_2026.md) first. Its
`PAPER`, `INTERPRETATION`, `VERIFIED LOCAL`, `USER DIRECTION`, `PROPOSED` and
`OPEN` distinctions apply here. Machine-readable fixed facts and unresolved decisions are in
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

**USER DIRECTION, 2026-10-14:** V2 will not use market capitalization. Its
starting feature set has 15 momentum/state inputs before extensions: five
normalized momenta, five cross-sectional means and five SDs. Omit the paper's
size decile, capitalization-based screens (including missing-cap and bottom-5%
filters) and cap-value weighting. Portfolio weighting remains OPEN; this
decision does not select an execution or weighting convention.

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
| Trading volume | `volume: double` and `adj_volume: double` in `intrinio_daily_prices`; both positive, finite and non-null in a bounded 256-row pre-holdout sample | Whole-history coverage and adjustment semantics remain unaudited; no supplied turnover-ratio or shares-outstanding column |
| Siblis historical constituents | Dated S&P 500/400/600 membership and change/identifier history; membership table 14,747,642 rows | Signal-time join rules, publication/availability versus effective dates, resolved ID ambiguity and coverage for the chosen universe |
| Unified index/identity release | Identity bridge with 20,283 rows plus historical index membership | Whether every proposed constituent maps uniquely at every required date; identity joins need temporal validation |
| Historical market capitalization | **No capitalization or shares-outstanding field found across the three release table schemas** | Deliberately excluded from V2 by user direction; no source procurement needed |
| Sadig feature code | PRICE5 and earlier 740-feature construction code in the pinned ignored snapshot | Neither is a model-ready V2 panel; the precomputed 740-feature warehouse was not downloaded |
| Sadig derived strategy panels | A prior read of the pinned calendar returned `403 / storage.objects.get`; derived panel downloads were incomplete | Access was not retested on 2026-10-14; use the ignored owner-request record if reuse is needed |

Product-wide stated coverage is Intrinio 1962-01-02 to 2026-07-24, Siblis
1970-03-05 to 2026-03-31 and unified membership 1970-03-31 to 2026-04-30.
These are release properties, not verified completeness for individual securities
or a license to inspect protected 2024+ financial values.

Market capitalization is required for the paper's size-decile input, cap-based
screens and value weighting, but these are deliberately omitted from V2.
The 15-input starting set is a declared adaptation (a size-feature ablation),
not the complete 16-input paper baseline. Capitalization is not a V2 blocker
and no additional source is being sought. Price, volume, liquidity or S&P index
labels must not be passed off as the omitted size feature.

Volume can support separately specified liquidity/activity features.
[Intrinio documents](https://docs.intrinio.com/documentation/web_api/GetSecurityStockPrices_v2)
raw volume as shares exchanged and adjusted volume as adjusted for splits and
dividends. Raw close times raw volume gives a dollar-volume approximation;
share turnover requires shares outstanding and the paper's market turnover
requires capitalization. Those ratios are unavailable with our current inputs
and scope. Strategy portfolio turnover is a later trading-ledger calculation,
not a missing source-data column. See [data status](DATA_STATUS.md) for the
bounded sample and its limits; no volume-feature formulas are frozen here.

## 3. Candidate universe: one primary choice before results

| Candidate | Signal-time rule | Implication |
|---|---|---|
| S&P 500 | Constituents known/eligible as of the signal cutoff | More liquid large-cap sample; excludes the small-stock channel emphasized by the paper |
| S&P 400 | Historical mid-cap constituents at the cutoff | Different size/liquidity distribution; no assumption of the same return effect |
| S&P 500 + 400 | Dated union, deduplicated by stable security identity | Larger domestic cross-section; one combined cohort is selected; target ranks use observed outcomes within the signal-time Paper15 cohort |

Never substitute today's constituents or require a future price to admit a
security. Decide whether membership needs an effective date, an availability
date, or both; an after-cutoff announcement cannot become an earlier signal.
Resolve overlaps, share classes, exchange/type eligibility, entries/exits and
identifier conflicts explicitly. Decide whether features and target ranks use
only the signal-time cohort or a broader reference market; this changes the
meaning of the ten shared state inputs. The user selected the historical S&P 500/400 union; the target-ranking convention is fixed by [the shared contract](TARGET_CLASSES_RET.md).

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
3. **Build the 15-input V2 starting set.** At end of month `t`, use
   `MOM1=r[t]`, and compound `t-m+1` through `t-1` for `m=3,6,9,12`.
   Normalize within the declared signal cohort; append five means and five SDs.
   Omit the paper's historical size decile. Freeze window completeness, SD
   divisor, zero dispersion and feature availability. Record per-feature missingness.
4. **Build supervised targets.** For eligible signal cohort `U[t]`, rank realized
   month `t+1` returns into ten groups only once outcomes complete. Persist
   best-to-worst class mapping and rank/tie rules. Keep missing/terminal names
   in coverage and economic ledgers, even if their supervised label is unavailable.
5. **Fit at annual cutoffs.** Only completed labels available by the cutoff
   enter training. The paper expands classifier history, randomly splits
   historical rows 80:20 for early stopping and repeats 100 fits. Pin software,
   parameters and seeds; decide a separately labeled chronological-validation
   adaptation. V2 explicitly selects rolling ten-year training while retaining
   annual refits and monthly predictions; this departure from expanding paper
   history is recorded in the chronology configuration. Do not silently change
   the refit cadence or present V2 rolling history as the paper specification.
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

**USER DIRECTION, 2026-10-15:** research initialization is 2010, with rolling ten-year
training / one-year outcome tests, annual refits and monthly predictions. Values live in
[`research_workflow.json`](../configs/research_workflow.json); the
[master notebook](../notebooks/00_research_master.ipynb) previews four nominal
2020–2023 tests: 2010–2019 training precedes 2020 testing, advancing through
2013–2022 training / 2023 testing. This explicitly adapts the paper's expanding
classifier history. The one-date engineering check is a separate control.
Raw pre-2010 history is retained for exact per-signal warmup (503 sessions for
the complete registered bank). The [2010–2023 endpoint audit](INITIALIZATION_2010.md)
measured the 15 paper inputs; full-bank coverage is now measured in [the full-panel audit](FULL_FEATURE_PANEL.md); the 80% primary/90% sensitivity policies are selected, with residual coverage limitations.
See [feature contracts](FEATURES.md).

The paper requires at least ten years before initial model fitting and uses a
rolling ten-year class-payoff estimate. A source product reaching back decades
does not establish ten usable years for our chosen universe and return rules.
The exact first admissible forecast is determined by verified feature, label,
identity, membership and payoff coverage.

The prior recovery used 1998 warmup, 1999-2023 accepted signal-panel coverage,
2009-2023 primary inputs and 2015-2023 test folds. These are different roles,
not interchangeable sample dates. January 1998 data alone cannot justify a
paper-style initial fit in 1999. The available releases may support a longer
history, but the new design does not adopt those dates. The measured paper-input
coverage does not certify daily technical/context history or economic labels.
The first inner pilot now fixes 2010–2017 training and 2018–2019 validation.
Later outer-evaluation design and operational execution timing remain separate.

Protected 2024+ financial values remain uninspected. Define the development
period, annual outer forecasts, inner validation, tuning budget, embargo/purge
rules, intended holdout and acceptance criteria before evaluation. Overlapping
momentum histories and common monthly state inputs matter for validation;
group by time when defining the proposed temporal variant. The paper's random
past-row inner split can be preserved in a clearly labeled reference arm;
it is not permission to mix future test rows into model development.

## 6. Model and feature extensions: isolate the source of any gain

**PROPOSED comparison sequence, not a frozen search:**

1. V2 paper-mechanism reference: the declared 15 momentum/state inputs, ten classes,
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

Compare `15 starting inputs × estimator change` separately from `expanded inputs ×
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
| Primary S&P universe and target-rank cohort | Defines the prediction distribution and eligible trades | USER SELECTED historical S&P 500/400 union; target ranks use valid observed outcomes inside the signal-time Paper15 prediction cohort; missing outcomes remain explicit |
| Return/terminal-event accounting and identity rules | Determines whether labels and portfolio cash flows are credible | Provisional identity admission selected; knowledge-time identity and economic accounting remain unverified |
| Coverage and experiment chronology | Ten-year startup, development/outer tests, holdout | 2010 initialization and rolling10/1 selected; 15-input endpoint coverage measured; full-bank coverage measured; 80% primary/90% sensitivity selected; 2024+ protected |
| Active V2 underspecification conventions | Ties, SD, winsorization, moments, seed/early-stop mechanics | Active normalization/missingness and exact midrank ties are fixed; the first pilot freezes versions, seed 1729 and fixed rounds without early stopping; later ensemble mechanics remain OPEN; size encoding applies only to the paper account |
| Execution/cost/exposure/borrow rules | Separates predictive results from investable returns | OPEN |
| Models, feature groups and trial bounds | Prevents results-driven expansion of the search | First pilot fixed: RF/XGBoost × Paper15/expanded, one fit per arm; no tuning or outer evaluation; later experiments OPEN |
| Calibration/ensemble weighting and acceptance criteria | RET relies on probabilities and payoff estimation | OPEN |
| Resources, full-build/backtest plan | Controls compute and reproducibility | Explicit local pilot has measured read/memory budgets and fixed execution ceilings; backtest/cloud workflows remain separate and OPEN |

The [monthly return engineering stage](MONTHLY_RETURNS.md) now constructs bounded
provider-adjusted ratios, retains full signal cohorts and preserves daily data.
These are not certified total returns or a complete model-ready panel. Next are
economic event/coverage accounting and continued dated identity evidence review.
The [accepted identity waiver](PROVISIONAL_IDENTITY_POLICY.md) permits provisional
development while those source limits remain explicit. The user has selected the historical S&P 500/400 union as the primary modelling
cohort; its target-ranking and missing-outcome rules are now fixed by [the shared target contract](TARGET_CLASSES_RET.md). Capitalization
source procurement is excluded from the work plan. These are design tasks,
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

## Feature engineering decision, 2026-10-14

**USER DIRECTION / IMPLEMENTED:** daily technical indicators then exact month-end
snapshots, alongside the15 exact monthly paper momentum/state variables. The
728-definition bank has715 eligible predictors after three constants and ten
exact rank aliases. Complete required windows and sample SD(ddof=1) are explicit
V2 engineering adaptations. The initial complete-original-cohort paper policy
was superseded on 2026-10-15 by the active common-five policy below. Context
now uses 80% original-cohort coverage as primary and 90% as sensitivity.
Features and future targets have separate tables, loaders and allowlists;
completed outcome cutoffs and canonical calendar windows guard later X/y
alignment. No fit, preprocessing, trial selection or full panel is authorized
by notebook Run All. Source knowledge-time and terminal-accounting limits remain
unresolved; see [FEATURES.md](FEATURES.md).

## Active engineering policy, 2026-10-15

The [normalization/missingness protocol](NORMALIZATION_PROTOCOL.md) requires
exhaustive admitted-original requests and one common population with all five
finite exact momenta and positive finite sample dispersions. All 15 paper inputs
use that population; shared mean/dispersion levels are retained. Arbitrary
two-ID samples withhold those statistics. Context requires exhaustive admitted-ID requests against original
cohorts,80% per-variable/date/group coverage and dated daily membership, with 90%
as sensitivity. Complete individual temporal windows remain unchanged. Added-feature medians, missing
indicators and all-missing/constant exclusions are fitted only on explicit
completed training rows, including within inner splits; prediction consumes no
labels. The 2010 endpoint audit is coverage evidence, not a fitted preprocessing
manifest or authorization for a full model/backtest run.

## Full feature panel verification, 2026-10-15

The [full 2010–2023 feature panel](FULL_FEATURE_PANEL.md) is now built and audited under the provisional identity policy. It retains 152,307 original rows and 168 monthly signals. Daily observations feed technical/context calculations and are also preserved independently for later backtesting, including complete annual warmup and signal anchors. All 715 eligible names remain in the dictionary; missing context is reported rather than filled or silently excluded. The master verifies saved coverage and exposes separate lazy feature/daily datasets. Notebook Run All does not relaunch the full build, fit imputation/models or backtest.

## Initial classifier comparison, 2026-10-15

**USER DIRECTION:** the primary modelling universe is the dated historical S&P
500 plus S&P 400 union. **PROPOSED:** compare XGBoost and random forest on Paper15
and the expanded 715-name candidate bank, sharing prediction eligibility, labels,
outer folds, preprocessing policy and RET payoff estimates. The
[modelling sketch](MODEL_COMPARISON.md) records the paper/V2 distinctions, proposed
chronological inner validation, pilot boundary and remaining label/compute rules.
Its standalone draft and metadata worker are displayed in the master notebook;
no financial values were read and no model, preprocessing or backtest was fitted
for this design task. This draft leaves existing pinned panel configs unchanged.

## Shared monthly class/RET engineering, 2026-10-15

The [shared workers](TARGET_CLASSES_RET.md) now implement exact tie-aware monthly
classes, original-cohort/missing-outcome ledgers, the label-aware training adapter
and strict 120-calendar-month realized-class RET payoffs. Forward-outcome requests
are distinguished from past feature requests. The master checks already loaded
bounded samples and keeps daily/feature data separate. Full historical label and
payoff coverage, estimator fitting and economic accounting were unverified at
that shared-worker stage; no model or backtest ran for that implementation.


## Full target/RET coverage, 2026-10-15

**USER DIRECTION / VERIFIED LOCAL:** the separately authorized
[full 2010–2023 target audit](FULL_TARGET_PANEL.md) preserves all 152,307 original
cohort rows and the same 140,777 Paper15 prediction rows. It releases 140,485
classes, with 276 missing-end and 16 source-quality eligible outcomes left
unlabelled. All 168 months contain all ten classes. All 48 nominal 2020–2023
RET states have complete preceding 120-month grids; the first available signal
is December 31, 2019 for January 2020. This resolves empirical structural
class/payoff coverage under the provisional policy, while terminal cash flows
and knowledge-time certification remain open. That target audit fitted no model
or preprocessor.


## Daily development portfolio decision,2026-10-15

USER DIRECTION: [daily development backtesting](DAILY_BACKTEST.md) is now
authorized for saved 2018–2019 probabilities, leaving 2020–2023 unevaluated.
The user selected 50% long / 50% short, about ten names per direction, and a
separate 10% stop per asset. The concrete protocol fixes ten per side within
mode-class 1/10 or separately2/9, ranked by corresponding class probability.
These are V2 breadth/capital adaptations. RET remains the intended primary
score but is withheld for this development period under the unchanged 120-month
gate. Daily adjusted-OHLC proxy fills, costs, drift, stop and fail-closed coverage
rules are recorded separately in `configs/daily_backtest.json`. Original model
fits, identity policy, feature/target recipes and daily inputs remain unchanged.
