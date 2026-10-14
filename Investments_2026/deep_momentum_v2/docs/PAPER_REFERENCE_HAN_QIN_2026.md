# Han-Qin Deep Momentum: source-locked paper reference

Reviewed on 2026-10-14. This is an original methodological analysis of the
user-supplied paper, not a reproduction of its PDF or a claim that its empirical
results have been replicated. Read this before designing V2 features or models.

## 1. Identity, version and evidence conventions

**Source:** Chulwoo Han and Chang Qin, *Bimodality Everywhere: International
Evidence of Deep Momentum*, cover date March 26, 2026, SSRN abstract 4452964.
The supplied file has **115 PDF pages**: main-paper pages 1-52 followed by
Internet Appendix pages 1-63. An appendix locator `IA p.12 / PDF p.64` means
both refer to the same physical page. Its SHA-256 is
`6156a8f6ae5e538901e0a014109b5477f328c662ca9dcfe83783850846fa15ed`.

The complete text, main tables, Appendix A, Internet Appendix tables IA1-IA55,
and rendered figures were reviewed. Equations on pp.10-15 and the schematic on
p.37 were checked visually against the extracted text. The original PDF is kept
locally under ignored `reference/papers/`; it is not included in Git.

[SSRN source page](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4452964).
The source page lists another, 122-page version posted in July 2026. That is not
the file supplied here and was not substituted for it. SSRN landing-page dates
and suggested citations do not override the date printed on our locked PDF.

Evidence labels used below:

- **PAPER:** stated in this specific PDF, with a page/section locator.
- **INTERPRETATION:** a mathematical implication or an operational reading.
- **VERIFIED LOCAL:** inspected in our source schemas or reference files.
- **PROPOSED:** a V2 adaptation, not something the authors did or we have frozen.
- **OPEN:** information missing from the paper or a project decision still needed.

## 2. The actual idea and the important naming distinction

**PAPER, pp.2-3 and 11-15:** both past winners and past losers can have sizeable
probability mass at both extremes of the next month's relative-return
distribution. A most-likely class can therefore be a poor portfolio-selection
criterion. The method predicts the *whole probability vector*, translates it
into a financial score, and sorts stocks on that score.

The main pipeline is:

```text
monthly stock information at t
    -> classifier for the ten cross-sectional return classes of month t+1
    -> probability vector p[i, 1:10]
    -> expected-return score RET (or DPR/SRP score)
    -> monthly long-short portfolio
```

**PAPER, section 3.3.1, p.12:** this version uses an **XGBoost multiclass
classifier**. It discusses Han (2022)'s neural-network predecessor but replaces
that classifier for practical tuning/computational reasons. It does not specify
an LSTM, a recurrent sequence architecture, or training by a portfolio-Sharpe
loss. The word "Deep" is the method's name; it is not an instruction to require
deep neural networks in V2.

**VERIFIED LOCAL:** Sadig's recent recovered U.S. comparison uses five
daily price-derived features, a 20-session next-open return target, and scalar
ridge/MLP/LSTM forecasts. That is a different prediction problem. Its data
audits and economic failure evidence are useful references; its target,
architecture, five-feature list and 20-offset schedule are not this paper's
replication specification. The legacy monthly implementation in
`../deep_momentum/` contains related classification/reclassification code, but
is retained as a reference and is not automatically accepted as correct.

**USER DIRECTION:** V2 will use U.S. stocks; candidate universes are historical
S&P 500, S&P 400, or their union. Feature expansion and forest ensembles are
intended extensions. They should preserve the probability-to-payoff mechanism
if they are presented as extensions of this paper.

## 3. Data, frequency and sample construction

### 3.1 What the authors used

**PAPER, section 3.1, pp.8-10:**

- International stocks: Refinitiv Datastream daily data from WRDS
  `wrds_ds2dsf`; monthly observations are the last stock record in each month.
- International monthly returns: ratios of Datastream Return Index `ri`.
- U.S. stocks: **CRSP monthly stock data**, substituted for Datastream.
- Book equity for factor construction: WRDS WorldScope `wrds_ws_funda`.
- The final sample has 45 countries. Countries need data from 2000 or earlier
  and at least 30 stocks per month, with the stated Greek-market exception.
- The U.S. raw sample is aligned to January 1965-December 2023, with the U.S.
  classifier retrained annually beginning January 31, 1975.
- Common-country headline strategy results are January 2010-December 2023.
  Appendix all-period results use each country's available out-of-sample history.

The paper's U.S. sample is a broad CRSP ordinary-share sample. It is not an
S&P-constituent sample and it does not begin in 1998. Intrinio plus Siblis and an
S&P 500/400 restriction are data/universe adaptations, even with the same model.

**PAPER/OPEN, pp.9-10 and 28:** the authors compare Datastream capitalization
with World Bank aggregates and report discrepancies in six countries; Argentina
and Brazil exceed the comparison by more than 100 times. The data section says
the discrepant countries are excluded from global portfolios, while the later
portfolio section explicitly names only Argentina and Brazil; country-model
results retain them. Preserve this exclusion-description inconsistency if
extending the global analysis. It does not resolve our missing U.S. historical
capitalization or validate every source record.

### 3.2 Cleaning rules as written

**PAPER, pp.8-9:** exclude observations when:

1. Current and previous monthly trading volumes are both zero and return is zero.
2. Current or previous monthly market capitalization is unavailable.
3. Market capitalization falls in the bottom 5% of the country's cross-section.
4. Monthly return is greater than +300% or less than -95%.

Remaining returns are winsorized at the 1st and 99th percentiles within country.
The stated share filters are Datastream `typecode='EQ'`, CRSP `shrcd` in
`(10, 11, 12)`, and primary country quotes `isprimqt=1`. Preserve those exact
codes in a paper account; do not silently replace the unusual CRSP list with a
remembered conventional list. The paper explicitly rejects a minimum number of
daily observations per month because that would wrongly remove holiday months.

**OPEN:** the precise time axis for winsorization thresholds, treatment of
missing volume, timing of the capitalization screen, and reconciliation of
share/quote filters between the two providers are not fully specified. Do not
use whole-history quantiles in a causal prediction pipeline without disclosing
the future-information problem. The text's "in any month" is not permission to
exclude a stock's entire history based on a future small-cap observation.

**V2 implication:** retain raw records and exclusion denominators. A cleaned
learning sample is not an economic accounting rule. A holding's delisting loss,
payout, or missing quote cannot be erased because the training-sample filter
rejects that observation. Winsorized learning returns must not silently replace
realized portfolio cash flows. Provider-adjusted prices need their own return
and corporate-action audit before being called CRSP-equivalent total returns.

## 4. Exact time indexing and the 16 features

### 4.1 Monthly clock

**PAPER, footnote 3, p.12:** at the end of month `t`, the algorithm predicts
the return during **month `t+1`**. `r[i,t]` is already observed when this
prediction is made. These are calendar-month variables, not 20-session
substitutes. A stock's missing month must remain missing rather than shifting
its remaining observations together.

### 4.2 Momentum windows

**PAPER, equations (6)-(7), p.12:**

\[
M_i^{(m)}(t)=\prod_{j=t-m+1}^{t-1}(1+r_{i,j})-1,
\quad m\in\{3,6,9,12\};\qquad M_i^{(1)}(t)=r_{i,t}.
\]

The long windows exclude the most recent month `t`; the one-month feature
includes it. The nominal `m` window has `m-1` compounded monthly returns:

| Feature | Included return months | Count |
|---|---|---:|
| `MOM1` | `t` | 1 |
| `MOM3` | `t-2` through `t-1` | 2 |
| `MOM6` | `t-5` through `t-1` | 5 |
| `MOM9` | `t-8` through `t-1` | 8 |
| `MOM12` | `t-11` through `t-1` | 11 |

**INTERPRETATION:** if `Q` is a valid monthly total-return index, the long-window
formula telescopes to `Q[i,t-1]/Q[i,t-m]-1`. In particular, `MOM12` is
`Q[i,t-1]/Q[i,t-12]-1`, not `Q[i,t]/Q[i,t-12]-1`.

### 4.3 Cross-sectional normalization and market state

**PAPER, equation (8), pp.12-13:** for each horizon, country and signal month,
compute the cross-sectional mean `a_m(t)` and standard deviation `s_m(t)`:

\[
zM_i^{(m)}(t)=\{M_i^{(m)}(t)-a_m(t)\}/s_m(t).
\]

The input set has **16 columns**:

- Five normalized stock momentum values: `zMOM1/3/6/9/12`.
- Five cross-sectional means: `meanMOM1/3/6/9/12`.
- Five cross-sectional standard deviations: `stdMOM1/3/6/9/12`.
- One market-capitalization size-decile feature, with values 1-10.

The introduction's "15 momentum-related features and a size feature" means
16 total. The ten cross-sectional mean/std inputs are shared by stocks at the
same signal month; they are time-varying market-state inputs, not extra
stock-specific ranks. Dropping them changes the paper model.

**OPEN:** the standard-deviation divisor, exact cross-section after missingness,
tie handling, size-decile direction, and numerical treatment of zero dispersion
are not specified. Define them before implementation. We can state a proposed
size convention of 1=smallest and 10=largest, but cannot attribute that direction
to the paper. Do not fabricate size with share price, dollar volume or index
membership. All normalization inputs must be available at the signal date.

## 5. Target and probability interface

**PAPER, section 3.3.1, p.12:** classify next-month stock returns into ten
cross-sectional return groups within country, then learn probabilities of those
classes from the signal-month features. The target is a categorical relative
return, not a direct return-regression target.

For clarity, our mathematical notation uses `k=1` for highest-return class `H`
and `k=10` for lowest-return class `L`, consistent with the descending-return
notation used in the DPR formula. This is a notation convention, not a claim
about the authors' undocumented software encoding. Any code encoding is valid
only if the label-to-column map is persisted and every score uses that same map.

**INTERPRETATION:** labels for signal `t` become observable only after the outcome
month `t+1` completes. At a training cutoff, include only rows with observed,
completed outcomes. The newest signal row normally has no known label. Future
monthly ranks are target construction, not a signal-time feature or selection
rule. Do not calculate targets against a full-period return distribution.

**OPEN:** target rank/tie algorithm, the exact cohort used for future ranks,
and behavior when classes contain few observations need explicit conventions.
For our adaptation, signal-time historical constituents determine the starting
cohort. Future membership or availability of a future quote must not determine
whether a stock could have been selected. Keep unlabelled names in coverage
and portfolio-accounting denominators.

Every fitted model must expose an aligned probability vector `p[i,k]`, with
finite nonnegative entries and sum one. Persist `classes_` or equivalent mapping.
Report any class absent from training; never assume column positions are
automatically comparable across fitted models.

## 6. Training schedule and the existing ensemble

**PAPER, section 3.3.3, pp.14-15:**

- At least ten years of source history before the first fit.
- Annual retraining, expanding the historical classifier-training sample.
- At each retraining date, random 80:20 split of historical rows into fit and
  validation sets; validation determines early stopping.
- XGBoost otherwise uses default hyperparameters, without an extensive search.
- Train **100 times at each annual training date**; average the class
  probabilities to make predictions during the ensuing monthly forecasts.

The footnote's minimum 1,400 fits = 14 years x 100, and the feature-importance
description's 100 fits per year on p.25, confirm annual rather than monthly
refitting. "Each training month" on p.15 identifies the scheduled training
month, not 100 new fits every forecast month.

**OPEN:** XGBoost version/interface, default parameter snapshot, boosting-round
cap, early-stopping patience/metric, split seeds/stratification, source of
variation across the 100 fits, final refit policy, and best-iteration prediction
behavior are not provided. Repeating identical deterministic defaults on an
identical split would not produce a meaningful new ensemble. Do not invent a
seed scheme or a numeric patience and present it as an author setting.

The random inner split is a stated paper choice. An outer forecast uses only
historically observed rows, so this does not automatically mean future test
data were used. However, random row splitting can mix correlated observations
from the same month and overlapping feature histories. Our chronological or
blocked-validation variant must be labeled as an adaptation, not quietly
described as the paper's split. Do not use random folds across future test years.

## 7. Reclassification: probability predictions into financial scores

### 7.1 RET: historical class payoffs

**PAPER, equations (9)-(12), pp.13-14:** let `g[k,s]` be the equal-weighted
average realized return of the stocks assigned to return class `k` in completed
month `s`. Estimate the class payoff `mu[k,t]` as its time-series mean over the
past ten years. The stock's score is

\[
RET_i(t)=\sum_{k=1}^{10}p_{i,k}(t)\,\widehat\mu_k(t).
\]

Important distinctions:

- The classifier's history expands; the class-payoff history is a **rolling
  ten-year window**. They are not the same estimator.
- Average the monthly class-return means across months. Pooling all stock rows
  over ten years instead weights months by their class sample size and differs
  from the specified time-series-average estimator.
- Estimate class payoffs from actual realized return-class membership, not
  predicted-mode buckets or subsequent strategy-portfolio returns.
- Use only completed historical outcomes. Test-month class payoffs are never
  allowed in that month's signal score.
- A vector of decile numbers or evenly spaced arbitrary payoffs is not RET.

**INTERPRETATION/OPEN:** monthly updating is the natural operational reading of
"past ten years" at a monthly prediction date. The paper does not give the exact
update cadence, endpoint inclusions, minimum-observation rules or handling of
missing months. A rolling 120 completed-calendar-month rule is a reproducible
proposal that must be recorded, not an undocumented numerical default.

The total-expectation formula motivates the score. Using unconditional
historical class means for feature-conditional stock predictions is an
approximation; class payoffs and calibration can change with regime. A RET
score is not a guarantee of an executable stock's expected realized return.

### 7.2 DPR: the printed probability-difference score

**PAPER, equation (11), p.14:** with our descending class convention,

\[
DPR_i=\sum_{k=1}^{5}(p_{i,k}-p_{i,11-k})(6-k).
\]

The printed weights are `(5,4,3,2,1,-1,-2,-3,-4,-5)`. Preserve this formula
if implementing the paper's DPR diagnostic. The accompanying text describes
linearly decreasing class means; a mathematically centered equally spaced
ten-class vector would have different middle spacing. This discrepancy is an
ambiguity, not permission to silently rewrite equation (11). RET does not need
that linear-spacing assumption.

### 7.3 SRP: mixture variance and individual-score Sharpe

**PAPER, equations (13)-(14), p.14:**

\[
v_i=\sum_kp_{i,k}(\widehat\sigma_k^2+\widehat\mu_k^2)-RET_i^2,
\qquad SRP_i=RET_i/\sqrt{v_i}.
\]

The score does **not** subtract a risk-free rate. The between-class term must be
included; `sum(p*sigma^2)` alone is wrong. It is not the standard deviation of
the probability vector, not an average of class Sharpes, and not a portfolio
Sharpe optimizer. No cross-stock covariance optimization is specified.

**OPEN:** the exact class-variance sample estimator is less explicit than the
class-mean estimator. State whether it is the temporal variance of monthly
class-average returns or a within-class individual-return variance, its divisor,
window and weighting. A literal class-return reading suggests the former, but
the mixture's interpretation needs scrutiny because diversified class-average
variance need not equal conditional individual-stock variance. Do not silently
substitute a pooled stock-return variance. Treat zero variance explicitly;
only negligible negative roundoff may be tolerated under a declared rule.

This PDF develops DPR, RET and SRP. CVaR/CVR scores found in legacy code are
extensions, not a method specified by this paper.

## 8. Portfolio formation, accounting and performance

**PAPER, sections 3.2-3.3 and 4.2, pp.10-15, 19-24:**

- `MOM`: rank on the lagged eleven-month momentum; buy the highest decile and
  sell the lowest decile.
- `XGB`: use the mode of the averaged probability vector; hold all names
  classified as highest or lowest class. Those groups need not contain 10% each.
- `RET`, `DPR`, `SRP`: rank their score and form top/bottom-decile portfolios.
- Rebalance monthly. Main analysis is equal-weighted, with value-weighted
  results reported separately.

Ranking `p(H)` alone, selecting a fixed top 20, holding only longs, or using
20 overlapping daily offsets changes the portfolio rule. A U.S. long-only
variant may be useful, but is a named extension, not the reported long-short
strategy. Empty mode-selected legs and score/rank ties require frozen rules.

**INTERPRETATION:** the standard H-L convention is a unit long leg minus a unit
short leg, gross exposure 2 and net exposure 0. Returns must be distinguished
from a 50%-long/50%-short gross-one normalization. Report convention, financing,
borrow availability/cost and short proceeds; the PDF does not fully specify an
operational execution ledger, same-close versus next-open fills, or borrowing.

**PAPER, p.24 and Table IA6 (PDF p.64):** turnover is the monthly change in
portfolio weights, with a maximum 400% when both unit legs are fully replaced.
Break-even flat cost is found in increments of 1 bp until net profit becomes
nonpositive. This is a sensitivity diagnostic, not a measured execution-cost
estimate or a borrow/impact model. Weight drift, precise cost timing and the
meaning of "net profit" in that search are not fully specified.

Unit trap: IA6's turnover cell `2.574` for U.S. equal-weight RET represents
257.4%, consistent with its prose convention; its break-even cost `0.900`
under the percent heading represents 0.9% = 90 bp. Do not read either as
2.574% turnover or 0.9 bp. Retain units explicitly in code and reports.

The paper reports **annualized mean return**, not CAGR. Table values must not be
relabeled as compounded growth. Conventional annualization is `12*monthly
mean` and `sqrt(12)*monthly Sharpe`; the precise risk-free and sample-SD choices
must still be declared. Appendices' cumulative figures are in log-scale plots.
Report arithmetic annualized mean, CAGR, volatility, Sharpe, drawdown, long and
short legs, gross/net results, and turnover separately in V2.

## 9. Bimodality: the diagnostic actually used

**PAPER, equations (1)-(5), pp.10-11; Table 2, pp.40-41; Figure 2, p.37:**
for a set predicted to be winners, calculate fractions that realize in highest
and lowest next-return deciles; do the analogous calculation for predicted losers.

\[
HH=P(actual\ H\mid predicted\ H)-0.1,\quad
HL=P(actual\ L\mid predicted\ H)-0.1;
\]
\[
LL=P(actual\ L\mid predicted\ L)-0.1,\quad
LH=P(actual\ H\mid predicted\ L)-0.1;
\]
\[
BM=-\{(HH-HL)+(LL-LH)\}/2.
\]

Use *all members of the predicted group* in each denominator, including names
that realize in one of the eight middle classes. The PDF's binary TP/FP
schematic omits those classes; a literal two-extreme-only denominator would
make the random baseline 0.5 rather than the stated 0.1 and contradict the
reported table values. Record this notation problem rather than implementing
the diagram as a two-class filter.

`BM=-1` corresponds to perfect persistence, `BM=1` to complete opposite-tail
outcomes. A value near zero alone is not proof of two modes: a distribution
concentrated in the middle can also produce weak persistence differences. The
authors acknowledge this limitation and inspect the distributions visually.
Do not substitute Sarle's skewness/kurtosis bimodality coefficient or a formal
modality test for this paper-defined diagnostic. For K-class extensions, the
random baseline is `1/K`, with tail/group definitions declared separately.

## 10. What the empirical evidence does and does not establish

### 10.1 Correct U.S. reference numbers

**PAPER, Tables 5 and 7, pp.44-48, country models, January 2010-December 2023:**

| U.S. strategy | EW annualized mean | EW Sharpe | VW annualized mean | VW Sharpe |
|---|---:|---:|---:|---:|
| MOM | 10.8% | 0.481 | 13.6% | 0.461 |
| XGB mode | 22.4% | 1.950 | 18.7% | 0.895 |
| RET | 28.9% | 1.765 | 20.0% | 0.833 |

RET raises U.S. mean returns over XGB in that period but **does not raise its
Sharpe** under either weighting. The value-weighted U.S. RET-XGB mean-return
difference has t-statistic 0.323. Do not claim universal or U.S.-specific
risk-adjusted superiority from the paper's cross-country average.

Table IA55 / PDF p.115 uses the longer available U.S. out-of-sample period.
Its EW RET mean 50.6% and Sharpe 2.855 are not the 2010-2023 statistics.
The global RET Sharpe 8.984 in Table 10 is an international Global-2,
local-selection result. It is not a target for our restricted U.S. universe.

### 10.2 Findings across the whole paper

- Bimodality is widespread and persistent; its cross-country association with
  momentum profits is strongly negative. Behavioral and market-state
  regressions are explanatory analyses, not extra classifier inputs.
- In the common 2010-2023 window, cross-country EW averages are MOM 20.7% /
  Sharpe 0.980, XGB 31.1% / 2.128, and RET 41.4% / 2.491 (Table 5). Use
  table-specific values: prose has small inconsistencies such as 20.9 versus
  20.7, and 16.8/20.4 versus IA2's 16.7/20.3 precision percentages.
- Value weighting reduces the average ML gains. Removing the smallest stocks
  also reduces performance. S&P 500/400 replication should not inherit the
  broad-sample/microcap-heavy effect size as an expected result (pp.23-24,
  IA5 and IA10).
- U.S. feature importance ranks normalized long momentum, size, and normalized
  one-month return highly (IA7 / PDF p.65). Importance is not causal proof and
  should not be used to redesign a supposedly untouched test retrospectively.
- Regional/global factor regressions use market, size, value and momentum
  factors. The main text uses WorldScope book equity for this extension;
  book equity and country behavior indices are not among the core 16 inputs.
- Monthly turnover is high. Break-even costs do not resolve real short
  borrowing, impact, execution delay or provider quality limitations.

### 10.3 Prediction diagnostics and explanatory evidence

**PAPER, pp.15-19, Tables 2-4 and IA2 / PDF p.58:** classifier accuracy averages
14.1% across countries versus the ten-class 10% reference. U.S. accuracy is
14.8%, with highest/lowest-class precision 17.8%/22.2%. Mode-predicted group
sizes need not match target deciles: across countries the classifier assigns
9.7% to the highest class and 17.8% to the lowest. This is another reason not
to replace the XGB mode rule with fixed deciles of `p(H)`.

The cross-country rank correlation of BM with momentum mean return is -0.856.
Persistence analysis sorts countries by past one-year mean BM into quintiles,
then examines the following year's BM; a Fama-MacBeth lagged-BM regression
provides another persistence check. The disposition index is examined similarly.
These are country-level repeated cross-sections, not evidence of independently
distributed stock-month training rows.

The determinants regression relates BM to country individualism, disposition,
market turnover and volatility. Turnover is market dollar volume divided by
capitalization; the paper's volatility proxy is the cross-sectional mean of
**squared monthly stock returns**, not rolling market-return volatility and
not a demeaned sample variance. Individualism/disposition coefficients are
negative; turnover/volatility coefficients are positive. Loss-associated
disposition is more relevant than gain-associated disposition. The proposed
behavioral interpretation is that underreaction preserves momentum while
correction/reversal increases opposite-tail outcomes. This is observational
association, not causal identification or a mandate to add these variables to
the 16-input classifier.

The Table 2 crash-rate diagnostic is the fraction of annual blocks with maximum
drawdown over 20%, averaged over twelve monthly offsets of the annual blocks.
It is not monthly loss probability or the maximum drawdown statistic itself.
Any replication of this diagnostic must specify block boundaries, wealth
construction and incomplete blocks; V2's ordinary drawdown report is separate.

## 11. Joint learning, explanatory appendix and robustness

**PAPER, section 4.3, pp.27-31:** Global 1 pools countries and defines feature
normalization, size and target classes globally. Global 2 pools countries but
keeps their local feature and class definitions. Both add country indicators.
Global and local portfolio selection are separate choices. The pooled model's
benefit is not evidence that a U.S. forest ensemble will achieve its numbers.
No cross-country model can be reproduced from our U.S.-only inputs. A pooled
S&P 500/400 model is a domestic adaptation, not the paper's global experiment.

**PAPER, Appendix A, pp.35-36:** the disposition index is 1,000 times the
coefficient on capital-gains overhang in a weekly cross-sectional predictive
regression. Its controls are recent four-week return, longer 52-to-5-week
return, mean turnover and log size. The reference price uses a turnover-weighted
52-week price history. Positive/negative-overhang versions are also examined.
These country-level behavioral diagnostics are not needed for the core U.S.
classifier; adding them would require new data and a separately defined feature.

**PAPER, section 4.4 and IA10 / PDF p.70:** robustness includes excluding the
smallest 10%/20%, K=5/20 classes, and 2000-2009 / 2010-2019 / 2020-2023
subperiods. Larger K also selects more extreme, smaller portfolio tails;
it is not a pure classifier-only comparison with fixed breadth. Some countries
are excluded from early subperiod comparisons due to later histories.

The entire Internet Appendix was reviewed: IA1 all-period momentum/bimodality;
IA2 classification; IA3-IA4 all-period EW/VW; IA5 size double sorts; IA6 costs;
IA7 importance; IA8-IA9 pooled-model country portfolios; IA10 robustness;
IA11-IA55 country-specific legs, risk/return metrics, cumulative curves and
return-class histograms. These are reported findings, not executable source code.

## 12. Reproducibility limits to carry into implementation

The following are not numerically resolved by this supplied PDF: software/default
versions; early-stopping settings; 100-fit seed mechanics; exact class ordering
in software and tie handling; size encoding; SD divisors; zero-dispersion rules;
class-variance estimator; rolling payoff endpoint/update/missingness rules;
winsorization time axis; sample-filter timing; exact execution and delisting
treatment; empty-leg policy; drift-adjusted turnover/cost ledger; portfolio
risk-free/capital convention; and Newey-West bandwidth. No author implementation
was verified in this review. Public third-party implementations are not evidence
of these missing author settings.

The source also contains the DPR-spacing and binary-confusion notation
ambiguities discussed above. We can reproduce the clearly specified mechanism
and document deterministic conventions. We cannot honestly promise an exact
numerical recreation of all published tables from this PDF and our data alone.

Use [the V2 adaptation plan](US_REPLICATION_PLAN.md) and the decision draft in
`configs/paper_replication_draft.json`. All proposed changes must remain
distinguishable from the locked paper account. No new model fits or backtests
were performed as part of this reading task.

## 13. Supporting primary sources

- [The supplied paper's SSRN record](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4452964).
- [Han's earlier Management Science article](https://pubsonline.informs.org/doi/10.1287/mnsc.2021.4189): background on the predecessor, not a substitute for this PDF's specification.
- [Official XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html): `multi:softprob` exposes a full multiclass probability output; library defaults must be pinned rather than attributed to the authors.
- [Official RandomForestClassifier documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html): forest probabilities average the trees' class probabilities, so probabilities can feed the same financial-score layer.
- [Official ensemble/voting documentation](https://scikit-learn.org/stable/modules/ensemble.html#voting-classifier): soft voting combines class probabilities; V2 probability alignment/calibration and chronological validation remain explicit design work.
