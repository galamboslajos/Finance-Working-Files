# Draft 13F Asset Embedding Model Comparison

This note is for a colleague reviewing our first model comparison. The immediate objective is to reproduce the *methods and benchmarks* of Gabaix, Koijen, Richmond, and Yogo's **Asset Embeddings** (September 2024 draft) as closely as the available data permit. We have built a point-in-time 13F adaptation of one paper benchmark: predicting a masked large holding from the rest of a manager's portfolio. The current scores measure portfolio-completion ability, not future returns, investment performance, or a full replication of the paper's results.

The quantitative results below are aggregate statistics from ignored local run files, checked against the complete locally decision-ready run cohort. The public repository contains the code and this explanation, but not the underlying holdings or executable outputs. The [results notebook](../notebooks/07_explain_13f_model_results.ipynb) reconstructs the tables and plots for a colleague with authorized local data access. All figures in this note are **draft findings under one fixed configuration**, not a selected final model.

## 1. Research question and the paper's argument

The paper asks whether investors' holdings reveal latent characteristics of assets that are not fully captured by conventional accounting variables. In its notation, investor $i$ holds asset $a$ in quarter $q$, and a low-dimensional vector $x_{a,q}$ represents the asset. Investors with different preferences and information sort assets into portfolios; the resulting cross-investor pattern can help identify those vectors. The theoretical demand model motivates the use of holdings, but an embedding's economic meaning has to be tested rather than assumed. See the paper's Sections 2–3 and [paper notes](PAPER_NOTES.md).

The paper tests three distinct uses of embeddings: relative valuation (RV), return comovement (RC), and asset similarity in managed portfolios (ASMP). RV is held-out cross-sectional valuation-residual explanation. RC uses preceding-quarter embeddings to explain held-out *cross-sectional comovement* in subsequent monthly returns after fitting that month's factor realization; it is not a live return forecast. ASMP hides a large holding and asks whether the remaining portfolio helps identify it. Our completed model runs address **ASMP only**.

## 2. Working definitions and notation

| Term | Symbol or field | Meaning in the current 13F work |
| --- | --- | --- |
| Manager | $i$, `manager_cik` | SEC filer identified by Central Index Key (CIK). It can aggregate multiple funds or strategies; it is not a fund identifier. |
| Asset | $a$, `typed_security_id` | A source-reported, normalized nine-character CUSIP **plus instrument channel**. The baseline uses the `cash_share_candidate` channel, not a verified common-equity or company identity. Raw identifiers are never included here. |
| Economic date | `period_of_report` | Quarter-end date to which the disclosed holdings refer. It is not the date on which we could have observed them. |
| Filing and availability | `filed_at`, `availability_timestamp_utc` | Source filing time and earliest permitted information-use time. A filing enters a historical state only after the relevant availability cutoff. |
| Decision time | $d_q$ | Standardized cutoff used to construct the point-in-time state for report quarter $q$. The decision-ready matrix cannot use a later filing or amendment. |
| Holding value and rank | $H_{ia,q}$, $r_{ia,q}$ | Positive disclosed position value and descending within-manager rank; deterministic security-ID order breaks ties. This is not an executed trade weight. |
| Fit and test managers | $F_q$, $T_q$ | Disjoint manager sets selected within quarter $q$ before fit-side universe filtering. |
| Scoreable test managers | $S_q\subseteq T_q$ | Test-query entrants whose rank-two target, visible rank-one asset, and candidate set pass the frozen-vocabulary checks. Report coverage alongside scores. |
| Fit vocabulary | $V_q$ | Typed securities retained after applying the universe rules to **fit managers only** in quarter $q$. It stays frozen for test managers. |
| Visible and candidate assets | $O_i$, $C_i=V_q\setminus O_i$ | Holdings shown to a model after rank two is hidden, and the eligible candidates not already visible. The true target must be in $C_i$. |
| Asset embedding | $e_a\in\mathbb{R}^K$ | Learned $K$-dimensional representation of a security. Distances or individual coordinates are not economically identified without a normalization and validation task. |
| Model score | $s_i(a)$ | A model's logit for candidate $a$ in manager $i$'s masked query. It becomes a probability after softmax over $C_i$. |

Our 13F filer-level, typed-security notation is deliberately different from the paper's company-level, mainly fund-level $i,a$ panel. The paper also defines shares $Q_{ia}$, price or market capitalization $P_a$, dollar holdings $H_{ia}=Q_{ia}P_a$, and log holdings $h_{ia}=\log H_{ia}$; its recommender-system level models use that richer information. Our current PCA baselines use only ownership or within-manager rank.

## 3. How a model-ready quarter is built

The sequence is **economic quarter-end → filing availability → decision-time manager state → fit/test manager split → fit-only security universe → model fit and masked test**. The availability ordering is essential: an amended filing changes the usable state only when that event becomes available, never retroactively at the quarter-end date. New-holdings amendments supplement a state, restatements replace it, and notices, combination reports, inconsistent joins, and incomplete states remain explicit or ineligible. See [data-spine decisions](13F_DATA_SPINE.md) and [data specification](DATA.md).

For the baseline, eligible positive `cash_share_candidate` lines from complete manager states are combined by typed security. We accept only a source-reported nine-character CUSIP in the permitted format; this is not an external validity or check-digit certification. We do not infer missing digits or silently join a present-day issuer mapping. On fit managers, the maximum-position concentration rule is 75%, and the minimum 20 assets per manager / 20 managers per asset conditions are reapplied together until they both hold. This produces a sparse manager-security matrix. These are sample-construction choices, not claims that every retained row is verified operating-company common stock.

Each quarter independently splits point-in-time-eligible manager IDs into fit and test sets using a deterministic seed-17 80/20 split **before** the universe filters. The surviving fit-filtered portfolios, left unmasked, define the vocabulary and train the models. Test managers do not contribute to the fit universe, model weights, or popularity counts. We hide each test manager's original second-largest positive eligible position; the target and rank-one holding must be in the fit vocabulary. Later out-of-vocabulary visible positions are dropped and counted, so evaluation is conditional on scoreability. All scorers see the same remaining holdings and candidate set within a quarter. This corrects the earlier saved-matrix pilot's all-manager prefilter leakage, but it does not create a chronological out-of-sample forecast. A manager may fall on different sides of the split in different quarters.

## 4. Models and actual building choices

| Model | How the candidate score is formed | What this run actually tests |
| --- | --- | --- |
| Uniform | Equal logits over $C_i$. | The ASMP likelihood zero point, not a fitted embedding. |
| Popularity | Fit-manager ownership frequency transformed to a log score. | Whether a simple tendency for widely held assets explains the target. |
| RS-Binary | Centered sparse PCA on fit-manager × asset ownership indicators. A held-out manager's visible holdings form a zero-filled frozen-vocabulary row projected into fixed loadings. | A global, low-rank co-ownership representation, at 4 and 10 dimensions. |
| RS-Ranks | The same PCA construction on descending within-manager percentile ranks, with unheld assets set to zero; test rows use the same zero-filled projection. | Whether portfolio ordering adds information beyond binary ownership, at 4 and 10 dimensions. |
| Word2Vec | Continuous-bag-of-words: average embeddings of visible assets within a rank window, then score a candidate with a tied embedding-vector dot product and full softmax. | Local rank-neighborhood substitution, with a 32-dimensional vector in the current neural comparison. |
| AssetBERT | Learned asset and rank embeddings enter a bidirectional transformer. The output at the masked rank-two slot scores the vocabulary. | Context-dependent portfolio completion, with a 32-dimensional input embedding in the current neural comparison. |

The paper's Word2Vec objective is a local rank-window prediction task. Its AssetBERT uses four encoder layers, two attention heads, balanced chunks of up to 64 holdings, and a 15% masked-token objective. Our implementation follows those headline choices and the 80/10/10 mask/unchanged/random replacement rule, while fixing implementation details the paper does not fully specify. The current neural run uses one Word2Vec training epoch and five AssetBERT epochs. This is **not equal-compute or hyperparameter-tuned**. AssetBERT attends only within its selected chunk at inference; Word2Vec uses its rank window. They share a test target and candidates, not an identical information-processing architecture.

The PCA test fold-in is a simplified zero-filled projection, **not** the paper's estimator that fits an investor's parameters using observed holdings only. Its predictions are passed directly as softmax logits without a validation-fitted temperature or the paper's full level-model logistic-residual mapping. Their likelihoods are therefore sensitive to score scale. A weak PCA ASMP likelihood next to a good PCA Hit@100 can indicate poor probability calibration, not necessarily poor ordering. We have not yet implemented the paper's RS-Level-0, RS-Level-Min, or RS-Level variants.

An earlier saved-matrix pilot also fitted these model families, with AssetBERT restricted to selected endpoint experiments. Its asset universe had already been filtered using all managers before the holdout. That pilot established feasibility but is **excluded** from the result table below; its scores must not be pooled with or directly subtracted from the corrected fit-manager-only runs.

## 5. What the performance measures mean

For scoreable test manager $i\in S_q$, let $a_i$ be its hidden rank-two asset. Every model's candidate probability is

$$
p_i(a_i)=\frac{\exp s_i(a_i)}{\sum_{a\in C_i}\exp s_i(a)}.
$$

The paper's normalized ASMP likelihood is

$$
\mathrm{ASMP}_q=1+\frac{\operatorname{mean}_{i\in S_q}\log p_i(a_i)}{\operatorname{mean}_{i\in S_q}\log |C_i|}.
$$

Uniform guessing scores zero; perfect target probability scores one; negative values would be worse than uniform on log loss. **ASMP = 0.38 does not mean a 38% chance of guessing the right stock**, a 38% return, or a 38% hit rate. It is normalized improvement in target log likelihood relative to uniform choice on that quarter's candidate sets. `Hit@100` is the fraction of targets ranked among the top 100 candidates, and mean reciprocal rank (MRR) averages $1/\text{target rank}$. These ranking measures can disagree with likelihood. Coverage is the fraction of eligible positive-position *test-query entrants* that remain scoreable after fit-vocabulary restrictions; it is not coverage of all raw 13F filings.

## 6. Current local results and interpretation

The table reports **medians of separate within-quarter scores**, weighting quarters equally rather than pooling managers across time. All rows within a quarter use identical test queries and candidate sets. Neural rows are 32D; PCA rows are 4D or 10D. The run used seed 17, one Word2Vec epoch, five AssetBERT epochs, and model-source commit `bc96215ce4bd`. Per-run data and code fingerprints are kept in the ignored local aggregate files; the tracked [results notebook](../notebooks/07_explain_13f_model_results.ipynb) contains no executed outputs or raw holdings.

| Model | Median ASMP | Median Hit@100 | Median MRR |
| --- | ---: | ---: | ---: |
| Popularity | 0.151 | 0.446 | 0.256 |
| RS-Binary 4D | 0.034 | 0.495 | 0.247 |
| RS-Binary 10D | 0.037 | 0.556 | 0.250 |
| RS-Ranks 4D | 0.028 | 0.545 | 0.271 |
| RS-Ranks 10D | 0.032 | 0.604 | 0.300 |
| Word2Vec 32D | 0.379 | 0.643 | 0.280 |
| AssetBERT 32D | 0.266 | 0.514 | 0.239 |

The expected chance Hit@100 is calculated from each query's candidate count, not from an arbitrary tie-broken uniform ranking. Coverage and exclusions are displayed by quarter in the local results notebook. The performance figures are conditional on the selected fit vocabulary, and skipped out-of-vocabulary cases must not disappear from interpretation.

Under **this** setup, Word2Vec has the stronger likelihood and top-100 ranking than AssetBERT on the matched held-out portfolios. Both neural models exceed the popularity baseline in median ASMP, but AssetBERT's median MRR is below popularity's. RS-Ranks 10D has a higher median MRR than Word2Vec 32D despite a much lower, uncalibrated ASMP likelihood. This is why a single metric is not an adequate model-selection rule. The quarter-to-quarter score spread shown in notebook 07 is dispersion of separate fits, not a confidence interval, a repeated-seed test, or a chronological investment record.

We can report a narrower result: the current Word2Vec configuration is a promising **within-quarter portfolio-completion** baseline on the available 13F panel. We cannot infer from this run that Word2Vec is intrinsically better than AssetBERT. The fund-versus-filer data grain, model dimensions, context rules, training budgets, calibration, and untried hyperparameters all matter; the present comparison isolates none of their effects.

## 7. Alignment with and departures from the paper

| Question | Paper | Current 13F work |
| --- | --- | --- |
| Empirical panel | Mainly disaggregated FactSet fund holdings, plus 13F-sourced hedge funds, combined with CRSP and Compustat; 2005Q1–2022Q4; companies are the asset grain. | SEC 13F filer-level managers; typed filing-reported securities. A manager can combine multiple funds or strategies. No licensed FactSet/WRDS replication sample is available. |
| Representation methods | Several recommender-system variants, Word2Vec, AssetBERT, and InvestorBERT; observed-characteristic, placebo, and text comparisons. | Uniform/popularity, RS-Binary/RS-Ranks PCA, Word2Vec, and AssetBERT are fitted for ASMP. The other paper methods and characteristic/text comparisons remain unrun. |
| Benchmarks | RV, RC, and ASMP. Paper ASMP reports Word2Vec near 0.26–0.29 at 4D/10D and AssetBERT near 0.35 at 4D, rising above 0.60 at higher dimension (Section 6.3, Figures 10–11). | ASMP only, under different data, security grain, dimensions, and training/evaluation choices. Numerical score levels must **not** be compared as a direct replication gap. |
| Time and information | Paper models can be trained within a quarter; its RC benchmark uses prior-quarter embeddings, but the paper does not document a tradable 13F filing-availability lag for the presented benchmarks. | Historical filing availability and decision cutoff are explicit. Every scored quarter has a separate within-quarter fit and manager holdout; there is no future-quarter forecast. |
| Economic outcome | Representation benchmarks; the paper discusses portfolio generation, investor similarity, and risk uses. | No market-return prediction, portfolio rule, transaction-cost model, or trading backtest. An ASMP improvement is not investable alpha. |

The paper's wider findings also matter for model selection. In RV, its selected observed characteristics explain about 15% of the valuation-residual variation, while its recommender-system level models do better (Section 6.1). In RC, a high-dimensional RS-Level-Min model reaches 7.2% out-of-sample cross-sectional $R^2$, versus 5.3% for its observed-characteristic comparator (Section 6.2). Neither figure has an equivalent result in our project yet. ASMP favors the paper's Word2Vec and especially AssetBERT models, but a model that completes portfolios well need not be best for valuation or return comovement.

The paper's reported ordering differs from ours: AssetBERT is strongest on its own ASMP sample, whereas our current 32D Word2Vec run scores higher than our current 32D AssetBERT run. That is a research question, not a contradiction that can be resolved by comparing the percentages. The paper itself notes that 13F filer aggregation can obscure fund-level substitution patterns (Section 5.1, footnote 17).

## 8. What must happen before stronger claims

1. Add a separate validation-manager split for hyperparameters and probability calibration, then freeze the protocol before a final test. Compare matched dimensions, repeated seeds, training effort, and query coverage; retain the simple popularity and PCA baselines.
2. Audit a point-in-time market security master before describing the typed 13F baseline as verified common equity or joining returns and fundamentals. Record how issuer/company aggregation changes the representation task.
3. Implement the paper's remaining recommender variants and, when point-in-time market and accounting data are ready, its RV and RC benchmarks. Treat the paper's original FactSet/WRDS sample as unavailable, not silently reproduced.
4. Only after representation benchmarks are stable should we specify a genuinely future-quarter forecast and, separately, an investment strategy with realistic execution, costs, and risk controls.

**Reproduction pointers:** the [project specification](PROJECT.md) defines the research gates; [matrix construction](../src/matrix_13f.py), [manager-only benchmark construction](../src/manager_only_asmp_13f.py), [ASMP evaluator](../src/asmp_13f.py), [Word2Vec](../src/word2vec_13f.py), and [AssetBERT](../src/assetbert_13f.py) contain the implemented contracts. The [runner](../scripts/run_13f_manager_only_asmp.py) records its configuration, seed, code provenance, and local data/run fingerprints. Per-run JSON files and private source-inventory details remain local; the aggregate performance medians above are the only run results published in this draft.
