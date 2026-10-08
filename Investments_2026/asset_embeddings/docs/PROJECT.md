# Project specification

## Objective

Determine whether holdings-derived asset embeddings provide economically useful information for
equity investment decisions after realistic availability lags, constraints, and costs.

The immediate objective is methodological replication of the paper's representation models and
benchmarks on the available 13F panel. A trading strategy is a later, separate decision.

## Hypotheses

1. Representation: holdings embeddings measure asset similarity beyond random embeddings and
   standard characteristics.
2. Prediction: lagged embeddings improve held-out valuation, comovement, risk, or return forecasts.
3. Investment: a frozen portfolio rule provides useful net risk-adjusted performance.

Failure at one stage blocks promotion to the next. A useful representation is not evidence of
tradable alpha.

## Initial scope

- U.S. equities.
- Quarterly 13F holdings as the first implementation.
- N-PORT deferred as a possible fund-level replication or extension.
- Filing-reported, typed CUSIP security grain for the first representation model; issuer and
  company aggregation remain robustness extensions requiring effective-dated mappings.
- Simple factor or recommender-system baseline first.
- Relative value among embedding peers as the first possible strategy family.

## Research safeguards

- Every observation has an economic date and an availability timestamp.
- Raw snapshots are immutable and identified by manifest hashes.
- Security mappings are effective-dated; current tickers are not timeless identifiers.
- Evaluation is chronological: training, validation, then an untouched final test.
- Random embeddings, standard characteristics, and a simple holdings model are mandatory baselines.
- Hyperparameters and portfolio rules are frozen before the final test.
- Results report gross and net returns, turnover, costs, exposures, concentration, drawdown,
  liquidity, capacity, and failed subperiods where applicable.
- Each reported run identifies its Git commit, data manifests, configuration, and random seed.

## Representation benchmarks

### Relative valuation

Test whether embeddings explain held-out market-value residuals after controlling for book equity.
This measures representation quality, not trading performance.

### Return comovement

Use embeddings available before a return period to explain held-out cross-sectional returns.
Compare with characteristics and combined characteristic-plus-embedding models.

### Managed-portfolio similarity

Mask a large holding and predict it from the remaining portfolio. Report likelihood, rank, top-k
accuracy, and improvement over random selection.

## Model ladder

1. Placebo and observed characteristics.
2. Binary ownership factor model.
3. Ranked-holdings factor model.
4. Centered holdings-level model with explicit missing-state treatment.
5. Word2Vec-style local portfolio context on the same held-out portfolios.
6. AssetBERT masked-token prediction on those same portfolios and candidate sets.

Models five and six are required paper-method comparisons even if they fail locally. Their
promotion to downstream prediction or investment use still requires stable held-out value.

## Roadmap

### Phase 0 - Foundation

Complete: source paper reviewed, research order agreed, cloud platform connected, and the repository
reduced to a transparent core.

### Phase 1 - Point-in-time data spine

Current phase:

- audit the provisional point-in-time amendment, notice, asset-grain, instrument, manager,
  coverage, and decision-time policies against the complete 13F history;
- construct filing-event ledgers and canonical manager-period states at historical decision
  cutoffs;
- validate accession joins, issuer identity snapshots, and manager identifier stability;
- validate the filing-reported CUSIP security key and quarantine identifier conflicts;
- materialize and fingerprint the sparse manager-security matrix for every decision-complete
  quarter;
- choose the permitted point-in-time issuer mappings and market common-equity security master;
- establish joins across holdings, mappings, market data, and fundamentals;
- build coverage and leakage diagnostics.

Gate: a reproducible snapshot passes identifier, timestamp, and reconciliation tests.

### Phase 2 - Simple embeddings

Build holdings matrices, iterative universe filters, factor models, normalized distances, and
placebo comparisons.

### Phase 3 - Representation benchmarks

Implement and freeze the three paper-aligned benchmarks. The first ASMP pilot on saved 13F
matrices holds out 20 percent of managers within each quarter and aligns test candidates across
PCA, Word2Vec, and AssetBERT. It is not sealed: all-manager filtering preceded the split. The
separate manager-only runner rebuilds a canonical PIT quarter, splits managers first, and fixes
the vocabulary and universe from fit managers only. It reports test coverage and visible-context
OOV loss; this corrects one leakage channel but does not yet settle model ranking. Add a separate
validation-manager split for calibration and hyperparameters, then expand across quarters and
document differences from the paper's FactSet fund/company sample and RV/RC asset split.

### Phase 4 - Investment strategy

Register, validate, and test an embedding-peer relative-value strategy after realistic costs and
constraints.

### Phase 5 - Further models and extensions

Consider investor embeddings, crowdedness, generative portfolios, stress testing, and text-assisted
interpretation only after earlier gates pass. Word2Vec and AssetBERT are already part of the
representation-replication comparison in Phase 3.
