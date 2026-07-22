# Project specification

## Objective

Determine whether holdings-derived asset embeddings provide economically useful information for
equity investment decisions after realistic availability lags, constraints, and costs.

## Hypotheses

1. Representation: holdings embeddings measure asset similarity beyond random embeddings and
   standard characteristics.
2. Prediction: lagged embeddings improve held-out valuation, comovement, risk, or return forecasts.
3. Investment: a frozen portfolio rule provides useful net risk-adjusted performance.

Failure at one stage blocks promotion to the next. A useful representation is not evidence of
tradable alpha.

## Initial scope

- U.S. equities.
- Quarterly 13F and N-PORT holdings.
- Company or security aggregation chosen only after identifier review.
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
5. Word2Vec-style local portfolio context.
6. AssetBERT only if it adds stable held-out value.

## Roadmap

### Phase 0 - Foundation

Complete: source paper reviewed, research order agreed, cloud platform connected, and the repository
reduced to a transparent core.

### Phase 1 - Point-in-time data spine

Current phase:

- inspect product schemas, manifests, and variable dictionaries;
- choose issuer-versus-security grain;
- define amendment and duplicate rules;
- establish joins across holdings, mappings, market data, and fundamentals;
- build coverage and leakage diagnostics.

Gate: a reproducible snapshot passes identifier, timestamp, and reconciliation tests.

### Phase 2 - Simple embeddings

Build holdings matrices, iterative universe filters, factor models, normalized distances, and
placebo comparisons.

### Phase 3 - Representation benchmarks

Implement and freeze the three paper-aligned benchmarks.

### Phase 4 - Investment strategy

Register, validate, and test an embedding-peer relative-value strategy after realistic costs and
constraints.

### Phase 5 - Advanced models

Consider Word2Vec, AssetBERT, investor embeddings, crowdedness, generative portfolios, stress
testing, and text-assisted interpretation only after earlier gates pass.
