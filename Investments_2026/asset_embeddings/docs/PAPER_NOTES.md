# Source paper notes

## Citation and provenance

Xavier Gabaix, Ralph S. J. Koijen, Robert J. Richmond, and Motohiro Yogo,
Asset Embeddings, draft dated 2024-09-13.

Reviewed file: AssetEmbeddings_preview (1).pdf, 52 pages.

SHA-256:
2b752f2e3010d34169023ef29ac23e27c7c00bd705c2175baf29fc5e82484b6d

The PDF is not committed because redistribution rights have not been established.

## Core idea

Investors organize assets in portfolios as documents organize words. Holdings can therefore reveal
latent asset characteristics and investor styles that observed accounting variables may miss.

## Models

- Recommender systems factor holdings into asset embeddings and investor loadings.
- Word2Vec uses a continuous-bag-of-words objective: within-portfolio holding ranks define the
  neighboring context, whose mean embedding predicts a hidden stock by full-vocabulary softmax.
- AssetBERT treats ranked stocks as tokens, learns position embeddings, and predicts masked
  holdings with bidirectional attention within balanced chunks of at most 64 stocks. The paper
  uses four encoder layers, two attention heads, and a 15 percent masked-token objective.
- InvestorBERT transposes the task and predicts investors holding an asset.

## Paper benchmarks

1. Relative valuation: explain held-out valuation residuals.
2. Return comovement: use prior embeddings to explain held-out returns.
3. Managed-portfolio similarity: predict a masked large holding.

The paper reports that recommender systems are strong for valuation and comovement, while
Word2Vec and especially AssetBERT are stronger for substitution or masked-holding prediction.
Higher-dimensional models generally perform better.

For managed-portfolio similarity (ASMP), the paper hides a large holding and evaluates its
probability among assets not already visible in that portfolio. Its normalized score is
`1 + mean(log hidden-asset probability) / mean(log candidate count)`; uniform guessing scores
zero. This is not the same quantity as Hit@100. The paper reports Word2Vec at 26-29 percent
and AssetBERT at 35 percent in four dimensions, rising beyond 60 percent at larger dimensions,
on its own data. Those figures are not targets or directly comparable scores for our 13F panel.

## Paper sample

The study combines CRSP, Compustat, and FactSet fund and hedge-fund holdings from 2005 Q1 through
2022 Q4. It aggregates to company level, removes micro caps, drops highly concentrated investors,
and iteratively requires at least 20 assets per investor and 20 investors per asset.

Our SEC-based sources differ, so this project is an adaptation, not a claimed replication.

## Implementation cautions

- Holdings are available after filing, not at quarter end.
- Missing holdings can mean zero ownership, mandate exclusion, short-sale constraints, or missing
  coverage.
- Scale and rotation must be normalized before interpreting embedding distance.
- A model trained on masked-holding prediction has an objective advantage on that benchmark.
- The first saved 13F matrices were filtered using all managers before the 80/20 manager holdout.
  Their cross-manager masked-holding results are provisional, not a sealed paper benchmark.
- The manager holdout is our 13F generalization diagnostic; it is not the paper's RV/RC 80/20
  asset split. All ASMP model fits use only complete training-manager portfolios, while the same
  test managers supply rank-two masked queries and explicitly reported OOV coverage.
- The paper leaves Word2Vec rank radius and several AssetBERT optimizer/architecture details
  unspecified; every local choice must be recorded as an adaptation.
- Contextual AssetBERT embeddings require a declared aggregation rule.
- Security-level and issuer-level embeddings are different research objects.
- Representation success does not establish investable alpha.
