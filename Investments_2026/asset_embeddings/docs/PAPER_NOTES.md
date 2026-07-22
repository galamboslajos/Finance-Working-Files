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
- Word2Vec ranks assets within portfolios and predicts assets from nearby positions.
- AssetBERT treats a ranked portfolio as a sequence and predicts masked holdings using the full
  portfolio context.
- InvestorBERT transposes the task and predicts investors holding an asset.

## Paper benchmarks

1. Relative valuation: explain held-out valuation residuals.
2. Return comovement: use prior embeddings to explain held-out returns.
3. Managed-portfolio similarity: predict a masked large holding.

The paper reports that recommender systems are strong for valuation and comovement, while
Word2Vec and especially AssetBERT are stronger for substitution or masked-holding prediction.
Higher-dimensional models generally perform better.

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
- Contextual AssetBERT embeddings require a declared aggregation rule.
- Security-level and issuer-level embeddings are different research objects.
- Representation success does not establish investable alpha.
