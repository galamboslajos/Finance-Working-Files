# Data and reference status

Independent copies of three repository-pinned source releases are in data/raw/:

- Intrinio daily market data, plus identity and quality metadata.
- Siblis S&P Composite 1500 historical membership.
- Unified historical index membership, used by the prior attempt for identity mapping.

All 149 source objects (1,792,717,079 bytes) were independently rechecked on
2026-10-14: sizes and SHA-256 matched the release manifests. Parquet footer row
counts reconciled. Exact private inventory, source pins and verification records
remain in ignored local files. This establishes integrity, not a complete
model-ready panel or verified returns for every security.

The source-product ranges are broader than the research period and do not imply
complete historical coverage. Metadata inspection does not inspect protected
2024+ financial values. See [the U.S. ledger](US_REPLICATION_PLAN.md) for verified
table/schema availability and the return/identity audits still needed.

**Deliberate V2 adaptation (user decision, 2026-10-14):** no historical
market-capitalization or shares-outstanding field was found in the downloaded
schemas, and V2 will not use capitalization. The paper's size-decile feature,
cap-based screens and cap-value weighting are omitted. V2 starts from the
15 momentum/state inputs before extensions; obtaining capitalization is not
a remaining task or blocker. Price, volume and index membership are not
substitutes for the omitted size data.

**Volume availability (verified 2026-10-14):** the local
`intrinio_daily_prices` Parquet schema contains `volume: double` and
`adj_volume: double`. Across all tables in the three downloaded releases, no
turnover-ratio or shares-outstanding column was found. In a bounded check of
the first 256 rows of one price shard, dated 2010-01-04 to 2011-01-05, both
volume columns were non-null, finite and positive in all 256 rows. This does
not establish complete volume coverage across securities or history.

[Intrinio's stock-price documentation](https://docs.intrinio.com/documentation/web_api/GetSecurityStockPrices_v2)
defines volume as shares exchanged during the period and describes adjusted
volume as adjusted for splits and dividends; those provider semantics have
not been independently validated here. Raw close times raw volume can give a
daily dollar-volume approximation, not exact traded notional. Share turnover
needs shares outstanding, and the paper's market-turnover ratio needs market
capitalization, so neither ratio is available under the current inputs/scope.
Portfolio turnover is a separate metric to calculate later from our trades
and portfolio weights.

Sadig's feature definitions and construction code are available under
reference/sadig_us_recovery/. Its SOURCE_SNAPSHOT.json records repository,
branch, commit and archive hash. The snapshot includes the daily five-feature
method and earlier 740-feature construction code. Neither is an active V2 panel.

Precomputed member_panel.parquet, valuation_opens.parquet and calendar.parquet
were not obtained during the prior download. A pinned calendar read returned
403 / missing storage.objects.get. Access was not retested on 2026-10-14.
The exact owner access request and run paths are in ignored
data/raw/strategy_panels/ records. The precomputed 740-feature warehouse was
not downloaded. Do not infer dataset absence from the permission failure.

The complete supplied paper was reviewed and copied locally to ignored
reference/papers/ssrn-4452964-2026-03-26.pdf. Its SHA-256/version and original
methodological notes are in [the paper reference](PAPER_REFERENCE_HAN_QIN_2026.md).
The PDF, raw data and source snapshot are not published to Git.

Data-loading and historical-universe foundations are implemented. The bounded
2010-01-29 S&P 500/400 snapshot retained all 902 constituents: 130 unique routes
without review flags, 755 needing review, 10 target collisions and 7 unlinked.
Two explicitly requested exact-date price/volume records loaded; unqueried
and unresolved names were retained. See [the foundation contract and evidence](DATA_FOUNDATIONS.md).
These are retrieval-time mappings, not certified knowledge-time PIT identities.

A bounded feature engineering stage is now implemented; see [FEATURES.md](FEATURES.md).
The separate [2010–2023 monthly-endpoint audit](INITIALIZATION_2010.md) measured
all 15 paper inputs under the accepted [provisional identity policy](PROVISIONAL_IDENTITY_POLICY.md):
140,777 of 152,307 original stock-months (92.43%) qualified. This is not a full
715-predictor/context panel, model fit or backtest. Provisional development may
continue without full identity certification; its original flags and missing,
ambiguous and collision exclusions remain explicit. A full added-feature build
requires the measured execution plan; historical-knowledge and economic
accounting limitations remain unverified.

The user-authorized [monthly return engineering stage](MONTHLY_RETURNS.md)
is now implemented. It preserves daily source tables separately, uses
preceding-month historical cohorts and emits exact provider-adjusted endpoint
ratios with missing/identity/coverage states. The local notebook example is
January–February 2010 with two requested price IDs, not a complete universe
panel or certified economic/PIT return history.

Current source definitions were separately audited at organizational commit
`e9a7a01881a46374694f60fc34eb7d7e4a26c2b8`. The private ignored inventory
records 745 candidates; our implemented bank omits22 external-source-dependent
columns and10 SP600 spreads and adds15 exact paper inputs. Formula availability
is separate from population/coverage verification.

The [identity evidence audit](IDENTITY_AUDIT.md) is implemented and run at
2000-01-31, 2005-01-31 and 2009-01-30. It retains all 2,701 original constituent
rows, verifies source ISIN syntax/check digits, distinguishes declared queries
from alias agreement and records same-date collisions without using eventual
price inventory for eligibility. No identities were promoted. Every current
identifier-history validity boundary is blank; native provider links remain
undated. The master exposes aggregate evidence and a private full review queue.
