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

## Full feature panel verification, 2026-10-15

The [full 2010–2023 feature panel](FULL_FEATURE_PANEL.md) is now built and audited under the provisional identity policy. It retains 152,307 original rows and 168 monthly signals. Daily observations feed technical/context calculations and are also preserved independently for later backtesting, including complete annual warmup and signal anchors. All 715 eligible names remain in the dictionary; missing context is reported rather than filled or silently excluded. The master verifies saved coverage and exposes separate lazy feature/daily datasets. Notebook Run All does not relaunch the full build, fit imputation/models or backtest.

The historical user-authorized 90% context rebuild is complete: all 402 previously empty
context predictors have observations somewhere in 2010–2023; none are entirely
missing over that full period. Paper15,298 technical features and every preserved
daily artifact match the previous run exactly. Among paper-qualified rows,
context-value coverage is 59.52% in 2010–2019 and 98.08% in 2020–2023. Early missingness
remains material; these are descriptive counts, not full-history feature selection.
See [the full-panel verification](FULL_FEATURE_PANEL.md) and the active
context contract for detailed annual/feature coverage.

## Active context panel roles, 2026-10-15

By user direction, the verified 80% original-cohort panel is now primary; the 90%
panel is retained as sensitivity. The [promotion/comparison record](CONTEXT_SENSITIVITY_80.md)
preserves original 90% →80% comparison tables and exact immutable run pins.
Context-value coverage among paper-qualified rows is 96.27% versus 71.06% over
2010–2023, and 92.09% versus 33.57% in 2010–2015 (80% versus 90%). Exhaustive requests,
original denominators, paper15, technical298, PIT cutoffs and separate daily
artifacts are unchanged. This is a user policy choice, not selection by model
performance or proof that remaining omissions are random.

## Shared monthly class/RET engineering, 2026-10-15

The [shared workers](TARGET_CLASSES_RET.md) now implement exact tie-aware monthly
classes, original-cohort/missing-outcome ledgers, the label-aware training adapter
and strict 120-calendar-month realized-class RET payoffs. Forward-outcome requests
are distinguished from past feature requests. The master checks already loaded
bounded samples and keeps daily/feature data separate. Full historical label and
payoff coverage is now measured by the separate explicit audit below.
No estimator or backtest has run for this implementation; economic accounting
remains unresolved.

## Full monthly target and RET coverage audit, 2026-10-15

The user-authorized [full target panel and coverage audit](FULL_TARGET_PANEL.md)
is complete for 2010–2023. It uses verified saved daily caches and the same
historical S&P 500/400-union cohorts and signal-time prediction eligibility as
the existing primary feature panel. Labels retain missing outcomes and their
original denominators. Class counts include all ten bins, including empty bins;
RET requires every cell of the trailing 120-calendar-month × ten-class grid.

The target portion retains **152,307 original rows**, including 2,357 identity
exclusions and 9,173 additional identity-admitted rows without Paper15 prediction
eligibility. Among the remaining **140,777 prediction-eligible rows**, **140,485**
receive labels: **99.79% of the prediction cohort**, or **92.24% of all original
rows**. The 292 eligible missing labels comprise 276 missing end quotes and
16 source quality flags. No prediction rows are removed because their later
outcomes are missing.

All 168 monthly rank/request guards pass, every month populates all ten classes,
and 43 months contain exact-return ties (maximum group size four). Every
identity-admitted original member has explicit past-feature and forward-outcome
request evidence; no unknown/unrequested cases or missing outcome records occur.
These are conditional observed-return labels, not certification that excluded or
missing outcomes are harmless.

All **48 nominal 2020–2023 test signals** have complete strict RET payoff vectors,
each checked against the canonical shared worker. The first available signal
is **2019-12-31**, predicting January 2020. The earlier 120 signals retain withheld
vectors because the 2010-start outcome history cannot yet supply 120 completed
months. The saved evidence includes all 201,600 month/class grid rows and 1,680
class-payoff rows, including explicit unavailable statuses.

The completed audit verified **475 recorded input files unchanged**. Existing
feature panels and their separate daily artifacts are preserved, and the audit
does not fit preprocessing, either model or a backtest. Successful cache hashes,
labels or payoff arithmetic would not certify identities, knowledge time,
missing terminal-event cash flows or executable economic returns.
