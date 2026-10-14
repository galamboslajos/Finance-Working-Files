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

**Paper-replication gap:** no historical market-capitalization or shares-outstanding
field was found in the downloaded table schemas. The Han-Qin paper requires
historical capitalization for size deciles, its bottom-5% screen and value
weighting. An approved additional source is needed for a complete baseline.
Price, volume and index membership are not equivalent size data.

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

No feature build, model fit or backtest has been run. Reusing/rebuilding derived
panels requires correct path, identity, calendar and accounting rules and the
project's agreed execution plan.
