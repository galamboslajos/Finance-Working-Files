# Data and reference status

Independent copies of all three repository-pinned source releases are in data/raw/:

- Intrinio daily market data, plus identity and quality metadata.
- Siblis S&P Composite 1500 historical membership.
- Unified historical index membership, used by the previous attempt for identity mapping.

The original downloads passed generation, byte-size and SHA-256 checks. Their
copies in this workspace are independently checked during setup. Parquet table
row counts were reconciled with the release manifests. Exact private inventory
and verification records remain in ignored data/raw/ files.

The source-product date ranges are broader than the strategy's research period
and do not imply complete historical coverage for every security.

Sadig's feature definitions and construction code are available under
reference/sadig_us_recovery/. The downloaded snapshot records its repository,
branch, commit and archive hash in SOURCE_SNAPSHOT.json. It includes both the
current five-feature method and the earlier 740-feature construction program.

The precomputed member_panel.parquet, valuation_opens.parquet and calendar.parquet
remain inaccessible to the configured collaborator account. A pinned calendar
read returned 403 / missing storage.objects.get. The exact owner access request
and run paths are preserved in ignored data/raw/strategy_panels/ records.
The precomputed 740-feature warehouse was not downloaded.

The source inputs are available locally. Reusing or rebuilding derived panels
still requires correct path/identity/calendar bindings and an agreed research plan.
