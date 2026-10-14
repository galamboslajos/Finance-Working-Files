# V2 data-loading and historical-universe foundations

Implemented 2026-10-14. This stage provides audited loading and constituent
reconstruction; it does not build returns, features, targets or backtests.
Market capitalization remains outside the project scope.

The loader is coded in `src/dmv2/loading.py`, with catalog/source/universe
helpers. A [master notebook](NOTEBOOK_WORKFLOW.md) now exposes those stages.
Research initialization is user-selected 2000 with rolling 10/1 chronology;
the 2010 audit below was a bounded test, not a source or study start date.

## Contracts

- `SourceCatalog` resolves only objects listed in pinned source manifests.
  It checks manifest SHA-256, path containment, sizes, declared physical
  schemas and footer row totals. Whole-file SHA-256 checks remain available
  through `verify-data --hashes`; a normal read does not claim to repeat them.
- `BoundedTableReader` requires explicit dates, columns and budgets. It reports
  source pins, object hashes, request settings and rows/bytes considered.
  Limits reject the whole request before a truncated result can look successful.
- `PriceLoader` requires stable Intrinio IDs. It neither selects members nor
  consults current `delisted`, last-price, future-price or security-master flags.
  Duplicate/null required keys and malformed dates fail instead of being cleaned away.
- `load_universe` reconstructs exact source-session membership for `sp500`,
  `sp400` or `combined`. It applies `is_member`, effective and availability
  dates; audited Siblis end dates are inclusive. Future or missing availability
  is excluded with a reason ledger. Absent daily records are not forward-filled.
- The combined cohort collapses only the same Siblis source security across
  indexes. Mapping uses `sp_composite1500` source ID → unified ID → Intrinio ID.
  Unmapped, ambiguous, review-required and conflicting target mappings remain
  in the constituent ledger and are quarantined from quote attachment.
- `attach_quotes` preserves the full cohort. It distinguishes `ok`,
  `missing_price`, `not_requested`, `identity_unresolved`, `invalid_price` and
  `source_quality_flag`. A small requested sample is not a coverage audit of
  the unrequested names. No zero-return imputation or stale-price fill is performed.
- `session_dates` and `month_end_sessions` use the pinned calendar, reject
  duplicates/unknown states and obvious missing months, and apply bounds after
  identifying source month ends. They do not independently certify every
  calendar session or an execution-time convention.

## What PIT means here

Use the explicit mode `historical_reconstruction` for the current sources.
It honors supplied historical effective/availability dates and causal query
cutoffs, but does **not** establish true historical knowledge-time correctness.
Siblis daily availability is reconstructed. Identifier-history validity dates
are blank, Intrinio links lack validity intervals, and adjusted prices are
retrieval-time history. A provider response marked `resolved` is a mapping
status, not historical identity certification. The `knowledge_time` mode fails
closed because the required publication/revision/identity evidence is absent.

The membership audit JSON contains source evidence, including later-known
interval endpoints. It is not a feature table. Do not feed future end dates,
eventual delisting flags, retrieval metadata or missing-future-quote outcomes
into signals, identity eligibility or model inputs. Entry timing within a
session is not inferable from date-only availability.

The code prevents common survivor/causality errors, but that is not a proof
that the vendor histories or economic outcomes are survivorship-bias free.
Identity repair, adjustment semantics, delisting cash flows, missing valuations
and a complete original-cohort outcome ledger remain required before economics.
Do not begin modelling on only the easily mapped subset.

## Holdout and storage bounds

Financial requests ending on or after 2024-01-01 fail before value loading.
Known financial fields activate this guard even if a generic caller omits
`financial=True`. Financial fields cannot be used as metadata keys or filters.
Date/ID masks are applied in Arrow before excluded financial observations
reach Python conversion, returned tables, joins or analysis.

Parquet column chunks can contain both development and holdout dates. Reading
an approved selection may transiently decode excluded values inside a block;
this is an output/analysis guard, not physical separation of the holdout bytes.
No protected financial values are returned, displayed or summarized.

Default limits are 128 footer files, 8 data files, 1,000,000 candidate rows,
100,000 output rows and 128 MiB projected compressed column bytes. Compressed
bytes are a column-chunk upper bound, not a RAM or total-I/O guarantee. A one-day
Siblis query is expensive relative to its output because the file is arranged
by security/index history rather than date. The measured 2010-01-29 combined
request considers 8,388,608 rows and 27,846,415 compressed projected bytes.
Its explicit 9,000,000 row budget is for that bounded engineering check, not
authorization to scan arbitrary periods or build a full feature warehouse.

## Repeat the checks

From the configured V2 folder, install the local package if needed:

```sh
.venv/bin/python -m pip install --no-deps -e .
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
PYTHONPATH=src .venv/bin/python -m dmv2 status
PYTHONPATH=src .venv/bin/python -m dmv2 universe --as-of 2010-01-29 --universe combined --mode historical_reconstruction --max-scanned-rows 9000000
```

The explicit source path runs the current project code and also avoids this
host's hidden editable-`.pth` file issue under Python 3.14. Installation itself
completed, but clearing the hidden flag did not persist. A new checkout
requires its own approved local source configuration; raw data are not in Git.
Selecting `--mode knowledge_time` intentionally fails for the current inputs.

A bounded Python engineering sample, with missing/unrequested names preserved:
launch Python using `PYTHONPATH=src .venv/bin/python` from the V2 folder.

```python
from pathlib import Path
from dmv2.catalog import SourceCatalog
from dmv2.data import load_universe
from dmv2.loading import PriceLoader, ReadBudget
from dmv2.universe import attach_quotes

catalog = SourceCatalog(Path.cwd())  # run in the configured V2 directory
loaded = load_universe(
    catalog, as_of="2010-01-29", universe="combined",
    mode="historical_reconstruction",
    budget=ReadBudget(max_scanned_rows=9_000_000),
)
cohort = loaded.snapshot
ids = sorted({r["intrinio_security_id"] for r in cohort.table.to_pylist()
              if r["identity_status"] == "resolved"})[:2]
if not ids:
    raise RuntimeError("No uniquely mapped engineering sample; review identities")
prices = PriceLoader(catalog).load(
    ids, cohort.as_of, cohort.as_of,
    ["close", "volume", "adj_close", "adj_volume", "price_quality_status"],
)
audit = attach_quotes(cohort, prices.table, requested_security_ids=ids)
print(cohort.summary)
print(audit.summary)  # all constituents remain; unqueried prices are not_requested
# Persist loaded.loading_provenance and prices.provenance with any audit artifact.
```

## Actual engineering evidence

On 2010-01-29 the combined snapshot has **902 source constituents** and no
membership-date exclusions. Identity statuses: 130 unique routes without a
review flag, 755 needing review, 10 target collisions and 7 unlinked. These
are engineering statuses; the 130 are not certified historically correct IDs.
All 902 remain in the output. A two-ID exact-date price sample returned two
accepted raw/adjusted close and volume records. The attached ledger reports
2 `ok`, 128 `not_requested` and 772 `identity_unresolved`.

The price sample inspected 105 footers, read one data shard with 255,503
candidate rows and about 5.09 MB compressed projected columns, and returned
two rows. Full historical price/volume coverage was not tested. Private audit
provenance and sample values remain in ignored outputs; public documentation
contains no raw market records or credentials.

The test suite uses tiny structural fixtures only. It checks manifest/path
drift, unpinned files, schema/key defects, read limits, protected dates, effective
and availability boundaries, inclusive removals, source-product identity joins,
ambiguous mappings, collisions, retained missing/unrequested quotes and calendar
boundaries. No synthetic data enter research outputs.

The next data-quality task is to explain/resolve the review flags and conflicting
historical identities using dated evidence, then audit return/terminal-event
accounting on bounded pre-holdout cases. The primary modelling universe and
evaluation chronology remain open.
