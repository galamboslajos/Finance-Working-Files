# Data specification

## Connected platform

The exact cloud project, bucket URI, object paths, authorized accounts, and inventory statistics are
private operational metadata. They must be supplied locally through an approved private channel and
must not be committed. Credentials and the local Cloud SDK live in ignored `.gcloud` and `.tools`
directories.

## Candidate products

| Product category | Intended role | Public status |
| --- | --- | --- |
| 13F holdings | Institutional holdings baseline | Validate locally |
| N-PORT fund holdings | Deferred disaggregated extension | Validate locally |
| Company mapping | Issuer and security mapping | Validate locally |
| XBRL financial statements | Fundamental characteristics | Validate locally |
| Shares and public float | Market-cap construction | Validate locally |
| U.S. equity daily market data | Prices, returns, and liquidity | Validate locally |
| Point-in-time index membership | Investable universe | Validate locally |
| Factor benchmarks | Factor and return baselines | Validate locally |

No payload is stored in Git.

## 13F product requirements

- Required grain: manager, filing, holding.
- Key point-in-time fields: period_of_report, filed_at, availability_timestamp_utc.
- Holding fields include manager, issuer, CUSIP, ticker, value, shares, put/call, voting authority,
  and identity confidence.

The filing date, not the portfolio quarter end, determines when a holding becomes usable. Amendments
must be handled explicitly.

## Timestamp contract

- Economic date: when the underlying state applies.
- Published or filed timestamp: when the source released it.
- Ingested timestamp: when the platform obtained it.
- Available timestamp: earliest permitted strategy use.
- Decision timestamp: when a portfolio is formed.

A feature is eligible only when available_at is no later than decision_at.

## Canonical research tables

- asset_master: stable internal identifiers and effective-dated source mappings.
- holdings: report date, availability, manager, asset, shares, value, weight, filing, amendment.
- prices_and_returns: observation and availability dates, prices, total returns, market cap,
  liquidity, corporate actions, and delistings.
- fundamentals: period end, filing and availability timestamps, issuer, field, value, and version.
- universe: decision date, membership, exclusion reason, liquidity, and borrow eligibility.

Zero, missing, not covered, and not eligible are different states.

## Initial paper-aligned filters

- Remove investors whose largest position exceeds 75 percent.
- Require at least 20 assets per investor.
- Require at least 20 investors per asset.
- Enforce the last two conditions iteratively by period.

These are starting assumptions and require sensitivity analysis.

## Required next review

For each candidate product, inspect README.md, schema.json, MANIFEST.json, and its variable
dictionary. Establish:

- stable keys and observation grain;
- timestamp and revision semantics;
- amendments and duplicates;
- value and share units;
- issuer-versus-security aggregation;
- coverage gaps;
- exact cross-product joins;
- licensing and collaborator-sharing constraints.

Raw data, credentials, local tools, and generated artifacts must never enter Git.

## Bounded exploration workflow

The first inspection is implemented in `notebooks/01_explore_13f_nport.ipynb` with tested helpers
in `src/holdings_exploration.py`.

- Private inputs are supplied as exact object URIs through environment variables.
- Bucket-wide listing is not required.
- Samples and variable dictionaries are downloaded into ignored `data/exploration/` paths.
- The committed notebook contains no executed outputs or raw rows.
- Schema completeness, timestamp ordering, identifiers, duplicates, signed values, categorical
  states, portfolio shape, and matrix density are reviewed separately.
- The paper-aligned 20-assets/20-investors conditions are applied iteratively and logged at every
  iteration.

The exploration does not yet choose an amendment rule, investor aggregation, security-to-company
mapping, or final equity filter. Those decisions require multiple-period evidence and a written
data-spine decision record.

## Full-history 13F orientation

The 13F-first workflow is implemented in `notebooks/02_explore_13f_full_history.ipynb` with tested
helpers in `src/full_history_13f.py`.

- Complete table shapes and schema stability are read from Parquet footers.
- The smaller filing index is loaded across all partitions for timing, cadence, form, and amendment
  diagnostics.
- Economic report dates, publication times, and permitted availability times remain separate.
- Older backdated records are separated from the inferred continuous product window.
- Holdings content is inspected through a deterministic bounded sample across filing-month
  partitions.
- Raw manager, issuer, CUSIP, ticker, and holding rows are not displayed.
- Detailed inventory statistics and executed outputs remain in ignored local paths.

This notebook is an orientation and audit tool. It does not resolve amendments, define the
long-equity universe, choose company-versus-security grain, or build the production holdings
matrix.

## Point-in-time 13F data-spine audit

The provisional policies agreed after the orientation are recorded in `docs/13F_DATA_SPINE.md`.
They cover amendment state transitions, notices and combination reports, issuer-level modeling,
typed instruments, manager CIK identity, the continuous coverage window, and the standardized
filing-deadline cutoff.

`notebooks/03_audit_13f_data_spine.ipynb` tests those policies against the complete local history
with helpers in `src/data_spine_13f.py`.

- Filings and cover pages are reconciled by accession before amendment metadata is joined.
- Restatements and new-holdings amendments remain separate event types.
- Boundary quarters are eligible only when the complete normal filing window was observable.
- Manager identifier instability is reported without displaying raw managers.
- Every holdings row is streamed through provisional instrument, issuer-identity, and timestamp
  audits.
- Mapping snapshot dates are audited separately from present-day identifier coverage.
- Aggregate scans are cached below ignored `data/audits/` paths using complete-input fingerprints.

The instrument classes are audit labels, not a final equity universe. A point-in-time security
master is still required to distinguish operating-company common equity from funds and other cash
share instruments reliably.

## Canonical point-in-time 13F state

`src/point_in_time_13f.py` and `notebooks/04_validate_13f_point_in_time_panel.ipynb` implement and
check the first canonical decision-time state:

- the filing table defines the event inventory and expected holding-line count;
- cover-page amendment and report metadata are joined one-to-one by accession;
- events are ordered by availability, filed time, and accession;
- originals initialize, new-holdings amendments supplement, and restatements replace;
- notices and combination reports stay explicit but do not enter the complete-portfolio baseline;
- missing joins, conflicting signals, count mismatches, and incomplete states are quarantined;
- every output row retains its source accession and source-event availability; and
- downstream modeling must use the explicit state-eligibility flag.

The local quarter builder streams only holdings partitions needed by visible candidate events. It
does not apply the current issuer mapping or define a final common-equity universe. The next
section resolves the first security-level representation key while leaving the market
common-equity gate explicit.

## Point-in-time 13F security identity

`src/security_master_13f.py` and `notebooks/05_audit_13f_security_master.ipynb` separate the first
model-ready security identity from the later common-equity and issuer-mapping questions.

- The first representation key is the normalized CUSIP reported with the filing, namespaced by
  instrument channel. It is knowable at the filing availability timestamp.
- The SEC filing contract and official 13(f) list specify a nine-character CUSIP. The baseline
  therefore admits only normalized, source-reported nine-character values. Shorter or otherwise
  nonstandard values remain in exclusion audits and are never padded, checksum-completed, or
  silently joined to another security.
- A typed key is mandatory because the same reported CUSIP can appear in cash-share and option
  channels. Those economic exposures are never added together.
- The first matrix uses only positive-value `cash_share_candidate` rows from complete,
  unquarantined point-in-time manager states.
- `cash_share_candidate` is not a claim of verified common stock. Current mappings, names, and
  tickers are not silent substitutes for a historical market security master.
- Reported issuer CIK remains audit metadata and a later aggregation grain. A resolved issuer
  mapping is usable only when its mapping snapshot predates the historical decision; a date-only
  same-day snapshot is conservatively unavailable.
- CUSIP changes remain separate security identities until effective-dated corporate-action links
  exist.

The full-history identity scan reads every holdings partition in bounded batches and caches only
aggregate diagnostics under ignored `data/audits/` paths. The first local matrix candidate jointly
reapplies the 75 percent concentration rule and the iterative 20-security/20-manager rules until
both hold, then recomputes weights after the final universe is fixed. It remains a sparse pair
table rather than a dense matrix.

`src/matrix_13f.py` materializes one ignored Parquet pair table per decision-complete quarter below
`data/model_inputs/13f_cash_share_pit/v1/`. Each partition includes local aggregate audits and a
manifest binding it to the data fingerprint, matrix-source fingerprint, identity policy, filter
parameters, and pair-file hash. Full-history validation checks expected-quarter coverage, stale
manifests, file integrity, Parquet row reconciliation, and every saved point-in-time matrix gate
without loading the entire pair history into memory.

## Verified commands

~~~bash
./scripts/gcloud auth list \
  --filter=status:ACTIVE \
  --format='value(account)'

./scripts/gcloud storage ls \
  "$ASSET_EMBEDDINGS_CURATED_URI"
~~~
