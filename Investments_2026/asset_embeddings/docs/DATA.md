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
| N-PORT fund holdings | Disaggregated fund holdings | Validate locally |
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

## Verified commands

~~~bash
./scripts/gcloud auth list \
  --filter=status:ACTIVE \
  --format='value(account)'

./scripts/gcloud storage ls \
  "$ASSET_EMBEDDINGS_CURATED_URI"
~~~
