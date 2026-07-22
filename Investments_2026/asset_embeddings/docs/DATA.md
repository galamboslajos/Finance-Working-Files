# Data specification

## Connected platform

Bucket:

gs://quant-research-trusty-agility-492109-v0

Curated prefix:

gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/

Access was verified with info@galamboscapital.com on 2026-07-16. Authentication, object reads, and
recursive listing succeeded. The namespace contained 20,630 objects across 79 dataset versions.
Credentials and the local Cloud SDK live in ignored .gcloud and .tools directories.

## Candidate products

| Curated root | Intended role | Status |
| --- | --- | --- |
| provider/sec_api/13f_holdings/v1 | Institutional holdings baseline | Passed |
| provider/sec_api/nport_fund_holdings/v1 | Disaggregated fund holdings | Passed |
| provider/sec_api/company_mapping/v1 | Issuer and security mapping | Passed |
| provider/sec_api/xbrl_financial_statements/v1 | Fundamental characteristics | Historical backfill incomplete |
| provider/sec_api/outstanding_shares_public_float/v1 | Shares and public float | Review pending |
| equity_us/equity_us_market_daily_unified/v1 | Prices, returns, and liquidity | Passed |
| equity_us/equity_us_index_membership_daily_unified/v1 | Point-in-time universe | Passed |
| factor_benchmarks/v1 | Factor and return baselines | Passed with legacy-metadata warnings |

No payload is stored in Git.

## 13F product snapshot

- Coverage: January 2013 through June 2026.
- Grain: manager, filing, holding.
- Holdings rows: 120,270,881.
- Filings: 388,292.
- Managers: 16,154.
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

## Verified commands

~~~bash
./scripts/gcloud auth list \
  --filter=status:ACTIVE \
  --format='value(account)'

./scripts/gcloud storage ls \
  'gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/**'
~~~
