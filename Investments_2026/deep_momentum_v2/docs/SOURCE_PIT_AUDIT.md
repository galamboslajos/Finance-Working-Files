# Local source and PIT audit for V2 foundations

Read-only review on 2026-10-14. No cloud/network access, price-value read,
full membership scan, feature build, model fit or backtest. Reviewed V2
instructions, paper skill, research/data/US ledgers; pinned manifest JSON,
Parquet footers, small metadata/dictionary/calendar/identity tables and 256
bounded pre-2024 membership rows. Membership samples are the first 128 rows
of Siblis row groups 0 and 7, covering S&P 500 1970 and S&P 400 1997–1998.
Several other tiny membership metadata samples contain post-2023 dates but
no financial values. Do not treat this as complete economic data validation.

## Pinned releases and useful relative paths

- `data/raw/equity_us_sp_composite1500_membership_daily_v1/professional-reference-20260807-r1/`
- `data/raw/equity_us_index_membership_daily_unified_v1/professional-reference-20260807-r1/`
- `data/raw/equity_us_intrinio_market_daily_v1/professional-reference-20260807-r1-pdrs-v2-r3/`

The private configuration pins canonical manifest SHA-256 values; load only
objects explicitly present in those manifests. Never discover unpinned nearby
files as implicit input. No new whole-file checksum work was performed.

## Membership fields and conventions

Use Siblis `tables/membership_by_index_by_date/part-00000.parquet` as the
membership source. Exact index IDs are `sp500`, `sp400`, `sp600`.
Columns: `index_id`, `index_name`, `trading_date`, `security_id`,
`membership_ticker`, `company_name`, `isin`, `is_member`,
`membership_effective_date`, `membership_end_date`,
`membership_availability_date`, `source_ref_id`, `source_workbook`,
`source_sheet`, `source_row`, `identity_confidence`, `pit_policy`.
Dates are ISO strings, not Arrow date types. Reject malformed dates rather
than comparing arbitrary strings or coercing errors to missing.

Manifest policy: use changes only from stated effective dates; do not infer
presence before earliest supported provider observation. Source availability
contract says component changes are after-close events effective on the next
XNYS session. Row `pit_policy` is `siblis_after_close_next_xnys_session`.
`component_changes_normalized` contains `change_date`, `announcement_date`,
`effective_date`, `action`, source and identity fields.

Bounded evidence establishes that `membership_end_date` is inclusive for the
sampled ADM spell: effective1970-03-31, end1970-04-23, still a member on
1970-04-23; removal event dated1970-04-23 becomes effective1970-04-24.
Do not claim every source interval has independently been validated.

All256 sampled pre-2024 membership rows have
`membership_availability_date == trading_date`. This is a reconstructed
daily availability field, not a timestamped historical announcement archive.
No within-session ordering is supplied. A date-level end-of-session API can
require exact trading-date rows, `is_member`,
`effective_date <= asof <= end_date` and `availability_date <= asof`.
A pre-open API cannot infer that same-day information was already known.
Do not promote future end dates into model features or a future-survival
eligibility screen. On absent exact-date rows do not silently forward-fill
membership or infer unsupported early history.

The file has14,747,642 rows in15 row groups, sorted broadly by index/security
spell rather than trading date. Date min/max in each row group is broad;
day filters can require many groups. Examples: row groups0–5 are `sp500`,
group6 crosses `sp500`/`sp400`, groups7–8 `sp400`, group9 crosses
`sp400`/`sp600`. Footer coverage includes S&P500 from1970-03-31 and
S&P400 from1995-01-02; those are evidence bounds, not research start dates
or certified complete constituent cohorts.

Footer preflight for either `2010-01-29` or `2009-01-30`, using conservative
lexicographic index/date min/max pruning:

| Requested index IDs | Potential row groups | Physical row upper bound | Compressed essential-column bytes |
|---|---|---:|---:|
| `sp500` | 2,3,4,5,6,9 | 6,291,456 | 21,759,607 |
| `sp400` | 6,7,8,9 | 4,194,304 | 12,665,954 |
| union | 2 through9 | 8,388,608 | 27,846,415 |

The ten columns in the byte estimate are index/date/source ID/ticker,
is_member, effective/end/availability dates, identity_confidence and
pit_policy. Row group9 has index range `[sp400,sp600]`, whose lexicographic
interval includes `sp500`; a generic min/max preflight must conservatively
retain it even though observed grouping suggests no sp500 rows there.
These counts are metadata-only potential work, not actual reads or constituent
counts. A one-date engineering audit can be bounded at this cost; require an
explicit adequate budget and report the broad layout rather than pretending
filtered output cardinality equals rows scanned. Do not silently expand the
request into multiple dates or an entire-history build.

`trading_calendar_sessions` contains `trading_date`, `calendar_name=XNYS`,
`is_trading_session`.14,445 rows. Read/filter this small metadata table for
calendar lookup. Session labels should still be checked against odd dates,
provider event expansion and execution rules before economic use.

Upstream QA records3,445 dropped cross-index rows across2,178 security-dates,
resolved by latest effective date then component priority
`sp500=300, sp400=200, sp600=100`. This is an upstream identity/membership
choice to preserve in provenance, not a reason to silently duplicate or pick
the first row during our union join.

## Identity join route and real ambiguity

1. Siblis `security_id` -> unified `security_identity_bridge.source_security_id`,
   **also constrain `source_product_id == 'sp_composite1500'`**.
2. Carry `unified_security_id`, `bridge_method`, `bridge_confidence`,
   `source_identity_confidence`, and source IDs.
3. Join Intrinio `intrinio_identity_links` on `unified_security_id`.
   Carry `intrinio_security_id`, `query_identifier_type`,
   `identity_review_required`, `resolution_status`, `http_status`, source
   response/run/manifest lineage. Provider `resolved`/HTTP200 is not proof
   of correct historical identity.
4. Intrinio price join uses `intrinio_security_id` and `trading_date`.
   Tickers are display/evidence attributes, never research keys.

Unified bridge has20,283 rows across3 upstream products. The Siblis product
has7,317 unique source IDs but7,281 unique unified IDs (36 duplicates).
Methods:2,665 `isin/high`,4,651 `ticker_company/medium`,1`source_local/low`.
2,441 Siblis bridge rows have blank first/last membership dates. These fields
are observed coverage summaries, not independently documented identity-valid
intervals or historical knowledge-time fields.

Critically, all24,616 Siblis `security_identifier_history` rows have blank
`valid_from` and `valid_to`. Its name does not establish usable dated aliases.
Intrinio identity links also have no validity/availability interval.

Intrinio links have14,297 rows, one per unique unified ID.9,564 do not require
review;4,733 do.3,820 Intrinio target IDs are shared by multiple unified IDs,
maximum9 aliases. Among Siblis source rows,6,647 link,670 do not;4,368
linked source rows require review. Distinct Siblis unified IDs linked:6,611.
These whole-metadata-table counts are not date-specific cohort coverage.

A loader should preserve all eligible source rows in an audit ledger. Mark
unmatched, non-resolved, review-required, low/medium-evidence policy states,
source duplicates and target collisions explicitly. Do not silently inner-join,
deduplicate by first row, merge economically distinct historical companies,
or treat a current ticker query as a time-valid match. Conservative quarantine
is an engineering default, not a claim that excluded names do not belong in
the universe. If model-ready rows require identity confidence, preserve the
excluded names and counts so a confidence filter cannot hide bias.

## Historical price and knowledge-time limits

Intrinio `intrinio_daily_prices` grain is `intrinio_security_id,trading_date`.
Price columns include raw/adjusted OHLCV, `factor`, `split_ratio`, `dividend`,
quality flags/status and `retrieved_at_utc`. First price shard contains255,503
rows in one group; footer ranges span1972-06-01–2026-07-24, and multiple IDs.
It is not a date partition. Restrict explicit columns, ID/date filters and
row budgets; do not use `read_table(...).head()` over entire history.

Manifest PIT rule only asserts `trading_date <= retrieved_at_utc`.
`source_refs.availability_contract` explicitly says: raw prices are event
dated and provider-adjusted history is retrieval-time data. The master and
price retrieval timestamps are July2026. Therefore these files support a
disclosed reconstruction with historical effective membership and backward
feature cutoffs, **not a certified historical knowledge-time/revision-as-of
data service**. Filtering `retrieved_at_utc <= historical cutoff` would
produce no historical dataset; pretending trading date proves publication
time would overclaim. An API should expose this distinction in reports.

Intrinio master:9,266 records,5,053 `delisted=True`,4,199 false,14 null.
Never filter current delisted status, last_stock_price, eventual quote
availability or terminal dates to choose historical signal cohorts. First/last
price summaries are current metadata, useful for evidence but not predictors.
Metadata retention of delisted names does not prove complete returns,
delisting cash flows, terminal prices or economically survivorship-free coverage.

The initial audit deferred return construction. The subsequent user-authorized
[monthly engineering stage](MONTHLY_RETURNS.md) implements explicitly named
provider-adjusted endpoint ratios while preserving daily source data and the
original cohort. Certified total returns, stale/missing-price resolution,
terminal-event accounting and date-specific identity repair remain separate
audits. Missing future labels/quotes must not erase original signal-time names
from eligibility/coverage or the later holding denominator.

## Recommended foundation contract

- Two explicit modes: historical-effective reconstruction (disclosed limits)
  and strict knowledge-time PIT (fail unless timestamp/revision evidence exists).
- Signal-time membership ledger independent of price and next-period labels.
- Distinct eligible, uniquely mapped, source-review-required, price missing,
  future-label missing and terminal-outcome states; no implicit drop path.
- Always include source manifest pins, requested dates/indexes/columns,
  budget, date cutoff, source identities and caveats in audit artifacts.
- Test entry/removal boundaries, after-close next-session behavior, inclusive
  ends, future availability, unsupported dates, source/target duplicates,
  ambiguous mappings, missing prices and a later-delisted member.
- Treat this as prevention of known causal/survivor errors, not a completed
  certification of absence of all survivorship bias.
