# Monthly research returns and preserved daily data

Implemented as a user-authorized engineering stage on 2026-10-14. The selected
research design remains rolling ten-year training, one-year testing and monthly
next-month predictions. Monthly return construction does not perform a model
fit, create ten-class labels or run a portfolio backtest.

USER UPDATE: the active notebook applies
[provisional identity admission](PROVISIONAL_IDENTITY_POLICY.md). Unique resolved
review-flagged routes may receive endpoint ratios under the accepted uncertainty;
their original state is retained alongside separate identity eligibility fields.
All members and missing outcomes remain in the ledger. Existing endpoint/path,
calendar, timing and independent daily-table rules are unchanged.

## Independent daily and monthly views

The master notebook keeps two named tables:

```python
daily_df = monthly_run.sample.daily_prices.table.to_pandas().copy(deep=True)
monthly_df = monthly_run.sample.panel.table.to_pandas().copy(deep=True)
```

`daily_df` retains the source daily raw/adjusted OHLCV, dividend, split ratio,
factor, frequency, completion and quality fields for the declared sample.
`monthly_df` is a separate derived historical-cohort/outcome ledger. Monthly
construction never resamples over, overwrites or truncates the original daily
Arrow table. Future backtesting can use daily paths once execution and economic
accounting are defined; monthly prediction frequency does not imply monthly
portfolio valuation. This engineering sample is not the eventual complete
daily backtesting dataset.

The underlying worker uses immutable Arrow tables; pandas is an optional
notebook dependency for the readable DataFrame views. Neither DataFrame is a
model-ready panel. In particular, the monthly ledger includes unresolved,
unrequested and missing outcomes rather than hiding them.

## Dates and cohort rule

For outcome month `m`:

1. Find the preceding month end `s` and current month end `e` in the pinned
   independent XNYS exchange calendar. Require session evidence in adjacent months;
   do not invent calendar-day closes or fill missing months.
2. Reconstruct membership at `s` using effective, inclusive end and supplied
   availability dates. The identity adapters use the reviewed source-to-provider
   route; current delisted/master flags never choose the cohort.
3. Preserve one row per original source constituent at `s`, independently of
   future membership and quotes. A removal during `m` remains in the outcome
   ledger. A constituent entering during `m` is not retroactively admitted.
4. Attach requested daily observations without removing original cohort rows.
   Keep every unavailable or rejected endpoint and identity state visible.

January 2000 therefore needs the December 1999 month-end cohort and price
anchor. This bounded anchor does not change the research initialization or
settle how much pre-2000 history future momentum features need.

The vendor calendar cannot be used as an exchange-session authority: it marks
2000-01-17 and 2000-02-21 as sessions, although those are NYSE holiday closures.
[Contemporaneous SEC-filed fund documentation](https://www.sec.gov/Archives/edgar/data/799295/000100547700000369/0001005477-00-000369-d1.html)
lists the relevant NYSE holiday observances. The monthly adapter therefore uses
the independent XNYS implementation in pinned
[`exchange_calendars` 4.13.2](https://pypi.org/project/exchange_calendars/4.13.2/),
and compares its dates with the original source calendar. It records date
differences, package/code/session hashes and retains both calendars. The
correction does not use quote availability, a survivor intersection or a
hard-coded two-holiday patch. Regression checks also cover Good Friday at a
month end and the September 2001 exceptional closure period. This library
reconciliation is not a certification of every exchange/venue or all-period
historical knowledge-time evidence; venue and execution rules remain open.

`outcome_month=m` is the realized month following `signal_date=s`. Its return
is known only after the outcome month has completed; it cannot be used as a
feature at `s`. Future feature construction at `e` must separately use the
then-eligible signal cohort and legitimate earlier price history, including a
new entrant's pre-entry observations. Membership must not truncate an otherwise
valid security's historical price series. No target ranks are constructed here.

## Return basis and statuses

The implemented numeric field is:

```text
provider_adjusted_return = adj_close[exact e] / adj_close[exact s] - 1
```

Both exact endpoints must have positive finite adjusted closes, `frequency`
equal to `daily`, `intraperiod` explicitly false and `price_quality_status`
equal to `accepted` (the pinned release's value) or `ok` (an explicit accepted
adapter state). Unknown quality/completion states are not accepted. Duplicate
price/date keys fail rather than being aggregated or silently deduplicated.
This is an explicit V2 endpoint rule. The paper describes selecting the last
stock observation within each month for its Datastream sample; that does not
authorize a silent stale-price fallback in our different daily source.

[Intrinio's API](https://docs.intrinio.com/documentation/web_api/get_security_stock_prices_v2)
describes adjusted closes and adjusted volumes as adjusted for splits and
dividends, and raw volume as shares exchanged. The worker does not add the
daily dividend field again or apply a second split adjustment. It preserves
those source fields for subsequent checks. The field is explicitly a provider
adjusted-close research ratio, not certified CRSP-equivalent total return or
an executable trading return.

| `return_status` | Meaning |
|---|---|
| `ok` | Exact accepted endpoints support the stated arithmetic |
| `identity_unresolved` | Missing, flagged, ambiguous or conflicting identity route is retained and quarantined |
| `not_requested` | Engineering sample did not request this otherwise mapped ID; source coverage is unknown |
| `missing_start` | Requested exact preceding-month endpoint is absent |
| `missing_end` | Requested exact outcome endpoint is absent; possible terminal-event review is needed |
| `invalid_endpoint` | Nonpositive/nonfinite endpoint or nonfinite ratio is not accepted |
| `source_quality_flag` | Endpoint frequency, completion or source quality is unacceptable or unknown |

Separate endpoint failure reasons retain multiple simultaneous problems.
Missing endpoints produce null returns. No forward fill, last-observed-price
substitution, zero return or removal from the denominator is performed.
Missing end is not evidence of a particular delisting or payout.

## Daily paths and volume

Expected daily counts come from the independent XNYS sessions in the outcome
month; no fixed 20/21-day threshold excludes short holiday months. The preceding
signal anchor is excluded from the monthly count. Requested IDs retain observed
counts, missing dates and invalid daily-bar dates. Unrequested or unresolved
IDs have unknown observed coverage (`None`), distinct from genuine missing
observations after a price request.

`daily_quote_path_complete` requires accepted endpoints and all valid listed
outcome sessions. A month-end ratio can still exist with an interior gap; it
does not become a complete daily path. The first outcome session is part of
the interior path between the preceding anchor and the outcome end.
`monthly_volume` and `monthly_adj_volume` are sums only when every outcome
session has valid observations and finite nonnegative volume. Incomplete or
invalid sums remain null, never partial sums or null-to-zero replacements.
These fields are volume aggregates, not share-turnover ratios or features.

`daily_backtest_ready`, `accounting_certified` and `knowledge_time_verified`
remain false for every row. Complete daily quotes do not establish execution,
dividend entitlement, borrowing or merger/delisting cash-flow completeness.
Future missingness or path-completeness flags are audit outcomes, not criteria
for admitting stocks at the earlier signal cutoff.

## PIT and survivorship limits

The supported mode is explicitly `historical_reconstruction`. Strict
`knowledge_time` mode refuses these sources. Membership availability is
reconstructed, identity links lack historical validity intervals and adjusted
prices are retrieval-time history. Review flags are not proof of incorrect
identity, but they cannot silently become accepted outcomes. Same-date target
collisions quarantine all affected source identities.

The pinned release has no separate terminal-payout, delisting-return, merger
or cash-entitlement table. Intrinio also documents that
[merger price histories can be stitched](https://help.intrinio.com/stock-price-adjustments-ipos-and-mergers),
so a stable provider ID alone is insufficient proof of historical economic
continuity. These limitations are separate from whether an endpoint ratio can
be calculated. Original-cohort retention prevents common selection mistakes;
it does not prove the entire vendor history is survivorship-bias free.

## Bounded notebook execution and private evidence

`configs/research_workflow.json` schema version 2 controls the stage. Its first
example uses January–February 2000, the combined historical index cohort and
two requested price IDs. Those IDs are selected from the first signal cohort
only, without requiring future membership or quote availability. This is an
engineering sample, not the modelling universe or a statistical subsample.

`monthly_loading.py` reads shared calendar/identity metadata once, batches the
explicit signal-date membership query and then makes one projected daily price
request. Row/file/output/byte ceilings are cumulative across this stage.
The original vendor calendar stays unchanged; a bounded independent session
table determines monthly endpoints and daily path expectations.
The local adapter caps samples at three outcome months and ten price IDs;
exceeding a budget raises before returning partial success. A full historical
build needs its own measured execution plan. The pure arithmetic worker is
reusable under that later plan; no heavy scan or cloud work is launched here.

Protected 2024+ financial requests/rows fail before arithmetic. Calendar
metadata can extend into 2024 to establish completion of December 2023.
Parquet may transiently decode excluded values in a mixed block, but the
reader masks them before returned values, Python conversion or analysis.

Private `outputs/notebook_audits/monthly-.../` artifacts contain independently
saved `daily.parquet`, `monthly.parquet`, `source_calendar.parquet`,
`exchange_calendar.parquet` and `audit.json`. The audit records
source/request/code/runtime pins, full signal cohort/exclusion ledgers, schemas,
table hashes and coverage counts. Identical reruns reuse evidence; changed
results under unchanged provenance are rejected. Nonfinite original source
values remain in Parquet rather than nonstandard JSON. The committed master
contains no execution outputs, downloaded data or generated research panels.

## Verified local execution

On 2026-10-14, the installed master completed all ten code cells in two fresh
V2 kernels, from the project root and notebook directory. The bounded sample
retained **82 daily observations** and **1,799 monthly cohort rows**: 899 January
constituents and 900 February constituents. Four requested endpoint ratios
were accepted; 1,192 identity-unresolved and 603 unrequested rows remained in
the ledger. All four requested monthly daily paths were complete against the
independent sessions, while accounting/PIT certification and daily backtest
readiness stayed false.

December 1999–March 2000 reconciliation retained 88 original vendor dates
and 85 independent sessions, with three source-only holiday dates listed
above. Both outcome months have 20 independent sessions. The stage considered
9,741,712 candidate rows across five source data files and 109 footers, with
43,454,076 projected compressed bytes; most rows considered were membership
metadata. Identical reruns reused the same private bundle and table/audit
hashes, and the committed master remained without outputs. Independent
DataFrame checks confirmed the daily view matched its original Arrow table.
All **164 tests** passed, using structural fixtures and date-only calendar
checks rather than financial-outcome-driven model tests. These results do not
certify complete 2000–2023 universe coverage or economic returns.
