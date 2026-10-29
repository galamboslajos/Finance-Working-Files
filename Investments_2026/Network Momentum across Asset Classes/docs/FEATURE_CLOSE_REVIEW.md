# Reviewed closing prices for the first network window

The separate close-only feature layer restores complete required price history
for **59 stocks** at the first-2016 graph anchor. The count rises from **370 to
429 of 505** original constituents. All 107 existing flagged observations pass
the scoped internal-consistency review; eight genuinely absent bars remain
unresolved across SPGI, CELG, COL and LVLT.

The layer is saved under `data/feature_close_review/v1/`. Its prices are the
original provider observations. No open, high, low, close or adjusted close was
replaced, interpolated or forward-filled. All original quality flags remain.

## Closing price review

The 107 reviewed observations comprise 106 provider-flagged open-range
anomalies and ADS on 2012-01-17, whose raw and adjusted lows are missing.
Their raw and adjusted closes are finite and positive. Available high/low
bounds are compatible with those closes. The suspect opens are not treated as
valid execution prices.

For each exact native/date target, the review verifies:

1. Daily, non-intraperiod frequency and the expected narrow anomaly profile.
2. Positive finite raw and adjusted closes, compatible with available ranges.
3. A common adjustment scale for high, low where available, and close.
4. Continuity from the immediately preceding admitted session: the change in
   adjusted/raw close ratio agrees with the recorded backward-adjustment factor.

The numeric review uses an explicit assumption of a +/-0.00005 uncertainty
envelope around stored adjusted prices, plus a 1e-12 arithmetic tolerance.
This is a declared precision model, not a provider guarantee. All 107 targets
pass. The two non-unit factors coincide with source-recorded dividends; those
factors are checked without adding the dividends again to returns. Large or
adverse returns are not clipped or used as admission criteria.

Only the target row and earlier admitted session determine admission. An
earlier reviewed target can support the next consecutive target; a missing
or unadmitted previous-session close causes refusal. Future observations do
not determine the decision.

This is **provisional internal consistency**, not independent confirmation of
the historical closing prices, adjustment vintage or economic total returns.
The original provider warnings and existing provisional identity limitations
remain visible.

## Feature and execution eligibility

The materialized `feature_prices/year=YYYY/` tables cover the initialization
and first-window scope, 2009-01-01 through 2016-01-04. They preserve all original
rows/columns and add:

| Column | Meaning |
| --- | --- |
| `feature_close_eligible` | Original accepted quote or an explicitly reviewed close. |
| `feature_close_status` | Original acceptance, provisional close review, or withholding. |
| `close_review_policy` | Review identifier on the 107 admitted observations. |
| `feature_return_1d` | Adjusted-close ratio with admitted current and exact previous-session closes. |
| `feature_log_return_1d` | Logarithmic version of that return. |
| `feature_return_status` | Explicit current/previous missing or withheld state. |
| `execution_bar_eligible` | Original whole-bar eligibility; none of the 107 warnings is cleared for execution. |

Recomputation makes **209 additional daily feature returns** available,
including returns immediately after reviewed dates. The eight absent keys
remain absent, and no return bridges a missing session. Existing
`provider_adjusted_return_1d` retains the original preparation policy; the new
feature worker must use `feature_return_1d` and apply `feature_close_eligible`
to closing-price inputs.

```python
from review_minor_closes import load_feature_prices

prices = load_feature_prices(
    "2009-01-01", "2016-01-04",
    columns=["date", "intrinio_security_id", "adj_close",
             "feature_close_eligible", "feature_return_1d"],
)
prices["feature_adj_close"] = prices.adj_close.where(prices.feature_close_eligible)
```

These are history tables, not a replacement investment universe. The original
PIT panel still selects each date's members and retains unresolved identities.
The first-2016 history table is an audit anchor rather than a survivor-filtered
cohort for all of 2011-2015. Complete price history remains a necessary check.
The subsequent [feature and network pilot](FIRST_NETWORK_PILOT.md) builds the
actual eight features and confirms all 429 histories under its declared
indicator conventions.

## Remaining source recovery

The eight absent native/date keys are missing from the original pinned
Intrinio curated release as well as the prepared dataset. The corresponding
four retained raw-page references were located through the original inventory.
The source manifest chain was retrieved and verified by its recorded hashes
and generations after the user refreshed the research account login.

The authenticated account `info@galamboscapital.com` can read the curated
metadata. Reads of the raw acquisition manifest and the exact SPGI raw-page
reference returned **HTTP 403**. No raw price page was downloaded. This is a
source-access limitation; another login refresh is not the established remedy.
The response also does not establish that a raw page contains the missing bars.

The recovery queue contains:

| Source membership label | Provider native ID | Missing sessions |
| --- | --- | --- |
| SPGI | `sec_gej9GX` | 2013-05-13, 2014-09-10, 2014-09-26 |
| CELG | `sec_bzqMBz` | 2014-10-23, 2014-10-24 |
| COL | `sec_5X7BGg` | 2014-10-23, 2014-10-24 |
| LVLT | `sec_zvN3Kz` | 2011-10-31 |

Next recovery requires readable raw-page exports/access, or actual historical
provider observations for these exact identities/dates. Compare any recovered
raw and adjusted values to the existing units and source lineage before a
separate version admits them. If the original raw pages lack those dates too,
their access alone will not supply a repair. The four names stay in the PIT
ledger with unresolved feature windows; none is silently removed or filled.

## Reproduction and validation

Run `review_minor_closes.py` with a fresh `--output` to reproduce the review
and materialization. The layer includes `policy.json`, the review ledger,
exact prior-session context, original source/output hashes, and
`first_2016_history_after_review.csv`. `missing_bar_recovery.json` and
`missing_bar_request.csv` retain the source-search results and outstanding keys.

All original price columns were compared unchanged across 1,610,463 output
rows. An independent check recomputed returns over 1,601,851 exchange-session
price rows, checked all 107 unchanged close pairs, confirmed that the eight
absent keys remain absent, and verified annual schema consistency and the
date-slice loader. All 28 structural tests pass, including future-prefix
invariance, missing-session handling, action-factor consistency and preservation
of execution rejections.
