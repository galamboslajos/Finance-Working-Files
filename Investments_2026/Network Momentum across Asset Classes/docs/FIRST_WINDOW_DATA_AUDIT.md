# S&P 500 completeness for the first network learning window

The 2011-2015 point-in-time universe has 91.45% usable member-day quotes in the
current assembled dataset. The principal problem is long missing blocks and
identity routes, rather than scattered missing days. Source and mapping repair
should precede interpolation or moving the research start.

The requested chronology uses 2009-2010 raw prices for initialization,
2011-2015 as the first network-learning history, and early 2016 for the first
full five-window ensemble. This is distinct from the later OLS fitting and
out-of-sample testing chronology, which remains to be selected.

## Scope and measured completeness

The audit preserves every dated constituent in the existing `sp500` panel.
There are 1,258 exchange sessions and 600 distinct source security IDs across
the five years, reflecting index additions and removals. No fixed survivor
universe replaces those daily cohorts.

| Year | Expected member-days | Usable quotes | Missing prices | Identity withheld | Invalid or flagged | Usable quote coverage |
| --- | --- | --- | --- | --- | --- | --- |
| 2011 | 126,000 | 113,733 | 9,756 | 2,511 | 0 | 90.26% |
| 2012 | 125,000 | 113,945 | 8,940 | 2,100 | 15 | 91.16% |
| 2013 | 126,000 | 115,292 | 8,909 | 1,794 | 5 | 91.50% |
| 2014 | 126,289 | 115,978 | 8,365 | 1,934 | 12 | 91.84% |
| 2015 | 126,720 | 117,181 | 7,632 | 1,869 | 38 | 92.47% |
| Total | 630,009 | 576,129 | 43,602 | 10,208 | 70 | 91.45% |

Of the 600 historical securities, 498 have usable quotes on every day they
were members in this window; 102 have at least one problem. The 498 count
does not imply 498 simultaneous graph nodes or complete pre-membership history.
Individual securities have different membership spans.

The 53,880 unusable member-days form 177 contiguous episodes, split when the
reason changes or membership ends:

| Episode length | Episodes | Unusable member-days |
| --- | --- | --- |
| One session | 97 | 97 |
| Two to five sessions | 12 | 27 |
| Six or more sessions | 68 | 53,756 |

Thus 99.77% of unusable member-days lie in long blocks. Only 124 observations
belong to episodes of one to five sessions. The large repair queue includes
XOM, BK, historical BBBY, DD, APC and other source identities. Tickers in the
queue are provider membership labels, not proof of historical identity.

## Price history for the first full graph

The separate anchor audit uses the 505 constituents on **4 January 2016 at
end of session**. It does not retrospectively replace the 2011-2015 membership
cohorts with those 505 names.

With full 63- and 252-observation MACD normalization windows, the first complete
eight-feature vector needs at least 314 price observations. The longest graph
requires 1,260 consecutive feature observations and therefore at least 1,573
usable prices. For this anchor, the minimal raw-price span is **5 October 2009
through 4 January 2016**. The corresponding 1,260-feature window starts
31 December 2010; the five calendar years 2011-2015 contain only 1,258 sessions.
These are exact session bounds for this anchor, rather than a different choice
of the intended approximate five-year history.

- **370 of 505** constituents have all required usable price observations.
- **63** additional constituents have only one to five unusable observations
  in that required raw window.
- The remaining **72** have larger history gaps or withheld identities.

These are necessary-history checks, not confirmed feature availability. The
audit does not calculate momentum features, inspect their zero-variance
denominators, or add an EWM burn-in requirement. It retains the existing
conservative OHLCV quality gate. Across the anchor histories, 158 rejected
observations still contain finite positive raw and adjusted closes. These are
field-specific review candidates, not automatically acceptable prices.

### Closer review of the 63 stocks with minor gaps

Those 63 stocks account for 115 unusable observations in the anchor's required
price history. Only eight bars are absent, across SPGI (three), CELG (two),
COL (two) and LVLT (one). The other 107 bars are present: 106 have raw and
adjusted opens outside their respective high-low ranges, while ADS on
2012-01-17 has unavailable raw and adjusted lows. All 107 have finite positive
raw and adjusted closes; 106 also have closes within both reported ranges.
No observed close is outside an available reported range. These checks do not
independently authenticate those closes or prove which source field is wrong.

For 59 of the 63 stocks, rejected existing OHLC bars are the only blockers;
there are no absent bars in their required windows. For 58, all such blockers
are open-range anomalies; ADS has the missing-low case. CELG has both a
rejected existing bar and two absent bars.

Retain all 63 in the PIT universe. A proposed research treatment is to review
close-price and adjustment continuity separately from execution OHLC quality.
After that review, an explicit `feature_close_eligible` policy can admit
supported closing prices while preserving original provider flags and blocked
execution fields. Recompute close-based daily returns and features under that
policy; merely retaining the original rows does not remove NaNs generated by
the existing whole-bar quality gate. The eight absent bars still need actual
source recovery or a declared missing-input treatment.

The subsequent [close-price review](FEATURE_CLOSE_REVIEW.md) admits all 107
existing closes under a separate provisional feature-only policy. It restores
complete required history for 59 anchor stocks, increasing the count to 429.
Original prices and execution quality flags remain unchanged; the eight absent
bars remain unresolved. Diagnostics are saved as
`minor_gap_summary.json`, `minor_gap_stock_triage.csv`, `minor_gap_keys.parquet`
and `minor_rejected_bar_diagnostics.parquet` in the audit directory.

## Action for missing observations

### Establish the expected observation

First check the independent exchange calendar and exact PIT membership.
Weekends, exchange holidays and dates outside a stock's membership are not
missing member-day quotes. Pre-entry prices are nevertheless needed when
they fall in the history of a currently eligible graph node. Verified listings,
halts, suspensions, mergers and delistings need event evidence; a first or last
captured quote does not establish those event dates.

### Repair identity and source capture

Review stable source/unified/native IDs and the historical ordinary-share
class before requesting or attaching prices. A reused ticker, a current
security master or a provider's generic `resolved` flag is insufficient.
Retain all original memberships, routes and review flags in the evidence.

Use existing pinned archives and reviewed prior repairs first. Then retrieve
actual missing observations from the verified original provider or a documented
independent source. Bind every replacement to an exact security, session,
adjustment convention and source version, and preserve the original values in
a versioned repair overlay. Do not silently splice raw and adjusted levels
between providers.

Prior Deep Momentum evidence identifies concrete leads:

- The original BK route selected preferred shares. A separately reviewed
  ordinary-share route has complete required history from 2008 and was used
  in a later research shadow. Its applicability and same-date collisions
  must be checked against this project's cohorts before import.
- The XOM original route lacks the requested earlier history. A reviewed
  independent source begins in 2016 and therefore cannot fill 2011-2015 or
  this first graph's warmup. Its earlier history remains a separate lookup.
- Historical BBBY has a confirmed wrong-issuer route; leave it withheld until
  a supported historical replacement exists. El Paso has separate reviewed
  source/route evidence that may support a bounded repair.

These are repair leads from the prior project, not amendments already applied
to this dataset. See the prior project's
[candidate validation](</Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2/docs/CANDIDATE_VALIDATION_BK_XOM_20261021.md>),
[BK and XOM shadow rebuild](</Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2/docs/BK_XOM_SHADOW_REBUILD_20261021.md>)
and [source-quality repair](</Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2/docs/SOURCE_QUALITY_REPAIR_20261016.md>).

### Handle isolated gaps and rejected fields

Inspect each short episode against the provider's actual records, calendar,
corporate actions and any verified trading halt. Re-download the actual bar
when a capture omission is established. A plausible interpolated price is not
an observed historical close.

Because this strategy's features use closing prices, a malformed open, high,
low or volume should prompt field-specific review. If the raw and adjusted
closes and their continuity are independently validated, a separate close-only
feature-quality policy may admit those fields while leaving affected execution
fields unavailable. Finite positive closes alone are not enough evidence to
make this change.

Default to **no forward fill, backward fill, interpolation or zero-return
substitution**. In a verified no-trade/halt case, an official published valuation
can be recorded under an explicit event policy; it must not imply an executable
trade or disguise unavailable observations as freshly traded prices.

### Preserve unresolved gaps during modeling

Keep every original PIT member/date and its reason code. Calculate which
features and graph windows actually depend on an unresolved observation.
Withhold the affected stock from a graph on dates when its required inputs
are invalid, preserving the full original-cohort denominator in coverage
reports. Reassess eligibility causally on each date; do not drop a stock from
all earlier training dates because it develops a later gap.

A strict complete-feature graph can lose a node for an extended period after
one missing observation, since the gap remains inside its trailing input
window. That consequence is why repairing genuine capture errors matters.
Graph estimation with missing entries, overlap-based distances or synthetic
prices changes the method and belongs in a separate declared experiment.

Do not delete a held position when its price or exit is missing. Retain the
position, flag the unresolved valuation and withhold affected continuous
performance where a valid path cannot be established. Missing prices must
never turn an adverse outcome into an omitted observation.

## Deliverables and verification

Run `audit_first_window.py` to reproduce the check in a new output directory.
Current outputs are under `data/first_window_audit/`:

| Artifact | Purpose |
| --- | --- |
| `coverage_by_year.csv` | Original annual denominators and quote/history counts. |
| `security_coverage.csv` | All 600 historical source securities and completeness while members. |
| `repair_queue.csv` | The 102 securities with problems, ordered by affected member-days. |
| `member_gap_runs.csv` | Exact missing/rejected episodes, reasons and session lengths. |
| `missing_member_days.parquet` | All 53,880 affected original member/date records. |
| `first_2016_graph_history.csv` | Required warmup/history for each of the 505 anchor constituents. |
| `anchor_gap_runs.csv` | Exact dates that block each anchor stock's required history. |
| `problem_states.csv`, `summary.json`, `validation.json` | State counts, source/script fingerprints, scope and verification. |

All yearly totals match the earlier full-dataset coverage report. Every missing
member-day reconciles to an episode. An independent distinct-session count
reproduces all 505 anchor security totals, including the 370 complete histories
and 63 near-complete cases. Nineteen structural tests pass. No source prices,
memberships, model fits or backtest results were changed by this audit.
