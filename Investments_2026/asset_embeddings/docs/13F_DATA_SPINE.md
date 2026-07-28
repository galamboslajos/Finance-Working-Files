# 13F point-in-time data-spine decisions

Status: **provisional, audit-gated**

This record converts the first full-history 13F review into explicit implementation policies.
Nothing in this document authorizes silent cleaning. A policy becomes final only after the
full-history audit passes and its evidence is reviewed.

## Purpose

The data spine must reproduce the information an investor could have known at each decision
timestamp. It also preserves a separate final-state view for audit. Final amended data must never
rewrite an earlier point-in-time observation used by a model or backtest.

The immutable source grain remains:

~~~text
manager CIK x filing accession x reported holding line
~~~

## 1. Amendments

Amendments are ordered by `availability_timestamp_utc`, then `filed_at`, with accession number as a
deterministic tie-breaker. The amendment number is diagnostic metadata, not an availability clock.

- `RESTATEMENT` replaces the complete previously observable state for that manager and reporting
  period.
- `NEW HOLDINGS` supplements the previously observable state.
- A future amendment never changes an earlier as-of snapshot.
- Missing or conflicting amendment metadata is quarantined for review.
- An amendment without an observable base is marked incomplete rather than treated as a complete
  portfolio.

The data spine will expose two distinct views:

- `holdings_asof`: the state knowable at a requested decision timestamp;
- `holdings_final`: the final corrected state for ex-post audit only.

The amendment type is joined from cover pages to filings and holdings by accession number. Join
coverage and duplicates must be reported before the amendment policy is applied.

## 2. Notices, combination reports, and confidential treatment

- A `13F NOTICE` is a delegated report, not a zero portfolio.
- A `13F COMBINATION REPORT` is partial until the other reporting-manager relationships are
  resolved.
- Delayed new-holdings amendments enter only after their availability timestamp.
- Confidential treatment is not inferred merely from a long delay; an explicit filing indicator or
  source statement is required.

Initial coverage states are:

~~~text
self_reported_holdings
partial_combination
delegated_notice
unknown_or_conflicting
~~~

The first complete-portfolio baseline uses only `self_reported_holdings`. Other states remain in the
audit tables and may enter later event-dated consolidation work.

## 3. Asset grain

The raw holding line is never discarded. The first modeled asset is a point-in-time issuer, not an
ultimate parent company.

- Eligible security lines are aggregated to a verified issuer identity.
- Security-level embeddings remain a required robustness comparison.
- Ultimate-parent or company consolidation is deferred until effective-dated corporate-action
  mappings exist.
- Present-day issuer or parent mappings must not rewrite historical identity.
- Unresolved identities remain unresolved; names and tickers are not silent fallback keys.

The mapping audit must review `reported_issuer_cik`, `resolved_issuer_cik`,
`issuer_provider_entity_id`, identity source and confidence, and `mapping_snapshot_date`.

## 4. Instrument scope

Every reported 13F instrument is preserved. The first model is a common-equity benchmark, followed
by typed instrument extensions.

Provisional instrument channels are:

~~~text
cash_share_candidate
call_option
put_option
convertible_or_debt
preferred_equity
warrant_or_right
unit
fund_or_etf_explicit
unknown
~~~

The full-history audit uses explicit filing fields and conservative title patterns. It does not
claim that a cash-share candidate is verified common stock or that all funds can be detected from
the reported class title. Final common-equity eligibility requires a point-in-time security master.

Options, convertibles, and funds must not be added to cash equities as if their dollar values were
identical economic exposures. Later broad models use instrument-specific matrices or typed graph
relations.

## 5. Manager identity

The first manager key is `manager_cik`, stored as a zero-preserving string.

- Name changes under one CIK remain one filing entity.
- Different CIKs remain separate without documented, effective-dated continuity.
- Manager names are labels, never join keys.
- CRD, SEC, and Form 13F file numbers are audit cross-checks.
- Mergers, acquisitions, and spinouts require `effective_from`, `effective_to`, `known_at`,
  relationship type, and source.
- Historical portfolios are not retroactively merged into a current parent organization.

Later research may add `manager_entity_id` and `manager_group_id`, while always preserving the raw
filer CIK.

## 6. Coverage window and partial quarters

The model does not use the minimum reported economic date as its history start. A quarter is
eligible only when the product was observed from no later than that quarter-end and through the
standardized information cutoff.

The first and latest eligible quarters are calculated locally from the observation start, data
freeze, and standardized cutoff; they are not hard-coded or published with the output-free source.

- Missing manager filings are unavailable, not zero.
- Previous portfolios are not silently forward-filled.
- A quarter can be decision-complete while still receiving amendments later.
- Final-state amendment analysis may require an additional seasoning window, but that state is
  never substituted into an earlier backtest snapshot.

## 7. Standardized decision time

The raw spine is event-time accurate. The baseline embedding uses synchronized quarterly
information:

1. determine the official filing deadline, normally 45 days after quarter-end;
2. include only events available by the end of that deadline date;
3. form the signal no earlier than the next U.S. trading session;
4. execute at a separately recorded, feasible market timestamp.

Production scheduling requires an exchange calendar and explicit timezone conversion. A record is
eligible only when:

~~~text
availability_timestamp_utc <= information_cutoff_utc < execution_timestamp_utc
~~~

Late filings and amendments remain available for event-driven or monthly extensions but cannot
retroactively modify the frozen quarterly snapshot.

## Audit gates before matrix construction

The first investor-issuer matrix is blocked until all of the following are reviewed:

1. accession joins reconcile filings, cover pages, and holdings;
2. amendment signals, types, numbers, sequences, and availability ordering are quantified;
3. the continuous coverage window and incomplete boundary quarters are identified;
4. every holdings row receives a provisional instrument class and explicit reason;
5. issuer identity coverage, confidence, and mapping snapshots are measured across all quarters;
6. manager identifier changes are quantified without fuzzy automatic consolidation;
7. timestamp violations and missing availability states are reported;
8. every filter reports rows, value, managers, and assets removed;
9. earlier as-of snapshots remain invariant when later events are added.

The committed notebook is output-free. Full inventory statistics, aggregate audit outputs, cache
files, raw identifiers, and executed notebooks remain under ignored local `data/` paths.
