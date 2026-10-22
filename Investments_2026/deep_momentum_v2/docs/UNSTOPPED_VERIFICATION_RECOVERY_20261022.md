# Unstopped target comparison: verification recovery — 22 October 2026

Status: **independent checks passed for all 20 saved RF fits, four shared
preprocessors and all 72 produced scenarios**. The original failed producer
ledger is preserved. A separate `verification_recovery/recovery_handoff.json`
binds the new independent proof to those original artifact bytes. No model was
refitted, selection changed, trade replayed, source corrected or financial
2024+ value read during this recovery. All older runs remain references.

## What the mismatch meant

The first gate compared two representations of the same SHA-256 digest. The
backtest producer stores plain 64-character hexadecimal logical hashes; the
checker incorrectly expected `sha256:` before those same digits. This affected
the plan and saved payoff fingerprints. Exact file hashes and the logical hash
digits matched. Model/key hashes still require their original prefixed format;
no generic prefix stripping or fingerprint waiver was introduced.

Later gates exposed three checker conversion bugs. Optional native security IDs
were converted from Arrow `None` into Pandas `NaN`; an exact four-column metadata
comparison found zero actual route/admission differences, with 230 ineligible
rows affected by that conversion in the diagnostic ledger. Arrow calendar dates
also needed strict ISO-date serialization, and an all-null annual report column
acquired a different Pandas dtype from its combined-report counterpart.

The final verifier keeps original Arrow identity metadata, accepts only canonical
calendar dates or naive midnight date values for the declared date fields, and
compares annual records in exact order with the full field set, native scalar
types, finite numerical values and unchanged nulls. It retains all probability,
score, original-cohort, basket, cost, stop and independent cash/units checks.
Missing numerical values are not imputed and generic NaN equality is not allowed.

The final worker is `scripts/verify_unstopped_target_comparison_v5.py`.
Versions 1–4 and failed attempts are preserved. Its 38 targeted tests and eight
passive-reader tests passed. Independent reconstruction checked 288 causal payoff
states, 432 original selection-month records, all nine selected horizons and all
72 complete-or-withheld daily accounting paths. A PASS on a withheld path means
that its exposures, failure and withheld metrics were preserved correctly.

## Comparable period and primary score baskets

All rows below cover **3 February 2020–29 December 2023: 47 months / 985 sessions**.
Training uses the preceding ten calendar outcome years with actual completion
cutoffs and one shared supervised population. Every forecast month remains;
January 2020 lacks the required completed payoff history. The old 48-month
results use a different interval and stay separate dated references.

Learning targets are unstopped. This table applies the unchanged **10% per-asset
execution stop**, 50% long/50% short and balanced up to ten names per side. Gross
cost is zero; net cost is 10bp per one-way traded notional. CAGR uses 252 daily
sessions/year; Sharpe uses daily returns and a zero cash rate.

| Target / score | Gross total return | Net total return | Net CAGR | Net daily Sharpe | Net max DD |
|---|---:|---:|---:|---:|---:|
| Endpoint control, matched payoff clock | +24.49% | +14.18% | +3.45% | 0.27 | -28.66% |
| Expected executable return | +19.71% | +9.64% | +2.38% | 0.22 | -31.99% |
| Expected return minus drawdown | Withheld | Withheld | Withheld | Withheld | Withheld |

Across the full predefined comparison, **32 paths are complete and 40
withhold full-period results**. The full 72-row comparison and 288-row annual
tables remain available through master section 29, including all withheld rows,
both stop variants, three baskets and four cost assumptions. Do not rank target
families on a favorable complete subset.

The endpoint control uses a matched lagged 120-month class-payoff clock. Its
expected-return score is distinct from archived original RET. The executable
return score is probability-weighted own-class cash return; expected utility
is probability-weighted own-class return-minus-drawdown, a different quantity.
Best/second-mode portfolios remain separate from these full-distribution scores.

The new proof establishes the specified calculations and preservation checks.
Provider identity, historical knowledge time, incomplete corporate-action/source
coverage, missing terminal observations and investability limitations persist.
Borrow/funding remain zero and daily-OHLC stop fills are idealized. This already
reviewed period is development evidence. No new search or tuning was performed.

## Reopen without computation

Use `dmv2.target_comparison_reporting_recovery.read_target_comparison_recovery`
with the saved run directory. It authenticates the original failed ledger, model
proof, separate backtest proof, read plans and byte-exact copied evidence before
opening only the 72-row and 288-row aggregate tables. It never loads a model,
scans source prices, fits, resumes or replays trades. Master section 28 retains
the original failed state; section 29 shows the separately verified recovery.
Private runs, receipts, models, source data and numerical ledgers stay outside Git.
