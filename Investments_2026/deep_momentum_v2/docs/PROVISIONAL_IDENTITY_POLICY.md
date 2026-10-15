# Provisional identity admission — user decision

USER DIRECTION, 2026-10-14: retry identification with less strict conditions
where defensible, accept the remaining identity uncertainty and continue the
research. This supersedes identity certification as a prerequisite to preparing
provisional research. It does not establish correct matches or remove source,
selection, knowledge-time, adjustment or terminal-accounting limitations.

The active notebook configuration is schema version 4 with
`identity.policy=provisional_unique_resolved_link`, policy version 1 and
`accept_identity_uncertainty=true`. Research status remains
`PROVISIONAL_IDENTITY_UNVERIFIED`. The pure Python APIs default to
`strict_unique_unflagged`; a provisional call must name its policy explicitly.
Legacy schema-3 configurations retain strict behavior.

## Precise rule

The original dated membership cohort is formed first. Admit an existing route
provisionally only when it has one unambiguous source → unified → Intrinio route,
nonblank unified/native IDs, a boolean producer review flag, published
`resolution_status=resolved`, and no native-ID collision with **any** other
original member at that date. A unique clean route remains admitted; an
otherwise qualified `needs_review` route is admitted only when its producer
flag is exactly true. Unknown flags, unresolved responses, missing routes,
ambiguous bridges/provider links and same-date collisions remain blocked.

This loosens the **producer review-flag exclusion**. It does not repair an
identifier, use current ticker/price continuity to guess a replacement, select
future survivors, or count source-alias agreement as independent confirmation.
No raw data, flags or source identity statuses are rewritten.

`identity_status`, `source_identity_status`, producer flags and resolution status
remain visible. Separate `identity_eligible`, `identity_eligibility_policy` and
`identity_eligibility_reason` fields govern the actual monthly/feature joins.
Every original member remains in coverage and outcome ledgers. The central
worker recomputes collisions and admission instead of trusting claimed flags.
It handles Arrow/pandas missing scalars without manufacturing string IDs.

Policy details and the explicit provisional warning travel with persisted
inspection, monthly-return, paper, context and feature artifacts. Historical
identity, knowledge time and survivorship remain unverified; later strategy
results must be identified as provisional and rerun after identity repair.
This uncertainty is no longer a repeated request to stop all development.

## Comparable flag rates and practical admission

VERIFIED LOCAL metadata: the same three dates, source releases and original
member denominators were compared for SP500/SP400 and SP1500. A flag is an
unresolved producer review requirement, **not a measured incorrect-match rate**.
S&P1500 counts include SP600 and cannot replace the chosen V2 cohort.

| Signal | SP500+400 producer flag rate | SP1500 producer flag rate | Strict SP500+400 admission | Provisional SP500+400 admission |
|---|---:|---:|---:|---:|
| 2000-01-31 | 559/900 = 62.11% | 1,066/1,500 = 71.07% | 302/900 | 831/900 = 92.33% |
| 2005-01-31 | 691/899 = 76.86% | 1,255/1,497 = 83.83% | 192/899 | 860/899 = 95.66% |
| 2009-01-30 | 750/902 = 83.15% | 1,333/1,502 = 88.75% | 142/902 | 878/902 = 97.34% |

These anchors do not reproduce a lower SP1500 flag rate. Sadig's observation
may concern other dates, releases, flag definitions or selection denominators;
it is not disproved for those unexamined settings. Raw flag counts include
flagged collision members, unlike mutually exclusive `needs_review` counts.

Remaining original SP500/SP400 blockers are respectively 32/28/16 collision
members and 37/11/8 unlinked members. The admitted counts are metadata route
coverage, not verified prices, complete feature history, correct matches or a
survivorship-bias-free universe. Actual mismatch rates remain unknown.

The comparison projects only exact-date membership, bridge and link metadata:
three files/footers, 13,666,068 candidate rows, 39,079 returned metadata rows and
44,970,470 projected compressed bytes, within a 20-million-row/100,000-output/
128-MiB cumulative ceiling. No prices, provider queries or cloud calls. Independent
SP500/SP400 route reconstruction agrees with the existing worker at all anchors.
Exact source/request/budget evidence is saved privately in
`outputs/identity_audits/provisional-v1/comparable_flag_rates.json`.

## Continue through the master

The master displays original identity evidence and a separate
`research_identity_df` admission ledger. Inspection, monthly price selection,
paper raw momentum, technical joins and monthly/DAILY context admission consume
the same explicit policy. Admission metadata never becomes a predictor.

The two-ID engineering sample chooses one unflagged control and one flagged
eligible link, when available, **before any price observation**. It is a
deliberately balanced integration smoke, not the future research selection
policy. Failed history does not trigger replacement by another stock.

VERIFIED LOCAL bounded check: January–February 2000 retained all 1,799 original
stock/signal rows. Provisional admission covers 830/899 on 1999-12-31 and
831/900 on 2000-01-31, including 525/529 flagged routes respectively. Both
sample IDs have all 298 technical values at both signals (1,192 finite values)
and four accepted provider-adjusted endpoint ratios. The flagged link stays
`needs_review` with its original flag while being explicitly provisionally
eligible. Daily outcome quotes remain a separate 82-row table; feature history
ends at 2000-01-31. This numerical check does not verify that the link is correct.

Neither a two-stock sample nor an arbitrary requested subset may redefine
cross-sectional statistics. The later [active normalization protocol](NORMALIZATION_PROTOCOL.md)
requires exhaustive requests over admitted original members and uses one common
five-momentum population at each signal. Its medians/schema fit training rows
only; full context keeps its explicit complete-cohort/daily-membership rules.
The prior proposed JSON and strict anchor bounds are historical evidence.
The user moved current initialization to 2010 on October 15, preserving rolling
ten-year training and monthly prediction. The measured
[2010 coverage](INITIALIZATION_2010.md) does not certify these admitted IDs or
remove missing-history selection bias. No new capitalization requirement.

This implementation runs small local reads only. It does not make Run All a
full-history feature warehouse, fit, backtest or cloud execution. Full builds
retain their measured execution plan and applicable cloud workflow. Protected
2024+ financial values remain outside development requests.

Verification: all **330 tests** passed, including explicit provisional-admission,
unknown metadata, Arrow/pandas null, collision, timing, predictor-separation and
legacy configuration checks. The installed master completed all **16 code cells**
in a fresh kernel. Its original notebook remained output-free; private execution
and separate daily/monthly/feature/admission evidence were retained. The full
history panel and model execution remain outside this bounded verification.
