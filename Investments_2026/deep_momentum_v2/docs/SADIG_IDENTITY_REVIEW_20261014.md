# Latest organizational identity-work review

VERIFIED GITHUB, 2026-10-14. Repository: `GalambosC/deep-momentum`, research
branch `codex/deep-momentum-v2-phase1`, pinned head
`f8f705a08ef06443c0aed3ff68216f713a35eef7`. The head is authored by `sadikhh`
at 2026-10-14 15:18:04 UTC and was rechecked at the end of the review.
This is a read-only review, not a rerun of Sadig's experiments or adoption of
his policy decisions. No provider outreach or dataset/model/cloud execution.

## Result and latest activity

No completed effective-dated Siblis-to-Intrinio mapping repair was found in
the checked branch, other branch differences, reports or activity. The latest
research documents explicitly retain **IDENTITY/PIT UNVERIFIED**. Existing
diagnostics are useful, but do not supply the missing qualified crosswalk.
This does not prove that no newer artifact exists outside the repository.

The [current work order](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/CURRENT.md)
and [2021 source acceptance](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/docs/v2/STRATEGY_REDESIGN_CATALOGUE_2021_ACCEPTANCE_20261014.md)
record three accepted source years (2020, 2021, 2023) and three running source
workers (2016, 2019, 2022). These are repository-reported checkpoints, not a live
cloud verification by V2. Accepted source delivery is separate from stock
identity correctness; both documents retain the identity/PIT warning.

Eight remote branches were inventoried. Five substantive research branch heads
are ancestors of the pinned phase1 head. The one divergent branch,
`codex/monthly-source-io`, has one unique commit changing four source-I/O scripts,
with no separate mapping changes. Master is older research plus a README-only
September 30 merge. Default-branch code search alone would miss current work.
The sole PR/issue is that README change; the releases collection is empty and
the sole Actions run is an old dependency-graph job. No mapping delivery was
found in these collections. Tags were not checked because the connector
rejected that collection route; no claim of tag absence is made.

The latest 205 branch commit records were examined through September 15. Recent
commit-title absence is only a discovery aid; the conclusion above relies on
actual source/docs/report review.

## Already documented identity findings

| Evidence | Repository-reported result | Use in our V2 |
|---|---|---|
| [Repair diagnostic](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/docs/v2/PHASE1_IDENTITY_REPAIR_PLAN.md#L109-L113) | 24,616 identifier-history rows; zero complete intervals and zero additional strong candidates | Same published histories cannot supply dated proof |
| [Local construction feasibility](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/docs/v2/PHASE1_LOCAL_IDENTITY_CONSTRUCTION_FEASIBILITY_PLAN.md#L138-L154) | Zero added strict PIT identities; optimistic 705-ID union explicitly not approved | Diagnostic upper bound only, not our population |
| [R14 inspection](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/docs/v2/PHASE1_R14_GCS_INSPECTION_PLAN.md#L96-L97) | Zero direct repairs of 433 unresolved IDs; 105 recovered aliases remain unresolved | Discovery leads; missing per-attempt lookup provenance persists |
| [September G1 assessment](https://github.com/GalambosC/deep-momentum/commit/d48e62b8c32edfe81757aaa563101833c627e000) | Zero fully dated identifier-history rows and zero newly approved identities | Later confirmation that same-input reconstruction did not close the gap |
| [Historical acquisition config](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/configs/v2/phase1_intrinio_historical_identity_acquisition.json) | `PLANNED_NOT_AUTHORIZED`; provider request `DRAFT_NOT_SENT` | Acquisition design, not a delivered historical master |
| [Exploratory exception](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/docs/v2/EXPLORATORY_IDENTITY_EXCEPTION.md) | Exploration permitted with unverified identities and rerun required | Their accepted limitation, not our gate change |

The repair and R14 `docs/v2/reports/.../artifact.json` identity summaries agree
with the documented results. Their ignored output ledgers are not delivered
by GitHub, so these counts are source-reported findings, not new numerical
measurements by our project. The 171 historical "approvals" refer to Sadig's
older discovery policy, not historical certification. His S&P1500 lifetime
populations differ from our SP500/SP400 dated anchors; neither 171 nor 705
may replace our cohort counts or become our eligibility list.

## Code and terminology

Git blob SHA-1 comparison confirms the three core modules are byte-identical
to our `a360112d8e56af0eb1df19f45a7bf44400706292` reference snapshot:

- `identity_adjudication.py`: `f3ee44d7cab567dabb5cdb158fba587d62b3806a`.
- `effective_identity_corroboration.py`: `327484c352499468a522da4ba4116c6ed6936d95`.
- `local_identity_construction_feasibility.py`: `170ac090955b7ac2e397d6667004e21068b10ce5`.

The [adjudication code](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/src/deep_momentum/identity_adjudication.py#L216-L285)
imports the producer flag and combines it with consumer rules. It does not
define the upstream producer's reason formula. Its ISIN comparison uses the
published alias arrays; acquisition/inventory success does not verify original
raw identity lookup bytes. Our independent alias-origin review still applies:
repeated source aliases cannot become independent provider confirmation.

Newer files titled runtime/execution "identity certification" concern source
commits, manifests, attempts and VM ownership. For example,
[runtime identity delivery](https://github.com/GalambosC/deep-momentum/blob/f8f705a08ef06443c0aed3ff68216f713a35eef7/docs/v2/PHASE2F_RUNTIME_IDENTITY_DELIVERY.md)
does not qualify historical stock identifiers. Distinguish these meanings.

## Consequence and narrowed request

Keep our full constituent ledger, unresolved/collision states and non-promoting
audit. No feature population, normalization policy, research chronology or model
gate changed. Reuse diagnostic taxonomy, documented failed approaches and source
pins as references; the checked material does not deliver a certified crosswalk.

The [updated draft request](IDENTITY_EVIDENCE_REQUEST.md) asks only whether newer
qualified artifacts or original capture/producer evidence exist outside the
checked branch, and for their exact immutable locations. It acknowledges the
already documented audits and provisional exception rather than asking Sadig
to repeat them. No message has been sent.
