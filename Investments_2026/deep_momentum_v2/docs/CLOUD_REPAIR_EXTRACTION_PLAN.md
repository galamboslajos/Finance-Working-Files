# Bounded 2018–2019 quote repair extraction proposal

Prepared 2026-10-15. **PROPOSED; no cloud launch, registration, upload or human
campaign authority has been created by this work.** This document is not a
sealed native runner packet. The final exact request, source-owner commit,
registered namespace, startup hash and native runtime availability remain
required before an executable approval request.

## Decision and narrow scope

Obtain missing execution evidence for the already selected 2018–2019 V2
holdings. Preserve the four frozen prediction arms, monthly selection/ranking,
10 long and 10 short slots, 50/50 capital, class 1/10 and 2/9 separation,
unstopped and 10%-per-asset stop alternatives, costs, original stock identities
and original input files. No new model fit, selection mask, target or
2020–2023 financial-value projection belongs to this extraction.

The full selected-holding audit found additional bad/missing bars behind the
first three visible OHLC errors. Prepare one request covering those exact
2018–2019 key failures and adjacent same-security controls, plus any necessary
2019 Ensco Rowan successor path and Dean Foods OTC path. Do not limit the
request to OAS/MNK/MPWR if that merely exposes the next known failure.

The extraction is source evidence. Admission to a versioned override ledger
is a later local validation step under the user's existing repair direction.
Missing, duplicate, invalid or conflicting records must remain explicit.
Do not select a ticker/series because its prices happen to produce a complete
path or attractive return.

## Exact source and bounded projection

Use only this already owned immutable object:

```text
gs://quant-research-trusty-agility-492109-v0/dataset-platform/raw/sharadar/bundle/v1/run_id=sharadar-bundle-full-20261008-r1/archives/stocks.csv.zip
generation: 1791455483855208
bytes: 990440028
SHA256: e76825c551dc179307a8264306f101a979d3995b489a045a044e017465070e6b
ZIP member: stocks.csv
member bytes: 3170006852
member CRC32: 4214241284
previously measured member records: 45412640
```

Read the compressed object once on one disposable worker; do not download the
990 MB archive or expanded 3.17 GB CSV to the laptop. Stream the single member
and select exact source-label/date keys before numeric conversion. Decode
financial values only for dates from 2018-01-01 through 2019-12-31 that appear
in the frozen request. Skip all other dates before financial conversion,
including all 2020–2023 and protected 2024+ records.

Freeze and hash a distinct request with **at most 256 exact source keys and
16 historical source labels**. This is a ceiling, not permission to fill the
request arbitrarily. Candidate mapping metadata must establish the exact
permaticker/security/share-class relationship separately; a ticker-only lead
does not establish continuity. The request should include original IDs,
reason, action relationship, target date, source label/permaticker and metadata
evidence. Publish actual requested/emitted/missing/duplicate/invalid counts.
If final inventory exceeds a bound, revise the plan before launch.

Required source columns are the captured ten-column contract:
`ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated`.
Retain original source values and the captured adjustment definitions.
Sharadar OHLC/volume are split adjusted; the existing source investigation
uses `closeunadj / close` as candidate raw OHLC multiplier and its reciprocal
for volume. Independently compare the same-security neighbouring controls
against pinned Intrinio raw bars and examine event/split witnesses before
admission. Do not substitute dividend-adjusted `closeadj` for an executable
open or add payouts twice. Disagreement is evidence, not an automatic repair.

## Reuse existing implementation, with fresh immutable identity

The organizational repository already contains:

- `src/deep_momentum/monthly_price_gap_probe.py`: exact-key, date-guarded
  streaming CSV projection, duplicate/invalid/missing accounting and ZIP
  integrity checks.
- `scripts/prepare_sharadar_price_gap_inputs.py`: small provenance-pinned real
  controls and parser projection export.
- `scripts/sharadar_price_gap_cloud.py`: bounded guest with existing governed
  GCS transport and separate resource, terminal, checkpoint and log evidence.
- `scripts/build_sharadar_price_gap_startup.py` and
  `scripts/check_sharadar_price_gap_launch.py`: startup construction and the
  actual five package checks.

These are source references; their old request hashes, run IDs, source commit,
package seals and authorities **must not be reused or edited in place**.
Adapt the narrow request through a new source commit and fresh immutable run
identity. Suggested ID: `dmv2-v2-daily-quote-repairs-2018-2019-20261015-r1`.
The owning repository must be registered accurately: this consumer workspace
is `galamboslajos/Finance-Working-Files`, not automatically the organizational
repository. An operator may instead prepare a separately approved source-owner
extraction in the organizational repo and deliver only the small result.

## Proposed resource, cost and cleanup bounds

| Item | Proposed hard envelope |
| --- | --- |
| Project | `trusty-agility-492109-v0` |
| Compute | One `e2-standard-2`, 2 vCPU / 8 GiB RAM, no GPU, no parallel workers |
| Region/zone | `us-central1-c`, conditional on source-location/network-cost preflight |
| Guest containment | 4 GiB memory, 200% CPU quota, I/O accounting |
| Disk | 40 GiB standard boot disk, auto-delete, no durable data disk |
| Processing | 25 minutes, plus 5 minutes publication reserve |
| Safety shutdown | 33-minute owned service cap; 60-minute provider DELETE fallback |
| Attempts | One, zero retries |
| Request | At most 256 exact keys / 16 labels, all financial dates 2018–2019 |
| Local payload | At most 1 MiB packaged code/request/real controls; 64 MiB assembly reserve |
| Cloud durable output | At most 4 MiB aggregate, at most 24 create-only objects |
| Human cost ceiling | USD 1.00 total / active / per run |
| Planning estimate | At most USD 0.25 for the full one-hour timeout envelope; billing unmeasured |

The prior same-archive extraction measured a complete scan in 264.813 seconds
on this tier, with approximately 1 GiB owned-cgroup peak. This is relevant
source workload evidence, not a guarantee of completion or whole-VM usage.
The numerical selection here is much smaller; preserve the original archive
byte bound rather than inventing a guessed row guard.

Current official Iowa on-demand [E2 pricing](https://cloud.google.com/products/compute/pricing/general-purpose?hl=en)
lists `e2-standard-2` at USD 0.06701142/hour. The [standard disk rate](https://cloud.google.com/compute/disks-image-pricing)
is USD 0.000054795/GiB-hour, or approximately USD 0.0022 for 40 GiB-hour.
The USD 0.25 planning envelope adds a conservative allowance for IP, GCS
operations, small output transfer and staging. Preflight must confirm storage
class/location and any retrieval/inter-region cost before authority. Do not
treat the USD 1 authorization limit as predicted billing or existing credit.

Create-only output must be below a **new exact hub-registered run prefix**;
the existing `strategy-research/active/deep_momentum/` parent alone grants no
write authority. Proposed parent/owner registration must be resolved explicitly
before any upload. Preserve all old registration rows and historical artifacts.
Publish immutable source rows, compact diagnostics, receipt, actual resource
measurements, checkpoint, terminal/heartbeat and run log. Retain durable evidence
after completion/failure. Delete only the exact VM and auto-delete disk through
the sealed normal cleanup gate; then verify absence and settle the campaign.
No raw-source overwrite, durable deletion, IAM change, purchase or other worker.

## Native workflow and authority that must still exist

The organizational [AGENTS.md Cloud Cost Gate](https://github.com/GalambosC/deep-momentum/blob/master/AGENTS.md)
states: “the existing remote-run override does not by itself authorize spending”.
Its Boundary section requires a fresh run root, exact hub registration, sealed
plan and sufficient current campaign or action-specific authority. This is an
explicit repository requirement, not an inference from hypothetical cloud risk.

Required sequence, following the installed native `quant-gcp-ops` / `gcp-run`
interface (do not invent or substitute CLI command syntax):

1. Finish the exact key manifest, immutable clean source commit, real control
   receipt, explicit unit/event rules and source-owner registration proposal.
2. Verify actual installed runner/version and read the current authoritative
   hub cloud policy. Native doctor/inventory, source metadata, fresh exact
   namespace absence and source/account/project preflight are read-only checks.
3. Register the exact owner/run/root and startup child through the authorized
   hub workflow. Build the small startup and independently verify its contents
   and five actual package checks without cloud execution.
4. Obtain any required exact create-only startup approval, then publish only
   that startup with generation precondition zero. Pin its observed generation,
   size, hashes and source commit in the native manifest; seal it.
5. Complete offline/live read-only preflight and actual package smokes. Obtain
   genuine action-specific campaign authority for the exact USD 1 / one-attempt
   scope if not already covered. Never fabricate human authority or reuse a
   consumed claim. A local test pass or startup approval is not compute approval.
6. Perform one native submit dry-run, then one governed submit only after the
   matching saved authority is verified. On ambiguous submission, inspect the
   claim/provider state rather than retrying.
7. Collect all declared final objects by exact observed generation and hashes;
   independently reconcile every request key. Complete normal finalization,
   owned cleanup, campaign settlement and completion attestation.
8. Only the small verified result is downloaded locally. Admit repairs via a
   separate immutable ledger, preserving original bars/flags, before the
   already authorized bounded local backtest rerun.

### Current capability and reuse limits

As checked here, the installed skill/tool catalog and local `~/.codex`/temporary
inventory do **not** contain `quant-gcp-ops` or a native `gcp-run` installation.
The org repository guest scripts are not that native authority layer. Its hub
policy is referenced on the source operator's Windows checkout; no policy file
is vendored in the org repo, and `GalambosC/quant-research-hub` returns 404.
This report has not accessed a personal GitHub hub as a workaround. Therefore
it cannot claim that native registration, sealing or submission is ready on
this machine. Runner and current policy availability are real dependencies,
not reasons to fall back to direct `gcloud compute instances create`.

The org's latest published CURRENT/GCS checkpoint records the earlier r2 probe
as CLEANED/SETTLED with consumed authority. Its 1,069 keys are a different
immutable request. The newer full-history check has a finalized startup and
SMOKE_PASSED state but was still at its final human AUTHORIZE prompt in that
checkpoint, with no saved compute authority or VM. Its intended output is
coverage metadata, not this missing execution-price slice. No authorized live
worker exposing our repair keys was established by this read-only inspection;
do not repurpose either run or message its owner automatically.

The smallest operational choices are enabling the governed runner/policy and
correct owner registration here, or asking the authorized source operator for
this exact small governed extraction. The latter requires an explicit message
instruction before sending; preparing a message is allowed. A USD 1 budget
approval alone does not install the runner or establish registration.

## Acceptance and stop conditions

Advance only when every original requested key is accounted for, all source
and output integrity checks pass, identity/unit/event ambiguities are explicit,
and the native run is ordinarily finalized and cleaned. Retain a missing quote
or unsupported halt valuation as unresolved rather than creating a flat return,
dropping a selected stock, swapping in a survivor or pretending an order filled.
No complete backtest performance is valid until each held liability/asset has
a justified daily valuation and executable exit or mandatory entitlement.

Stop on source pin/schema/member-integrity mismatch, request/date violation,
duplicate/conflicting quote, unit conflict, missing native authority, unsafe
namespace, resource/cost bound or publication failure. Preserve evidence;
no automatic retry or replacement campaign. The alternative is an honest
partial-development report with unchanged stocks and withheld complete P&L.
