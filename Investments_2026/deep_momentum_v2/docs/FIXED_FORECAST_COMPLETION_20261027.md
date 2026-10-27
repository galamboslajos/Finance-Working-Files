# One observed boundary point completes the close-clock controls

The original fixed-forecast run completed independent checks of all 28 paths:
22 complete and six withheld. All eight original-cache close controls reproduce
the monthly reference, and all twelve next-open/cost/stop paths are complete.
The six unavailable rows are the two execution-quote close controls for
XGBoost5, RF10 and XGBoost10.

Every failure is one newly selected PBF boundary: 30 December 2022,
source key siblis_ticker_company_9f3c03dbe95bb1e9c67d, native sec_z9QxBz.
The required close is absent from both the 2022 and 2023 execution tables.
It is present in the independently reconstructed full-history OC cache.
The source cohort's 2023 signal metadata identifies the membership alias PBF.
This is a difference in saved request coverage, not proof that the historical
market quote is absent. The original next-open paths already have their
required January 2023 execution prices.

## Necessary additive completion within the approved reconciliation

Use configs/fixed_forecast_completion_20261027.json. Preserve the completed
original run and its six withheld controls. A separate plan authenticates the
original input/accounting proofs and seals the exact existing cache key before
reading its numerical fields. Append only its observed raw/adjusted open and
close to a separate execution adapter. No value is interpolated, forward-filled,
invented or taken from a newly downloaded source.

The adapter represents every old row's exact daily/completion/quality/bounds
admission as a strict boolean price_accepted. The added point retains the
independently checked OC cache acceptance. It supplies no new high/low values
and only the six no-stop close-clock controls consume it. Reject an existing
execution-key overwrite, invalid/negative/nonfinite values, ambiguous keys,
nonboolean quality flags, new high/low values or any protected date.

Replay just those six cases with the same fixed baskets, old event definitions,
unit-factor choices, cash/share worker and terminal marks. Independently check
the observed point and every old execution row, reproduce the six daily
ledgers with the original separate oracle, and retain the other 22 comparison
rows exactly. A composite view may show all 28 rows only after the completion
proof passes. It always exposes the original six withheld rows as reference.

This extension supersedes the original recipe's no-additional-point boundary
only for this necessary observed PBF entry close. It changes no original file,
next-open result, stop path, label, RET estimate, model or forecast. The
authorization is the user's 27 October request to complete the fixed-forecast
reconciliation; no new portfolio strategy or model trial is introduced.
Original source identity, adjustment/knowledge-time and investability limits
persist. No raw fallback, cloud mutation or 2024+ financial read occurs.

The completion uses a separate zero-fit two-phase controller. Per-pass derived
input reservation is 64 MiB, RSS 4 GiB, outputs 512 MiB, independent derived
verification 1 GiB, 20 minutes per phase and 40 active minutes total. These are
ceilings, not an expected model computation. Numerical input reads use the
existing verified small OC/execution projections. All raw prices and ledgers
remain private.
