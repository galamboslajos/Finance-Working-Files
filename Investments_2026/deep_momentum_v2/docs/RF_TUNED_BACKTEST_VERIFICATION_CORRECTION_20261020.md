# Verification-only Parquet encoding correction — 20 October 2026

The first independent verifier stopped before its numeric-read reservation.
It required byte-identical selection files between preflight and replay.
The replay had read and saved those tables through pandas; the `side` text
column changed from Arrow `string` to `large_string`, changing four Parquet
hashes without changing any table value, row order, null or pandas dtype.
Both baskets' original ledgers and selected tables have that physical change;
their monthly and rebalance files remain byte-identical.

The bounded diagnostic compared all eight table pairs exactly. Its receipt,
the first verifier and the failed verification log/resource receipt are
preserved. The original replay, forecasts, source prices, selected stocks,
weights, cash accounting, stops, costs and root identities are unchanged.
There is no additional model fitting, inference or backtest replay.

The additive V2 verifier authenticates both raw artifact pins, permits only
that specific `side: string -> large_string` storage-width change, and checks
exact complete frame equality after its read reservation and date guards.
It then performs the same independent selection/cash/units/cost/action/stop/
metric checks. It does not coerce, sort, fill or tolerate differing values.
A separate correction authority binds the unchanged old root and old verifier,
the reviewed new verifier, the diagnostic and the preserved failure evidence.
The old sealed code files are not edited to make verification pass.

The diagnostic's conservative 2,973,088 compressed bytes are charged to the
same cumulative 512 MiB ledger. The first watchdog consumed 2.098476417 seconds;
charge three seconds, plus a conservative sixty-second diagnostic allowance,
against the original thirty-minute verification ceiling. The new supervisor
and verifier therefore allow at most 1,737 seconds. Keep the same 4 GiB RSS
stop, 3 GiB warning, normal-pressure/7 GiB available-memory gate, one numeric
thread and cumulative 256 MiB output limit. Preserve this verification attempt
if it fails; no retry or resource reset is implicit.

This is a verifier serialization correction, not an economic data repair or a
change in the research protocol. All sixteen scenarios still require an actual
independent PASS before verified performance is reported. Existing identity,
knowledge-time, survivorship, missing-price and fill assumptions remain.
