# Corporate-action development rerun: 2018–2019

USER DIRECTION, 2026-10-15: resolve corporate actions and the three original
OHLC failures, rerun the same daily development strategies, retain the selected
stocks, and leave 2020–2023 untouched. This is an additive experiment under
[`action_backtest.json`](../configs/action_backtest.json). The original
[`daily_backtest.json`](../configs/daily_backtest.json), workers, saved source
artifacts and `outputs/daily_backtest/LATEST.json` remain unchanged.

## Selection and scope remain fixed

Reuse the **bytes** of each of the eight saved selected-stock artifacts and
eight rebalance schedules. Also copy their original-cohort ledgers unchanged.
No selection worker is called, no probability is reranked, and no estimator or
preprocessor is loaded or fitted. Verify the original and copied hashes before
reporting. There are the same 64 scenarios: four model arms × separate class
1/10 and 2/9 baskets × unstopped/10%-per-asset-stop variants × 0/5/10/25 basis
points per one-way executed market notional.

The original intended basket remains ten longs and ten shorts, each originally
allocated 5% of pre-cost NAV. An order for a security already retired when that
order reaches its scheduled execution open remains in the intended order
ledger as `unfilled_retired_security`. Leave its slot in cash; do not replace it,
reweight the remaining names, or give a newly selected order a merger payout
that belongs only to shares held before the event. This can reduce actual gross
exposure below 100% without changing signal-time selection or denominators.

RET remains withheld in all 24 signals under its unchanged complete
120-calendar-month gate. No earlier payoff extension, model fit, later test, or
2020–2023 financial analysis is included.

## Dated corporate-action accounting

The eight reviewed BIVV, ESRX, IDTI, GXP, RDC, ESL, BID and LLL events have
explicit legal terms and first application sessions. See the separate
[primary-source evidence record](CORPORATE_ACTION_EVIDENCE.md). Missing quotes
never create an event or establish a payout. For a pre-event held lot:

1. Convert adjusted proxy units into signed raw predecessor shares using the
   exact preceding session's `adj_close / close` ratio.
2. Record signed cash entitlements or short-side cash liabilities at face value.
   This is an economic receivable/liability under zero cash/funding rates;
   physical broker payment or delivery timing is not certified.
3. Convert the stock entitlement into successor adjusted units using that
   session's actual `adj_open / open` ratio. Keep the original source-lot identity
   and its successor exposure until a scheduled rebalance, actual stop, or the
   December 31, 2019 terminal close.

The source `factor` field is retained as evidence and is **not** assumed to be
that paired-price ratio. Do not add ordinary dividends, special distributions
already reflected by adjusted prices, or splits a second time. In particular,
RDC's reviewed exchange uses the post-consolidation successor share basis;
wrong-security aliases and obsolete ADRs cannot substitute for the actual
2019 successor.

Theoretical fractional successor shares remain fractional. No cash-in-lieu
amount is invented. Mandatory legal conversions do not create market turnover
or transaction costs. At a common actual open/close, aggregate same-provider
lot instructions into net market orders before charging cost; keep the
original lots separately visible. Intraday stop instructions are not netted
with a different threshold or a terminal-close fill merely because the date is
the same. Borrow remains the inherited gross-short-lot convention, assumed
available at zero charge in this diagnostic.

The stop retains the original monthly holding episode's economic threshold:
subtract the mandatory cash component and scale the remaining threshold into
successor units. A long stop made unreachable by an already-received cash
component does not invent a trade. Merger gaps can exceed a stop; a mandatory
cash payout is not magically repriced to the stop level. Old episode stops
still cancel at the scheduled monthly rebalance open. Same-session chained
legal actions require explicitly ordered terms and are rejected by this
protocol rather than ordered by an event-name sort.

When the legally established RDC successor lacks an admissible quote, the
explicit `unquoted_successor:CUSIP:G3166L100` asset identity cannot supply a
price. Affected held paths retain unresolved exposure and withhold continuous
full-period results. The retained predecessor lot is a trace of the unresolved
legal conversion and exposure, not a claim that retired RDC remains tradable
or that its old units supply a valid successor valuation. Unaffected scenarios can continue. Do not drop RDC, assign
zero, cash out at its stale last quote, or publish a model ranking conditioned
on which scenarios happen to have complete paths.

## Actual-source OHLC overlays

`reviewed_ohlc_inputs.json` pins a finite published Alpaca SIP raw daily
projection, source-generation declarations, a range receipt, the original
admitted source/native mapping and explicit review notes. This is an
**alternative-provider overlay**, not a claim that Intrinio corrected its
original files. The original three required keys are:

| Native ID | Session | Original issue |
| --- | --- | --- |
| `sec_g0wpMy` (OAS) | 2019-01-18 | Missing open/high/low |
| `sec_ge8Rby` (MNK) | 2019-08-12 | Close above recorded high |
| `sec_gVBv5g` (MPWR) | 2018-08-02 | Open below recorded low |

Before portfolio results, freeze raw-close agreement at **both USD 0.05 absolute
and 0.5% relative**, with maximum original raw/adjusted pair residual **0.0002**.
The close-derived adjusted-unit basis is transported onto actual replacement
raw OHLC. Every repair pins the exact rejected original bar and its evidence;
all original values and quality flags remain in separate overlay columns.
All three original required keys must pass before a run is created. A missing
source bar or conflicting close is never replaced by a made-up open/close.

Additional reviewed rejected bars can be repaired under the identical limits.
Nonmandatory close/unit conflicts remain rejected, with the source disagreement
recorded in the review ledger; affected held paths still fail closed. Evidence
hash, identity, date or structural-integrity failures stop preparation. Do not
relax tolerances based on portfolio performance. Missing entire quote keys use
a separate explicitly reviewed append-only admission path, not an OHLC overwrite.

A separately pinned optional `reviewed_pcg_inputs.json` admits only the existing
PCG native identity and its two original source aliases, using the identical
repair rules. It is a second independent published product, not a universe or
model-selection change. The strict append-only adapter is separately frozen to
four FRT/FIVE/WCG source/date keys. Their actual raw bars use a preceding accepted
adjusted/raw unit ratio confirmed by a following same-source overlap control;
nonzero dividends, nonunit splits or a discontinuous paired-price basis refuse
admission. No raw OHLC price is interpolated. Its accepted/refused cases remain
in `supplemental_bars.parquet`.

Local projection and receipt bytes are authenticated. The original publication
manifest/generation and declared source-table SHA are recorded separately from
full-file verification. Partial remote range reads do not verify a whole source
table SHA, retained raw-page lineage, or an independent historical identity.
The exporter discloses that mixed Parquet row groups may decode other values
inside Arrow before the date/ID filter. Only authorized 2018–2019 records are
returned or used; this disclosure is distinct from the local backtest's two
physically bounded annual caches.

## Coverage and outputs

Use new `action_loading.py`, `action_accounting.py`, the source-review workers,
and `action_backtest.py`. Missing unexplained selected entries or held valuations
still retain positions and expected marks, fail the continuous path closed and
never restart after a gap. Full-period return, volatility, Sharpe, drawdown,
turnover and cost summaries remain null for incomplete paths. Valid prefixes are
coverage diagnostics only. Complete proxy paths still do not certify economic
coverage, historic identity, borrowing, knowledge-time prices or investability.

Keep all 64 scenarios in the comparison. Report applied action counts, retired
unfilled orders, successful overlays, unresolved source/admission cases, actual
coverage and first failures alongside the metrics. A table of complete cases
must not become a model-selection leaderboard.

Saved output is private under `outputs/action_backtest/`. It includes:

- Byte-identical original cohort/selected/rebalance artifacts and their hashes.
- Open/close NAV, per-lot instructions, net market trades, holdings, stops,
  dated entitlements, intended order statuses and remaining failures.
- Daily-close-based metrics, monthly returns, source-review/repair ledgers,
  configuration/worker identities and unchanged input pins.

The local ceiling remains one million candidate rows, 128 MiB projected
compressed input, 2 GiB peak RSS, 900 seconds and 512 MiB output. No cloud
mutation or implicit notebook execution is introduced. From V2:

```sh
PYTHONPATH=src .venv/bin/python -B -m dmv2.action_backtest plan
PYTHONPATH=src .venv/bin/python -B -m dmv2.action_backtest run
PYTHONPATH=src .venv/bin/python -B -m dmv2.action_backtest report
```

An identical complete build is verified and reused; an incomplete identical
attempt requires review. `read_action_backtest` reads verified small reports
only. It never loads raw prices, runs accounting, or fits a model. The master
notebook calls this reader; Run All is not authority to launch the rerun.

## Engineering validation before the real rerun

Thirty focused action-accounting tests cover signed cash/stock conversions,
retired orders, original ten-slot preservation, economic stop carry, event
chains, fractional units, costs, borrowing and fail-closed exposure. Thirty-one
frozen-engine cases also replay against the additive engine. An independent
raw-share oracle reconciled 160 synthetic cases and 1,280 marks, including 72
stop cases and 112 two-action chains. These tests use no research data and are
not a claim that the actual 2018–2019 paths are complete.
