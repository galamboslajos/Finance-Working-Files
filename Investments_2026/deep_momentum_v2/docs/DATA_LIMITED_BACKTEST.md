# Data-limited development backtest — October 16, 2026

## Decision and interpretation

The user requested a practical solution now because the source owner is
unavailable, explicitly allowing a documented exclusion rather than an
indefinite wait for replacement data. The separately frozen contract is
`configs/data_limited_fallback.json`, protocol
`dmv2-data-limited-restricted-universe-2018-2019-v1`.

This creates a **data-limited, restricted-universe development experiment**.
It does not certify a repaired old-BBBY price route, point-in-time availability,
complete historical-universe coverage, delisting accounting or investability.
The data-quality restrictions were chosen in October 2026 after inspecting the
2018–2019 development problems. They are retrospective restrictions and can
introduce selection bias. Retaining membership rows makes that loss of coverage
visible; it does not eliminate the bias.

No new result is asserted by this document. Completion, measured populations,
resource use and performance must come from the new saved phase audits.

## Three fixed security decisions

| Security | Feature and target treatment | Trading treatment in this experiment |
| --- | --- | --- |
| Historical Bed Bath & Beyond, ISIN `US0758961009` | Quarantine both demonstrated wrong source routes. Preserve their historical membership and original flags, but leave unavailable features and outcomes missing. Do not use historical Overstock prices or invent a replacement native ID. | No prediction or order without a valid baseline feature row. |
| Dean Foods, native `sec_gjP4xM` | Retain valid historical features, observed outcomes and labels. Do not censor its losses or remove it from shared statistics. | Exclude from trading across all 2018–2019 signals, before choosing the ten names on each side. |
| Rowan, native `sec_zmYNnz` | Retain valid historical features, observed outcomes and labels. Do not censor its losses or remove it from shared statistics. | Exclude from trading across all 2018–2019 signals, before choosing the ten names on each side. |

The two old-BBBY source IDs are
`siblis_ticker_company_00326c7d9ad13903e5e3` and
`siblis_isin_c297a5f81a23c9f90503`. Their invalid route is `sec_ydbQnj`.
The reviewed [identity amendment](BBBY_IDENTITY_AMENDMENT.md) supplies the
evidence and keeps the replacement explicitly unresolved.

Dean Foods and Rowan restrictions address the missing suspension/OTC and
successor execution records in this dataset. They are **trading-only**
restrictions, independent of estimator, class, direction, stop variant, costs
and monthly scores. Their prediction rows and probabilities remain visible.
This is a fresh restricted-universe selection, not deletion of holdings or
losses from an existing portfolio. The original portfolios remain unchanged.

The new version does not manufacture the absent Dean Foods bars, a halt exit,
Rowan successor prices, merger consideration beyond verified entitlements,
or an assumed stop fill. Missing dataset observations alone do not establish
that the historical security could not trade.

## Rebuilding features and targets

Build historical S&P 500 plus S&P 400 cohorts from their original dated
membership before price joins. Retain every original source-ID/signal-date row,
review flag, daily membership record and original cohort denominator.

The original audit identified 120 old-BBBY monthly rows in the development
history: 96 training examples and 24 validation predictions. Those are original
dependency counts, not a claim that each newly built financial value changes.

Recompute the frozen feature formulas from the independently bounded annual
daily-history caches after applying the reviewed quote amendments and the
explicit wrong-route quarantine. In particular:

- Recompute Paper15's common-five population, shared normalization statistics
  and qualification. Its fifteen formulas remain unchanged.
- Recompute daily technical indicators from past daily observations, then take
  their stated signal-date snapshots. Missing prices are not forward-filled.
- Recompute cross-sectional, market and relative-context dependencies using the
  active 80% **original-cohort** coverage rule and exhaustive admitted-ID
  requests. Keep the 90% panel as its existing sensitivity reference.
- Recompute monthly observed returns, class ranks, exact-equal tie treatment
  and class payoffs from the new eligible observed data. Missing targets remain
  missing; an unavailable return is never zero or a bottom-class label.

The class universe is consequently conditional on observed valid outcomes.
Keep the original population and missingness diagnostics alongside class
counts rather than describing the labels as complete full-index outcomes.

The daily feature-history view ends at each signal's available history. The
separate outcome view contains the preceding signal anchors and the complete
following-month daily sessions; monthly returns use its exact endpoints.
Daily sources remain separate for execution and eventual backtesting. Neither
future outcomes nor audit flags enter the predictor matrix.

## Four classifier fits

The new model phase uses the unchanged shared classification adapter and exact
hyperparameters, package pins, seed and estimator settings from the frozen
`configs/classifier_pilot.json`, copied into the new outer contract. This outer
contract authorizes four fresh fits; it does not change or rerun the original
pilot in place.

| Model phase | Outcome years | Model updates |
| --- | --- | --- |
| Training | 2010–2017 | One fit for each of RF/Paper15, XGBoost/Paper15, RF/expanded and XGBoost/expanded |
| Development prediction and evaluation | 2018–2019 | Monthly predictions from those four fixed fitted models; no retraining within these two years |

Both feature sets and estimators share the same supervised training mask and
the same signal-time prediction population. Added-feature missingness must not
remove baseline-qualified prediction rows. Medians, missing indicators and
feature retention are fitted only on completed supervised training rows. Labels
are never imputed. Validation labels are supplied only to saved evaluation,
never to fitting, early stopping, calibration or parameter selection.

The model phase permits four sequential single-thread fits, at most 8 GiB peak
RSS and 90 minutes. It records an independent phase identity containing the
actual model-stage worker hashes, configuration, runtime and new annual input
pins. Attempt markers prevent an incomplete attempt from silently fitting again.

## Daily backtest contract

Run the same 24 development months, January 2018 through December 2019, with
the newly fitted models and the fixed trading restrictions. Recompute selection
from the new monthly probabilities; do not force the old holdings or predictions
onto corrected inputs.

There are **64 predefined scenarios**: four model arms × two separate class
baskets × two stop variants × four trading-cost assumptions.

- Class pairs 1/10 and 2/9 remain separate portfolios.
- Select ten long and ten short names within the corresponding predicted mode
  classes after the fixed trading restrictions, using the original ranking and
  tie rules. Do not select names by later price completeness.
- Keep 50% long and 50% short, equally weighted within each side, with the
  established next-session-open execution rule.
- Keep unstopped and 10% per-asset stop alternatives, and 0/5/10/25 bp one-way
  traded-notional costs. Stops remain subject to the stated daily-bar and gap
  fill assumptions; they do not guarantee a portfolio loss limit.
- Apply the same reviewed corporate entitlements and actual quote amendments,
  including the separately reviewed ACC/K/SO correction.
- RET retains its 120-month gate and is withheld in this development period.

If a newly selected or held security encounters another unexplained valuation
gap, retain the exposure and withhold its continuous full-period metrics.
Do not add a new exclusion after seeing its performance, silently drop its
losses, invent an exit, substitute a different stock, or relax a price check to
make the result complete. Report incomplete scenarios alongside complete ones.

## Storage, preservation and chronology

The additive CLI worker is `src/dmv2/data_limited_fallback.py`. New private
outputs belong under
`outputs/data_limited_backtest/2010-2019-<feature-identity-prefix>/`.
Its `years/YYYY/` directories hold newly computed features, coverage, original
cohort metadata and verified target states. `models/` contains the new shared
preprocessing states, ledgers, four estimators, probabilities and reports;
`backtest/` holds the separate execution phase.

Within each year, `daily_prices_raw.parquet` is the disjoint outcome-year daily
core. `daily_history_raw.parquet` contains that outcome view plus the preceding
December signal anchor; it is **not** a saved 503-session feature-warmup table.
The full past daily history used to rebuild indicators remains in the original
hash-pinned annual cache and can be reproduced with the same reviewed amendments.
Future consumers requiring the full warmup must use that documented source.

Each phase records its own input/code identity. Code completed after feature
processing begins is not claimed to have been part of that earlier identity.
Use the `plan`, `build`, `models` and `backtest` CLI phases explicitly. The
last two require the new run directory. Annual source planning uses metadata
and Parquet footers without decoding annual source prices. However, compiling
or loading the reviewed quote-amendment bundle also materializes its approved
2018–2019 price corrections. The CLI plan reports
`annual_source_financial_values_read=False` and
`reviewed_quote_amendment_values_loaded=True` to distinguish these operations;
the entire CLI invocation must not be described as purely metadata-only. A plan
does not fit a model. Passive notebook readers must avoid quote compilation.

An initial `e3318c7fb505486f` build attempt was stopped during metadata planning
after approximately 86 seconds because of memory pressure, before any financial
result was produced. It remains preserved as an incomplete attempt. Sequential
metadata planning produced the separately versioned active feature identity
`sha256:50aa6e09f11b687c36cd3cf52ed8eb51fdde591fb2391cd58162d02edbb29b10`,
under `outputs/data_limited_backtest/2010-2019-50aa6e09f11b687c/`.
The financial formulas, population rules and model settings did not change.

Master notebook section 21 reads the explicitly pinned active run with
`fallback_reporting.read_data_limited_run`. Its default checks bounded saved
identity/audit/JSON summaries and the small backtest comparison when available.
It does not certify full source bytes. Set `verify_sources=True` explicitly for
the more expensive byte-integrity checks; those checks still do not deserialize
estimators or evaluate future financial periods. Final feature/model/backtest
audit SHAs can be supplied in `expected_receipts` after each phase completes.

Keep original sources, feature/target panels, model files, selections, backtest
outputs and their existing `LATEST.json` pointers unchanged. New raw evidence,
financial artifacts and estimators remain private and outside Git. The master
notebook may verify saved small reports; Run All must not start this workflow.

No 2020–2023 financial model or strategy evaluation is part of this run, and
2024+ financial values remain protected. The nominal later plan stays rolling
ten completed training years, one outcome year at a time, with annual refitting
and monthly predictions. That extension needs a separately frozen execution
contract after this development experiment is reviewed.

Compare the original and new experiments as a **joint change in repaired
inputs and declared trading restrictions**. Any performance difference cannot
be attributed solely to the estimator or to BBBY's removal.
