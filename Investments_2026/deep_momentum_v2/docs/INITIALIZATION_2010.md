# Analysis initialization: 2010

USER DIRECTION, 2026-10-15: move initialization from 2000 to 2010 and measure
the resulting usable history. Retain rolling ten-year training / one-year
outcome testing, annual refits and monthly predictions. Nominal test years are
now 2020–2023 (four folds). Training for 2020 uses 2010–2019 outcomes; the final
fold uses 2013–2022 outcomes for 2023. No model has been fitted or backtested.

The cutoff in 2023 preserves the existing 2024+ holdout. Changing initialization
does not truncate raw files or remove earlier warmup. January 2010 outcomes
use December 31, 2009 signals. Exact paper endpoints start December 31, 2008;
daily technical windows retain their independent 503-session warmup.

## Verified local endpoint coverage

The explicit audit reconstructs all historical SP500/SP400 original cohorts
before consulting prices. It uses the accepted provisional identity policy,
exact independent XNYS month ends and the shared common-five normalizer. Every
intermediate monthly endpoint is required; no stale-price fill or compression
of missing months is permitted. Outcomes are a separate input and do not select
the signal population. No capitalization, current-survivor filter or model
scores enter this check.

| Outcome period | Complete paper inputs | Original stock-months | Coverage |
|---|---:|---:|---:|
| 2010 | 9,226 | 10,824 | 85.24% |
| Initial training, 2010–2019 | 98,626 | 108,652 | 90.77% |
| Nominal tests, 2020–2023 | 42,151 | 43,655 | 96.55% |
| Full development, 2010–2023 | 140,777 | 152,307 | 92.43% |

For comparison, the separately reproduced 2000 baseline was 4,840/10,799
stock-months (44.82%). These percentages count original constituent-month rows,
not distinct securities or the fraction of calendar dates present.

The remaining 2010 exclusions are 1,052 missing exact signal quotes (9.72%),
312 incomplete momentum lookbacks (2.88%), 226 identity-blocked rows (2.09%)
and eight invalid signal quotes (0.074%). Moving the start improves availability;
it does not establish correct historical identities or remove missing-data
selection bias. Absent quotes are absent under the currently admitted native
IDs; a wrong/later mapping and limited provider coverage remain separate
possible causes. The qualified subset is not asserted to be the complete index.

Across the full new window, 292 common-five-qualified rows lack paired future
endpoints. They remain signal-eligible in the ledger. Missing labels must not
be imputed, and later economic accounting must address absent valuations and
terminal events rather than silently dropping those predictions.

## Execution and evidence

This is an explicit monthly-endpoint audit, separate from the two-ID notebook
sample. The metadata budget is 20 million candidate rows, 250,000 returned
rows and 128 MiB of projected chunks. The price budget is 30 million candidate
rows, one million returned rows, 128 files and 256 MiB of projected chunks.
Actual reads returned 201,332 metadata rows and 208,920 exact endpoint quotes;
the price projection considered approximately 180.9 MB across 104 data files.
No financial 2024+ rows reach Python analysis. Mixed Parquet chunks may decode
excluded dates transiently inside Arrow, as documented in DATA_FOUNDATIONS.

Private versioned evidence lives under `outputs/initialization_coverage/`.
The master checks current source/release/object-size pins, code, runtime,
calendar, request, artifact hashes and ledger denominators before displaying
the selected audit. Run All does not repeat the endpoint scan. Earlier audit
versions are retained with their original code pins. Full raw-object hashes
are not reread on each notebook run.

This check establishes complete paper-input coverage and paired endpoint
availability. It does not establish all715 predictor coverage, complete daily
paths, knowledge-time PIT, economic returns or backtest readiness. Daily data
remain separate for later backtesting. The active
[normalization/missingness rules](NORMALIZATION_PROTOCOL.md) forbid using this
full-period diagnostic to fit medians or select predictors for earlier folds.

The fresh daily notebook run exposed provider quotes on independent exchange
holidays, including January 19, 2009. The adapters now partition quotes against
the independent calendar inside Arrow before numerical workers run, reporting
each excluded ID/date and the source/analysis row counts. The monthly raw daily
view remains unchanged; monthly arithmetic uses the exact-session view. Source
files are preserved and the core date/target validators remain strict.

## Sadig's recent work

The [October 15 organizational review](SADIG_UPDATES_20261015.md) pins the latest
merged research state. It adds catalogue delivery progress, with nine reported
accepted source years, but no changed identity resolver or demonstrated repair
of our historical-price gap. Complete original-key catalogue counts preserve
missing observations and must not be compared to our finite paper-input ratio.
Their different universe and training/validation schedule do not replace ours.
