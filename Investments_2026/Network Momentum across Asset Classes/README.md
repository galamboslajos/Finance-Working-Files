# Network Momentum across Asset Classes

The paper's equations, training procedure, equity adaptation and replication
tests are recorded in [Network Momentum methodology and S&P 500 replication](docs/NETWORK_MOMENTUM_REPLICATION.md).
The corresponding [parameter record](configs/network_momentum_paper.json)
distinguishes author settings from proposed implementation conventions.
`audit_warmup_coverage.py` measures the uninterrupted price histories needed
by the long graph windows; its results are saved in `data/warmup_coverage/`.
This metadata diagnostic is a necessary-history screen, not a feature build.
The [first-window audit](docs/FIRST_WINDOW_DATA_AUDIT.md) checks 2011-2015
membership, 2009-2010 initialization and the first 2016 graph history, with
per-security repair queues and exact missing-session lists.
The [reviewed close-price layer](docs/FEATURE_CLOSE_REVIEW.md) admits the 107
scoped existing closes for feature calculations, restoring 59 complete
histories. Use `review_minor_closes.load_feature_prices` for this first-window
scope; eight absent observations still require source recovery.

The [first feature and network pilot](docs/FIRST_NETWORK_PILOT.md) is complete:
all eight features qualify **429 of 505 PIT members** for the five-network
ensemble after the close on **2016-01-04**. Run `build_momentum_features.py`,
`learn_momentum_network.py` and `validate_momentum_pilot.py` with fresh output
directories to reproduce it. The graphs use untuned `alpha=beta=1`; no
regression or performance backtest has been fitted. All 42 tests pass.

`prepare_sp500_daily.py` assembles a daily **S&P 500 only** research dataset from
the same pinned local Siblis membership, unified identity bridge and Intrinio
price releases used by Deep Momentum V2. It defaults to their full common
history, with earlier price history retained for feature warmup. Sources are
read only; there is no download, model fitting or backtest.

## Build

```bash
python -m pip install -r requirements.txt
python prepare_sp500_daily.py
```

The existing Deep Momentum V2 environment also has these dependencies:

```bash
/Users/lajosgalambos/Desktop/Finance-Working-Files/Investments_2026/deep_momentum_v2/.venv/bin/python prepare_sp500_daily.py
```

Use `--source-root /path/to/deep_momentum_v2` if the source project moves.
`--start` and `--end` accept inclusive ISO dates. `--preflight` verifies the
source pins without assembling the dataset. Existing output directories are
never overwritten; use `--output data/sp500_daily_v2` for a separate build.

## Outputs in `data/sp500_daily/`

| File/directory | Contents |
| --- | --- |
| `panel/year=YYYY/` | One row per historical source security and eligible member date, left joined to exact-date OHLCV, adjusted OHLCV, action fields, returns and explicit identity/quote statuses. Missing quotes remain rows. |
| `price_history/year=YYYY/` | Source price history for mapped ever-members, including dates before/after membership and earlier warmup. It includes rejected observations with status flags; it is **not** a dated investment universe. |
| `identity_mapping.parquet` | Source/unified/native IDs, original review flags, mapping eligibility and known wrong-issuer conflicts. |
| `membership_exclusions/year=YYYY/` | Supplied membership rows rejected by exact-date eligibility or calendar checks, with reasons. Created only when needed. |
| `coverage_daily.csv`, `coverage_annual.csv` | Original cohort counts, admitted identities, usable quotes/returns and missing source-membership sessions. |
| `manifest.json` | Successful-build marker, exact source/output hashes, dates, preparation policy, calendar vintage, counts and limitations. |

## Read a slice

```python
from prepare_sp500_daily import load_panel

daily = load_panel(
    start="2010-01-01", end="2010-12-31",
    columns=["date", "source_security_id", "membership_ticker", "adj_close",
             "provider_adjusted_return_1d", "identity_status", "quote_status",
             "analysis_eligible"],
)
# Keep the original rows when measuring coverage; select admissible observations
# explicitly for an individual calculation.
quotes = daily.loc[daily.analysis_eligible]
```

Use `source_security_id` and `date` as panel keys. Tickers are display attributes.
Do not collapse separate historical companies sharing a ticker. For wide return
matrices, pivot `provider_adjusted_return_1d` by these keys, retaining NaNs; never
fill unavailable returns with zero or carry constituents across missing dates.

## Timing, returns and limits

Membership uses `sp500` exclusively, exact daily records, inclusive interval ends
and availability no later than the observation date. Date-only availability
supports an **end-of-session** convention. Future membership end dates and
identity/quality diagnostics are audit fields, not predictors. Delisted and
removed names remain in their historical cohorts. There is no current-constituent,
current-survival, complete-history or future-return eligibility filter.

`provider_adjusted_return_1d = adj_close[t] / adj_close[previous XNYS session] - 1`.
Returns are computed on price history **before** membership filtering, including
an exact previous-year session anchor. Missing sessions, bad endpoints and
non-session bars cannot create a multi-day return labeled daily. No filling,
clipping, winsorizing or additional dividend credit is applied. These ratios are
provider-adjusted research proxies, not certified total returns or terminal
delisting proceeds. Price observations and their source lineage are preserved.

The default identity policy follows the prior project's exploratory admission
of unique resolved routes with review flags. `--identity-policy strict` withholds
review-flagged routes, while retaining their constituent rows. Ambiguity,
same-date native-ID collisions and previously confirmed wrong historical BBBY /
El Paso issuer routes cannot attach prices. Confirmed OPCH corrupt dates, if
requested, remain in price history with `known_bad_quote` status. Prior bounded
repair overlays are not silently extrapolated to this full-history dataset.

Historical membership prevents using today's survivors as the whole universe.
It does **not** guarantee an unbiased complete return sample: missing prices,
unverified historical identities, retrieval-vintage adjustments and incomplete
terminal cash flows remain visible limitations. Selecting only
`analysis_eligible` observations can introduce coverage bias. Preserve the full
panel and its denominators. The history table can support holding valuations
after removal, but a backtest still needs separate corporate-action and exit
accounting. No historical knowledge-time certification is claimed.

## Completed full-history build

The local build covers **31 March 1970–31 March 2026** and retains **7,074,237
member-days across 1,733 source security IDs**. The separate price-history table
contains 7,070,378 observations, including available warmup from the price
release's 1962 start. Together the generated files occupy approximately 686 MiB.

Of the supplied S&P 500 membership rows, 152,500 fall on dates outside this XNYS
calendar and are retained in the exclusion ledger. Every supplied row reconciles
to either the panel or that ledger. Every XNYS session in the requested interval
has source membership. This verifies the ledger's structure, not the economic
accuracy of every provider membership record.

**Full membership history does not mean complete price history.** Usable quotes
cover 3.5% of member-days in 1970, 48.9% in 1999, 90.6% in 2010, 97.2% in 2019,
98.6% in 2023 and 99.4% in the available portion of 2026. Check the daily and
annual coverage files before choosing an analysis period. In total, 2,743,901
member-days lack a matched price, 649,960 have withheld identity routes and 8,504
have invalid OHLCV. These rows remain in the dataset; their returns are not
imputed. The full history contains 3,671,872 usable quote rows and 3,666,516
available daily returns.

## Verification

```bash
python -m unittest discover -s tests -v
```
