# 5min_momentum

Research project for momentum strategies using five-minute US stock bars and historical S&P 500 membership.

The current implementation downloads a published Alpaca five-minute release, verifies its files, selects S&P 500 member-days, and prepares compatible historical membership tables. Strategy development and backtesting are the next stage.

## Local setup

Use Python 3.14 and an authenticated Google Cloud CLI:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-data.txt
gcloud auth login
gcloud config set project trusty-agility-492109-v0
```

`gcloud` is discovered on PATH. Set `GCLOUD_BIN` if it is installed elsewhere. Access tokens stay in memory.

The local project can reuse an existing private data directory:

```sh
ln -s /absolute/path/to/existing/private/data data
```

The `data` directory, its symlink, virtual environment, credentials, caches and Parquet files are excluded from Git. This public repository contains source code and documentation. Licensed vendor data stays in Galambos Capital's private storage and local workspace.

## First notebook: loading and coverage

Open `01_load_sp500_5min.ipynb` using the project's `.venv` kernel. It loads the historical S&P 500 five-minute prices and both membership tables, audits the full available raw history by default, checks exchange-calendar coverage and missing bars, and displays a bounded OHLCV sample with price/volume charts. Change the first cell to choose another date range, adjustment or sample ticker.

Install notebook dependencies and run the full audit:

```sh
.venv/bin/python -m pip install -r requirements-notebook.txt
.venv/bin/python scripts/run_coverage_notebook.py
```

The runner preserves the output-free source notebook and writes `data/reports/01_load_sp500_5min/01_load_sp500_5min.executed.ipynb`. That private directory also contains `SUMMARY.json`, coverage tables, quality totals, exact missing-slot examples, the sample and charts. Report files are replaced on rerun; the JSON records the selected parameters. Keep notebook outputs containing licensed data in this ignored directory.

The expected grid uses the XNYS regular-session calendar, including early closes, with start-labelled five-minute slots. The notebook separates unresolved membership identities, whole days without daily prices, absent intraday bars and present five-minute buckets containing fewer than five source minutes. No prices or gaps are filled.

## Published data

Pinned five-minute release:

```text
gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/equity_us_intraday_5min_alpaca/v1/_releases/professional-source-build-20261103-r1/MANIFEST.json
```

| Local directory | Contents |
| --- | --- |
| `data/alpaca_5min_20261103/` | Complete published multi-index release, raw and adjusted bars, source metadata and download report |
| `data/sp500_5min_20261103/` | S&P 500-only copy, partitioned by adjustment and month |
| `data/sp500_pit_20261103/` | Historical membership series and a unique security/date table for joins |

The full release and S&P 500 copy overlap: the copy is selected from the full release. They are separate local files, rather than additional historical coverage. Read the local manifests for counts, sizes, checksums and caveats.

## Download and preparation

For a fresh data directory, download the release and upstream membership metadata:

```sh
mkdir -p data/alpaca_5min_20261103/membership_support

gcloud storage cp \
  gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/equity_us_intraday_5min_alpaca/v1/_releases/professional-source-build-20261103-r1/MANIFEST.json \
  data/alpaca_5min_20261103/MANIFEST.json

gcloud storage cp \
  gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/equity_us_intraday_1min_alpaca/v1/_releases/professional-source-build-20261102-r2/MANIFEST.json \
  data/alpaca_5min_20261103/SOURCE_1MIN_MANIFEST.json

gcloud storage cp \
  gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/equity_us_market_daily_unified/v3/_releases/professional-source-build-20261102-r2/MANIFEST.json \
  data/alpaca_5min_20261103/MEMBERSHIP_V3_MANIFEST.json

gcloud storage cp \
  gs://quant-research-trusty-agility-492109-v0/dataset-platform/runs/datasets-platform-20261102-125411-fe7ec96/runs/datasets-platform-20261102-125411-fe7ec96/results/content/tables/membership_matches/part-00000.parquet \
  data/alpaca_5min_20261103/membership_support/MEMBERSHIP_MATCHES.parquet

gcloud storage cp \
  gs://quant-research-trusty-agility-492109-v0/dataset-platform/runs/datasets-platform-20261102-125411-fe7ec96/runs/datasets-platform-20261102-125411-fe7ec96/results/content/tables/membership_price_gaps/part-00000.parquet \
  data/alpaca_5min_20261103/membership_support/MEMBERSHIP_PRICE_GAPS.parquet

.venv/bin/python scripts/download_published_5min.py
.venv/bin/python scripts/prepare_published_sp500.py
```

The downloader makes GCS GET requests only, pins object generations, and checks each file's size and SHA-256 against the manifest. Existing files are verified before reuse. It refuses releases above 50 GB; check transfer costs before a fresh download. Source objects are never uploaded, moved or deleted.

Preparation copies the published five-minute prices unchanged. It selects date-specific S&P 500 membership without filling gaps or resampling minute data. All membership intervals reconcile to the upstream member-date counts, including members with unavailable prices.

## Read a bounded sample

```python
import duckdb

con = duckdb.connect()
con.sql("""
    SELECT security_id, ticker, bar_timestamp_utc,
           open, high, low, close, volume, minute_bars
    FROM read_parquet(
        'data/sp500_5min_20261103/adjustment=raw/month=2026-10/*.parquet',
        hive_partitioning=false
    )
    WHERE ticker = 'AAPL'
    ORDER BY bar_timestamp_utc
    LIMIT 10
""").show()
```

Join bars to `membership_by_security_date.parquet` using `security_id` and `trading_date`. `membership_daily.parquet` preserves the full membership-series/date grain, including unresolved identities and conflicting series; it can have multiple membership series for a security/date.

## Data limits

- Price coverage is restricted to index-member days. Pre-entry and post-exit stock history is absent.
- Membership is a retrospective snapshot with date-level availability. After March 2026, S&P 500 membership uses Sharadar lists and changes; their change dates can precede the effective session by up to one week.
- Adjusted prices inherit different source-retrieval vintages. Source changes can create steps in a series.
- Bar-end availability is modeled; it is not an archived historical feed-receipt time.
- Extended-hours bars are retained. Regular-session flags use the New York clock; apply an exchange calendar for early closes.
- Empty intervals have no row. Preserve gaps and unresolved identities rather than replacing them with zero returns.

Consult each local source manifest before using the data in a backtest.
