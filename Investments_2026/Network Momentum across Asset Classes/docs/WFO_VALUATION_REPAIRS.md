# Held-valuation repairs for the original WFO

The latest [continuous accounting-scenario report](CONTINUOUS_WFO_PERFORMANCE.md)
handles the two remaining unquoted rights with explicit long-zero/short-cap
marks. It covers every session and month. This earlier audit preserves which
source valuations were actually recovered and which remain assumptions.

The missing months were caused by a strict completeness rule: one unavailable
held stock's next-session return withheld the entire common-cohort portfolio
day, and any such day withheld that month's compounded return. The market data
for whole months had not disappeared.

The original 2019–2025 WFO had 62 missing stock/date labels: 27 existing bars
rejected by the OHLC gate and 35 absent native bars. The latter comprise six
interior gaps and 29 merger/conversion events. All 62 original forecasts remain
in the experiment. Repairs recover 60 labels, leaving two contingent-right
valuations unresolved.

| Coverage | Original | After held-valuation repair |
| --- | ---: | ---: |
| Missing stock/date outcomes | 62 | 2 |
| Unavailable portfolio days | 38 | 2 |
| Complete portfolio days | 1,722 / 1,760 | 1,758 / 1,760 |
| Complete months | 53 / 84 | 82 / 84 |
| Uninterrupted return segments | 37 | 3 |
| Longest complete segment | 140 sessions | 1,002 sessions |

## Repair rules and evidence

Thirty exact raw closes came from the separately pinned Alpaca SIP daily-price
object, including all 27 rejected bars and the LIN, BIIB and PSA interior gaps.
The download's complete SHA256 is
`4a4919d48128d5c573ddbca64649d8cc78783f7bbb818219a3ad5f34728e38a0`;
GCS generation is `1783344954247444`. Each recovery matches the original
security key, symbol and two existing prior raw-price controls. The supplied
OHLC bounds and raw adjustment basis are checked. Source raw-page bodies were
not independently fetched again.

Three predecessor quotes require separate reviewed transcriptions because
the current-symbol SIP object omits those earlier dates:

| Universe label | Date | Historical symbol | Raw close | Quote source | Dated identity evidence |
| --- | --- | --- | ---: | --- | --- |
| WBD | 2022-04-04 | DISCA, Series A | $25.80 | [Issuer historical table](https://ir.wbd.com/stock-information/stock-quote-and-chart/default.aspx) | [Nasdaq conversion notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-63) |
| ELV | 2022-06-27 | ANTM | $482.58 | [HistoricalStockPrice table](https://www.historicalstockprice.com/antm-historical-stock-prices/) | [Issuer ticker-change notice](https://www.elevancehealth.com/newsroom/elevance-health-rings-in-rebrand-with-nyse-opening-bell) |
| RVTY | 2023-05-15 | PKI | $115.24 | [HistoricalStockPrice table](https://www.historicalstockprice.com/pki-historical-stock-prices/) | [Issuer SEC filing](https://www.sec.gov/Archives/edgar/data/31791/000003179123000011/pki-20230702.htm) |

Each transcription matches four prior original raw closes. HistoricalStockPrice
attributes its data to Ticker Technologies and Mergent. These are reviewed
web-table observations, not archived provider responses; direct HTTP retrieval
was refused. The configured values and source URLs are preserved in
`configs/valuation_manual_quotes_v1.json`.

Ordinary quote returns use contemporaneous raw units. Existing bad bars must
have unit action factors and zero dividends, with consistent rounded
adjusted/raw scales. Absent bars must have an unchanged adjusted/raw scale
between the original entry and a nearby following accepted quote, with no
recorded intervening split or dividend. This is a reviewed continuity
assumption: a complete independent action feed for every absent bar is not
claimed. Future bracket observations check units retrospectively and never
change signal-date selection or supply interpolated prices. PSA also has a
[documented one-for-one reorganization](https://www.sec.gov/Archives/edgar/data/1393311/000119312523211436/d542893d8k12b.htm).

Twenty-seven fixed merger entitlements use
`(cash + exchange_ratio × acquirer_raw_USD_close) / old_raw_entry_close - 1`.
Cash-only consideration is a cash receivable; stock consideration is the
replacement shares at the exact valuation-date close. These marks are not
fabricated tradable stock bars and never enter momentum features. Terms and
primary issuer, exchange or SEC references for every event are recorded in
`configs/valuation_events_v1.json` and the output valuation ledger. Final
[ANSS closing terms](https://www.sec.gov/Archives/edgar/data/883241/000114036125026139/ef20051970_8k.htm)
are $199.91 plus 0.3399 SNPS shares, rather than the initial announcement.

Acquirer prices use accepted pinned Intrinio raw records; the available SIP
cross-checks agree. Historical XOM is absent from the pinned Intrinio release,
so the PXD conversion uses its exact independent SIP raw close of $116 on
2024-05-03. Adjusted acquirer prices are never mixed with raw exchange ratios.
Cash/share receivables are valued without settlement friction, taxes, fees or
fractional-share processing. Original entry closes around completion can be
stale or untradeable; this correction does not establish executable trading.

## The two remaining dates

| Return date | Stock | Consideration | Why still unresolved |
| --- | --- | --- | --- |
| 2022-12-22 | ABMD | $380 cash plus one non-tradeable right, up to $35 | No observed liquidation price or carried-right valuation supplied. |
| 2025-08-28 | WBA | $11.45 cash plus one non-transferable right, up to $3 | No observed liquidation price or carried-right valuation supplied. |

The cash-plus-right terms are documented by the
[Abiomed closing announcement](https://www.sec.gov/Archives/edgar/data/200406/000119312522311072/d428734dex991.htm)
and [Nasdaq's Walgreens notice](https://classic.nasdaqtrader.com/TraderNews.aspx?id=ECA2025-463).
Neither a zero mark, maximum mark, last-price carry-forward, nor an eventual
realized payout is silently substituted for a contemporaneous valuation.
Non-transferable rights also cannot be assumed sold at an invented exit price.

The saved sensitivity varies each right between zero and its contractual
maximum, retaining all holdings and accounting for position sign. GMOM is
short ABMD and long WBA on these dates:

| Stock | GMOM lower one-day mark | GMOM upper one-day mark | Portfolio range |
| --- | ---: | ---: | ---: |
| ABMD | -0.03224% | -0.02845% | 0.380 bp |
| WBA | -0.12456% | -0.06563% | 5.893 bp |

These are hypothetical one-day mark sensitivities, not observed returns,
realized liquidation prices, complete wealth bounds, or a subsequent
contingent-right cash-flow ledger. All six strategies' sensitivities are saved
in `contingent_claim_sensitivity.csv`.

## Results and scope

On the now 1,758 complete dates, before costs, GMOM Sharpe is **0.736**,
LinReg **0.753**, and matched-cohort LongOnly **0.776**. The original figures
used only 1,722 dates. Recovering 36 additional days changes the sample;
every originally observed portfolio return is preserved numerically.

The revised monthly table is complete except for December 2022 and August
2025. Equity and drawdown now have three complete runs: 2019-01-02–2022-12-21
(1,002 sessions), 2022-12-23–2025-08-27 (670), and 2025-08-29–2025-12-31 (86).
Each run starts at 100. Full-period CAGR and maximum drawdown remain withheld
across the two unresolved marks.

**This is a retrospective realized-label correction of the frozen experiment.**
No prices, features, graph eligibility, validation selection, models, scalers,
forecasts, positions or original observed labels are changed. It does not
repair every feature-history gap or refit the strategy on a newly corrected
training population. A subsequent ranker must explicitly version its training
labels, feature repairs and rights/exit policy; these corrections must not be
silently described as a fully rebuilt clean-data WFO.

## Verification and outputs

The independent replay checks all 62 repair keys against the original missing
set, derives 60 repaired values separately, preserves every original artifact,
and recomputes all 10,560 strategy/date portfolio returns and exposures.
Maximum return discrepancy is `1.01e-16`. The numerical verification of the
original 21 OLS models and 771,773 forecasts still applies to the unchanged
original files. The expanded structural suite has **69 passing tests**.
The evidence preparer also reconstructed all 159,148 native-context rows from
SHA-verified source objects; reconstructed event-price records match field by
field.

Review the [repaired output bundle](../outputs/valuation_repaired_wfo/README.md),
[performance definitions](WFO_VALUATION_REPAIR_PERFORMANCE.md), or the
[unchanged original output bundle](../outputs/original_wfo/README.md).

Local generated directories are `data/walk_forward/valuation_results_v1/`,
`valuation_verification_v1/` and `valuation_performance_v2/`.

For a fresh checkout with the pinned local data and the original WFO run:

```bash
python audit_wfo_availability.py
python prepare_valuation_evidence.py
gcloud --configuration=deep-momentum-research storage cp 'gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/equity_us/sp_composite1500_market_daily/v1/tables/daily_bars/part-00000.parquet#1783344954247444' data/performance_gap_audit/v1/independent_daily_bars.parquet
python repair_walk_forward_valuations.py
python verify_valuation_repairs.py
python analyze_walk_forward.py --results data/walk_forward/valuation_results_v1 --verification data/walk_forward/valuation_verification_v1/report.json --output data/walk_forward/valuation_performance_v2
python export_valuation_outputs.py
python -m unittest discover -s tests -v
```

Existing generated directories are never overwritten; use fresh version paths
when rebuilding. Source data and large prediction tables are required for a
full replay and are excluded from the compact review export. A selected report
can be viewed without those raw inputs.
