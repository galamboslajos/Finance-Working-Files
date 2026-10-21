# RET backtest completed — 20 October 2026

**All 40 execution paths are complete and independently verified. Zero paths remain withheld for the selected 2020–2023 portfolios.** The first RET attempt remains unchanged as a reference of the diagnosed failures; this additive execution repair did not change its scores or selections.

## What changed

The original 146 missing stock/date keys across eleven names came from eight cash acquisitions, Grubhub's stock conversion, the historical RCM native route and five pre-transition Discovery observations. Cash/stock entitlements replace terminated predecessor exposure; they do not fabricate quotes for stocks that stopped trading. Forty actual observed quote rows were appended (22 RCM, five DISCA, thirteen new ADS observations). Nine legal actions supplement the previous nine, with all eleven historical retirements and the PENN exception preserved.

For the main tuned RF expanded model, its entire original gap list was **Legg Mason from August 3, 2020 and Mimecast from May 19, 2022**, both short positions. They require signed cash settlement liabilities. Legg Mason also has a newly selected already-retired order; its original selection remains an unfilled cash slot, with no replacement or old-holder entitlement.

See [the frozen repair method and exact legal terms](RET_REPAIRS_20261020.md) for the company-by-company treatment, price-unit controls, dated execution identity exceptions and primary sources.

## Results

All results cover 2 January 2020–29 December 2023: 48 months, 1,006 trading sessions.

RET uses the original saved forecasts and 120-month realized class payoffs to select ten long and ten short names. This table uses the fixed 10% per-asset stop and 10 bp per one-way traded notional. Returns compound over the full four years.

| Model | RET net return | CAGR | Annual vol | Sharpe | Max DD | Stop exits | Prior class 1/10 net return |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| RF Paper15 | -23.73% | -6.56% | +19.68% | -0.25 | -49.94% | 408 | withheld |
| XGBoost Paper15 | +16.69% | +3.94% | +20.70% | 0.29 | -34.43% | 415 | +23.45% |
| RF expanded | -2.36% | -0.60% | +17.33% | 0.05 | -33.73% | 366 | +18.82% |
| XGBoost expanded | +7.26% | +1.77% | +17.67% | 0.19 | -32.90% | 393 | +3.41% |
| RF expanded tuned | +23.70% | +5.47% | +17.63% | 0.39 | -24.12% | 328 | +42.78% |

The prior class-basket result is preserved context. New execution evidence was applied only to this RET replay; this is not a claim that all historical learning inputs or comparator execution evidence were rebuilt.

| Model | RET SL gross return (0 bp) | RET SL net return (10 bp) | Stop exits, gross / net |
| --- | ---: | ---: | ---: |
| RF Paper15 | -16.29% | -23.73% | 408 / 408 |
| XGBoost Paper15 | +27.95% | +16.69% | 415 / 415 |
| RF expanded | +6.68% | -2.36% | 366 / 366 |
| XGBoost expanded | +17.27% | +7.26% | 393 / 393 |
| RF expanded tuned | +35.15% | +23.70% | 328 / 328 |

Verified complete paths: 40/40; withheld paths: 0.

The original gap inventory contained 146 native/date keys across 11 companies and 208 stock-month episode rows. The repair adds nine legal actions and 40 observed quote rows, preserving the previous actions, retirements, selections and scores.

| Company | Ticker | Gap dates | Native/date keys |
| --- | --- | --- | ---: |
| CDK Global Inc | CDK | 2022-07-06–2022-08-01 | 19 |
| Activision Blizzard Inc | ATVI | 2023-10-13–2023-11-01 | 14 |
| Warner Bros. Discovery Inc | WBD | 2022-04-04–2022-04-08 | 5 |
| Cerner Corp | CERN | 2022-06-08–2022-07-01 | 17 |
| Syneos Health Inc | SYNH | 2023-09-28–2023-10-02 | 3 |
| National Instruments Corp | NATI | 2023-10-11–2023-11-01 | 16 |
| Mimecast Ltd | MIME | 2022-05-19–2022-06-01 | 9 |
| CyrusOne REIT Inc | CONE | 2022-03-25–2022-04-01 | 6 |
| Legg Mason Inc. | LM | 2020-08-03–2020-09-01 | 22 |
| GrubHub Inc. | GRUB | 2021-06-15–2021-07-01 | 13 |
| R1 RCM Inc | RCM | 2021-11-01–2021-12-01 | 22 |

GRUB and RCM execution identity reconstruction remains provisional. Historical identity, knowledge-time, survivorship and investability are not certified. Features, labels, models and RET payoff inputs were not rebuilt; the potential historical learning-input impact remains unresolved.

Previously examined development years, provider-adjusted price proxies, retrospective DF/RDC restrictions, zero borrow/funding and idealized daily OHLC stop fills remain limitations. Withheld metrics remain null; no ranking drops incomplete models.

The JSON includes all 40 RET scenarios, annual results, selection coverage, failures and all 80 preserved class-basket scenarios. This passive report performed no fitting, inference, price-source read or accounting replay.

## Main RF comparison

The tuned RF RET strategy has lower cumulative return than the preserved tuned RF class1/10 strategy, and a smaller maximum drawdown. This is a selection-strategy comparison on the same saved forecast models; it is not a new fitted model or a label experiment.

| Tuned RF expanded, 10% asset stop / 10 bp one-way cost | Four-year return | CAGR | Annual volatility | Daily annualized Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| RET | +23.70% | +5.47% | 17.63% | 0.39 | -24.12% |
| Preserved class1/10 reference | +42.78% | +9.33% | 20.69% | 0.53 | -35.14% |

Annual returns below are slices of the same continuous RET NAV, with no annual liquidation or reset. They compound to the full-period return; annual drawdown slices do not replace the full-period maximum drawdown.

| Outcome year | Tuned RF RET net return |
| --- | ---: |
| 2020 | +23.34% |
| 2021 | +4.58% |
| 2022 | -5.92% |
| 2023 | +1.93% |

## Verification, preservation and scope

- 195 focused installed tests passed (69 new repair/supervision/verifier cases and 126 existing RET/target cases). Two additional synthetic production→Parquet→independent unit/overlay checks passed in peer review. The passive reporter also passed three synthetic integration checks.
- The actual independent verifier reconstructed the quote overlays and units, action registry, all forty daily cash/units paths, stops, fees, metrics and continuous annual slices using the preserved independent equations. It did not call the production accounting or overlay for expected answers.
- The exact previous RET PASS is inherited for its 48 causal payoff vectors and full-probability scoring. All twenty original selection/ledger/rebalance/monthly files are preserved by byte pins. No feature, target, payoff, model or preprocessor was rebuilt, and no estimator was reloaded for inference.
- All 110 original code/protocol pins, all 609 artifacts of the first RET run, and all 57 newly installed source/configuration/evidence files reconciled after verification and reporting.
- The coverage preflight had zero missing keys and zero quality flags across every selected maximum holding horizon in all five arms. This establishes coverage for these portfolios, not every stock or every bar in the dataset.
- RCM's specific historical route and GRUB's dated successor ADS execution rebind remain provisional. In particular, the Russell bridge's old Grubhub ISIN did not independently identify the successor. Official listing/share-class evidence and literal dated SIP raw/all observations support the stated execution reconstruction.
- **The effect of these newly recovered histories/terminal events on saved features, monthly labels and RET payoff inputs is not repaired by this execution rerun.** Assess that impact before claiming an end-to-end repaired learning pipeline or launching a dependent new fit. Preserve this fixed-model comparison as its own experiment. Alternative labels remain on hold pending that methodological decision.
- Zero borrow/funding, retrospective DF/RDC restrictions, adjusted-price proxies, fractional entitlements and idealized daily OHLC stop fills remain inherited simulation assumptions. These examined development years are not a fresh untouched test.

All four supervised phases passed their resource limits. Watchdog-sampled peak process-tree RSS was 2.13 GiB; the verifier's own reported peak was 2.29 GiB, both below 4 GiB. Cumulative conservative production/verification/reporting compressed-read reservations were 283,528,149 bytes against 1 GiB. Run/control output size was 52.44 MiB before this handoff, against 512 MiB. The earlier bounded source inspection separately projected 80,765,930 compressed financial bytes against 128 MiB. No 2024+ financial rows were returned or analyzed; mixed Arrow-block decoding limits remain documented.

## Saved artifacts and repeat workflow

- Protocol: `docs/RET_REPAIRS_20261020.md`.
- Configuration: `configs/ret_repair_backtest_20261020.json`.
- Immutable reviewed quote/action package: `data/derived/ret_repairs/20261020-v1/repair_manifest.json`.
- New run: `outputs/ret_repair_backtest/2020-2023-20261020-v1`.
- Independent PASS: `verification/ret_repair_backtest.json` within that new run.
- Passive full comparison: `reports/verified_20261020/verified_ret_repair_comparison.md` and `.json`; all forty RET and eighty preserved class-basket scenarios are retained.
- Separate resource controls: `outputs/ret_repair_backtest_control/20261020-v1`.
- Entry points: `scripts/run_ret_repair_backtest.py`, `scripts/watch_ret_repair_backtest.py`, `scripts/verify_ret_repair_backtest.py`. A repeat requires fresh run/control directories and an explicit scope; do not overwrite the completed attempts. No rebuild or fit is needed to read these saved reports.

Independent receipt SHA-256: `8253c45415a9766172c6955b58906c6d5762e53853e50ab50048996259ebf3a0`.

These changes are local. No commit, push or merge was performed in this repair round.
