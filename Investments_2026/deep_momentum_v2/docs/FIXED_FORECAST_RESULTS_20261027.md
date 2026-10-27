# Monthly versus daily backtests: 47-month cumulative returns

**Return units and period:** Every performance percentage in the cumulative-return tables below is a **compounded total over 47 months, February 2020–December 2023**. These tables cover nearly four years. Annualized CAGR appears in its own table. The **5-year / 10-year labels describe rolling training windows**, while the backtest period is the same 47 months for every arm.

VERIFIED LOCAL, 27 October 2026. The question is whether the same saved predicted names earn similar realized returns when the monthly-return engine is replaced with daily cash/share accounting. This is a backtest-realization comparison; no individual training-label certification is claimed.

**Answer: yes when dates and conventions match.** All eight original-cache daily close controls (four arms, zero/10 bp) reproduce all 47 monthly portfolio NAV observations within the frozen tolerance. The largest absolute NAV discrepancy is 7.55e-15 on initial capital 1. The observed execution quote view also gives the same gross totals under the same close conversion units. Next-open execution changes the returns modestly; the asset-stop rule is a materially different payoff.

The same monthly-model forecasts, RET scores and selected baskets are fixed in all stages. Zero new fits, probabilities, RET estimates, substitutions or rebalancing choices. This uses the isolated Paper15 RF/XGBoost 5/10-year experiment, not the earlier expanded-RF experiment.

## Gross cumulative total returns over 47 months

All values are cumulative total returns over **February 2020–December 2023: 47 monthly outcomes and 985 following trading sessions**. Initial wealth is measured at the 31 January 2020 close and terminal holdings are marked at the 29 December 2023 close without forced liquidation.

| Model / training window | Monthly engine: 47-month gross total | Daily same-close: 47-month gross total | Daily next-open: 47-month gross total |
|---|---:|---:|---:|
| RF · 5-year training | +27.05% | +27.05% | +25.72% |
| XGBoost · 5-year training | +21.77% | +21.77% | +20.77% |
| RF · 10-year training | -5.05% | -5.05% | -5.86% |
| XGBoost · 10-year training | +55.11% | +55.11% | +52.36% |

The same-close equality is checked month by month, rather than inferred from matching final totals. Daily valuation still reveals intramonth risk that monthly observations miss. For example, XGBoost 10-year same-close gross drawdown is −16.57% on daily closes versus −11.75% on month-end observations in this matched period.

## Similarity after next-open execution

These compare the 47 monthly returns from the monthly engine with daily-backtest returns compounded to the corresponding calendar month. Gross, no stops, same saved baskets; differences below are percentage points of monthly return.

| Model / training window | Monthly correlation | Mean absolute monthly difference | Largest absolute monthly difference | Same monthly sign |
|---|---:|---:|---:|---:|
| RF · 5-year training | 0.9954 | 0.433 pp | 2.909 pp | 95.7% |
| XGBoost · 5-year training | 0.9946 | 0.419 pp | 3.185 pp | 95.7% |
| RF · 10-year training | 0.9867 | 0.477 pp | 3.503 pp | 95.7% |
| XGBoost · 10-year training | 0.9883 | 0.521 pp | 2.753 pp | 100.0% |

These are similarity diagnostics on already examined development results, not significance tests or fresh out-of-sample evidence. A high correlation does not mean each monthly return is identical.

## Net cumulative total returns over 47 months

Ten basis points are charged one way on actual net physical trading notional. The monthly control and daily next-open paths use the same cost rate. The stop variant adds the original 10% per-asset threshold, with gap-at-open and idealized high/low fills and stopped capital retained in cash.

| Model / training window | Monthly: 47-month net total | Daily next-open: 47-month net total | Daily next-open + 10% asset stop: 47-month net total |
|---|---:|---:|---:|
| RF · 5-year training | +16.96% | +15.74% | -1.63% |
| XGBoost · 5-year training | +12.36% | +11.44% | -4.02% |
| RF · 10-year training | -12.56% | -13.30% | -36.36% |
| XGBoost · 10-year training | +42.93% | +40.39% | -8.02% |

For XGBoost 10-year, +55.11% gross at the close becomes +52.36% gross at the next open (−2.75 pp), then +40.39% after costs (−11.97 pp), then −8.02% with asset stops (a further −48.42 pp). These are sequential differences in compounded total return, not additive cash P&L contributions.

A stop-path result should not be expected to equal an unstopped monthly-return backtest. Likewise, the earlier executable-target models used different forecasts and RET baskets. Their comparison with these monthly-model paths mixes learning and execution changes. This fixed-forecast result does not identify a causal target-label effect or explain the paper’s performance by itself.

## Annualized CAGR, shown separately

Daily next-open execution, 10 bp one-way costs. The verified daily-report convention annualizes geometric NAV growth with 252 / 985 trading years: `CAGR = (ending NAV / starting NAV) ** (252 / 985) - 1`. This is geometric annual growth, distinct from the paper’s annualized arithmetic mean return. The cumulative totals below cover the same 47 months as the preceding tables.

| Model / training window | No-stop 47-month net total | No-stop annualized CAGR | 10% stop: 47-month net total | 10% stop: annualized CAGR |
|---|---:|---:|---:|---:|
| RF · 5-year training | +15.74% | +3.81% | -1.63% | -0.42% |
| XGBoost · 5-year training | +11.44% | +2.81% | -4.02% | -1.04% |
| RF · 10-year training | -13.30% | -3.59% | -36.36% | -10.92% |
| XGBoost · 10-year training | +40.39% | +9.07% | -8.02% | -2.12% |

Example: XGBoost with a ten-year training window turns starting NAV 100 into 140.39 over these 47 months without stops, after 10 bp trading costs. That is +40.39% cumulative and +9.07% annualized CAGR under the stated daily convention.

The private chart `outputs/fixed_forecast_anchor_completion/20261027-v1/reports/same_baskets_total_return_47_months.png` explicitly labels the 47-month cumulative totals. The editable slide revision is under outputs/presentations/20261027_monthly_daily_comparison_clarified. The original presentation remains a dated reference.

## Boundary completion and verification

The preserved original run independently checked 28 paths: 22 complete and six unavailable diagnostic close-clock controls. All twelve next-open/cost/stop paths were already complete. The six controls lacked only PBF’s observed 30 December 2022 entry boundary in the saved execution requests. A separately sealed, independently checked completion adds exactly that cached accepted raw/adjusted open/close point; no high/low is manufactured. The other 22 rows remain exactly unchanged. The composite report has **28 complete / 0 withheld**, while the original 22/6 report remains visible.

The quote-view close controls agree with the monthly benchmark. Changing only the corporate-action successor conversion factor from effective close to effective open leaves small residuals: up to 1.16e-8 NAV units and 0.0000011 percentage points of final total return. Some exceed the original tight reconciliation tolerance. They remain recorded as mismatches; no tolerance was relaxed and no quote, term or basket was adjusted to force equality. The open-unit stage is a changed convention, not an additional proof of exact close-clock equality.

The main independent checker reconstructs input projections and checks signed cash/share actions, holdings, net market orders, fees, stops, failure retention, every valid mark and risk/return metrics with a separate accounting oracle. The completion independently reconstructs the one-point adapter and rechecks six controls. The passive reader also checks that displayed month-end curves are derived exactly from the independently checked daily NAV ledgers. The auxiliary quote-difference inventory is not used as an independently certified numerical result.

This verifies the bounded arithmetic and preserved forecasts. It does not fully certify source identity, comprehensive corporate-action/distribution coverage, historical knowledge-time availability or investability. Same-close execution remains idealized; raw/adjusted unit conventions, fractional successor shares and zero borrow/funding remain assumptions. Retired new orders remain unfilled cash slots without a replacement or redistribution. All original source/run bytes are preserved; no raw download, cloud mutation or protected 2024+ financial value is read.

## Why the run ends in 2023, and the longer available horizon

VERIFIED METADATA, 27 October 2026. December 2023 is the old experiment's frozen research/holdout boundary. `configs/research_workflow.json` sets `last_test_year = 2023`, and each executed recipe explicitly limits its evaluation period. Rolling walk-forward learning has no inherent 48-month limit. The previously completed monthly-only run contains January 2020–December 2023 (48 months); the matching daily comparison starts in February (47 months). The first ten years from the user-selected 2010 initialization supply training, so the common ten-year test starts in 2020.

A bounded date/membership-key check of the downloaded pinned files, plus live cloud pointer/manifest checks after reauthentication, found:

| Input | Local dates actually checked | Current cloud release metadata | Meaning for this strategy |
|---|---|---|---|
| Intrinio prices | Date keys extend through 24 July 2026 | Same price release/pin, declared end 24 July 2026 | Later daily prices already exist locally; July is incomplete |
| Primary Siblis S&P 500/400 membership | Both sleeves have month-end member rows through 31 March 2026 | New repaired release, declared end 31 March 2026 | March-end membership can define April's predicted basket |
| Unified product | Product-wide dates reach 30 April 2026, but the old local S&P 500/400 flags are empty in April | New full release, product-wide end 30 April 2026 | Other index coverage does not establish later S&P 500/400 eligibility |

**Candidate extension with the existing local source version:** February 2020–April 2026, **75 months (6 years 3 months), 28 additional months**. The monthly-only January 2020 start would contain **76 months**. The candidate terminal mark is 30 April 2026, using picks formed from 31 March membership. This is a date-availability ceiling to validate, not a newly completed backtest or certification of every stock's prices/actions. Keeping the 2010 initialization and 5/10-year histories preserves the current experiment. Moving the initialization earlier would be a separate chronology/coverage decision.

Annual rolling refits continue as follows:

| Forecast year | Five-year training outcomes | Ten-year training outcomes |
|---|---|---|
| 2024 | 2019–2023 | 2014–2023 |
| 2025 | 2020–2024 | 2015–2024 |
| 2026, through April | 2021–2025 | 2016–2025 |

Retain the same Paper15 target, RET ranking, monthly predictions, matching past-history eligibility and unchanged daily cost/stop/basket rules. If the old source version and saved models remain reusable, four arms across these three new forecast years need twelve incremental annual fits. Adopting corrected historical membership/identity inputs may invalidate old training cohorts, so that fit count must follow a source-impact check. No tuning or numerical post-2023 price/feature/label read occurred in this availability audit.

The cloud check found newer Siblis and unified releases named `professional-reference-20260807-r1-pdrs-v2-r1`; their object bytes differ from the downloaded versions. Their declared horizons do not extend the price/membership ceiling above. The account can read the curated manifests/certificates, but three exact referenced objects (primary membership table, identity bridge and index coverage registry) return **403 / storage.objects.get denied** under their `dataset-platform/runs/` URIs. That is separate from authentication and is not evidence that the data is absent. The old local source pins are unchanged; neither the new release nor its rows have been promoted. A prepared, unsent read-only access request and the exact URIs/errors are in the private audit evidence.

The metadata-only pass measured 51,000,593 compressed key-column bytes, considered 30,152,869 candidate rows, and used approximately 151 MiB peak RSS. It selected date keys and membership booleans only, no financial columns. The private evidence is `outputs/backtest_horizon_audit/20261027-v1`. The proposed next-step configuration is `configs/walkforward_extension_20261027.proposed.json`; it launches no computation and does not weaken the frozen old guards. **All performance tables above still end in December 2023.**

## Reproduce the comparison

Open [passive notebook 13](../notebooks/13_fixed_forecast_reconciliation.ipynb). Its ignored local settings point to both completed private runs. Run All authenticates independent proofs and reads saved reports; it never loads fitted models, refits, downloads or replays trades. A fresh clone without private evidence reports that it is unavailable.

Frozen methods: [main reconciliation](FIXED_FORECAST_RECONCILIATION_20261027.md) and [one-point control completion](FIXED_FORECAST_COMPLETION_20261027.md). Private main run: outputs/fixed_forecast_reconciliation/20261027-v1. Private completion and presentation-ready CSV/PNG reports: outputs/fixed_forecast_anchor_completion/20261027-v1/reports. The active branch remains codex/paper15-ret-5y-10y; no merge is implied by this check.
