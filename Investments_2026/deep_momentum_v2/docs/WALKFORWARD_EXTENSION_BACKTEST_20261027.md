# Continuous Paper15 realization through April 2026

This is the side experiment on `codex/paper15-ret-5y-10y`: RF and XGBoost, each with rolling five- and ten-year training windows and only the agreed fifteen paper inputs. It is separate from the expanded-RF experiments. The 2010 research initialization and shared February 2020 evaluation start stay unchanged.

## Period and return units

The common comparison covers **February 2020–April 2026: 75 monthly outcomes and 1,569 trading sessions**, 3 February 2020–30 April 2026, with starting NAV at 31 January 2020 close. A total return is cumulative compounded over all 75 months. CAGR is separately annualized using the frozen daily convention, `terminal_NAV ** (252 / 1569) - 1`. The five/ten-year model labels are training-window lengths. They are not holding periods or result durations.

The original 47-month February 2020–December 2023 results and 48-month January 2020–December 2023 monthly-only experiments remain separate references. April 2026 is the last holding month supported by the pinned S&P500/400 membership ending March 2026. Prices through July alone do not establish later membership. No current cloud release replaces the pinned inputs.

## Models, selection and continuous accounting

The run reuses the independently verified twelve additional annual fits for 2024, 2025 and 2026, plus the sixteen original annual fits for 2020–2023. Predictions remain monthly; the four arms retain their separate five/ten-year complete past daily-OHLC eligibility and cohorts. No new fitting, tuning, source upgrade or label repair occurs in this realization phase.

The target is the strict adjusted-close return from the preceding signal month-end to the outcome month-end, ranked into deciles. RET is the probability-weighted expected return computed from trailing actual class means, with 60/120 completed monthly cells according to the arm. The independently frozen top/bottom RET baskets contain ten longs and ten shorts every month. Each rebalance allocates 50% long and 50% short. No stock is dropped, replaced or screened using its subsequent quote availability.

Each daily path maintains one cash/holdings book through all year boundaries, including December 2023. Next-open execution uses the first session of the outcome month. Voluntary market orders net native-security quantities and pay 0/5/10/25 bp per one-way notional; mandatory conversions incur no invented market trade. Borrow, financing and cash interest remain zero. The final close marks surviving holdings without forced liquidation.

The stop alternative retains the original entry-relative 10% per-asset rule. A gap beyond the threshold exits at observed open; an intraday adverse high/low crossing uses the idealized threshold fill. Monthly rebalance occurs before the old month's open-stop logic. Stopped capital remains cash until the next scheduled rebalance. These rules do not cap portfolio losses or guarantee attainable fills.

## Frozen grid and direct monthly/daily control

There are **48 daily paths**, four arms × twelve declared stages:

| Stage | Convention |
|---|---|
| Cache close gross / 10 bp net | Same quote cache and close-based successor units as the monthly accounting |
| Execution close gross | Execution quotes; close-based successor units |
| Execution close, open successor units | Separate conversion-factor control |
| Next-open at 0, 5, 10, 25 bp, each without/with stops | Eight unchanged execution/cost/stop combinations |

Sixteen independent monthly-return references cover four arms × four costs. Their full 75-month signed cash/share accounting preserves the original four EV/CDK/XEC/XLNX realized-return amendments. The same-cache, same-close daily gross and 10 bp paths must reproduce their monthly reference at every month-end within the original tolerance. The changed-convention paths report differences; they are not forced to equal the monthly benchmark. Both daily and month-end risk measures remain separately labelled.

## Source and execution-only repair

The sealed recipe reserves 415,785,272 source-file bytes per pass, below its 1 GiB ceiling, before decoding daily values. It uses verified old OC/execution projections, the independently checked PBF OC completion, and the separately verified extension cache's 584 sessions and eight raw/adjusted OHLC fields. Only required January 2024 names' 29 December 2023 OC anchors may supplement the old projection, directly from the immutable observed OC cache; high/low values are not invented. The full source/native/date inventory is fixed before reading returns.

U.S. Steel (X), native `sec_g4KJZy`, is the one added legal event: $55 cash per common share, removal before the 18 June 2025 open. The official [SEC 8-K, Introductory Note and Item 3.01](https://www.sec.gov/Archives/edgar/data/1163302/000110465925060674/tm2517314d1_8k.htm) is captured and pinned privately as complete HTML. The predecessor's observed 17 June paired raw/adjusted close converts signed adjusted units into raw-share entitlements. A short position owes the cash; it is not deleted with zero payoff. Any earlier valid stop takes precedence because no position then survives to conversion. Cash is an effective-date, zero-rate economic entitlement proxy rather than a claim about the actual broker payment date.

Only realized holdings and the monthly realized-payoff comparison use this event. The strict learning label, historical RET cell population, saved probabilities and selections remain unchanged. The pre-existing 22 events, eleven historical retirements and exact PENN handover remain unchanged. Neither provisional source identity nor all possible future distributions are newly certified by this repair.

## Verification, resources and reproducibility

The separate `extension_accounting.py` copies the parent financial equations unchanged; only its explicit authorized calendar is different. Existing workers' pre-2024 guards remain. Independent input checking reconstructs all old/new quote projections and acceptance masks, exact selections, observed anchors, outcome projections and the new raw-share cash formula. A separate transition oracle imports no producer accounting and checks daily NAV, signed PnL/cash, held units, voluntary trades/fees, mandatory actions and stops. It independently calculates all metrics and checks all 28 earlier 47-month control curves. Missing required outcomes retain unavailable full-period metrics; favorable subsets are never used to rank arms.

The controller has prepare → independent input check → replay → independent replay check phases, exclusive locking and clean-boundary pause/resume. A consumed failure remains preserved. The 8 GiB worker ceiling, 45-minute phase ceiling, two-hour total ceiling and 4 GiB output ceiling bound this zero-fit replay. There is no artificial 7 GiB machine RAM claim or estimator-memory restriction; the measured peak is reported from the saved run.

The original controller's metadata-plan subcommand saved its valid plan and then hit an obsolete console-diagnostic key. The plan was validated before launching; no financial phase failed. Its sealed source bytes remain unchanged. Use the additive `scripts/seal_extension_backtest.py` planner for a fresh attempt, followed by the original controller's `run` command, with a new versioned output and plan path. Never overwrite this run or silently resume a consumed failure.

Private evidence and outputs are under `outputs/walkforward_extension_backtest/20261027-v1`; the plan is under `outputs/walkforward_extension_backtest/plans/20261027-v1.json`. The passive [notebook16](../notebooks/16_walkforward_extension_backtest.ipynb) authenticates independent proofs, saved aggregates and checked chart inputs. Run All builds, downloads, loads or fits nothing. Code and aggregate documentation belong in Git; data, models, raw evidence, generated plots and ledgers stay local. No main-branch merge is implied.

## Results

**Complete and independently verified: 48 daily paths / zero withheld, plus 16 monthly references.** All 28 prior control curves preserve the earlier 47-month results. The maximum same-cache, same-close month-end NAV difference over all 75 months and both zero/10 bp costs is `1.3322676295501878e-15` per unit starting NAV.

### Cumulative compounded total returns · February 2020–April 2026, 75 months

| Model / training | Top-down monthly gross | Daily same-close gross | Daily next-open gross | Daily next-open net, 10 bp | Daily net, 10 bp + 10% asset stops |
|---|---:|---:|---:|---:|---:|
| RF · 5-year | +40.81% | +40.81% | +42.45% | +24.67% | -12.37% |
| XGBoost · 5-year | +63.09% | +63.09% | +64.51% | +44.49% | -6.14% |
| RF · 10-year | -0.83% | -0.83% | +4.43% | -8.47% | -42.58% |
| XGBoost · 10-year | +95.23% | +95.23% | +98.46% | +74.04% | -8.72% |

All columns are whole-period cumulative compounded returns over **February 2020–April 2026, 75 months**. Annualized CAGR appears separately below. The top-down monthly column comes from the separate verified monthly-return backtest. Daily same-close uses matching prices, action treatment and timing. Daily next-open then changes execution timing. Net columns charge 10 bp one-way costs.

Top-down monthly refers to **realized portfolio returns**, including the approved execution-only cash/share entitlements for missing delisting endpoints. RET remains the expected-payoff ranking score used to choose stocks. These columns all hold the monthly-model forecasts and selections fixed; earlier executable-target and expanded-RF experiments remain separate.

The display-only [notebook17](../notebooks/17_full_period_monthly_daily_report.ipynb) shows these distinct columns and the saved charts. The matching eight-slide local report is `outputs/presentations/20261027_full_period_monthly_daily/Deep_Momentum_V2_Full_Period_Monthly_Daily_20261027.pptx`. Its editable tables and charts use the verified full-period aggregates. The original 47/48-month deck remains preserved.

### Annualized CAGR and daily risk · 10 bp one-way costs

| Arm | Stop | Cumulative total return | Annualized CAGR | Daily annualized Sharpe | Daily-close max drawdown | Stop exits |
|---|---|---:|---:|---:|---:|---:|
| RF · 5-year | None | +24.67% | +3.61% | 0.271 | -28.46% | 0 |
| RF · 5-year | 10% per asset | -12.37% | -2.10% | -0.027 | -41.65% | 594 |
| XGBoost · 5-year | None | +44.49% | +6.09% | 0.371 | -33.13% | 0 |
| XGBoost · 5-year | 10% per asset | -6.14% | -1.01% | 0.040 | -38.87% | 657 |
| RF · 10-year | None | -8.47% | -1.41% | 0.001 | -28.14% | 0 |
| RF · 10-year | 10% per asset | -42.58% | -8.53% | -0.502 | -50.09% | 657 |
| XGBoost · 10-year | None | +74.04% | +9.31% | 0.593 | -17.72% | 0 |
| XGBoost · 10-year | 10% per asset | -8.72% | -1.45% | -0.008 | -29.25% | 623 |

The unchanged asset-stop rule reduced net cumulative returns for all four arms. This is the observed fixed-rule comparison, not an optimized choice based on these evaluation outcomes. The notebook contains continuous annual slices, with eleven months in 2020 and four in 2026.

### Cost sensitivity · cumulative total returns over the same 75 months

| Arm | Stop | 0 bp | 5 bp | 10 bp | 25 bp |
|---|---|---:|---:|---:|---:|
| RF · 5-year | None | +42.45% | +33.27% | +24.67% | +1.99% |
| RF · 5-year | 10% per asset | +1.08% | -5.88% | -12.37% | -29.32% |
| XGBoost · 5-year | None | +64.51% | +54.18% | +44.49% | +18.83% |
| XGBoost · 5-year | 10% per asset | +8.27% | +0.81% | -6.14% | -24.30% |
| RF · 10-year | None | +4.43% | -2.23% | -8.47% | -24.97% |
| RF · 10-year | 10% per asset | -33.65% | -38.27% | -42.58% | -53.82% |
| XGBoost · 10-year | None | +98.46% | +85.86% | +74.04% | +42.80% |
| XGBoost · 10-year | 10% per asset | +5.41% | -1.90% | -8.72% | -26.50% |

The replay and independent checks used 550.50 active seconds, with 1.36 GiB observed peak worker RSS. Forty focused accounting/guard/reader tests passed. All four notebook code cells passed passive execution: four table displays, two chart displays, zero estimator imports, zero source-array/table reads and zero subprocesses. Notebook outputs remain clear.

Optional chart/table packages are isolated in the private `.reporting` folder. `requirements-reporting.txt` pins their versions. The fitted-model runtime and NumPy/pandas/RF/XGBoost versions remain unchanged. The saved report reader uses only the standard library; notebook display uses pandas and the optional formatting runtime.

The same-close equality establishes the consistency of these fixed baskets under matching inputs, clocks and action conventions. It does not establish prediction quality, complete source-identity certification, realizable same-close fills, complete historical knowledge-time provenance, or unrestricted stock borrow. Keep these source and execution limits in every interpretation.
