# Continuous WFO accounting scenario

The latest report compounds all **1,760 sessions and 84 months** without equity resets. Two unquoted contingent-right valuations use a disclosed conservative convention; they have not become observed prices.

Long rights are marked zero; short rights are reserved at the full contractual payout cap. Cash consideration is recognized. No subsequent claim payouts, financing, revaluations or reserve releases are credited. Rights are retained in a separate register and no sale is assumed.

| Strategy | Total return* | CAGR* | Max drawdown* | Sharpe* |
| --- | ---: | ---: | ---: | ---: |
| LongOnly | 54.52% | 6.43% | -16.40% | 0.766 |
| MACD | -3.66% | -0.53% | -8.52% | -0.157 |
| LinReg | 48.91% | 5.87% | -16.18% | 0.746 |
| GMOM | 46.82% | 5.65% | -16.05% | 0.733 |
| RegCombo | 45.46% | 5.51% | -15.93% | 0.728 |
| SignCombo | 47.87% | 5.76% | -16.12% | 0.740 |

*Conditional accounting-scenario metrics, before trading costs. The sensitivity below varies only the two initial rights marks and shares the assumption of zero subsequent claim P&L.

| Stock | Assumed date | GMOM right mark | GMOM portfolio-day return |
| --- | --- | ---: | ---: |
| ABMD | 2022-12-22 | $35.00 | -0.03224% |
| WBA | 2025-08-28 | $0.00 | -0.12456% |

GMOM CAGR across all four event-mark combinations is **5.6523%–5.6618%**. These are not bounds on actual wealth with later rights cash flows.

The other 1,758 dates, all portfolio weights, forecast cohorts, features and fitted models are unchanged. The two raw stock labels remain unavailable in the source audit.

[Review the continuous curves and tables](../outputs/continuous_wfo/README.md). The [earlier valuation audit](WFO_VALUATION_REPAIRS.md) preserves the two source gaps and the preceding 60 recoveries.

*All path metrics are conditional on the declared contingent-right accounting scenario, including subsequent claim P&L being zero. Two source valuations are still unobserved; those dates are visibly flagged as assumptions.

Cash consideration is recognized at face value. Long non-traded rights are marked zero; short obligations are reserved at their full contractual payout cap. The rule follows position sign, never whichever mark improves test performance.

The claim register retains the rights/obligations. No right is assumed sold. Later payouts, fair-value changes, reserve releases and financing are excluded; this is not a fully reconciled realized cash-flow ledger.

The four alternative zero/face event-mark combinations use the same exclusion of subsequent claim P&L. Their envelope measures only the two initial-mark assumptions and is not a bound on actual strategy wealth with later cash flows.

All 1,758 observed or previously recovered portfolio days, every forecast, position, cohort and original model are unchanged. No stock label is invented and no whole portfolio return is replaced by zero, dropped or reweighted.

Equity starts at 100 once and compounds all 1,760 scheduled daily returns. Drawdown uses the running high-water mark including initial capital. All 84 monthly returns compound every scheduled session.

CAGR annualizes with 252 trading sessions. Sharpe uses sample daily volatility and zero risk-free rate. Sortino uses a zero return threshold. LongOnly is the inverse-volatility-sized cohort, not a capitalization-weighted index.

The existing research-proxy limits remain: executable fills, trading costs, borrow/financing, source identity certification and complete training-feature repairs are not supplied by this report.

## Reproduce

```bash
python continuous_walk_forward.py
python verify_continuous_wfo.py
python report_continuous_wfo.py
python export_continuous_wfo.py
```

Existing versioned output directories are preserved; choose fresh paths when rebuilding. Viewing the compact export does not require raw data, but independent reconstruction requires the pinned local sources.
