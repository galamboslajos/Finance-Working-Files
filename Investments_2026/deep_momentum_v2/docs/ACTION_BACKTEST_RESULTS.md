# Corporate-action rerun coverage and remaining data issues

VERIFIED LOCAL, 2026-10-15. The fixed 2018–2019 rerun completed execution of all
64 scenarios under `configs/action_backtest.json`. Saved run:
`outputs/action_backtest/2018-2019-b088d2328a8aea03`, build identity
`sha256:b088d2328a8aea033547fd5e03a076160276e2f50a9ffe7983bda34aeec670ce`.
This is an accounting/coverage result, **not a valid strategy-performance comparison**.

## What changed and what was verified

- All three original OAS/MNK/MPWR failures have independent actual raw OHLC
  overlays. Their original raw closes agree exactly across vendors.
- Additional compatible DF and PCG bars were repaired. Four missing FRT/FIVE/WCG
  bars were appended from actual raw quotes with separately labelled provisional
  adjustment-unit transport. Original price rows were not overwritten.
- Eight verified corporate-action definitions are available to the engine;
  only events encountered by an actually held lot are applied. Signed cash/stock
  entitlements, short liabilities, successor stop basis and retired new orders
  are handled separately from ordinary market trading.
- All eight selected-stock files and eight rebalance schedules match the original
  bytes. No stocks were dropped, reranked or replaced. The ten intended slots per
  side, 50%/50% weights, stops and costs remain unchanged. No models were refitted.
- Original daily, feature, target and pilot files and the first backtest pointer
  are unchanged. RET remains withheld at all 24 signals. No 2020–2023 financial
  observations were returned or used in this work. Remote mixed-date Parquet
  blocks may transiently decode excluded observations inside Arrow; this physical
  limitation is disclosed in the projection receipts and is distinct from analysis.

## Continuous-path coverage

| Result | Cost-inclusive scenarios | First blocking issue |
| --- | ---: | --- |
| Complete adjusted-price proxy ledger | 12 | No missing daily valuation encountered |
| Withheld | 16 | 2019-04-11: missing correctly identified Ensco Rowan successor quotes |
| Withheld | 32 | 2019-08-12: unresolved ACC/K/SO price disagreement or BBBY identity conflict |
| Withheld | 4 | 2019-11-12: Dean Foods suspension/OTC valuation gap |

The 12 arithmetically complete paths are XGBoost/Paper15 class 2/9 (both stop
variants, all four cost assumptions) and RF/expanded class 1/10 (stop variant,
all four cost assumptions). They must not form a conditional model leaderboard.

## Known historical BBBY identity conflict

The old Bed Bath & Beyond source route is linked to native `sec_ydbQnj`, whose
downloaded identifier evidence is `US6903701018`. A contemporaneous
[SEC filing identifies CUSIP 690370101 as Overstock.com](https://www.sec.gov/Archives/edgar/data/1130713/000110465919008562/a19-4181_30sc13g.htm).
This establishes a historical issuer conflict, beyond an unexplained rounding
difference. Current ticker reuse cannot establish continuity with old BBBY.

The route occurs in 71 original selected stock-months across the four class 1/10
arms: 15 RF/Paper15, 16 XGBoost/Paper15, 20 RF/expanded and 20 XGBoost/expanded.
Thus the four completed RF/expanded paths have direct exposure to this conflict.
Class 2/9 has no direct selected exposure, but the route may also affect previous
features, cross-sectional statistics, targets, training and predictions; the full
upstream impact has not been measured. Arithmetic completion is not strategy
validation. Headline model-performance interpretation is withheld pending a
separately versioned identity correction and affected rebuilds. No frozen cohort
or model was changed as part of this audit.

The complete exact-key metadata audit is private at
`data/derived/daily_action_inputs/known_identity_conflict_exposure.json`.
The [remaining-data request](REMAINING_BACKTEST_DATA_REQUEST.md) is prepared to
forward; it has not been sent.

## Verification

All **883 tests passed**. An independent saved-ledger replay checked all 64
scenarios, 51,920 valid open/close marks, 44,592 lot/market instructions, 56
signed event applications, 25,512 fixed-weight orders and 3,992 stop fills.
These counts repeat across cost/variant scenarios; they are not unique trades
or unique real-world corporate actions. Eight retired order slots stayed in cash.
The replay independently checks actual ordinary fill prices, raw-share event
units, signed cash, net-order fee groups, holdings, NAV and failed-path metric
suppression. It does not certify historical security identity or vendor economics.

The run took 50.83 seconds with 1.37 GiB peak RSS, within its local budget.
An earlier engineering attempt (`2018-2019-eb491e56d691d0af`) rejected every
first-day quote because a supplemental acceptance column introduced nulls on
original rows. That bug was fixed and covered by three regression tests. The
attempt is retained for audit, has no valid P&L and is superseded by the run above.

Repeat the passive report with
`PYTHONPATH=src .venv/bin/python -B -m dmv2.action_backtest report`.
Repeat the independent replay with
`PYTHONPATH=src .venv/bin/python -B scripts/verify_action_backtest.py .`.
Master section 19 reads verified saved coverage only and does not rerun accounting.
