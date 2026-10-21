# Corrected-input walk-forward backtest

All results cover **2 January 2020–29 December 2023: four years, 48 monthly outcomes, 1,006 trading sessions**. NAV is continuous across annual refits; capital is never reset.
Independent verification retained all120 paths: **56 complete / 64 withheld**. A dash denotes unavailable performance, not a zero return.
RF expanded, the RET basket and the 10% per-asset stop remain the stated primary focus. The class1–10 and class2–9 baskets remain separate comparators. “Baseline” and “prior CV settings” are distinct frozen estimators; no new tuning or strategy selection is performed here.

## 10% per-asset stop: primary comparison

Gross is 0bp; net uses10bp per one-way traded notional. Borrow/funding are0 in this frozen proxy. Sharpe uses daily returns and a zero cash rate; CAGR annualizes the full1,006-session path. Drawdown is shown as a negative return.

| Model | Basket | Gross total | Net10bp total | Net CAGR | Net vol | Net Sharpe | Net max DD | Gross / net status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RF · Paper15 | ret | — | — | — | — | — | — | withheld / withheld |
| RF · Paper15 | class_1_10 | — | — | — | — | — | — | withheld / withheld |
| RF · Paper15 | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| XGBoost · Paper15 | ret | +39.41% | +27.16% | +6.20% | +21.17% | 0.39 | -25.56% | complete / complete |
| XGBoost · Paper15 | class_1_10 | +22.08% | +11.78% | +2.83% | +21.79% | 0.23 | -31.87% | complete / complete |
| XGBoost · Paper15 | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| RF · expanded baseline | ret | +0.18% | -8.35% | -2.16% | +18.90% | -0.02 | -31.01% | complete / complete |
| RF · expanded baseline | class_1_10 | +27.76% | +17.52% | +4.13% | +20.35% | 0.30 | -37.58% | complete / complete |
| RF · expanded baseline | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| XGBoost · expanded | ret | — | — | — | — | — | — | withheld / withheld |
| XGBoost · expanded | class_1_10 | +60.90% | +47.87% | +10.30% | +19.75% | 0.59 | -34.52% | complete / complete |
| XGBoost · expanded | class_2_9 | +13.29% | +3.49% | +0.86% | +11.50% | 0.13 | -14.61% | complete / complete |
| RF · expanded, prior CV settings | ret | +26.69% | +15.87% | +3.76% | +19.11% | 0.29 | -22.71% | complete / complete |
| RF · expanded, prior CV settings | class_1_10 | +38.13% | +27.11% | +6.19% | +20.41% | 0.40 | -37.58% | complete / complete |
| RF · expanded, prior CV settings | class_2_9 | — | — | — | — | — | — | withheld / withheld |

## Unstopped comparator

Gross is 0bp; net uses10bp per one-way traded notional. Borrow/funding are0 in this frozen proxy. Sharpe uses daily returns and a zero cash rate; CAGR annualizes the full1,006-session path. Drawdown is shown as a negative return.

| Model | Basket | Gross total | Net10bp total | Net CAGR | Net vol | Net Sharpe | Net max DD | Gross / net status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| RF · Paper15 | ret | — | — | — | — | — | — | withheld / withheld |
| RF · Paper15 | class_1_10 | — | — | — | — | — | — | withheld / withheld |
| RF · Paper15 | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| XGBoost · Paper15 | ret | +68.24% | +54.72% | +11.55% | +24.04% | 0.57 | -29.52% | complete / complete |
| XGBoost · Paper15 | class_1_10 | +54.64% | +43.73% | +9.51% | +23.73% | 0.50 | -29.69% | complete / complete |
| XGBoost · Paper15 | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| RF · expanded baseline | ret | +4.57% | -3.60% | -0.91% | +23.30% | 0.08 | -34.71% | complete / complete |
| RF · expanded baseline | class_1_10 | +49.08% | +39.75% | +8.75% | +21.78% | 0.49 | -35.49% | complete / complete |
| RF · expanded baseline | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| XGBoost · expanded | ret | — | — | — | — | — | — | withheld / withheld |
| XGBoost · expanded | class_1_10 | — | — | — | — | — | — | withheld / withheld |
| XGBoost · expanded | class_2_9 | — | — | — | — | — | — | withheld / withheld |
| RF · expanded, prior CV settings | ret | +58.69% | +46.29% | +10.00% | +23.25% | 0.52 | -23.61% | complete / complete |
| RF · expanded, prior CV settings | class_1_10 | +79.66% | +68.61% | +13.98% | +21.82% | 0.71 | -35.49% | complete / complete |
| RF · expanded, prior CV settings | class_2_9 | — | — | — | — | — | — | withheld / withheld |

## Trading-cost sensitivity

Each cell is the complete four-year total return and its availability. All four frozen cost assumptions and both variants are retained.

| Model | Basket | Stop | 0bp | 5bp | 10bp | 25bp |
|---|---|---|---:|---:|---:|---:|
| RF · Paper15 | ret | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · Paper15 | ret | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · Paper15 | class_1_10 | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · Paper15 | class_1_10 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · Paper15 | class_2_9 | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · Paper15 | class_2_9 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| XGBoost · Paper15 | ret | 10% | +39.41% (complete) | +33.15% (complete) | +27.16% (complete) | +10.72% (complete) |
| XGBoost · Paper15 | ret | none | +68.24% (complete) | +61.35% (complete) | +54.72% (complete) | +36.38% (complete) |
| XGBoost · Paper15 | class_1_10 | 10% | +22.08% (complete) | +16.82% (complete) | +11.78% (complete) | -2.12% (complete) |
| XGBoost · Paper15 | class_1_10 | none | +54.64% (complete) | +49.09% (complete) | +43.73% (complete) | +28.72% (complete) |
| XGBoost · Paper15 | class_2_9 | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| XGBoost · Paper15 | class_2_9 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · expanded baseline | ret | 10% | +0.18% (complete) | -4.18% (complete) | -8.35% (complete) | -19.84% (complete) |
| RF · expanded baseline | ret | none | +4.57% (complete) | +0.41% (complete) | -3.60% (complete) | -14.72% (complete) |
| RF · expanded baseline | class_1_10 | 10% | +27.76% (complete) | +22.54% (complete) | +17.52% (complete) | +3.64% (complete) |
| RF · expanded baseline | class_1_10 | none | +49.08% (complete) | +44.35% (complete) | +39.75% (complete) | +26.80% (complete) |
| RF · expanded baseline | class_2_9 | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · expanded baseline | class_2_9 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| XGBoost · expanded | ret | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| XGBoost · expanded | ret | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| XGBoost · expanded | class_1_10 | 10% | +60.90% (complete) | +54.25% (complete) | +47.87% (complete) | +30.23% (complete) |
| XGBoost · expanded | class_1_10 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| XGBoost · expanded | class_2_9 | 10% | +13.29% (complete) | +8.28% (complete) | +3.49% (complete) | -9.69% (complete) |
| XGBoost · expanded | class_2_9 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · expanded, prior CV settings | ret | 10% | +26.69% (complete) | +21.16% (complete) | +15.87% (complete) | +1.29% (complete) |
| RF · expanded, prior CV settings | ret | none | +58.69% (complete) | +52.37% (complete) | +46.29% (complete) | +29.41% (complete) |
| RF · expanded, prior CV settings | class_1_10 | 10% | +38.13% (complete) | +32.51% (complete) | +27.11% (complete) | +12.18% (complete) |
| RF · expanded, prior CV settings | class_1_10 | none | +79.66% (complete) | +74.05% (complete) | +68.61% (complete) | +53.26% (complete) |
| RF · expanded, prior CV settings | class_2_9 | 10% | — (withheld) | — (withheld) | — (withheld) | — (withheld) |
| RF · expanded, prior CV settings | class_2_9 | none | — (withheld) | — (withheld) | — (withheld) | — (withheld) |

## Annual returns: 10% stop, 10bp

These are slices of the same continuous NAV. If any part of a full path fails, every annual return for that path is withheld.

| Model | Basket | 2020 | 2021 | 2022 | 2023 | Status |
|---|---|---:|---:|---:|---:|---|
| RF · Paper15 | ret | — | — | — | — | withheld |
| RF · Paper15 | class_1_10 | — | — | — | — | withheld |
| RF · Paper15 | class_2_9 | — | — | — | — | withheld |
| XGBoost · Paper15 | ret | +48.51% | -6.39% | -8.92% | +0.43% | complete |
| XGBoost · Paper15 | class_1_10 | +28.79% | +2.00% | -15.18% | +0.32% | complete |
| XGBoost · Paper15 | class_2_9 | — | — | — | — | withheld |
| RF · expanded baseline | ret | +16.08% | -4.03% | -11.91% | -6.61% | complete |
| RF · expanded baseline | class_1_10 | +7.39% | -11.15% | +1.84% | +20.94% | complete |
| RF · expanded baseline | class_2_9 | — | — | — | — | withheld |
| XGBoost · expanded | ret | — | — | — | — | withheld |
| XGBoost · expanded | class_1_10 | +43.99% | +1.19% | -3.38% | +5.04% | complete |
| XGBoost · expanded | class_2_9 | +1.59% | +4.19% | -5.51% | +3.48% | complete |
| RF · expanded, prior CV settings | ret | +16.08% | +8.11% | -12.36% | +5.34% | complete |
| RF · expanded, prior CV settings | class_1_10 | +7.39% | -3.10% | +7.73% | +13.39% | complete |
| RF · expanded, prior CV settings | class_2_9 | — | — | — | — | withheld |

## Selection and quote coverage

| Model | Basket | Withheld selection months /48 | Names per side, min–max | Missing quote keys | Quality-flag keys | Retired new orders |
|---|---|---:|---:|---:|---:|---:|
| RF · Paper15 | ret | 0 | 10–10 | 24 | 0 | 0 |
| RF · Paper15 | class_1_10 | 0 | 10–10 | 21 | 1 | 0 |
| RF · Paper15 | class_2_9 | 0 | 6–10 | 39 | 0 | 0 |
| XGBoost · Paper15 | ret | 0 | 10–10 | 0 | 0 | 1 |
| XGBoost · Paper15 | class_1_10 | 0 | 10–10 | 0 | 0 | 1 |
| XGBoost · Paper15 | class_2_9 | 0 | 5–10 | 1 | 0 | 0 |
| RF · expanded baseline | ret | 0 | 10–10 | 0 | 0 | 0 |
| RF · expanded baseline | class_1_10 | 0 | 10–10 | 0 | 0 | 0 |
| RF · expanded baseline | class_2_9 | 0 | 7–10 | 20 | 0 | 0 |
| XGBoost · expanded | ret | 0 | 10–10 | 21 | 1 | 0 |
| XGBoost · expanded | class_1_10 | 0 | 10–10 | 0 | 0 | 0 |
| XGBoost · expanded | class_2_9 | 0 | 2–10 | 1 | 0 | 0 |
| RF · expanded, prior CV settings | ret | 0 | 10–10 | 0 | 0 | 0 |
| RF · expanded, prior CV settings | class_1_10 | 0 | 10–10 | 0 | 0 | 0 |
| RF · expanded, prior CV settings | class_2_9 | 0 | 4–10 | 41 | 0 | 0 |

RCM/DISCA OHLCV learning candidates failed frozen volume compatibility; no new learning bars admitted. Existing execution-only repairs remain separate.
Conditional historical proxy; identity knowledge-time, survivorship, investability and economic-return certification remain unresolved.
This passive report reads saved summary tables only. It does not fit, infer, replay, choose a new strategy, certify the source data, or claim improvement against an older run. Recorded stop/trade counts and turnover/cost totals are retained in JSON; failed paths expose no full-period or annual performance.
