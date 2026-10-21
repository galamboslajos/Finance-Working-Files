Verified saved prediction diagnostics for January2020–December2023 (48 outcome months).
All five arms use the same prediction and observed-label source/signal keys. Monthly return classes are unchanged; class1 is best and class10 worst.

| Model | Pooled log loss | Equal-month log loss | Pooled accuracy | Equal-month accuracy | Label coverage |
|---|---:|---:|---:|---:|---:|
| RF · Paper15 | 2.283152 | 2.283136 | 12.34% | 12.34% | 99.85% |
| XGBoost · Paper15 | 2.284890 | 2.284871 | 12.13% | 12.13% | 99.85% |
| RF · expanded | 2.268530 | 2.268524 | 13.26% | 13.26% | 99.85% |
| XGBoost · expanded | 2.268343 | 2.268338 | 13.40% | 13.40% | 99.85% |
| RF expanded · prior CV settings | 2.268729 | 2.268725 | 13.28% | 13.28% | 99.85% |

Pooled weights each observed stock-month equally. Equal-month weights each observed month equally. Label coverage is observed outcomes divided by prediction-eligible stock-months; missing outcomes stay in that denominator.

| Model | Year | Observed / predicted | Pooled log loss | Equal-month log loss | Accuracy |
|---|---:|---:|---:|---:|---:|
| RF · Paper15 | 2020 | 10,502 / 10,518 | 2.275753 | 2.275747 | 12.56% |
| RF · Paper15 | 2021 | 10,483 / 10,502 | 2.283281 | 2.283275 | 12.12% |
| RF · Paper15 | 2022 | 10,547 / 10,566 | 2.288551 | 2.288561 | 12.13% |
| RF · Paper15 | 2023 | 10,644 / 10,655 | 2.284974 | 2.284961 | 12.55% |
| XGBoost · Paper15 | 2020 | 10,502 / 10,518 | 2.280179 | 2.280168 | 12.51% |
| XGBoost · Paper15 | 2021 | 10,483 / 10,502 | 2.280882 | 2.280874 | 12.29% |
| XGBoost · Paper15 | 2022 | 10,547 / 10,566 | 2.291890 | 2.291906 | 11.83% |
| XGBoost · Paper15 | 2023 | 10,644 / 10,655 | 2.286550 | 2.286534 | 11.90% |
| RF · expanded | 2020 | 10,502 / 10,518 | 2.275863 | 2.275867 | 12.75% |
| RF · expanded | 2021 | 10,483 / 10,502 | 2.258868 | 2.258874 | 14.04% |
| RF · expanded | 2022 | 10,547 / 10,566 | 2.270105 | 2.270116 | 13.29% |
| RF · expanded | 2023 | 10,644 / 10,655 | 2.269250 | 2.269239 | 12.97% |
| XGBoost · expanded | 2020 | 10,502 / 10,518 | 2.273798 | 2.273795 | 13.58% |
| XGBoost · expanded | 2021 | 10,483 / 10,502 | 2.259214 | 2.259219 | 14.45% |
| XGBoost · expanded | 2022 | 10,547 / 10,566 | 2.272541 | 2.272556 | 12.35% |
| XGBoost · expanded | 2023 | 10,644 / 10,655 | 2.267792 | 2.267781 | 13.23% |
| RF expanded · prior CV settings | 2020 | 10,502 / 10,518 | 2.275863 | 2.275867 | 12.75% |
| RF expanded · prior CV settings | 2021 | 10,483 / 10,502 | 2.259916 | 2.259923 | 13.89% |
| RF expanded · prior CV settings | 2022 | 10,547 / 10,566 | 2.269933 | 2.269946 | 13.19% |
| RF expanded · prior CV settings | 2023 | 10,644 / 10,655 | 2.269175 | 2.269166 | 13.28% |

| Model | Class1 P / R (actual n) | Class2 P / R (actual n) | Class9 P / R (actual n) | Class10 P / R (actual n) |
|---|---:|---:|---:|---:|
| RF · Paper15 | 15.19% / 27.24% (4,215) | 10.23% / 5.33% (4,220) | 11.07% / 5.36% (4,215) | 15.37% / 19.95% (4,220) |
| XGBoost · Paper15 | 15.72% / 23.82% (4,215) | 10.42% / 7.09% (4,220) | 10.72% / 6.71% (4,215) | 15.19% / 21.94% (4,220) |
| RF · expanded | 15.41% / 33.31% (4,215) | 11.60% / 6.23% (4,220) | 10.88% / 9.80% (4,215) | 15.70% / 28.89% (4,220) |
| XGBoost · expanded | 16.50% / 34.02% (4,215) | 10.57% / 6.78% (4,220) | 9.36% / 6.45% (4,215) | 16.38% / 30.69% (4,220) |
| RF expanded · prior CV settings | 15.33% / 32.62% (4,215) | 10.66% / 5.40% (4,220) | 10.33% / 9.56% (4,215) | 15.93% / 29.93% (4,220) |

These are ten-class mode precision and recall, with separate classes and no merged tails. They are not top-K basket statistics or portfolio returns.
Tail precision/recall are aggregated from pinned saved row scores and cross-checked against saved mode tables, following the independently verified upstream probability lineage. This reporter does not reconstruct per-row scores from probabilities.
The fifth arm uses prior annual CV settings, newly fitted to the repaired inputs. No new parameter search or repaired-input CV selection was performed.
Inputs retain the provisional BK/XOM provider-adjusted endpoint policy. The proposed RCM/DISCA learning repairs were rejected by frozen volume controls; earlier execution repairs remain separate.
Historical knowledge-time, provider adjustment vintage, terminal-return and survivorship limitations remain provisional. Passing these checks does not certify investable performance.
Old runs are preserved. This report makes no paired-cohort improvement claim against historical runs and launches no RET scoring, model fitting or backtest.
