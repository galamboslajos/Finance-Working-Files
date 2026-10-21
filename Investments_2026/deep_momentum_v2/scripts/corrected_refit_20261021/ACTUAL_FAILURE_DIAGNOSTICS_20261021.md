# Actual first-failure diagnostics

All120 paths have independent verification. This file reads only the saved date/identity/reason metadata from their failure tables. It contains no price values or partial-path performance. Each grouped row lists the exact cost scenarios with that same first failure.

| Model | Basket | Stop variant | First failure | Reason | Native ID | Costs, bp |
|---|---|---|---|---|---|---|
| rf_extended | class_2_9 | asset_stop_10pct | 2022-04-04T00:00:00 | missing_quote | sec_gBbwwg | 0,5,10,25 |
| rf_extended | class_2_9 | unstopped | 2022-04-04T00:00:00 | missing_quote | sec_gBbwwg | 0,5,10,25 |
| rf_paper15 | class_1_10 | asset_stop_10pct | 2020-10-02T00:00:00 | source_quality_flag | sec_y3QbRX | 0,5,10,25 |
| rf_paper15 | class_1_10 | unstopped | 2020-10-02T00:00:00 | source_quality_flag | sec_y3QbRX | 0,5,10,25 |
| rf_paper15 | class_2_9 | asset_stop_10pct | 2021-12-13T00:00:00 | missing_quote | sec_gjv90g | 0,5,10,25 |
| rf_paper15 | class_2_9 | unstopped | 2021-12-13T00:00:00 | missing_quote | sec_gjv90g | 0,5,10,25 |
| rf_paper15 | ret | asset_stop_10pct | 2021-03-01T00:00:00 | missing_quote | sec_yRvj1y | 0,5,10,25 |
| rf_paper15 | ret | unstopped | 2021-03-01T00:00:00 | missing_quote | sec_yRvj1y | 0,5,10,25 |
| rf_tuned_extended | class_2_9 | asset_stop_10pct | 2021-08-04T00:00:00 | missing_quote | sec_gBDMez | 0,5,10,25 |
| rf_tuned_extended | class_2_9 | unstopped | 2021-08-04T00:00:00 | missing_quote | sec_gBDMez | 0,5,10,25 |
| xgb_extended | class_1_10 | unstopped | 2022-10-03T00:00:00 | selected_provider_held_source_collision | sec_ylkLlz | 0,5,10,25 |
| xgb_extended | class_2_9 | unstopped | 2022-12-14T00:00:00 | missing_quote | sec_z5xK8O | 0,5,10,25 |
| xgb_extended | ret | asset_stop_10pct | 2020-10-02T00:00:00 | source_quality_flag | sec_y3QbRX | 0,5,10,25 |
| xgb_extended | ret | unstopped | 2020-10-02T00:00:00 | source_quality_flag | sec_y3QbRX | 0,5,10,25 |
| xgb_paper15 | class_2_9 | asset_stop_10pct | 2020-03-31T00:00:00 | missing_quote | sec_XnJnDN | 0,5,10,25 |
| xgb_paper15 | class_2_9 | unstopped | 2020-03-31T00:00:00 | missing_quote | sec_XnJnDN | 0,5,10,25 |

64 scenarios have a saved failure record. Full-period and annual performance for those paths remains unavailable. Holdings and unvalued lots are retained, not dropped. Descriptive security names and the complete maximum-horizon gap list are in SELECTED_HORIZON_GAPS_20261021.md. Failure metadata alone does not independently establish a legal corporate-action entitlement.

Evidence: run/verification/root_failure_metadata_v1.json SHA256 74cbb2012efdf32e7e9ee62bdebef1b46b6a20ca8812385a0c09428598fe826c; final independent proof SHA256 a0374f97f6193969f741cd2f59b19ea8e367eb851bdd48276b6763c04fe56aea.
