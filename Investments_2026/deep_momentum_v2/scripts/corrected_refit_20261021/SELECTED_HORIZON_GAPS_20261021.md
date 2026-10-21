# Newly selected-horizon quote diagnostics

These are descriptive metadata observations from the corrected forecasts, before performance reporting. The saved preflight checks the maximum unstopped holding horizon for both stop variants. Actual path completeness is established only by replay and independent verification; an earlier stop can remove a later quote requirement.

All 15 model/basket selection sets have zero unavailable selection months. Eight sets have at least one missing or quality-flagged price key. Across sets there are 168 missing episode rows, 127 distinct native/date keys and nine native IDs, plus two quality-flag episode rows referring to the same additional native/date key. Selection remains unchanged and no position is dropped.

| Descriptive provider ticker / cohort name | Native ID | Required missing dates (inclusive) | Distinct missing dates | Affected model/basket |
|---|---|---|---:|---|
| PBCT / People's United Financial | sec_gBbwwg | 2022-04-04–2022-05-02 | 20 | RF expanded baseline and prior-CV, class2/9 |
| EV / Eaton Vance | sec_yRvj1y | 2021-03-01–2021-04-01 | 24 | RF Paper15, RET |
| NBL / Noble Energy | sec_y3QbRX | 2020-10-05–2020-11-02 | 21 | RF Paper15 class1/10; XGBoost expanded RET |
| HRC / Hill-Rom | sec_gjv90g | 2021-12-13–2022-01-03 | 15 | RF Paper15, class2/9 |
| PSB / PS Business Parks | sec_gNZrEg | 2022-07-20–2022-08-01 | 9 | RF Paper15, class2/9 |
| WWE / World Wrestling Entertainment | sec_yRLZxX | 2023-09-12–2023-10-02 | 15 | RF Paper15, class2/9 |
| WRI / Weingarten Realty | sec_gBDMez | 2021-08-04–2021-09-01 | 21 | RF expanded prior-CV, class2/9 |
| FBIN / Fortune Brands Innovations | sec_z5xK8O | 2022-12-14 | 1 | XGBoost expanded, class2/9 |
| HWM / Howmet Aerospace | sec_XnJnDN | 2020-03-31 | 1 | XGBoost Paper15, class2/9 |

NBL also has an unaccepted source-metadata key on 2020-10-02, affecting two episodes. A missing quote is not evidence of an exchange session, a company failure or an entitlement. Published cohort names and provider identifier-resolution evidence are descriptive labels; they do not independently certify historical identity, a merger/spin date, or a cash/stock entitlement. The longer post-date gaps need dated legal-action/terminal-route review as well as quote review, rather than fabricated bars or ticker substitution. The single-day cases need their own observed-price/action evidence.

Evidence: plans/root_preflight_summary_read_v1.json and plans/root_preflight_gap_read_v1.json bind the charged saved-diagnostic reads; run/verification/root_preflight_gap_read_v1.json preserves exact episode rows. plans/gap_identity_labels_v1.json and run/verification/gap_identity_labels_v1.json bind the bounded 4/5-column metadata lookup. All nine native/source pairs returned one descriptive Siblis row and one Intrinio evidence row each. No price values, memberships or model outputs were amended by that lookup.

The next execution-data repair cycle should obtain independent dated official action evidence or observed native quotes; check predecessor/successor identity and units, cash/stock/distribution entitlements, and effective timing; freeze a new versioned execution manifest; and replay the same saved selections with unchanged trading rules. No zero terminal return, forced liquidation, backfill, held-position removal, relaxed quality threshold or model refit follows automatically from this list. A separate learning/label/RET-payoff impact audit must precede any claim that those inputs are fully repaired. Source/knowledge-time and provisional identity limitations remain.
