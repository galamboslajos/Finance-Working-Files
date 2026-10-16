# Source-quality review and versioned repair — October 16, 2026

This review distinguishes incorrect observations and issuer routes from legitimate extreme returns. It preserves the existing experiments as references. It does not authorize removing suspicious returns, selecting stocks by future quote availability, changing model parameters, or presenting an unretrained input amendment as a corrected strategy result. The repaired experiment requires its own source pins, dependent input rebuild, model fits, daily replay and verification receipts before new performance is available.

The effective-session screen covers **3,792,168 dated observations** in the bounded cached research inputs, including the earlier daily history required for feature warmup. It observes **1,440 native IDs out of 1,531 requested IDs**. The remaining **91 IDs have no observation in these bounded caches**; this is not a finding that their historical prices are unavailable everywhere. This scope covers the project's saved research inputs, rather than every object in Google Cloud or an independently sourced market-wide history. No protected 2024+ market observations were returned or analyzed.

The screen produces **9,797 flag records, 3,490 diagnostic clusters and 1,029 flagged native IDs**. These are review counts, not error counts. A security/date can have several reasons; overlapping spans can describe the same event. The screen includes existing rejected or unusable source states, return jumps, abrupt changes followed by reversals, and other candidate anomalies. A flagged observation is not necessarily an input used by a fitted model or an exposure actually held. Original monthly cohorts, dated daily cohorts, feature lookbacks and potential holding horizons require separate intersection checks. Warmup dates outside a security's current membership can still affect later indicators.

| Finding | Evidence and disposition |
| --- | --- |
| OPCH price contamination | Fifteen 2022 daily observations contain a materially different price regime from independently acquired actual raw SIP bars. This includes a false large opening gap that entered the stopped backtest. The separate repair replaces raw/adjusted OHLC only after identity, date, frequency and paired-unit checks. Original values remain in audit evidence. |
| OPCH volume | The contaminated observations also contain unreliable volume. Even on good OHLC controls the two feeds have different volume coverage, so their volumes cannot be treated as interchangeable. On the fifteen confirmed contaminated bars, volume and adjusted volume become unavailable; original values remain in audit. Good original volumes remain unchanged. No scale factor, interpolation or blended feed volume is invented. |
| El Paso versus Empire Petroleum | The original historical El Paso route points to the native history of a different issuer using the same ticker. Historical El Paso common stock is CUSIP **28336L109**; Empire Petroleum's later EP listing uses **292034303**. The correct El Paso native history is already downloaded. The separate route amendment must preserve membership rows, producer review flags and the original route evidence, and must requalify joins rather than silently editing the published mapping. |
| LNC and HIG crisis moves | The bounded March 2020 checks confirm all ten raw closing prices against independent SIP observations. All five LNC OHLC bars match exactly. HIG has two small opening-price feed differences; its extreme closing moves match. These observations remain unchanged. A large fall followed by a large rebound is not sufficient evidence of corruption. |
| KEM / KEM (KEMP) | The dated source cohort contains both KMPR and KEM (KEMP) as separate S&P 400 members. Routing the ambiguous source to the KMPR native would create a same-date collision. The source company/alias interpretation remains unresolved; no automatic native override or price repair is justified. Matching prices from two vendors queried with KEM would not establish that the intended issuer was Kemper. |
| Other flagged windows | Ten of the thirteen fixed comparison windows precede the pinned independent SIP products' declared 2016 start. Their absence from those products is a coverage limit, not proof that the original observations are wrong. Colonial Bancgroup's later CBCGQ quote tail also requires distinction from its earlier index membership; post-membership OTC observations are not automatically invalid. No flags trigger automatic exclusions or manufactured exits. |

The OPCH review checked **1,256 original native/date observations** over the required cached span, January 4, 2019 through December 29, 2023. Independent actual SIP observations cover **985 dates**, February 3, 2020 through December 29, 2023. Among these, 964 match original OHLC within the declared $0.0002 comparison tolerance, fifteen show the confirmed contamination, and six have small feed differences that were not promoted to confirmed corruption. The earlier 271 BIOS-period observations remain outside this independently acquired quote coverage. These counts do not certify every original observation or every engineered feature.

The OPCH / BIOS distinction is material to units. Nasdaq documents the **one-for-four reverse split, BIOS-to-OPCH ticker change and new CUSIP 68404L201 effective February 3, 2020**. The original source's paired adjusted/raw units change across that event. A single unit-one multiplier must not be applied to all earlier history, and a ticker change must not be guessed from the 2019 corporate-name history. The confirmed 2022 replacement targets have preceding and following full-OHLC matching controls, unit-one paired multipliers and zero following-control residual. These checks establish the bounded transport of the actual replacement observations; they do not independently certify the complete corporate-action feed. [Nasdaq notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2020-21), [issuer SEC filing](https://www.sec.gov/Archives/edgar/data/1014739/000110465920009797/tm206375d1_8k.htm).

El Paso's identifier is documented in a contemporaneous SEC-filed voting record for its May 17, 2011 meeting. Empire's own 2022 annual report identifies its distinct EP listing and CUSIP beginning March 8, 2022. These are documented primary-source issuer/identifier findings, **not authenticated raw provider responses**. The correct downloaded El Paso candidate contains **1,111 dated source records: 1,107 XNYS sessions and four non-session observations**. Non-session records remain source evidence and cannot become trading-session quotes. Historical monthly and daily collision checks accompany the route review. A correct native route does not supply missing terminal consideration or authorize a synthetic month-end quote after retirement. [Historical El Paso identification](https://www.sec.gov/Archives/edgar/data/225777/000092881611001038/a_43npx.htm), [Empire issuer identification](https://www.sec.gov/Archives/edgar/data/887396/000107261323000285/empire_form10k123122.htm).

The separate identity metadata review covers **1,600 source/native/unified route cases**. Of these, 145 have a valid declared source ISIN, 140 have a valid producer-link query ISIN, and 743 have a valid ISIN in the retained native-master lookup path. These counts overlap and must not be added. None independently proves the ISIN returned in an original provider response or its dated historical assignment: the reviewed schema has **zero independently returned-provider ISIN cases**. Query-type/literal disagreements and differences among aliases are review evidence, not automatic proof of a wrong issuer. Primary-source legal and identifier evidence is required to distinguish genuine issuer conflicts from legitimate share-class, ticker and corporate-action changes.

The repair retains the historical S&P 500/400 membership keys and original denominators. Paper15 definitions, individual lookbacks, the 80% context policy, training-only missingness/preprocessing rules, ten-class target construction, rolling ten-year annual fits, monthly prediction timing, balanced baskets and stop rules stay fixed. Corrected prices and routes can legitimately change feature values, signal-time qualification, cross-sectional statistics and target classes. Those changes require an explicit dependency audit and recomputation, not a claim that all numeric Paper15 values or every prediction row must remain identical. Missing outcomes remain missing and do not remove forecast rows; a repaired route does not authorize using future labels for eligibility.

Daily inputs feed both past-only feature calculations and the separate execution view. Consequently, fixing only a backtest fill would leave potentially contaminated predictors and labels in place. The versioned rebuild must follow affected own-price features, shared daily context, month-end endpoints, cross-sectional class boundaries, training preprocessing, fitted models and forecasts. Refit only under the frozen experimental parameters; no result-driven tuning, trimming of adverse returns or replacement of selected stocks is part of this correction. Earlier experiments and unsuccessful paths remain preserved with their original source identities and limitations.

Independent arithmetic and inference receipts establish that saved models reproduce saved probabilities and that the declared cash/units rules reconcile. They do not establish correct source history. The existing independent feature verifier recomputes all Paper15 variables and shared target classes, but independently samples only six daily formulas; it does not certify every expanded formula. These sources support a historical reconstruction. Provider adjustment vintage, original response-body authentication, complete dated identity, delisting/distribution accounting, borrow availability and historical knowledge-time certification remain limited. No 100% clean-data, leakage-free or survivorship-free guarantee follows from the screen or the repaired subset. Current corrected-model performance remains unavailable until the required refit, replay and their separate audits complete.

## Implemented local handoff

The confirmed corrections are installed as an additive package at
`data/derived/data_repairs/source-quality-20261016-v1`. Frozen source objects,
earlier annual outputs and model/backtest runs retain their original bytes.
New workers are `quality_audit.py`, `opch_amendment.py`, `el_paso_route.py` and
`source_repair_loading.py`. The latter composes the reviewed corrections after
the existing quote bundle, preserves the old cache whitelist and admits the
correct El Paso history through a separate pinned source. It refuses automatic
reapplication. It supplies no fetching, model fitting or synthetic exit.

The corrected input rebuilds are complete for **2010–2013 and 2022–2023**.
The other eight annual inputs remain pinned references. The six rebuilt views
preserve all original membership keys and keep feature history separate from
daily outcome/execution data. The El Paso retirement outcome remains missing.
Paper15 and all expanded formulas retain their original definitions; repaired
inputs legitimately change their values, shared statistics and target ranks.
Private per-column impact ledgers and independent verification receipts are in
`outputs/source_quality/confirmed-repairs-20261016-v1`.

These are the primary **80%** context views. The original **90%** sensitivity
outputs remain references and have not been rebuilt with these corrections;
do not compare corrected-primary values with old-sensitivity values as though
they share the same source history. The separately admitted correct El Paso
native also passed the unchanged quality screen over its full required span:
1,107 XNYS observations, no missing sessions or invalid OHLC/factor/duplicate
flags. Eight raw/adjusted roundtrip flags describe four 2008 warmup dates and
three review episodes. They remain unverified review candidates, not errors
or exclusions. Four source observations on closed sessions remain outside
the numerical trading-session view.

The focused new suite passed **107 tests**, and the existing loader, feature,
monthly, target, identity and preprocessing regression selection passed
**143 tests**. Synthetic tests are structural evidence; the independent actual
input review is recorded separately. No model was fitted and no backtest was
replayed in this source-repair phase. All **16 saved 2020–2023 model fits and
their dependent forecasts/performance are stale for the corrected experiment**.
The earlier four-arm 2018–2019 development run also needs retraining if it is
to be regenerated with the repaired 2010–2013 history. Keep both original runs
as dated references. Do not combine their old forecasts with new source data
and call that a corrected full-model result.

These source corrections do not supply the previously unresolved CIT, WBD,
GRUB, FBIN or VYX execution marks or terminal entitlements. The old 28-complete /
36-withheld count belongs to its saved forecast selections. A fresh selected-
holding-horizon preflight must establish the coverage of new forecasts; neither
the old count nor the complete-only subset is a valid new model comparison.

Read the pinned handoff from the V2 directory without fitting or loading models:

```python
from pathlib import Path
from dmv2.source_quality_reporting import read_source_quality_repair

repair = read_source_quality_repair(Path.cwd())
print(repair["status"])
print(repair["summary"])
```

Launch Python with `PYTHONPATH=src .venv/bin/python` from this directory.
The handoff verifies saved file hashes and reports `REPAIRED_INPUTS_READY_MODELS_STALE`.
It does not read financial columns, download data, fit preprocessing or start
a model/backtest. `corrected_annual_sources.json` records the exact fourteen
annual references, and `handoff.json` records the sixteen affected walk-forward
fits. The next economic comparison must fit new training-only preprocessing
and all four frozen estimator arms for each fold, then generate monthly
forecasts and replay the unchanged baskets, costs and per-asset stop rules.
No parameter, loss-function or selection change belongs in that correction.
