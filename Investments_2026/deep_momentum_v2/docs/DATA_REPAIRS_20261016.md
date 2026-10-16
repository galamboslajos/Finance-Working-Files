# Targeted 2018–2019 data repairs — October 16, 2026

This pass prepares corrected quote inputs and narrows the remaining source
requests. Original releases, panels, models, selections, backtests and output
pointers remain reference evidence. No model was fitted, no backtest was rerun
and no 2020–2023 financial evaluation was performed.

## Current disposition

| Item | Verified progress | Remaining dependency |
| --- | --- | --- |
| Historical BBBY | Known wrong route disabled by the separate October 16 identity amendment; fresh organizational and curated-source checks complete | Correct old-issuer provider identity, original lookup evidence and matched historical prices including pre-2010 warmup |
| ACC/K/SO, August 12, 2019 | Separate full-OHLC replacement adapter and pinned SIP evidence prepared; preceding valid units checked against a following valid control | Consume the reviewed amendment in a future separately versioned run; original runs have not changed |
| Rowan → Ensco Rowan | Successor security and required share basis remain explicit | Actual April 11–May 1, 2019 successor ordinary-share quotes, CUSIP G3166L100 |
| Dean Foods | November 13 OTC start independently documented; all eleven locally available bars already match the existing cache | Actual November 18–21 OTC bars and a separately stated November 12 halt valuation/execution treatment |

The private evidence package is
`data/derived/data_repairs/2018-2019-resolution-20261016-v1/`.
It contains small, selected 2019 quote evidence, unchanged-source receipts,
Cloud metadata receipts, the narrowed terminal-gap review and reproducible
audit scripts. It is excluded from Git.

## ACC/K/SO: a separate reviewed full-OHLC amendment

The three original August 12 bars were already rejected for invalid raw and
adjusted OHLC bounds. Their closes also disagree with independent raw SIP bars.
Their internal raw/adjusted proportionality does not establish correct prices;
changing only an adjustment multiplier cannot reconcile the vendor closes.

`src/dmv2/bracket_ohlc_repairs.py` applies only the three named original rejected
keys. It replaces open/high/low/close with actual pinned raw SIP values, and
transports those prices into the existing adjusted proxy units using the
**preceding August 9 valid close pair**. August 13 is an independent validation
control; its values do not determine the multiplier. Both control bars' complete
raw OHLC must match the SIP source. Target and control pins, frequency, action
fields, source metadata, dates and scope are checked before acceptance.

| Security | Preceding multiplier | Maximum adjusted control residual |
| --- | ---: | ---: |
| ACC | 0.8964770053475936 | < $0.000045 |
| K | 0.7531360566098424 | < $0.000068 |
| SO | 0.7644831133207612 | < $0.000087 |

All residuals pass the existing $0.0002 adjustment-rounding ceiling. Control
close tolerances are unchanged. ACC's August 9 dividend is already reflected
in that valid paired quote; no target/following split or dividend is admitted
through the bracket. The exact upstream cause of the malformed bars remains
unknown. Original and adjusted volume fields are unchanged and not certified
by this OHLC amendment.

The old `ohlc_repairs.py` close-match policy and existing action configuration
remain frozen. This amendment is a different, explicit rule for replacing an
entire previously rejected OHLC observation. Original values survive in audit
columns and a separate three-key ledger. No replacement stock, synthetic raw
price, interpolation or price-driven model selection is introduced.

The underlying source is a selected curated Alpaca SIP projection. Its source
URI, object generation, declared whole-object hash, publication pin and local
projection hash are retained. This pass does not reread the whole remote table
or authenticate its original raw page bodies. Mixed remote row groups could
have transiently decoded excluded values inside Arrow during the original
projection; excluded values did not reach the selected table or Python analysis.
These are historical reconstruction inputs with provisional adjustment-unit
transport, not certified historical knowledge-time data or actual trade fills.

A later corrected workflow must distinguish execution overlays from rebuilt
predictors. Corrected August 12 daily inputs can affect expanded indicators and
context at later signals. Applying only an execution overlay does not repair an
already saved feature panel or its predictions. Monthly endpoint-based Paper15
inputs and return labels do not automatically change from an interior-day OHLC
replacement; the separate BBBY correction has broader normalization/label/model
dependencies documented in [the identity amendment](BBBY_IDENTITY_AMENDMENT.md).

## Dean Foods: an actual halt followed by incomplete OTC coverage

The issuer's [SEC securities description](https://www.sec.gov/Archives/edgar/data/931336/000093133620000006/ex44123119.htm)
states that DFODQ quotation began November 13, 2019, and NYSE delisting became
effective November 27. The issuer's [November 19 Form 8-K](https://www.sec.gov/Archives/edgar/data/931336/000095010319015638/dp115992_8k.htm)
records suspension at the November 12 market opening.
These dates distinguish a documented nontrading halt on November 12 from four
unexplained missing OTC observations on November 18–21.

The pinned native source has eleven bars from November 8 through December 2:
November 8, 11, 13, 14, 15, 22, 25, 26, 27, 29 and December 2. Every date and all
twenty compared fields match the existing annual cache; this pass discovered
**zero new usable quote dates**. Around the halt/reopening, raw and adjusted
prices share units, with unit split ratios and no recorded dividend. Local bar
metadata alone does not independently certify the OTC execution venue.

Two additional bounded curated metadata checks found no exact four-date keys in
the pinned S&P1500 source, or in the broader unified Dean Foods candidate with
ISIN US2423702032. Only identifier/date/source metadata columns were read;
neither check opened OHLC columns. This narrows the published sources checked,
without establishing that OTC data are absent elsewhere.

A proposed halt treatment would retain signed positions and stop basis, forbid
entry/exit/stop fills during a confirmed halt, and resume only against actual
subsequent quotes under an explicit OTC execution convention. The baseline
continues to record the missing valuation and withhold continuous performance.
A last-observed carrying mark would be an explicitly assumed, nonexecutable
valuation, not a new quote, zero actual return or guaranteed stop fill. No such
alternative has been implemented or used in performance.

Saved dependency metadata shows Dean Foods first blocks four cost versions of
RF/Paper15 class 1/10 with the 10% per-asset stop. It was a short holding. A fall
on OTC reopening therefore does not by itself imply a short stop or remove the
later missing-date dependency. No counterfactual completion count or P&L was
computed from these observations.

## Fresh source checks and remaining request

After interactive login refresh, three exact curated pointers were successfully
read as `info@galamboscapital.com`, configuration `deep-momentum-research`,
project `trusty-agility-492109-v0`. Their combined content was 2,914 bytes, and
their manifest pins match the existing releases. A bounded, complete,
nonrecursive product-name listing returned thirteen curated U.S. product
prefixes, without a new Sharadar or dedicated permanent-identity product.
No raw archive, bulk data, cloud compute, IAM or GCS write was used.

The organizational review covered head
`f5063d9fdd6fc1a4d610a6236a1e5ccb58f768f3`. Its newer catalogue/Sharadar work
does not publish the needed old-BBBY native crosswalk in the inspected sources.
This is a bounded source finding, not a claim that no private correction exists.
Earlier authentication failures remain preserved alongside the fresh success.

A small local metadata diagnostic also checked the ISIN-query-label failure
pattern across 1,377 distinct development source IDs. It found five other
source IDs whose declared ISIN query is a ticker literal. This is a provenance
review queue, **not proof of five wrong issuers** or authority to reject them.
Source aliases do not constitute independent provider identity evidence. No
additional identity decision or admission rule changed.

The [updated, unsent owner request](REMAINING_BACKTEST_DATA_REQUEST_20261016.md)
asks only for the unresolved data and evidence. Original requests and runs remain
available. The next production rebuild still depends on the old-BBBY historical
route; arithmetic quote repair alone does not produce corrected trained models.

## Verification

All **958 regression tests passed** in the staged V2 tree. The new public tests
use synthetic prices and temporary evidence files, so a clone needs no private
quote fixture to run them. Separately, 31 actual-evidence checks, an independent
Decimal calculation, and an independent audit-field/overlay-stacking replay
passed for the three-key amendment. The unchanged source-input, frozen-worker
and original-pointer hashes are recorded in the private package. These checks
establish the bounded amendment's behavior; they are not a new strategy result.
