# RET execution repairs — frozen 20 October 2026

The user authorized fixing the gaps and completing the RET comparison, with the existing scores, selections and trading rules unchanged. This is an additive execution-only experiment. Preserve the first independently verified RET run and its withheld performance at `outputs/ret_backtest/2020-2023-20261020-v1`.

## What the gaps actually were

A maximum-horizon metadata audit identified 146 unique native/date keys across eleven names and 208 stock-month episode rows. These are not 146 missing companies or 40 independent bad stocks. Several models share the same missing observation; potential stopped-path requirements can end earlier.

| Company | Dates with missing predecessor/route keys | Keys | Repair |
| --- | --- | ---: | --- |
| Legg Mason (LM) | 2020-08-03–2020-09-01 | 22 | $50 per raw share cash acquisition; first retired session August 3 after July 31 trading |
| Mimecast (MIME) | 2022-05-19–2022-06-01 | 9 | $80 cash acquisition, May 19 |
| CyrusOne (CONE) | 2022-03-25–2022-04-01 | 6 | $90.50 cash acquisition, March 25 |
| CDK Global (CDK) | 2022-07-06–2022-08-01 | 19 | $54.87 cash acquisition, July 6 |
| Cerner (CERN) | 2022-06-08–2022-07-01 | 17 | $95 cash acquisition, June 8 |
| Syneos Health (SYNH) | 2023-09-28–2023-10-02 | 3 | $43 cash acquisition, September 28 |
| National Instruments (NATI) | 2023-10-11–2023-11-01 | 16 | $60 cash acquisition, October 11 |
| Activision Blizzard (ATVI) | 2023-10-13–2023-11-01 | 14 | $95 cash acquisition, October 13 |
| Grubhub (GRUB) | 2021-06-15–2021-07-01 | 13 | Convert held old common into 3.355 Just Eat Takeaway Nasdaq ADSs per raw share; successor quote history required |
| R1 RCM (RCM) | 2021-11-01–2021-12-01 | 22 | Actual paired bars from the older RCM native route; provisional dated execution-only remap |
| Discovery / Warner Bros. Discovery | 2022-04-04–2022-04-08 | 5 | Actual DISCA Series A bars before the 1:1 WBD transition; regular WBD trading starts April 11 |

For tuned RF expanded, the entire original gap list is LM and MIME, both short selections. LM includes an inherited short that must settle at the August 3 open and a separate new order for a retired issue that must remain an unfilled cash slot. Do not delete either selected row or confer a prior-holder payout on the new order.

## Data and identity evidence

Nine new dated actions (eight cash, one GRUB stock conversion) supplement the nine original current actions. Preserve all eleven historical retirements and the exact prior PENN execution exception. Merger cash is a signed face-value receivable/liability under the inherited zero-funding convention; physical broker payment times are not simulated. No transaction fee is invented for the legal conversion itself.

Eight cash actions passed two preceding-session raw/adjusted OHLC checks against primary closing/trading dates; all observed factors are one, with no separate recorded split/dividend to double count. Primary issuer/SEC/Nasdaq references and exact private evidence pins are retained in `data/derived/ret_repairs/20261020-v1`.

The raw/all quote supplement comes from the owner's approved, exact-generation unified-market GCS object, with dated metadata reproduced before financial conversion. Only the selected two row groups and 42 authorized observations were returned: 21 raw/all pairs covering GRUB and DISCA/WBD plus controls. The broader initial financial acquisition was rejected by its budget before financial reads; it is preserved separately. Every observed paired OHLC factor is one with zero residual; this was measured, not assigned as a default. No dividend or split values are invented for these published pairs.

DISCA April 1 and WBD April 11 raw and adjusted OHLC controls match all eight existing fields exactly. The official Nasdaq/SEC transition is Series A common 1:1: pre-April 11 missing dates use DISCA, never the separate WBDWV when-issued instrument.

The Grubhub predecessor June 14 raw OHLC control matches exactly. Nasdaq explicitly identifies the new June 15 GRUB listing as Just Eat Takeaway ADS, CUSIP 48214T305 / ISIN US48214T3059, not old Grubhub common. The Russell source bridge still carries the old Grubhub ISIN. It therefore does not independently verify the ADS. The repair explicitly rebinds only the post-June 15 dated SIP GRUB observations to `execution_cusip:48214T305`, using the official listing/share-class evidence and 3.355 ratio. This is a provisional execution reconstruction, not promotion of the publisher's old identity alias. It keeps separate raw/adjusted units for the successor rather than splicing pre-merger prices into the new issue.

RCM uses actual `sec_yKbEk4` paired observations for the missing `sec_zMM1vd` history. Both October 28/29 controls match all eight OHLC fields exactly. The target route has no post-gap observations through June 21, 2022 while the old route does, so an after-gap overlap control cannot be fabricated. The explicit exception requires both pre-gap controls, every copied paired row, immutable absence metadata, and primary evidence of the equivalent-share June 2022 holding-company reorganization continuing RCM's trading history. The older Accretive ISIN is a provider alias; it is not claimed to be RCM's unchanged official 2021 identifier. This exception is specific and provisional; the default two-sided control rule remains for other repairs.

Exactly forty observed rows are appended: 22 RCM, five DISCA and thirteen new ADS observations. Original accepted observations are unchanged. Do not insert fabricated continuing prices for acquired predecessor shares.

## Frozen comparison and limits

Run five saved model arms × unstopped/10% per-asset stop × 0/5/10/25 bp one-way cost = forty scenarios over 2020–2023, 48 monthly outcomes and 1,006 daily sessions. Ten long/ten short RET selections, 50%/50% initial allocation, next-session open execution, cash after stops, continuous NAV and terminal close liquidation remain unchanged. Reuse all twenty parent selection/ledger/rebalance/monthly artifacts with exact byte pins. No feature, target, payoff, prediction, model or preprocessor is rebuilt or refitted; there is no new inference.

These execution repairs do not resolve the separate effect of missing historical routes or terminal labels on saved features, training labels or RET payoffs. Preserve and disclose that limitation instead of silently claiming an end-to-end repaired learning pipeline. The comparison isolates the requested selection strategy and new execution evidence against the same saved model inputs.

The configuration is `configs/ret_repair_backtest_20261020.json`; reviewed private data is bound through its immutable repair manifest. New workers use fresh `outputs/ret_repair_backtest` and separate controls. Read reservations precede decoding; use one numerical thread, 4 GiB process-tree RSS stop, 7 GiB available-memory dispatch, 30 GiB free disk, 1 GiB cumulative compressed read reservations and 512 MiB new outputs. Phase ceilings remain 300/900/1200/3600 seconds for prepare/preflight/replay/verification. Preserve stopped/failed attempts with no automatic retries.

Require a complete metadata preflight before performance, then independently reconstruct quote normalization, overlays, action definitions and all forty daily cash/units paths. Do not call new performance verified until that independent receipt passes. Opening the master notebook starts no computation. Protected 2024+ financial rows remain outside returned tables/analysis; mixed compressed Arrow blocks may decode before exact row filtering, as documented by the source reader. Partial cloud range hashes do not establish a full downloaded-object digest or original provider-response identity certification.
