# Narrowed owner data request — October 16, 2026

Prepared for forwarding; **not sent**. This updates the earlier request while
preserving that original record.

Hi Sadig,

We have narrowed the remaining 2018–2019 backtest data issues. Could you provide
the following small, identified extracts/evidence under the approved curated
namespace?

1. **Historical Bed Bath & Beyond**, ISIN `US0758961009`, CUSIP `075896100`:
   the correct historical provider security ID, original exact-identifier
   response/path/hash and dated issuer/share-class evidence, plus the matched
   historical price route, including the missing 2008–2015 history and required
   pre-2010 feature warmup. Our published old-ISIN
   link is labelled as an ISIN query but actually queried `BBBY`; it points to
   `sec_ydbQnj`, the historical Overstock route. We have blocked that wrong
   assignment separately and preserved the old models/runs. Details are in
   [the precise BBBY lookup request](BBBY_LOOKUP_REQUEST.md).
2. **Rowan → Ensco Rowan**, successor ordinary shares, CUSIP `G3166L100`:
   actual daily raw and adjusted OHLC for **April 11–May 1, 2019**, with dated
   native identity and the post-consolidation share basis. The entitlement is
   0.6875 successor shares per old RDC share. Old ESV ADR history ending in 2016
   and post-2021 VAL are different securities and cannot substitute these quotes.
3. **Dean Foods / DFODQ**, native `sec_gjP4xM`: actual OTC OHLC for
   **November 18, 19, 20 and 21, 2019**, plus source/venue and adjustment evidence.
   The other eleven available dates from November 8 to December 2 are already
   cached; we do not need those duplicated. November 12 is a documented halt,
   not an ordinary missing quote. Any official halt/reopening evidence or stated
   historical valuation convention would help us specify treatment separately.

For the extracts, please include exact identifiers/dates, units/currency,
raw/adjusted OHLC, adjustment and action semantics, source manifest/object
generation and content hashes. We will retain missing positions and coverage;
we are not requesting invented exits, zeros or guaranteed stop prices.

The current curated stable pointers were rechecked after login and still match
the published releases we already use. Please identify any newer correction
outside GitHub explicitly. A small curated extract is sufficient; we are not
requesting a bulk archive, credentials, broader administrative roles or a cloud
compute launch. Our active account is `info@galamboscapital.com`, project
`trusty-agility-492109-v0`.

The ACC/K/SO August 12 OHLC cases now have separately reviewed SIP replacement
inputs and are excluded from this outstanding-data request. Corrected model and
backtest runs have not yet been generated.
