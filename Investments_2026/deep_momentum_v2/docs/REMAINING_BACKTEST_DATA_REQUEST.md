# Prepared message for the dataset owner — not sent

Hi Sadig,

We repaired the original three OHLC issues using pinned, dated Alpaca SIP bars
and added verified corporate-action accounting to our fixed 2018–2019 backtest.
We retained every original selected stock and did not change the models.
Could you help with these remaining items?

1. **Historical BBBY mapping:** the old Bed Bath & Beyond membership source
   `siblis_ticker_company_00326c7d9ad13903e5e3` currently maps to Intrinio
   `sec_ydbQnj`, whose identifier evidence is `US6903701018` / historical
   Overstock. Please provide the correct dated old-BBBY identity mapping, original
   lookup evidence and full historical price-source route. Current ticker/name
   reuse must not substitute the newer issuer. This can affect features and
   training, not just the daily backtest.
2. **Rowan → Ensco Rowan:** correctly identified 2019 successor ordinary-share
   quotes for April 11–May 1, 2019, CUSIP `G3166L100`. The entitlement is 0.6875
   post-consolidation shares per old RDC share. Our downloaded ESV ADR ends in
   2016 and post-2021 VAL is a different security; neither is an acceptable alias.
3. **Dean Foods:** dated suspension/OTC identity and actual quote evidence around
   November 12–December 2, 2019, including DFODQ where applicable, plus the halt
   information needed for a stated valuation/exit convention. A missing quote
   must not become an invented zero, stale-price exit or stop fill.
4. **ACC/K/SO on August 12, 2019:** authoritative OHLC and adjustment-basis
   clarification. Their original rejected bars have closes that disagree with
   the independent raw SIP source beyond our frozen compatibility tolerance.

A small filtered extract in the approved curated GCS namespace is sufficient
for the daily quote gaps. Please include instrument identifiers, exact dates,
raw/adjusted OHLC, adjustment semantics and action/quality flags, source manifest,
object generation and hashes. For BBBY we need the historical mapping and source
route, not only an isolated execution-day correction.

The active account is `info@galamboscapital.com` in project
`trusty-agility-492109-v0`. Approved curated reads work. A GET of the exact raw
Sharadar archive below returned HTTP 403 identifying `storage.objects.get`:

`gs://quant-research-trusty-agility-492109-v0/dataset-platform/raw/sharadar/bundle/v1/run_id=sharadar-bundle-full-20261008-r1/archives/stocks.csv.zip`

That error does not establish that data are absent. Publishing the small reviewed
extract under our existing approved scope would avoid a bulk download or broader
role request. We have not launched compute, changed IAM or written to GCS.
