# 2018–2019 corporate action evidence

Reviewed 2026-10-15 for the bounded development backtest. This is a dated event
input review, not a change to historical cohorts, identity decisions, model
predictions, targets or original provider bars. Completion evidence below
supersedes proposed transaction announcements. All amounts are USD per actual
pre-event ordinary share, before any provider adjustment. Entitlements exclude
treasury, affiliated, dissenting or other exceptional holdings described in the
filings; the development simulation assumes ordinary public-market holdings.

## Verified event terms

| Old ticker / provider security | Economic effective date | Mandatory ordinary-share entitlement | First session without old listed shares |
| --- | --- | --- | --- |
| BIVV / `sec_gedWvo` | 2018-03-08 | $105 cash | **2018-03-08 before open**, per issuer 8-K; local cache has a conflicting March 8 bar |
| ESRX / `sec_zrJv8z` | 2018-12-20 | $48.75 cash + 0.2434 New Cigna ordinary shares | 2018-12-21 before open |
| IDTI / `sec_gVbw5g` | 2019-03-29 US date | $49 cash | 2019-04-01, documented exchange suspension date; exact intraday legal effectiveness on March 29 is not established here |
| GXP / `sec_XWOwJX` | 2018-06-04 | 0.5981 Evergy ordinary shares | 2018-06-05 before open |
| RDC / `sec_zmYNnz` | 2019-04-11 before open | 2.750 pre-consolidation Ensco ordinary shares, immediately consolidated four-for-one: **0.6875 post-consolidation shares** | 2019-04-11 before open |
| ESL / `sec_XnlJ4y` | 2019-03-14 | $122.50 cash | 2019-03-14 before open, per issuer 8-K |
| BID / `sec_Bgk1Bz` | 2019-10-03 before open | $57 cash | 2019-10-03 before open |
| LLL / `sec_zEMdWz` | 2019-06-29 | 1.30 L3Harris ordinary shares | 2019-07-01, the next exchange session |

The session dates inferred from missing local quotes are labelled separately
from documented event dates. They are not evidence that an unavailable bar can
be valued at the last close.

## Primary sources and settlement distinctions

- **BIVV:** [Sanofi completion release filed with the SEC](https://www.sec.gov/Archives/edgar/data/1121404/000119312518074806/d538875dex991.htm), dated March 8, 2018. The tender offer expired late March 7; tendered shares were accepted March 8 for prompt payment. Untendered ordinary shares converted to the same $105 cash right through the completed merger. [Bioverativ's own completion 8-K](https://www.sec.gov/Archives/edgar/data/1681689/000110465918015757/a18-7726_18k.htm), Item 3.01, establishes suspension **before March 8 open**. Its cached March 8 bar is not proof of a tradable session. Actual account payment time is not established.
- **ESRX:** [Express Scripts completion 8-K](https://www.sec.gov/Archives/edgar/data/1532063/000114036118045477/form8k.htm), introductory note. The completed transaction created New Cigna, which commenced NYSE trading December 21 under CI. Old Cigna shares converted one-for-one; ESRX shares converted to the cash-plus-stock entitlement in the table. Fractional stock interests were paid in cash. Delivery and individual cash-payment times are not established; effective-date recognition is not proof of settled buying power.
- **IDTI:** [IDT completion 8-K](https://www.sec.gov/Archives/edgar/data/703361/000119312519093186/d929249d8k.htm), Item 2.01, confirms both final $49 cash terms and March 29 completion. [Renesas completion announcement](https://www.renesas.com/en/about/press-room/renesas-completes-acquisition-integrated-device-technology-0) distinguishes March 29 US time from March 30 Japan time. Neither establishes individual payment timing. [Nasdaq's updated corporate-action notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2019-13) confirms suspension effective April 1 and predecessor CUSIP `458118106`.
- **GXP:** [Great Plains completion 8-K](https://www.sec.gov/Archives/edgar/data/54476/000119312518182966/d454249d8k.htm), Item 8.01, confirms the final ratio and June 5 EVRG trading start. [Evergy shareholder exchange FAQ](https://investors.evergy.com/static-files/1a95cb3d-c3ac-4d98-a8a9-8ca2986d1869) states brokerage/book-entry shares convert automatically upon closing, with cash in lieu of fractions. Certificate holders had separate surrender requirements. Exact broker delivery time is not established.
- **RDC:** [Ensco completion 8-K](https://www.sec.gov/Archives/edgar/data/314808/000110465919020867/a19-7374_18k.htm), Item 2.01, explicitly sequences the 2.750 issuance before four-for-one consolidation, affecting former Rowan holders too. [Rowan completion 8-K](https://www.sec.gov/Archives/edgar/data/85408/000114420419019311/tv518654_8k.htm) confirms no RDC trading from April 11 open and cash in lieu of fractions. The raw physical ratio is 0.6875. **Do not use the $43.67 value in retrospectively restated accounting exhibits as an old raw RDC terminal price.** Exact broker delivery and fractional-cash times remain unverified.
- **ESL:** [TransDigm completion 8-K](https://www.sec.gov/Archives/edgar/data/1260221/000126022119000011/a8-kesterlineclosing31419.htm), Item 2.01, confirms $122.50 cash. [Joint completion release](https://www.sec.gov/Archives/edgar/data/1260221/000126022119000011/exhibit991jointpressreleas.htm) says the paying agent would provide surrender instructions. [Esterline's own 8-K](https://www.sec.gov/Archives/edgar/data/33619/000003361919000003/eslform8-kxmergerclosing.htm), Item 3.01, confirms a halt before March 14 open. No exact payment date is established.
- **BID:** [Sotheby's completion 8-K](https://www.sec.gov/Archives/edgar/data/823094/000082309419000038/a8-kclosingxbrl.htm), introductory note and Item 2.01, confirms final $57 cash terms. The [NYSE removal notice filed with SEC](https://www.sec.gov/Archives/edgar/data/823094/000087666119001019/ruleprovisionnotice.htm) establishes effectiveness before October 3 opening. Actual stockholder payment time is not established; equity-award payment deadlines are not ordinary-share settlement dates.
- **LLL:** [Harris completion 8-K](https://www.sec.gov/Archives/edgar/data/202058/000114036119012139/form8k.htm), introductory note and Item 2.01, establishes June 29 effectiveness and 1.30 L3Harris shares for each L3 share. L3 became a subsidiary of renamed Harris. Fractional-share treatment and broker delivery times need the merger agreement if an integer-share simulation is introduced.

## Successor identity: verified facts versus local candidates

SEC evidence identifies the delivered securities economically. The downloaded
Intrinio master supplies these **candidate provider mappings**, which remain
subject to the existing provisional identity policy; this review does not
upgrade an identity flag:

| Delivered security | Candidate local provider ID | Local metadata constraint |
| --- | --- | --- |
| New Cigna, NYSE CI | `sec_Qz8jlg` | Master response queried ISIN `US1255231003`; provider history extends before the new issuer existed, so post-event date and identifier checks remain necessary. |
| Evergy, NYSE EVRG | `sec_Xn1EW4` | Master response queried EVRG and provider history extends before the merger. Read only correctly dated post-event bars. |
| L3Harris, NYSE LHX | `sec_gYrYW2` | Master query identifier `US4138751056`; provider history includes predecessor Harris. |
| Ensco Rowan, NYSE ESV, post-consolidation | **No valid 2019 candidate found in downloaded master** | `sec_gk5pAW` is an ESV ADR whose metadata ends in 2016. `sec_Xa9e96` is post-2021 VAL and cannot represent the 2019 delivered security. `sec_gx2bWz` is unrelated VAL ending 2017. |

The [SEC Q2 2019 13F security list](https://www.sec.gov/files/divisions/investment/13f/13flist2019q2.pdf)
adds Ensco Rowan Class A, CUSIP `G3166L100`, while deleting predecessor Ensco
CUSIP `G3157S106`. This is a useful exact-identifier retrieval lead, not a
substitute for 2019 successor quotes or an executed trade.

## Dean Foods is a halt/OTC case, not a cash merger

For `sec_gjP4xM` (dated membership ticker DF, current downloaded master ticker
DFODQ), [Dean's contemporaneous November 19, 2019 8-K](https://www.sec.gov/Archives/edgar/data/931336/000095010319015638/dp115992_8k.htm)
states NYSE suspension at the November 12 opening after Chapter 11 filing.
There is no zero-dollar cancellation or cash entitlement in this evidence.
Do not remove the position, fabricate a 10% stop fill, assume zero, or carry its
last close as an observed fresh quote. The missing 2019 OTC bars and an explicit
halt valuation/exit policy remain required for a complete economic path.

## Unit conversion and engine requirements

Let a verified pre-event row satisfy `P_adjusted = f_old * P_raw` for the
same field and day, and let the old engine hold `q_adjusted` proxy units.
Then `q_old_raw = q_adjusted * f_old`. A mandatory event with raw cash `c` and
raw successor ratio `r` creates signed cash entitlement
`q_old_raw * c` and signed successor shares `q_old_raw * r`. For a successor
row satisfying `P_successor_adjusted = f_new * P_successor_raw`, its proxy
units are `q_new_adjusted = q_old_raw * r / f_new`.

Use the last verified pre-event factor at the conversion boundary; establish
its exact raw/adjusted semantics from the pinned bars. Multiplying cash by an
adjustment factor that includes a further merger transform can double-count
the event. No original source factor should be silently overwritten.
Short positions create the corresponding liabilities rather than positive
cash receipts. Do not independently add dividends already represented by the
adjusted-price proxy. The provider's retrospective factors still do not prove
historical knowledge-time availability or certified economic total returns.

A newly selected stock that has ceased trading **before its next-open entry**
does not confer merger consideration: preserve its selected row as unfilled
and its allocation as cash, with no substitute or false entry. Existing
pre-event holdings receive the entitlement. Keep effective-date receivables,
actual settlement, any successor sale and cash availability separate. Exact
settlement is unverified above, so a development convention must be labelled
as an assumption and must not free unverified cash for interim reinvestment.

The frozen stop, selected-name, weighting and rebalance rules must survive
conversion. Mandatory conversion is not a stop fill at the old price. Keep
the original economic cost basis for any continuing successor position and
do not reset its stop to the successor opening price. Missing delivered-share
quotes remain a hard coverage failure; public merger terms do not provide a
replacement observed successor market price.

## Additional dated identity conflict found during quote review

VERIFIED LOCAL source disagreement, not a new cohort rule: the August 12, 2019
BBBY membership route currently points to `sec_ydbQnj`. Its downloaded identity
evidence includes ISIN `US6903701018`, historical Overstock, whereas the independent
curated dated BBBY bar is old Bed Bath & Beyond. Their raw closes are $24.97 and
$8.41. This is a potential historical ticker/identity substitution, not acceptable
rounding. The overlay rejects it; original flags, selected rows, source files,
features and model fits remain unchanged. An affected continuous backtest path
must remain withheld. The original provisional identity waiver does not certify
this mapping or authorize silently replacing the native instrument.

Other additional August 12 close disagreements are ACC $46.77 versus $46.84,
K $61.62 versus $61.53, and SO $58.68 versus $58.50 (original versus Alpaca raw).
They exceed the frozen absolute compatibility limit and require a separate
source/unit review. They are not repaired by widening a tolerance after results.
