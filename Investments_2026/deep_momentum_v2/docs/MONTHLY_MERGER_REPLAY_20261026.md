# Monthly comparison: realized merger payoffs, fixed models and RET

USER DIRECTION, 26 October 2026: reopen EV, CDK, XEC and XLNX to obtain a full
comparison. The user explicitly selected **repair realized portfolio returns
only; keep saved forecasts and RET selections fixed**. This supersedes the
earlier deferral for these four realized-return cases only. Preserve the strict
monthly benchmark and its common fourteen-month report as historical evidence.

## Scope frozen before observing repair values

The recipe is `configs/monthly_merger_replay_20261026.json`. Reuse the sixteen
saved annual Paper15 RF/XGBoost five/ten-year models, original predictions,
probabilities, strict training labels, features, RET payoff grids, scores and
selected baskets byte-for-byte. Maximum new fits is **zero**. No searches,
reclassification, new trade selections, exclusions, replacement stocks or
portfolio reweighting are authorized by this realized-payoff replay.

Keep forty-eight monthly returns January 2020–December 2023, plus the separately
initialized forty-seven-month February-start comparison, the same fifty/fifty
long/short target weights and ten selected names per side. There are no stops
in this monthly experiment. Costs remain 0/5/10/25 bp per one-way voluntarily
traded notional. All thirty-two scenarios and annual slices must be reported,
including any still-withheld row. There is no forced annual NAV reset or
terminal liquidation. Protected 2024+ financial values stay unread.

## Primary legal terms

| Former security | Effective session | Entitlement per former raw common share | Source |
| --- | --- | --- | --- |
| EV | 1 March 2021 | $28.25 cash plus 0.5833 MS shares, documented non-election mixed consideration | [OCC 48362](https://infomemo.theocc.com/infomemos?number=48362) |
| CDK | 6 July 2022 | $54.87 cash | [Nasdaq effective-merger notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=eca2022-164) |
| XEC | 1 October 2021 | 4.0146 Coterra/Cabot common shares | [Cimarex closing 8-K](https://www.sec.gov/Archives/edgar/data/1168054/000110465921122062/tm2129003d1_8k.htm) |
| XLNX | 14 February 2022 | 1.7234 AMD shares | [Xilinx closing 8-K](https://www.sec.gov/Archives/edgar/data/743988/000110465922021762/tm226258d1_8k.htm), [Nasdaq timing notice](https://www.nasdaqtrader.com/TraderNews.aspx?id=ECA2022-23) |

The [EV closing release](https://www.sec.gov/Archives/edgar/data/350797/000095010321003212/dp147036_ex9901.htm)
places its separate $4.25 dividend on 18 December 2020, before the March holding
month. Do not add it again. EV's mixed consideration is the documented default
for non-electing shares; do not invent an all-cash/all-stock election or its
proration. Existing verified CDK/XEC terms are inherited unchanged. MS, AMD and
Coterra routes remain the original provisional cached identities, with no
claim of new provider certification. Theoretical fractional successor units
are retained instead of inventing actual cash-in-lieu proceeds.

## Monthly return and unit reconstruction

The original fixed monthly strategy assumes entry at the preceding signal
close. Therefore the old shares are held before these legal conversions; this
differs from the earlier next-open EV order that could not execute on March 1.
No nonexistent month-end price is manufactured for a retired predecessor.

Start with one unit of capital at the exact adjusted signal close A0. The old
adjusted units are 1/A0. At the last verified predecessor trading close, let
raw and adjusted prices be Ppre and Apre. Raw shares delivered to the merger
are `(1/A0)*(Apre/Ppre)`. Apply the documented cash c and stock ratio r to those
signed raw shares. Cash remains cash at zero funding through month-end.

For a delivered successor, the paired raw/adjusted close on the effective
session supplies factor `fnew=Asucc/Psucc`. Adjusted successor units are
`raw_shares*r/fnew`; value them at the exact successor month-end adjusted
close. This paired close factor is a unit conversion, not a trade at that
price. Asset return equals terminal cash plus successor value minus one.
Ordinary distributions already represented by the source adjusted prices are
not added again. The adjustment vintage/dividend-feed coverage remains a
provisional research proxy, not certified CRSP cash-flow accounting.

Require every explicitly planned paired close to have positive finite raw and
adjusted prices and accepted completion/quality/bounds evidence in the verified
cache. The plan enumerates twelve exact date/native close points and the four
missing original terminal endpoints. Source metadata and immutable file pins
are checked before numeric reads. Missing successor/basis evidence still fails
closed; legal terms alone cannot supply a market price.

## Portfolio cash, successor holdings and costs

A short position creates signed cash and share liabilities. Mandatory mergers
are not sales of the retired security or new voluntary purchases: they incur
no market-trading fee. Successor exposure survives and nets with any existing
exposure in the same exact cached native security. The next regular rebalance
trades against actual cash/successor holdings. Do not retain a fictional old
stock notional, charge a cash balance as a stock sale, or reset converted
exposure to a new independent trade. Numerical target weights and one-way fee
rates remain the previous contract; corrected post-conversion holdings govern
its turnover calculation.

Before replay, verify both directions of consecutive canonical/native key
continuity in every original selected basket. Any unrelated alias/route change
requires separate review; it must not be silently handled by this merger fix.
Unamended stock returns remain the original strict adjusted-close ratios.
Every selected return must be supported; no zero fill, future filter or omitted
stock is allowed. Monthly risk measures use sqrt(12) annualization and
month-end-only drawdowns, with idealized same-close entries and zero borrow.

## Verification and preserved learning scope

Independent checks reconstruct all twelve quote points and four unit returns
from exact cache bytes. A separate oracle reconstructs each monthly cash/share
ledger, signed action, voluntary trade, fee, contribution, annual return and
aggregate without importing producer accounting. The first fourteen months
must reproduce the prior common-period report at every cost. Preserve hashes
of all original selected ledgers, sixteen forecast/fit receipts, strict feature
and label panels, RET grids and payoff vectors. No estimator is loaded or fit.

A metadata-only dependency audit records which training windows could be
affected if these terminal returns were ever promoted to learning. **They are
not promoted here.** Strict learning labels and RET still use their original
observed-return population. This is the full realized-return comparison of
those frozen models, not a refit on repaired labels or a claim that all prior
missing training returns have been resolved. Such a research rerun remains a
separate decision.

Cache-file reservation is 512 MiB per pass, worker RSS 2 GiB, outputs 256 MiB,
600 seconds per phase and 2,400 active seconds total. Only cached paired closes
are decoded; no raw price download, cloud mutation, full price scan, financial
2024+ value or change to earlier artifacts is included. Use the explicit
`scripts/run_monthly_merger_replay.py` plan/run/pause/resume controller. A passive
reader/notebook will show results only after both independent proofs pass.
