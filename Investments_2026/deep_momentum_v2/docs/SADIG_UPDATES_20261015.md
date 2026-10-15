# Sadig's latest organizational updates — October 15, 2026

VERIFIED GITHUB, 2026-10-15. Read-only review of `GalambosC/deep-momentum` through
`2d65c8bd8846018ffb48332f05e46aee794dca71`, authored/merged by `sadikhh` at
2026-10-15 08:00:54 UTC (10:00:54 Europe/Vienna). Both `master` and
`codex/deep-momentum-v2-phase1` point to this commit. Compare base:
`f8f705a08ef06443c0aed3ff68216f713a35eef7`, reviewed on October 14.
Only the organizational repository was inspected. No messages, cloud calls,
dataset downloads or external mutations were made.

## What changed and why it matters

The repository has newer source-delivery progress, but this review found no
new historical stock-identity mapping repair or demonstrated replacement for
our missing 2010 historical quotes. The comparison from the previous research
head contains **14 additional graph commits and 14 changed files, all Markdown**.
No numerical loader, identity resolver, feature formula or price source code
changed in that comparison.

[PR #2](https://github.com/GalambosC/deep-momentum/pull/2) was merged at
08:00:54 UTC on October 15. It brings the previously published research branch
onto the default branch. Its large default-branch diff is principally older
research becoming visible on master; it is not 1,233 newly written fixes.
The current research head's net changes since our previous review are the
14 documentation files above.

The latest [work order](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/CURRENT.md)
reports **nine accepted source years out of 28**, up from three:
2014, 2015, 2016, 2017, 2019, 2020, 2021, 2022 and 2023. It reports
3,235,046 original member-days out of 9,460,346. These are repository-reported
delivery checks; this project did not repeat their GCS or numerical validation.

| Latest repository-reported checkpoint | Meaning for our work |
| --- | --- |
| 2014 accepted at 07:49:04 UTC on October 15; 357,236 original keys, 745 candidate columns and 416 bases | A catalogue delivery with preserved keys, not proof that every price or feature is available |
| 2017 accepted; 357,516 original keys | Another year of the same catalogue, not a new identity crosswalk |
| 2013 and 2018 reported running at the pinned checkpoint | Neither was an accepted year in this commit; live state was not checked |
| 2010 and 2011 startup packages/preflights prepared; the 2010 upload still pending finalization | No newly accepted 2010 catalogue is established by this checkpoint |
| No broad-redesign model fits or funded backtest yet | Source preparation should not be reported as strategy performance |

Exact 2014 archive locator, as documented in the
[acceptance record](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/docs/v2/STRATEGY_REDESIGN_CATALOGUE_2014_ACCEPTANCE_20261015.md):

```text
gs://quant-research-trusty-agility-492109-v0/strategy-research/active/deep_momentum/runs/dmv2-catalogue-year-2014-20261014-r1/catalogue/block.tar
generation 1792048838580195
size 4,006,748,160 bytes
sha256 1a71d5f2c2ad91ee90fe814fa798a5df6d6e4d91d4b27577a5a2f56b20e22edd
```

This is a locator from GitHub, not an independently read or verified cloud
object. The full numerical archive was not downloaded.

## Identity and historical-price implications

The same three core identity modules still have exactly the Git blob SHA-1
values recorded in our October 14 review:

| Module under `src/deep_momentum/` | Git blob SHA-1 |
| --- | --- |
| `identity_adjudication.py` | `f3ee44d7cab567dabb5cdb158fba587d62b3806a` |
| `effective_identity_corroboration.py` | `327484c352499468a522da4ba4116c6ed6936d95` |
| `local_identity_construction_feasibility.py` | `170ac090955b7ac2e397d6667004e21068b10ce5` |

The [current work order](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/CURRENT.md)
retains IDENTITY/PIT UNVERIFIED, PROVISIONAL_EXPLORATION, ALPHA_NOT_VALID and
RERUN_REQUIRED. It does not claim a repaired historical crosswalk.
Their accepted identity exception remains an explicit research waiver.

The [source-wave specification](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/docs/v2/STRATEGY_REDESIGN_CATALOGUE_HISTORY_WAVE_PREPARATION_20261014.md)
states that all original 105 remote price pins and 11 metadata pins are
preserved. This is computation and delivery of features from existing inputs;
the checked new changes do not deliver a new raw price feed or corrected native
security IDs. We did not independently match all those remote pins to our local
release, so identical object inventories between projects are not asserted.

The [catalogue price-prefix loader](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/src/deep_momentum/monthly_backtest_catalogue_source.py)
retains known-membership selectors even when a quote is absent and explicitly
leaves price gaps unfilled. Complete original-key counts therefore cannot be
compared to our percentage of stock-months with all five valid paper momenta.
The newer [compatibility check](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/docs/v2/STRATEGY_REDESIGN_ACCEPTED_SOURCE_COMPATIBILITY_20261014.md)
checks definitions, calendars and company/macro source qualifications. It
explicitly does not validate full numerical arrays or admit predictors.

## Useful material and methodological boundaries

The catalogue's compact delivery metadata, family masks, normalized column
semantics and full-key reconciliation could help a future comparison of our
feature definitions and coverage reasons. A bounded metadata read would be the
next useful step if a delivered year is needed; multi-gigabyte archive hydration
is not necessary to understand these contracts. No such GCS read happened here.

Sadig's [current method](https://github.com/GalambosC/deep-momentum/blob/2d65c8bd8846018ffb48332f05e46aee794dca71/docs/v2/STRATEGY_REDESIGN_METHOD_CHOICES_20261010.md)
uses historical S&P 1500, rolling 36 TRAIN / six validation / three TEST months,
quarterly model updates, monthly rankings and a broad technical/company/macro
catalogue. It also records training-only feature admission and missingness
choices. Those choices belong to that separate research protocol; they do not
replace our paper's 15 inputs, selected S&P 500/400 engineering union or rolling
10-year training / 1-year testing with monthly predictions. The user's present
2010 initialization is recorded before model results, with earlier raw data
retained for warmup.

The updated AGENTS guidance is prospective research/operator guidance, not
authorization to use Sadig's compute campaign, send messages, mutate cloud
resources or import his scientific choices. Our current task remains the
2010-start coverage check and reproducible local workflow.

## Completeness and reproducibility

All eight branch heads were inventoried. Five other substantive branch heads
are ancestors of the pinned main/research head. The sole divergent branch,
`codex/monthly-source-io` at
`51f205ad7eb233d8680352fad578560c6e274733`, has one unique commit changing four
source-I/O scripts; no separate mapping repair is in that diff. The master
and phase1 branch heads agree.

The prior head, descendant commits, branch comparisons, PR #2 metadata, exact
file blob hashes and selected source-document snapshots are retained privately
under `outputs/github_review/20261015/`. The STATUS snapshot is explicitly
limited to lines 1–190. These records support what was visible at the pinned
commit; they do not establish that newer external or unpublished mappings do
not exist.

The practical conclusion is to retain our provisional identity warnings and
measure the later-start coverage directly. Nothing in the new checked pushes
establishes that the historical-price deficit has been repaired.
