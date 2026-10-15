# Context comparison and promotion: 80% primary, 90% sensitivity

**CURRENT USER DIRECTION, 2026-10-15:** use 80% as the primary original-cohort
rule and retain 90% as sensitivity. The already verified panels keep their
original bytes, paths and identities. `configs/context_panel_roles.json` records
both explicit roles and immutable pins. The master reads them through
`dmv2.context_panel_roles.read_context_panel_roles(catalog)`; `full_feature_run`
and `full_feature_dataset` refer to 80%, while `context_sensitivity_run` and
`context_sensitivity_dataset` refer to 90%. Raw daily quotes remain separate.

The earlier request built 80% as a sensitivity alongside then-primary 90%.
This file and saved comparison tables preserve that historical **90% → 80%**
orientation: gains denote additional values released at 80%. Original filenames
such as `context_sensitivity_80.json` and `LATEST_80.json` identify the historical
build, not its current role. `configs/context_panel_roles.json` is the active
role selector; neither historical output pointer selects current roles. Promotion does not recalculate features or select
thresholds from model results. Paper15, technical298, original cohorts, PIT
cutoffs, exact temporal windows and preserved daily artifacts are unchanged.

## Fixed comparison

The original `configs/context_sensitivity_80.json` recipe pins the completed 90%
reference by audit and build-identity hashes. Role verification additionally
checks the archived 80% recipe and both original worker snapshots, source pins,
runtime, chronology, paper/missingness contracts and artifact hashes against
the explicit promotion contract. Historical source is verified, never executed.
The strict current-code reader remains available for runs built by current code;
it does not silently accept historical recipes. Both original panel artifacts
and original build pointers remain unchanged; the role contract is separate.

The 80% panel uses `available_cohort_80pct`: all admitted IDs must be requested,
and `10 * observed >= 8 * original` for each feature/date/cohort. All original
members, including blocked identities, stay in the denominator. Exact temporal
lookbacks, same-sleeve continuity and every required stock/benchmark daily pair
remain unchanged. Each historical daily benchmark has its own coverage gate.
No missing stock value is filled before a rank or aggregate calculation.

Both policies calculate on the same valid observed population when their gates
pass. The lower threshold releases additional statistics; it does not add new
contributors to a group already passing 90%. Every shared finite value must be
exactly equal, and every 90%-finite cell must remain finite at 80%. These are
tested and checked across all 415 context definitions, while coverage summaries
focus on the 402 eligible names.

## Historical cached build and bounded execution

Use verified annual signal-only panels, original historical cohorts, independent
calendars and past-only daily feature histories from the 90% run. Recalculate
context only. Copy paper15 and all298 technical columns unchanged, and preserve
raw daily/core/annual-history artifacts byte-for-byte as independent files.
No outcome values, protected2024+ financial values or new provider data are read.

The completed historical build used one annual local worker with the existing 4GiB peak-RSS, 10GiB output,
20GiB conservative projected-read and three-hour limits. Projected cache reads
budget five artifact passes; these are not measurements of physical disk I/O.
Per-year checkpoints bind sources, reference, code, runtime and policy. A partial
run cannot update `outputs/context_sensitivity/LATEST_80.json`. The then-primary
90% pointer was checked unchanged. Neither notebook Run All nor loading reports
implicitly starts a build, preprocessing fit, model fit or backtest.

The original `dmv2.context_sensitivity` build/compare workflow was executed
under its archived 90%-primary recipe. It is not the command for promoting
roles or rebuilding under the active 80% configuration. To read the promoted
pair in a configured workspace:

```python
from dmv2.context_panel_roles import read_context_panel_roles

panels = read_context_panel_roles(catalog)
primary = panels["primary"]       # verified 80%
sensitivity = panels["sensitivity"]  # verified 90%
comparison = panels["comparison"]  # historical 90% → 80% orientation
```

## Reports and limits of interpretation

Comparison reports provide original and paper-qualified coverage by feature,
year and full/early/later/train/test period. “Before 2016” means outcome years
2010–2015; a December signal belongs to the following outcome year. All 402
names already have observations somewhere under 90%; the question is additional
dates and values, not a larger feature-definition bank.

For each original variable/date/cohort, compare valid and omitted members,
S&P500/S&P400 composition, identity-admitted and provisionally flagged counts,
and reasons for omitted inputs. Available signal-time return, volatility and
log-dollar-volume snapshots provide descriptive profiles with explicit finite
counts. They cannot reveal the economic characteristics of unmapped/missing
stocks. Sector, capitalization, distress and actual listing dates are not
inferred. Membership/feature slots repeat across variables; they are not unique
stocks. Daily warmup overlap is reported separately from primary signal dates.

Newly released values have per-feature distributions in the report. Combining
values across unlike units is not an economic comparison. No target or test
performance chooses the threshold. More computable values do not establish
representativeness, historical identity/PIT certification or investable returns.

Comparison completion requires unchanged original keys, identity flags, paper
eligibility, paper15, technical298 and 140 preserved annual artifact files.
Both root audits and every comparison CSV/JSON are hashed in a separate manifest,
whose hash is recorded in the sensitivity pointer. The master verifies this
evidence and displays the two policies separately.

## Verified results, 2026-10-15

Both panels are complete. The now-primary 80% panel is
`outputs/context_sensitivity/80pct-2010-2023-c39631454bfea7da`; the now-sensitivity 90% panel
remains `outputs/full_feature_panel/2010-2023-9992dcb5c83acdf2`. Both retain
168 signals,152,307 original stock-months and 140,777 paper-qualified stock-months.

Finite values across the 402 eligible context predictors, among paper-qualified rows:

| Outcome period |90% sensitivity (historical reference)|80% primary (historical sensitivity)|
|---|---:|---:|
|2010–2015|33.57%|92.09%|
|2010–2019 initial training|59.52%|94.99%|
|2020–2023 nominal tests|98.08%|99.27%|
|2010–2023 full history|71.06%|96.27%|

Against all original rows, the full-period fractions are 66.69% and 90.62%.
All 402 definitions have observations somewhere in both panels. The 80% panel
adds 14,653,642 finite original context cells, including13,942,798 before 2016
(95.15% of the gains). It loses no 90%-finite cell and changes no shared finite
value across any of the 415 context definitions. The additional values therefore
extend the information set on previously withheld dates/groups; shared ranks,
market aggregates and relative statistics do not shift.

In 2010, among paper-qualified rows, coverage rises 2.52%→63.27% and available
context names rise 56→356. All 402 names have observations in 2012 under 80%, versus
2016 under 90%; this does not imply full coverage at every intervening date.

### Cohort composition before 2016

There are 5,760 union cross-sectional base/date groups in 2010–2015. Of these,
1,720 pass both thresholds,3,531 are newly released at 80%, and 509 pass neither.
The newly released groups pool3,187,249 original member/feature/date slots:
2,803,748 observed and 383,501 omitted, or 87.97% observed. These slots repeat stocks
across features/dates and are not a count of unique securities.

- S&P400 comprises 44.57% of original slots,43.56% of observed slots, and 51.89%
  of omitted slots. This is an approximately1.00 percentage-point shortfall in
  observed S&P400 share; shared 90%-passing groups have a0.55 point shortfall.
- Of omitted slots,17.31% are identity-blocked and 82.69% are admitted/requested
  but lack a valid base-feature input. There are no unrequested admitted omissions.
  These are variable-specific unavailable inputs, not proof that every affected
  security lacks all historical prices.
- 85.77% of observed slots retain provisional identity-review flags, versus 86.77%
  in groups passing both thresholds. Lowering coverage does not certify identity.
- The report records past return,20-session volatility and log-dollar-volume
  profiles with separate finite counts. Most omitted slots lack those profiles;
  do not infer missing stocks' distress, sectors or listing histories.

The coverage gain is substantial, particularly before 2016. These descriptive
results do not establish that omitted observations are random or that 80% is the
better modelling policy. The user subsequently chose 80% as primary and 90% as
sensitivity; no model performance was evaluated to select the threshold.

### Verification and handoff

All 15 paper and 298 technical values, original keys, identity/paper eligibility
flags and 140 preserved artifact files are unchanged. The separate raw daily
core still contains3,563,099 observations; complete annual raw and past-only
histories retain9,507,584 and 9,224,882 rows, respectively, including annual overlap.
Both original panel pointers and all panel bytes are unchanged by role promotion. No target, preprocessing fit, model fit,
backtest or cloud mutation was used. Historical knowledge-time and economic
completeness remain unverified.

469 structural/adversarial tests passed. The cached build used1,945,939,027 output
bytes,2,594,750,464 peak RSS bytes and 9,148,299,260 conservatively projected cache
read bytes, completing in 638.77 seconds. The read metric budgets five artifact
passes and is not measured physical disk I/O.

Inside the original 80% run, `comparison/` contains per-feature/year/period value
and coverage tables, monthly/daily gate tables, detailed cohort profiles,
`preserved_artifact_hashes.csv`, `comparison_summary.json` and its pinned
manifest. The master shows the promoted primary/sensitivity datasets and verified
comparison tables; its versioned outputs remain empty.
