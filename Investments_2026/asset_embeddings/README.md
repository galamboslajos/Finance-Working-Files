# Asset Embeddings

This project tests whether asset representations learned from investor holdings improve equity
research, risk measurement, and eventually a realistic trading strategy.

The central hypothesis is that holdings contain point-in-time information about asset similarity
that standard firm characteristics do not fully capture. An embedding is not automatically an
alpha signal: representation quality, predictive value, and net investment performance are tested
separately.

## Current position

- Research design: defined.
- Google Cloud data platform: connected and readable.
- Candidate data: 13F, N-PORT, company mappings, XBRL fundamentals, U.S. market data, membership,
  and factor benchmarks.
- Current phase: first holdings-representation smoke tests on saved point-in-time 13F matrices;
  N-PORT is deferred as a possible later extension.
- Trading backtests: not started.

## Repository map

| File | Purpose |
| --- | --- |
| README.md | Fast orientation and current status |
| docs/PROJECT.md | Hypotheses, safeguards, benchmarks, and roadmap |
| docs/DATA.md | Cloud access, dataset inventory, and data rules |
| docs/13F_DATA_SPINE.md | Provisional point-in-time 13F decisions and audit gates |
| docs/PAPER_NOTES.md | Exact source paper and implementation lessons |
| AGENTS.md | Rules for AI-assisted work in this repository |
| scripts/gcloud | Safe wrapper using ignored project-local credentials |
| notebooks/01_explore_13f_nport.ipynb | Bounded, point-in-time holdings exploration |
| notebooks/02_explore_13f_full_history.ipynb | Memory-safe orientation to the complete local 13F product |
| notebooks/03_audit_13f_data_spine.ipynb | Full-history audit of the provisional 13F data-spine rules |
| notebooks/04_validate_13f_point_in_time_panel.ipynb | First decision-time 13F state validation |
| notebooks/05_audit_13f_security_master.ipynb | Full-history security identity and first sparse matrix audit |
| notebooks/06_build_13f_pit_matrices.ipynb | Resumable full-history point-in-time matrix build and validation |
| src/holdings_exploration.py | Tested diagnostics used by the notebook |
| src/full_history_13f.py | Tested full-history Parquet and 13F timing diagnostics |
| src/data_spine_13f.py | Tested amendment, coverage, identity, and instrument audits |
| src/point_in_time_13f.py | Canonical availability-ordered 13F state transitions |
| src/security_master_13f.py | Point-in-time security identity and audited matrix-candidate rules |
| src/matrix_13f.py | Fingerprinted quarterly matrix materialization and full-history gates |
| src/representation_13f.py | Sparse centered-PCA and target-excluded masked-holding diagnostics |
| scripts/run_13f_pca_smoke.py | One-quarter local RS-Binary/RS-Ranks smoke-test runner |
| tests/test_holdings_exploration.py | Unit tests for timing, missing-state, and matrix filters |
| tests/test_full_history_13f.py | Unit tests for inventory, filing timing, and bounded sampling |
| tests/test_data_spine_13f.py | Unit tests for point-in-time data-spine audit rules |
| tests/test_point_in_time_13f.py | Unit tests for amendment state and no-lookahead behavior |
| tests/test_security_master_13f.py | Unit tests for security identity, mapping cutoffs, and matrix gates |
| tests/test_matrix_13f.py | Unit tests for full-history matrix construction and safe resume |
| tests/test_representation_13f.py | Unit tests for masking, rank ties, and sparse centered PCA |
| data/README.md | Safe local-data workflow |
| .env.example | Placeholder-only private input configuration |
| requirements.txt | Minimal Python dependencies |
| .gitignore | Prevents credentials, data, tools, and artifacts entering Git |

The notebook is an inspection tool, not a production pipeline. Its outputs remain local and
uncommitted.

## Research order

1. Validate point-in-time holdings, identifiers, prices, and fundamentals.
2. Build a simple holdings factor or recommender-system baseline.
3. Test relative valuation, return comovement, and masked-holding prediction.
4. Add Word2Vec or AssetBERT only if simpler models leave validated value.
5. Test a frozen strategy after costs, turnover, exposures, liquidity, and capacity.

## Cloud access

The cloud project, bucket URI, and authorized account are private operational configuration. Set
the curated URI locally from a value shared through an approved private channel:

~~~bash
export ASSET_EMBEDDINGS_CURATED_URI='gs://<private-bucket>/<private-prefix>/'
./scripts/gcloud auth list
./scripts/gcloud storage ls \
  "$ASSET_EMBEDDINGS_CURATED_URI"
~~~

Credentials, account identifiers, cloud resource identifiers, source data, generated artifacts,
and the local Cloud SDK must remain outside Git.

## First exploration

Install the small Python environment, configure exact private sample objects in an ignored `.env`,
and launch the notebook from this directory:

~~~bash
python -m pip install -r requirements.txt
set -a
source .env
set +a
jupyter lab notebooks/01_explore_13f_nport.ipynb
~~~

The notebook keeps economic, filing, and availability timestamps separate; profiles identifier
coverage and duplicates; preserves zero, negative, missing, and excluded states; and reports every
filter used to form a provisional investor-asset matrix.

## Full-history 13F orientation

After mirroring the complete 13F product, open:

~~~bash
jupyter lab notebooks/02_explore_13f_full_history.ipynb
~~~

The notebook uses Parquet footers for complete table shapes, loads the smaller filing index for
full-history timing analysis, and reads only a bounded cross-partition holdings sample. It explains
the difference between the portfolio date, filing time, and permitted availability time without
defining production amendment, asset-mapping, or equity-universe rules.

Executed outputs contain private inventory statistics and remain local and uncommitted.

## Full-history 13F data-spine audit

The seven provisional point-in-time decisions are recorded in `docs/13F_DATA_SPINE.md`. To test
them against the complete local history, open:

~~~bash
jupyter lab notebooks/03_audit_13f_data_spine.ipynb
~~~

The first run streams every holdings row in bounded Parquet batches. Only aggregate results are
retained, and a fingerprinted ignored cache under `data/audits/` prevents unchanged inputs from
being rescanned. The notebook remains output-free in Git and does not build a modeling matrix.

## Canonical point-in-time 13F panel

`src/point_in_time_13f.py` implements the first decision-time state builder. It:

- orders filing events by permitted availability, filed time, and accession;
- supplements with `NEW_HOLDINGS` amendments and replaces with `RESTATEMENT` amendments;
- retains notices, combination reports, future filings, and quarantines in an event ledger;
- marks incomplete or ambiguous manager-period states as ineligible rather than silently fixing
  them; and
- preserves filing-line identifiers without applying a present-day issuer mapping.

Run the local validation with:

~~~bash
jupyter lab notebooks/04_validate_13f_point_in_time_panel.ipynb
~~~

The notebook visualizes the full-history relationship between economic report dates and filing
availability, derives the first decision-complete quarter from the local product, streams only the
required filed-month holdings partitions, and exposes `raw_holdings_asof_local` for private Data
Wrangler inspection. Executed outputs remain local and uncommitted.

## Point-in-time security identity

`src/security_master_13f.py` and `notebooks/05_audit_13f_security_master.ipynb` establish the first
model-ready asset grain without applying a future issuer mapping. They:

- use the normalized CUSIP reported with each filing, namespaced by instrument channel;
- preserve options, debt-like instruments, preferred shares, units, warrants, and explicit funds
  as separate channels;
- use `cash_share_candidate` as the first holdings-representation baseline without calling it
  verified common equity;
- audit identifier coverage and collisions across the complete holdings history with an ignored,
  fingerprinted aggregate cache;
- exclude mapping snapshots that were not knowable before the historical decision time; and
- build a sparse first-quarter matrix candidate whose 75 percent concentration and iterative
  20-manager/20-security coverage rules hold jointly at convergence.

The notebook exposes `security_rows_local` and `matrix_pairs_local` for private Data Wrangler use.
Those frames contain raw identifiers and must remain local. A point-in-time market security master
is still required before describing the universe as common equity or joining returns for a trading
test.

## Full-history point-in-time matrices

`src/matrix_13f.py` and `notebooks/06_build_13f_pit_matrices.ipynb` materialize the representation
input for every locally derived decision-complete quarter. The baseline:

- accepts only a normalized, source-reported nine-character CUSIP and never guesses a repair for a
  shorter identifier;
- builds each manager state from events available by that quarter's standardized cutoff;
- keeps one sparse typed cash-share pair per manager and security;
- jointly reapplies the 75 percent concentration and iterative 20-by-20 degree rules until both
  hold at convergence;
- records data, source-code, policy, parameter, and output-file fingerprints; and
- blocks handoff when a quarter is missing, stale, modified, duplicated, or fails a no-lookahead,
  channel, degree, concentration, uniqueness, or weight-reconciliation check.

Generated partitions live below ignored `data/model_inputs/`. The notebook exposes one selected
quarter as `matrix_pairs_local` for private Data Wrangler inspection and a partitioned Arrow
dataset as `matrix_dataset_local` for memory-bounded full-history work.

## First representation smoke tests

Run the paper's simpler RS-Binary and RS-Ranks PCA constructions on a saved quarterly matrix:

~~~bash
python -m scripts.run_13f_pca_smoke --quarter YYYYQX --save
~~~

Replace `YYYYQX` with a locally available report quarter.

The runner masks each manager's second-largest retained holding **before** fitting, recomputes
within-manager ranks on the remaining holdings, and compares 4- and 10-dimensional PCA scores
with asset popularity. It reports top-100 recovery, mean reciprocal rank, and rank percentile.
The seed, matrix-file hash, code hash, and limitations accompany each ignored local JSON run.
The saved matrix universe was filtered before masking, so this is a smoke test, **not** a sealed
paper benchmark or evidence of tradable alpha. A later benchmark must split before universe
selection and use the paper's actual likelihood protocol. No raw manager or security identifiers,
matrix rows, or run outputs are committed.

## Transparency standard

- Every meaningful milestone is committed and pushed.
- Inputs and results identify their Git commit and data manifest.
- Availability timestamps are mandatory.
- Negative results and limitations stay visible.
- No raw or licensed data, credentials, cloud resource identifiers, or generated artifacts are
  committed.

## Git and collaboration policy

This project is not a standalone repository. Its permanent location is:

- Repository: https://github.com/galamboslajos/Finance-Working-Files
- Directory: Investments_2026/asset_embeddings
- Default branch: main

For substantive work:

1. Verify that the Git top level is Finance-Working-Files and origin points to the repository above.
2. Update main and create an agent/ feature branch.
3. Stage only explicit files under Investments_2026/asset_embeddings.
4. Review the staged diff and run relevant validation.
5. Commit and push each coherent milestone.
6. Open or update a draft pull request against main.
7. Report the branch, commit hash, checks, and PR URL at handoff.

Never initialize Git inside this directory, use repository-wide bulk staging, or claim that local
work is shared before the remote commit has been verified.
