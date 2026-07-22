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
- Current phase: bounded 13F/N-PORT schema exploration before building the point-in-time data spine.
- Models and backtests: not started.

## Repository map

| File | Purpose |
| --- | --- |
| README.md | Fast orientation and current status |
| docs/PROJECT.md | Hypotheses, safeguards, benchmarks, and roadmap |
| docs/DATA.md | Cloud access, dataset inventory, and data rules |
| docs/PAPER_NOTES.md | Exact source paper and implementation lessons |
| AGENTS.md | Rules for AI-assisted work in this repository |
| scripts/gcloud | Safe wrapper using ignored project-local credentials |
| notebooks/01_explore_13f_nport.ipynb | Bounded, point-in-time holdings exploration |
| src/holdings_exploration.py | Tested diagnostics used by the notebook |
| tests/test_holdings_exploration.py | Unit tests for timing, missing-state, and matrix filters |
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
