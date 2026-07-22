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
- Current phase: inspect schemas and build the point-in-time data spine.
- Models and backtests: not started.

## The seven repository files

| File | Purpose |
| --- | --- |
| README.md | Fast orientation and current status |
| docs/PROJECT.md | Hypotheses, safeguards, benchmarks, and roadmap |
| docs/DATA.md | Cloud access, dataset inventory, and data rules |
| docs/PAPER_NOTES.md | Exact source paper and implementation lessons |
| AGENTS.md | Rules for AI-assisted work in this repository |
| scripts/gcloud | Safe wrapper using ignored project-local credentials |
| .gitignore | Prevents credentials, data, tools, and artifacts entering Git |

Implementation files will be added only when the first data pipeline is built.

## Research order

1. Validate point-in-time holdings, identifiers, prices, and fundamentals.
2. Build a simple holdings factor or recommender-system baseline.
3. Test relative valuation, return comovement, and masked-holding prediction.
4. Add Word2Vec or AssetBERT only if simpler models leave validated value.
5. Test a frozen strategy after costs, turnover, exposures, liquidity, and capacity.

## Cloud access

~~~bash
./scripts/gcloud auth list
./scripts/gcloud storage ls \
  gs://quant-research-trusty-agility-492109-v0/dataset-platform/curated/
~~~

Credentials, source data, generated artifacts, and the local Cloud SDK are ignored by Git.

## Transparency standard

- Every meaningful milestone is committed and pushed.
- Inputs and results identify their Git commit and data manifest.
- Availability timestamps are mandatory.
- Negative results and limitations stay visible.
- No raw or licensed data, credentials, or generated artifacts are committed.
