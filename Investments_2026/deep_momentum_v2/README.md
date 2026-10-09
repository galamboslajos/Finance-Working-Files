# Deep Momentum V2

This is the new U.S. equity research workspace. We will develop a modeling
approach inspired by the Deep Momentum paper and the previous U.S. attempt.
The existing `../deep_momentum/` folder is retained for observation and reference.

This directory belongs to the existing `Finance-Working-Files` Git repository.
It does not contain a separate Git repository.

## Start here

The local workspace already has a Python environment and authorized source data.
For a fresh checkout, create the environment first:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -e .
```

The data and local source pins are excluded from Git. Obtain the approved source
releases through your own cloud access, copy `configs/data_sources.example.json`
to `configs/data_sources.local.json`, and populate the local paths and canonical
manifest hashes from those releases. A clone alone does not include source data.

From a configured local workspace:

```sh
.venv/bin/python -m dmv2 status
.venv/bin/python -m dmv2 schema intrinio
.venv/bin/python -m dmv2 verify-data --hashes
```

The first command checks input files against the source manifests. The second
shows the price schema without reading financial values. The last verifies
every local file's size and SHA-256. Local reads do not require Google login.

## Layout

- `src/dmv2/`: our new code; currently a data catalog and verification CLI.
- `configs/data_sources.local.json`: private local paths and source pins.
- `data/raw/`: independent copies of the downloaded Intrinio, Siblis membership,
  and unified identity source releases, including manifests and certificates.
- `data/derived/`: future feature and model-input panels.
- `reference/sadig_us_recovery/`: a source snapshot of Sadig's U.S. recovery
  branch, including the five-feature and 740-feature construction code.
- `docs/RESEARCH_PLAN.md`: the starting research workflow.
- `docs/DATA_STATUS.md`: source availability, chronology and known gaps.
- `outputs/`: future reports and experiment artifacts.

Data, downloaded reference code, credentials, and generated outputs are ignored
by Git. The raw download report and a local relocation report retain provenance.

## Current state

SETUP COMPLETE. The new project is initialized and its source data verified.
No new features, model fits or backtests have been run. The reference code is
available locally; its historical paths have not been wired into this project.
The research design, paper interpretation, model and evaluation dates still need
to be specified before experiments begin.
