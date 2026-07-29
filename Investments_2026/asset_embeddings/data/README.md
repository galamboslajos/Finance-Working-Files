# Local research data

This directory is intentionally ignored except for this README. Raw holdings, cloud metadata,
licensed data, and notebook-generated extracts must never be committed.

The first exploration notebook uses these local files:

| Local path | Purpose |
| --- | --- |
| `data/exploration/13f_holdings_sample.parquet` | Bounded 13F holdings sample |
| `data/exploration/nport_holdings_sample.parquet` | Bounded N-PORT holdings sample |
| `data/exploration/13f_variable_dictionary.parquet` | Optional 13F field dictionary |
| `data/exploration/nport_variable_dictionary.parquet` | Optional N-PORT field dictionary |

To fetch them safely:

1. Copy `.env.example` to the ignored `.env` file and fill in exact private object URIs.
2. Export those variables in the terminal used to launch Jupyter.
3. Open `notebooks/01_explore_13f_nport.ipynb`.
4. Set `FETCH_PRIVATE_OBJECTS = True` only for the first local download, then return it to `False`.

The notebook deliberately avoids bucket-wide listing and downloads only explicitly configured
objects. Clear notebook outputs before committing so sample rows and private inventory statistics
do not enter Git.

## Full-history mirror

The complete products can be mirrored into the ignored `data/full_history/` directory without
bucket-listing permission. Configure the two private manifest URIs shown in `.env.example`, then
validate the download plan:

~~~bash
python3 scripts/sync_full_history.py --dry-run
~~~

Start or resume the mirror with:

~~~bash
python3 scripts/sync_full_history.py
~~~

The downloader validates manifest paths, reserves local disk headroom, skips exact-size files,
resumes partial objects, refreshes short-lived Google access tokens, and verifies every downloaded
file by its manifest byte size. The data and downloaded private manifests remain ignored by Git.

## Full-history 13F notebook

`notebooks/02_explore_13f_full_history.ipynb` reads the complete local 13F mirror without loading
the full holdings payload into memory. If an executed copy is needed for local review, store it
below `data/exploration/`; that directory is ignored because the outputs contain non-public
inventory statistics.

`notebooks/03_audit_13f_data_spine.ipynb` streams all holdings rows in bounded batches and writes
only aggregate, fingerprinted cache files below `data/audits/13f_data_spine/`. Those caches and any
executed notebook remain ignored and must not be committed.

`notebooks/04_validate_13f_point_in_time_panel.ipynb` builds one decision-complete quarter from
local Parquet partitions. Store its executed copy below `data/exploration/`. The in-memory
`raw_holdings_asof_local` DataFrame is suitable for private Data Wrangler inspection but must never
be exported into a tracked path.
