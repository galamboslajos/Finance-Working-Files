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
