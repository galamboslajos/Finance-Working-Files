"""Execute one source notebook, saving licensed outputs only under private data/."""
from pathlib import Path
import argparse
import os
import json
from datetime import datetime,timezone

REPORTS = {"01": "01_load_sp500_5min", "06": "06_build_volume_and_overnight_features",
           "07": "07_train_xgb_ret_volume_overnight", "09":"09_train_xgb_ret_joint"}


def run(project, number):
    project = Path(project).resolve()
    matches = list((project / "notebooks").glob(number + "_*.ipynb"))
    if len(matches) != 1 or number not in REPORTS:
        raise ValueError("Select a single active notebook: 01, 06, 07 or 09.")
    private = project / "data/reports" / REPORTS[number]
    private.mkdir(parents=True, exist_ok=True)
    for key, name in [("IPYTHONDIR", ".ipython"), ("MPLCONFIGDIR", ".matplotlib"),
                      ("JUPYTER_RUNTIME_DIR", ".jupyter_runtime")]:
        folder = private / name
        folder.mkdir(exist_ok=True)
        os.environ[key] = str(folder)
    os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      VECLIB_MAXIMUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    import nbformat
    from nbclient import NotebookClient
    notebook = nbformat.read(matches[0], as_version=4)
    # Inspection must never turn into training merely because a flag was edited.
    for cell in notebook.cells:
        if cell.cell_type == "code":
            for flag in ("RUN_TRAINING", "BUILD_IF_MISSING", "RUN_IF_MISSING"):
                import re
                cell.source = re.sub(rf"(?m)^(\s*{flag}\s*=\s*)True\b", r"\1False", cell.source)
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"Executing {number}: cell {cell_index+1}/{len(notebook.cells)}", flush=True)
    client = NotebookClient(notebook, timeout=None, kernel_name="python3",
        resources={"metadata": {"path": str(project)}}, on_cell_start=progress, record_timing=True)
    state=private/"REPORT_STATUS.json"
    def report_status(status,**details):
        state.write_text(json.dumps(dict(status=status,updated_at_utc=datetime.now(timezone.utc).isoformat(),**details),indent=2)+"\n")
    report_status("executing")
    try:
        client.execute()
    except Exception as error:
        nbformat.write(notebook, private / f"{matches[0].stem}.failed.ipynb")
        report_status("failed",error=str(error))
        raise
    destination = private / f"{matches[0].stem}.executed.ipynb"
    nbformat.write(notebook, destination)
    report_status("complete",executed_notebook=str(destination))
    print(f"Saved {destination}; report kernel shut down.", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("number", choices=sorted(REPORTS))
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    run(args.project, args.number)
