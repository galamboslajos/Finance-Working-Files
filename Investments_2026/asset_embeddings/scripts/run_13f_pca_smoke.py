"""Run local, non-trading PCA diagnostics on one saved 13F PIT quarter.

Run from the project directory with ``python -m scripts.run_13f_pca_smoke``.
The pair table and optional output stay in ignored local data directories.
"""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import re
import subprocess

from src.representation_13f import load_pairs, run_smoke_test


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MATRIX_ROOT = PROJECT_ROOT / "data" / "model_inputs" / "13f_cash_share_pit" / "v1"
DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "model_runs" / "13f_pca_smoke"
MATRIX_CONTRACT = "13f-pit-matrix-v1"


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_matrix_manifest(
    matrix_root: Path, quarter: str, pair_sha256: str
) -> None:
    """Reject a missing, mismatched, or unregistered private pair table."""

    manifest_path = matrix_root / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("The matrix run manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("contract_version") != MATRIX_CONTRACT:
        raise ValueError("The matrix contract does not match this runner")
    records = [
        record
        for record in manifest.get("materialized_quarters", [])
        if Path(record.get("quarter_directory", "")).name == f"report_quarter={quarter}"
    ]
    if len(records) != 1:
        raise ValueError("The requested quarter is not uniquely registered")
    if records[0].get("pair_file_sha256") != pair_sha256:
        raise ValueError("The pair table does not match its registered file hash")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quarter", required=True, help="Report quarter in YYYYQ[1-4] form")
    parser.add_argument("--matrix-root", type=Path, default=DEFAULT_MATRIX_ROOT)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--save", action="store_true", help="Save JSON in ignored local data")
    arguments = parser.parse_args()
    if not re.fullmatch(r"\d{4}Q[1-4]", arguments.quarter):
        parser.error("--quarter must have YYYYQ[1-4] form")
    pair_path = arguments.matrix_root / f"report_quarter={arguments.quarter}" / "pairs.parquet"
    if not pair_path.is_file():
        parser.error("The requested quarterly pair table is not available locally")
    pair_sha256 = file_sha256(pair_path)
    validate_matrix_manifest(arguments.matrix_root, arguments.quarter, pair_sha256)
    pairs = load_pairs(pair_path)
    metrics = run_smoke_test(pairs, seed=arguments.seed)
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    model_code_status = subprocess.run(
        ["git", "status", "--porcelain", "--", "src/representation_13f.py", "scripts/run_13f_pca_smoke.py"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "purpose": "representation_smoke_test_not_strategy_or_paper_replication",
        "report_quarter": arguments.quarter,
        "matrix_contract": MATRIX_CONTRACT,
        "source_pair_file_sha256": pair_sha256,
        "source_code_commit": git_commit,
        "model_code_committed_at_head": not bool(model_code_status),
        "model_code_sha256": file_sha256(PROJECT_ROOT / "src" / "representation_13f.py"),
        "seed": arguments.seed,
        "masked_edge_policy": "one_second_largest_retained_position_per_manager",
        "rank_tie_policy": "ascending_typed_security_id",
        "limitations": [
            "The saved universe was filtered before edges were masked.",
            "A 13F manager CIK is coarser than the paper's usual fund-level investor.",
            "These are within-quarter representation diagnostics, not return forecasts.",
        ],
        "results": metrics,
    }
    if arguments.save:
        DEFAULT_RUN_ROOT.mkdir(parents=True, exist_ok=True)
        destination = DEFAULT_RUN_ROOT / f"{arguments.quarter}_seed{arguments.seed}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Saved local ignored run summary: {destination}")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
