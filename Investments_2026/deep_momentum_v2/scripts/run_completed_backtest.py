"""Run the separate VVC/TSS action completion using saved quotes and forecasts."""
import argparse
import json
from pathlib import Path
import pickle
import sys
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--quote-cache", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    run = (root / args.run).resolve()
    if not run.is_relative_to(root / "outputs/independent_backtest"):
        parser.error("--run must identify an existing independent_backtest run")
    sys.path.insert(0, str(root / "src"))
    from dmv2.catalog import canonical_sha256
    from dmv2.classifier_pilot import _ResourceGuard
    from dmv2.completed_backtest import run_completed_backtest
    cfg = json.loads((root / "configs/independent_backtest_completion.json").read_text())
    identity = json.loads((run / "build_identity.json").read_text())
    pin = canonical_sha256(identity)
    started = time.monotonic()
    guard_cfg = {"execution": {"maximum_wall_seconds": cfg["execution"]["backtest_maximum_wall_seconds"],
        "maximum_peak_rss_bytes": cfg["execution"]["backtest_maximum_peak_rss_bytes"]}}
    # Trusted locally compiled receipt; the object checks its exact source pins.
    with _ResourceGuard(guard_cfg, started, run):
        with args.quote_cache.open("rb") as stream:
            quotes = pickle.load(stream)
        quotes.verify(root)
        result = run_completed_backtest(root, run, cfg, pin, quote_amendments=quotes,
            sni_evidence_directory=root / cfg["sni_evidence_directory"],
            vvc_evidence_directory=root / cfg["vvc_evidence_directory"],
            tss_evidence_directory=root / cfg["tss_evidence_directory"])
    print(json.dumps({"phase": "action_completion", "output": str(result),
        "new_model_fits": 0, "new_inference_runs": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
