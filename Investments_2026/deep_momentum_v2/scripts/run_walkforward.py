"""Explicit, checkpointed 2020–2023 workflow; never invoked by notebook Run All."""
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("plan", "build", "fold", "finalize-models", "preflight", "backtest"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--run", type=Path)
    parser.add_argument("--year", type=int, choices=(2020, 2021, 2022, 2023))
    parser.add_argument("--event-registry", type=Path)
    parser.add_argument("--backtest-output", type=Path,
        help="A new directory within --run for a separately versioned action/quote rerun")
    parser.add_argument("--allow-withheld-paths", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    sys.path.insert(0, str(root / "src"))
    from dmv2.catalog import canonical_sha256
    from dmv2.walkforward_panel import validate_walkforward_configuration
    cfg = validate_walkforward_configuration(json.loads((root / "configs/walkforward_2020_2023.json").read_text()))
    if args.phase in ("plan", "build"):
        from dmv2.walkforward_panel import plan_walkforward_panel, build_walkforward_panel
        # This trusted local bundle checks all its receipt/source hashes.
        cache = (root / cfg["quote_cache"]).resolve()
        if not cache.is_relative_to(root / "data/derived"):
            parser.error("Quote bundle must be a private derived-data artifact")
        if cache.stat().st_size > 16 * 1024**2:
            parser.error("Compiled quote bundle exceeds the bounded local receipt size")
        if hashlib.sha256(cache.read_bytes()).hexdigest() != cfg.get("quote_cache_sha256"):
            parser.error("Compiled quote bundle bytes differ from the frozen protocol")
        with cache.open("rb") as stream:
            quotes = pickle.load(stream)
        quotes.verify(root)
        if args.phase == "plan":
            identity, pin = plan_walkforward_panel(root, cfg, quotes)
            print(json.dumps({"phase": "plan", "build_identity_sha256": pin,
                "original_stock_months": identity["original_stock_months"],
                "projected_compressed_bytes": identity["projected_compressed_bytes"],
                "source_price_cache_tables_decoded": False,
                "compiled_quote_overlay_loaded_and_verified": True,
                "new_model_fits": 0}, sort_keys=True))
        else:
            result = build_walkforward_panel(root, cfg, quotes, approved_financial_read=True)
            print(json.dumps({"phase": "features_and_targets", "output": str(result), "new_model_fits": 0}, sort_keys=True))
        return
    if args.run is None:
        parser.error("--run is required after the feature build")
    run = (root / args.run).resolve()
    if not run.is_relative_to(root / "outputs/walkforward"):
        parser.error("--run must be a private walkforward run")
    identity = json.loads((run / "build_identity.json").read_text())
    pin = canonical_sha256(identity)
    if identity["configuration"] != cfg:
        parser.error("Frozen run configuration differs from the current protocol")
    if args.phase == "fold":
        if args.year is None:
            parser.error("--year is required for one explicit annual fold")
        from dmv2.walkforward_models import run_walkforward_fold
        result = run_walkforward_fold(root, run, cfg, pin, args.year)
    elif args.phase == "finalize-models":
        from dmv2.walkforward_models import finalize_walkforward_models
        result = finalize_walkforward_models(root, run, cfg, pin)
    else:
        from dmv2.walkforward_backtest import run_walkforward_backtest
        registry = None
        if args.event_registry is not None:
            path = (root / args.event_registry).resolve()
            if not path.is_relative_to(root / "data/derived"):
                parser.error("Action registry must be a private derived-data artifact")
            registry = path
        output = None
        if args.backtest_output is not None:
            output = (root / args.backtest_output).resolve()
            if not output.is_relative_to(run):
                parser.error("--backtest-output must remain within the selected private run")
        result = run_walkforward_backtest(root, run, cfg, pin,
            event_registry=registry, preflight_only=args.phase == "preflight",
            output_directory=output,
            allow_withheld_paths=args.allow_withheld_paths)
    print(json.dumps({"phase": args.phase, "output": str(result), "year": args.year}, sort_keys=True))


if __name__ == "__main__":
    main()
