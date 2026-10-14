"""Pinned source checks and bounded historical-universe reconstruction."""
import argparse
import json
from pathlib import Path

from .catalog import SourceCatalog
from .data import load_universe
from .loading import PriceLoader, ReadBudget


ROOT = Path(__file__).resolve().parents[2]


def catalog(root=None):
    """Compatibility iterator for earlier local metadata scripts."""
    for source in SourceCatalog(root or ROOT):
        yield source.config, source.base, source.manifest


def _budgets(parser):
    parser.add_argument("--max-scanned-rows", type=int, default=1_000_000)
    parser.add_argument("--max-data-files", type=int, default=8)
    parser.add_argument("--max-output-rows", type=int, default=100_000)
    parser.add_argument("--max-read-mib", type=int, default=128)


def _budget(args):
    return ReadBudget(max_scanned_rows=args.max_scanned_rows,
                      max_data_files=args.max_data_files,
                      max_output_rows=args.max_output_rows,
                      max_read_bytes=args.max_read_mib * 1024 * 1024)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT,
                        help="V2 project root containing private source configuration")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    check = commands.add_parser("verify-data")
    check.add_argument("--hashes", action="store_true")
    schema = commands.add_parser("schema")
    schema.add_argument("source", choices=["intrinio", "siblis", "identity"])
    schema.add_argument("--table")
    universe = commands.add_parser("universe", help="Audit one historical constituent cohort")
    universe.add_argument("--as-of", required=True)
    universe.add_argument("--universe", choices=["sp500", "sp400", "combined"], required=True)
    universe.add_argument("--mode", choices=["historical_reconstruction", "knowledge_time"], required=True)
    _budgets(universe)
    prices = commands.add_parser("prices", help="Read declared pre-holdout IDs/dates only")
    prices.add_argument("--security-id", action="append", required=True)
    prices.add_argument("--start", required=True)
    prices.add_argument("--end", required=True)
    prices.add_argument("--column", action="append", required=True)
    _budgets(prices)
    args = parser.parse_args(argv)
    try:
        source_catalog = SourceCatalog(args.root)
        if args.command in ("status", "verify-data"):
            reports = source_catalog.verify(getattr(args, "hashes", False))
            for report in reports:
                print(f"{report.label}: {report.object_count} files; {report.release_id}")
            print(f"Verified {sum(r.object_count for r in reports)} files / "
                  f"{sum(r.size_bytes for r in reports):,} bytes.")
        elif args.command == "schema":
            table = args.table or source_catalog.source(args.source).config["sample_table"]
            print(source_catalog.table_schema(args.source, table))
            print("All pinned table footers checked; no data values read.")
        elif args.command == "universe":
            loaded = load_universe(source_catalog, as_of=args.as_of, universe=args.universe,
                                   mode=args.mode, budget=_budget(args))
            print(json.dumps({"as_of": loaded.snapshot.as_of,
                              "universe": loaded.snapshot.universe,
                              "summary": loaded.snapshot.summary,
                              "pit": loaded.snapshot.provenance,
                              "read_stats": loaded.loading_stats}, indent=2))
        elif args.command == "prices":
            loaded = PriceLoader(source_catalog, _budget(args)).load(
                args.security_id, args.start, args.end, args.column)
            # A declared bounded sample is displayed; never a full-table head scan.
            print(json.dumps({"stats": loaded.stats, "provenance": loaded.provenance,
                              "rows": loaded.table.to_pylist()}, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Data check failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
