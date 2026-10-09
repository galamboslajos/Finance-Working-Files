"""Metadata and integrity checks for the new U.S. workspace."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def local_path(relative):
    path = (ROOT / relative).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Input path leaves the project directory")
    return path


def catalog():
    config = json.loads((ROOT / "configs/data_sources.local.json").read_text())
    for source in config["sources"]:
        path = local_path(source["manifest"])
        manifest = json.loads(path.read_text())
        raw = json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        if "sha256:" + hashlib.sha256(raw).hexdigest() != source["manifest_sha256"]:
            raise ValueError(f"Manifest pin mismatch: {source['key']}")
        yield source, path.parent, manifest


def verify(hashes=False):
    count = size = 0
    for source, base, manifest in catalog():
        for obj in manifest["objects"]:
            path = (base / obj["relative_path"]).resolve()
            if not path.is_relative_to(base):
                raise ValueError("Object path leaves its release directory")
            if not path.is_file() or path.stat().st_size != obj["size_bytes"]:
                raise ValueError(f"Missing or wrong-size input: {path}")
            if hashes:
                h = hashlib.sha256()
                with path.open("rb") as f:
                    for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
                        h.update(block)
                if "sha256:" + h.hexdigest() != obj["sha256"]:
                    raise ValueError(f"Input hash mismatch: {path}")
            count += 1
            size += obj["size_bytes"]
        print(f"{source['label']}: {len(manifest['objects'])} files; {manifest['release_id']}")
    print(f"Verified {count} files / {size:,} bytes ({'SHA-256 and size' if hashes else 'size and presence'}).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("status")
    check = commands.add_parser("verify-data")
    check.add_argument("--hashes", action="store_true")
    schema = commands.add_parser("schema")
    schema.add_argument("source", choices=["intrinio", "siblis", "identity"])
    args = parser.parse_args()
    try:
        if args.command in ("status", "verify-data"):
            verify(getattr(args, "hashes", False))
        else:
            import pyarrow.parquet as pq
            for source, base, manifest in catalog():
                if source["key"] != args.source:
                    continue
                prefix = f"tables/{source['sample_table']}/"
                obj = next(o for o in manifest["objects"] if o["relative_path"].startswith(prefix))
                parquet = pq.ParquetFile(base / obj["relative_path"])
                print(parquet.schema_arrow)
                print(f"This file: {parquet.metadata.num_rows:,} rows; no financial values read.")
                break
        return 0
    except (OSError, ValueError, KeyError, StopIteration) as error:
        parser.exit(1, f"Data check failed: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
