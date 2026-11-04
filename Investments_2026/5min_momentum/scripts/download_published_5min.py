"""Download the supplied published release, with pinned generations and SHA256.

Only HTTP GET requests are made to GCS. Credentials remain in memory.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

from gcs_readonly import CTX, token


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/alpaca_5min_20261103"
MANIFEST = DEST / "MANIFEST.json"
BUCKET = "quant-research-trusty-agility-492109-v0"


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def fetch(obj):
    relative = Path(obj["relative_path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Invalid manifest-relative path")
    uri = obj["uri"]
    if not uri.startswith(f"gs://{BUCKET}/dataset-platform/runs/"):
        raise ValueError("Unexpected source outside this dataset's GCS run folder")
    destination = DEST / relative
    if destination.exists():
        if destination.stat().st_size != obj["size_bytes"] or sha256(destination) != obj["sha256"]:
            raise ValueError(f"Existing file does not match manifest: {relative}")
        return {"relative_path": str(relative), "bytes": obj["size_bytes"], "cached": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    bucket, name = uri[5:].split("/", 1)
    url = ("https://storage.googleapis.com/storage/v1/b/" + quote(bucket, safe="")
           + "/o/" + quote(name, safe="") + "?alt=media&generation=" + str(obj["generation"]))
    for attempt in range(5):
        try:
            request = Request(url, headers={"Authorization": "Bearer " + token(),
                              "Accept-Encoding": "identity"})
            h = hashlib.sha256()
            size = 0
            with urlopen(request, timeout=90, context=CTX) as response, temporary.open("wb") as output:
                if response.status != 200 or response.headers.get("x-goog-generation") != str(obj["generation"]):
                    raise ValueError(f"Source generation mismatch: {relative}")
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    h.update(chunk)
                    size += len(chunk)
            if size != obj["size_bytes"] or "sha256:" + h.hexdigest() != obj["sha256"]:
                raise ValueError(f"Downloaded bytes do not match manifest: {relative}")
            os.replace(temporary, destination)
            return {"relative_path": str(relative), "bytes": size, "cached": False}
        except HTTPError as error:
            if error.code == 401:
                token(refresh=True)
            elif error.code not in (408, 429, 500, 502, 503, 504):
                raise
            if attempt == 4:
                raise
        except Exception:
            if attempt == 4:
                raise
        time.sleep(min(2 ** attempt, 8))


def main():
    manifest = json.loads(MANIFEST.read_text())
    if manifest["dataset_id"] != "equity_us_intraday_5min_alpaca_v1" or manifest["release_id"] != "professional-source-build-20261103-r1":
        raise ValueError("Unexpected dataset or release")
    objects = sorted(manifest["objects"], key=lambda x: (x["relative_path"].startswith("tables/bars_5min/"), x["relative_path"]))
    if len({x["relative_path"] for x in objects}) != len(objects):
        raise ValueError("Duplicate download destinations")
    total = sum(x["size_bytes"] for x in objects)
    if total > 50_000_000_000:
        raise ValueError("Release exceeds the authorized download-size threshold")
    started = time.monotonic()
    completed = []
    report = {"release_id": manifest["release_id"], "complete": False,
              "expected_objects": len(objects), "expected_bytes": total,
              "manifest_sha256": sha256(MANIFEST)}
    token()
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(fetch, obj): obj for obj in objects}
        for future in as_completed(futures):
            completed.append(future.result())
            report.update({"completed_objects": len(completed),
                           "verified_bytes": sum(x["bytes"] for x in completed),
                           "elapsed_seconds": round(time.monotonic() - started, 1),
                           "objects": completed})
            temporary = DEST / "DOWNLOAD_REPORT.json.tmp"
            temporary.write_text(json.dumps(report, indent=2) + "\n")
            os.replace(temporary, DEST / "DOWNLOAD_REPORT.json")
            print(json.dumps({k: v for k, v in report.items() if k != "objects"}), flush=True)
    report.update({"complete": True, "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    (DEST / "DOWNLOAD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "objects"}), flush=True)


if __name__ == "__main__":
    main()
