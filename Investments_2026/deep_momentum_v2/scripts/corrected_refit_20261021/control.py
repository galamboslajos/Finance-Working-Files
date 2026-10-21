"""Small immutable-plan and cumulative-resource helpers for the corrected rerun."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import fcntl

ROOT = Path(__file__).resolve().parent
CAPS = {"compressed_source": 12 * 1024**3, "derived_matrix": 32 * 1024**3}


def need(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def pin(path):
    p = Path(path).resolve()
    need(p.is_file() and not Path(path).is_symlink(), "Pin requires a regular file")
    return {"path": str(p), "sha256": digest(p), "size_bytes": p.stat().st_size}


def check(item):
    p = Path(item["path"])
    need(p.is_absolute() and not p.is_symlink(), "Absolute regular pin required")
    need(pin(p) == {key: item[key] for key in ("path", "sha256", "size_bytes")},
         "Pinned bytes changed: " + str(p))
    return p


def read(path):
    return json.loads(Path(path).read_text())


def save_new(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return pin(p)


def replace_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temporary = p.with_name(p.name + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, p)


def reserve_read(root, label, amount, detail, category="compressed_source"):
    """Charge before decoding. Repeated labels and budget resets are forbidden."""
    root = Path(root).resolve()
    need(root == ROOT, "Unexpected rerun root")
    need(isinstance(label, str) and label and category in CAPS, "Unknown reservation")
    need(type(amount) is int and amount >= 0, "Nonnegative byte count required")
    directory = root / "run"
    directory.mkdir(exist_ok=True)
    with (directory / ".read_usage.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = directory / "read_usage.json"
        usage = read(path) if path.exists() else {
            "schema": "dmv2-corrected-refit-read-usage/v1", "limits": CAPS,
            "totals": {key: 0 for key in CAPS}, "reservations": []}
        need(usage["limits"] == CAPS, "Read limits changed")
        need(not any(x["label"] == label for x in usage["reservations"]),
             "Reservation already consumed; preserve the attempt")
        need(usage["totals"][category] + amount <= CAPS[category], "Read budget exhausted")
        if "plan_pin" in detail:
            check(detail["plan_pin"])
        usage["reservations"].append({"label": label, "category": category,
            "bytes": amount, "detail": detail,
            "reserved_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()})
        usage["totals"][category] += amount
        replace_json(path, usage)
        return usage["reservations"][-1]


def require_reservation(root, label, amount, plan_pin, category="compressed_source"):
    usage = read(Path(root) / "run/read_usage.json")
    rows = [row for row in usage["reservations"] if row["label"] == label]
    need(len(rows) == 1, "Missing exact read reservation")
    row = rows[0]
    need(row["category"] == category and row["bytes"] == amount
         and row["detail"].get("plan_pin") == plan_pin, "Read reservation differs from plan")
    check(plan_pin)
    return row
