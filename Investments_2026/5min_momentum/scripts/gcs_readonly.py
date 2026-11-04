"""Read-only GCS authentication; access tokens remain in process memory."""
import os
from pathlib import Path
import shutil
import ssl
import subprocess
import threading
import time

import certifi

GCLOUD = (os.environ.get("GCLOUD_BIN") or shutil.which("gcloud")
          or str(Path.home() / "google-cloud-sdk/bin/gcloud"))
CTX = ssl.create_default_context(cafile=certifi.where())
TOKEN_LOCK = threading.Lock()
TOKEN = None
TOKEN_AT = 0


def token(refresh=False):
    global TOKEN, TOKEN_AT
    with TOKEN_LOCK:
        if refresh or TOKEN is None or time.monotonic() - TOKEN_AT > 2400:
            result = subprocess.run([GCLOUD, "auth", "print-access-token", "--quiet"],
                                    capture_output=True, timeout=60)
            if result.returncode:
                raise RuntimeError("Unable to obtain a gcloud access token; check gcloud auth login")
            TOKEN = result.stdout.decode().strip()
            TOKEN_AT = time.monotonic()
        return TOKEN
