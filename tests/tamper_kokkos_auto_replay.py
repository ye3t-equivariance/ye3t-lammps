#!/usr/bin/env python3
"""Create a self-consistent replay with one intentionally wrong identity."""

import hashlib
import json
import sys
from pathlib import Path


if len(sys.argv) not in (3, 4):
    raise SystemExit(
        "usage: tamper_kokkos_auto_replay.py INPUT OUTPUT [executable|device]"
    )

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
identity = sys.argv[3] if len(sys.argv) == 4 else "executable"
if identity == "executable":
    payload["runtime"]["lammps_executable_sha256"] = "0" * 64
elif identity == "device":
    payload["device"]["device_class_sha256"] = "0" * 64
else:
    raise SystemExit("identity must be executable or device")
unsigned = {key: value for key, value in payload.items() if key != "replay_sha256"}
canonical = json.dumps(
    unsigned,
    allow_nan=False,
    ensure_ascii=True,
    separators=(",", ":"),
    sort_keys=True,
).encode("ascii")
payload["replay_sha256"] = hashlib.sha256(canonical).hexdigest()
Path(sys.argv[2]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
