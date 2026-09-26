#!/usr/bin/env python3
"""Refresh the checksum-managed public LAMMPS example inventory."""

import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "examples" / "PACKAGES" / "ye3t"
MANIFEST = ROOT / "YE3T_EXAMPLE_MANIFEST.sha256"


def main():
    rows = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path == MANIFEST:
            continue
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
    MANIFEST.write_text("\n".join(rows) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
