#!/usr/bin/env python3
"""Verify promoted cost-comparison model bytes and compact evidence."""

import argparse
import hashlib
import json
from pathlib import Path


SYSTEMS = ("Li", "Mo", "Cu", "Ni", "Si", "Ge")
MAX_GIT_BYTES = 100_000_000


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksums(result_root):
    checksum_path = result_root / "checksums.sha256"
    if not checksum_path.is_file():
        return 0
    checked = 0
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = checksum_path.parent / relative
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Evidence checksum mismatch: {path}")
        checked += 1
    return checked


def verify_models(root):
    checked = 0
    for system in SYSTEMS:
        directory = root / system
        manifest_path = directory / "model_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["system"] != system:
            raise RuntimeError(f"Wrong system in {manifest_path}")
        for record in manifest["artifacts"]:
            path = directory / record["path"]
            if not path.is_file():
                raise RuntimeError(f"Missing promoted model file: {path}")
            if path.stat().st_size != int(record["bytes"]):
                raise RuntimeError(f"Size mismatch: {path}")
            if sha256(path) != record["sha256"]:
                raise RuntimeError(f"Model checksum mismatch: {path}")
            checked += 1
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("results"),
        help=(
            "Result directory containing checksums.sha256; relative paths are "
            "resolved from --root."
        ),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    result_root = (
        args.result.resolve()
        if args.result.is_absolute()
        else (root / args.result).resolve()
    )
    oversized = [
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.stat().st_size >= MAX_GIT_BYTES
    ]
    if oversized:
        raise RuntimeError(f"Files exceed the public Git limit: {oversized}")
    result = {
        "schema": "ye3t_cost_comparison_bundle_verification_v1",
        "root": str(root),
        "model_artifacts_verified": verify_models(root),
        "evidence_artifacts_verified": verify_checksums(result_root),
        "result_root": str(result_root),
        "oversized_files": oversized,
        "passed": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
