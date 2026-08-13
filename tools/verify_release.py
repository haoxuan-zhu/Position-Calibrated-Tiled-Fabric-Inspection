#!/usr/bin/env python3
"""Verify the immutable SHA-256 inventory of this release repository."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "release_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = payload["files"]
    failures: list[str] = []
    for relative, digest in expected.items():
        path = ROOT / relative
        if not path.is_file():
            failures.append(f"missing: {relative}")
        elif sha256(path) != digest:
            failures.append(f"hash mismatch: {relative}")

    if failures:
        raise SystemExit("RELEASE_VERIFY_FAILED\n" + "\n".join(failures))
    print(f"RELEASE_VERIFY_OK files={len(expected)} source_commit={payload['source_commit']}")


if __name__ == "__main__":
    main()
