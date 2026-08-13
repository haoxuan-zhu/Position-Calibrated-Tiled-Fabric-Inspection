#!/usr/bin/env python3
"""Build the deterministic SHA-256 inventory used by verify_release.py."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "release_manifest.json"
IGNORED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "outputs",
    "venv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def included(path: Path) -> bool:
    relative = path.relative_to(ROOT)
    return (
        path.is_file()
        and path != OUTPUT
        and not any(part in IGNORED_PARTS for part in relative.parts)
        and path.suffix not in {".ckpt", ".log", ".pt", ".pth", ".pyc"}
    )


def main() -> None:
    files = {
        path.relative_to(ROOT).as_posix(): sha256(path)
        for path in sorted(ROOT.rglob("*"))
        if included(path)
    }
    payload = {
        "schema_version": 1,
        "source_commit": (ROOT / "SOURCE_COMMIT").read_text(encoding="utf-8").strip(),
        "files": files,
    }
    with OUTPUT.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(payload, indent=2) + "\n")
    print(f"RELEASE_MANIFEST_BUILT files={len(files)}")


if __name__ == "__main__":
    main()
