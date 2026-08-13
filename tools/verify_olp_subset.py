"""Build or verify the exact byte manifest for the frozen OLP subset."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def inspect(root: Path, relative_paths: list[str]) -> dict[str, object]:
    files: dict[str, dict[str, object]] = {}
    geometries: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    total_bytes = 0
    for relative in relative_paths:
        path = root / Path(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        size = path.stat().st_size
        record: dict[str, object] = {"bytes": size, "sha256": sha256_file(path)}
        if path.suffix.lower() == ".png":
            with Image.open(path) as image:
                geometry = f"{image.width}x{image.height}"
                record.update({"width": image.width, "height": image.height, "mode": image.mode})
                geometries[geometry] += 1
                modes[image.mode] += 1
        files[relative] = record
        total_bytes += size
    return {
        "schema_version": 1,
        "purpose": "Frozen OLP scene-grouped front-light subset byte manifest",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "png_geometries": dict(sorted(geometries.items())),
        "png_modes": dict(sorted(modes.items())),
        "files": files,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("members", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    relative_paths = [
        line.strip()
        for line in args.members.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    current = inspect(args.root, relative_paths)
    if args.check:
        expected = json.loads(args.manifest.read_text(encoding="utf-8"))
        if current != expected:
            changed = [
                relative
                for relative in sorted(set(current["files"]) | set(expected["files"]))
                if current["files"].get(relative) != expected["files"].get(relative)
            ]
            raise RuntimeError(f"OLP subset manifest mismatch: {changed[:10]}")
        print(
            json.dumps(
                {
                    "state": "verified",
                    "file_count": current["file_count"],
                    "total_bytes": current["total_bytes"],
                },
                indent=2,
            )
        )
        return
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "state": "created",
                "file_count": current["file_count"],
                "total_bytes": current["total_bytes"],
                "png_geometries": current["png_geometries"],
                "png_modes": current["png_modes"],
                "manifest": str(args.manifest),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
