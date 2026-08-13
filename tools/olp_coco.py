"""Small dependency-free helpers for the compressed COCO masks in OLP."""

from __future__ import annotations

from typing import Any

import numpy as np


def decode_compressed_coco_rle(segmentation: dict[str, Any]) -> np.ndarray:
    """Decode COCO's compressed RLE into a boolean H-by-W array."""

    height, width = map(int, segmentation["size"])
    encoded = segmentation["counts"]
    if not isinstance(encoded, str):
        raise TypeError("OLP is expected to use compressed string COCO RLE")
    counts: list[int] = []
    position = 0
    while position < len(encoded):
        value = 0
        shift = 0
        more = True
        while more:
            code = ord(encoded[position]) - 48
            position += 1
            value |= (code & 0x1F) << (5 * shift)
            more = bool(code & 0x20)
            if not more and (code & 0x10):
                value |= -1 << (5 * (shift + 1))
            shift += 1
        if len(counts) > 2:
            value += counts[-2]
        if value < 0:
            raise ValueError("Negative run length in compressed COCO RLE")
        counts.append(value)

    flat = np.zeros(height * width, dtype=np.uint8)
    cursor = 0
    foreground = False
    for run in counts:
        end = cursor + run
        if end > flat.size:
            raise ValueError("COCO RLE exceeds declared mask geometry")
        if foreground:
            flat[cursor:end] = 1
        cursor = end
        foreground = not foreground
    if cursor != flat.size:
        raise ValueError(
            f"COCO RLE covers {cursor} pixels, expected {flat.size}"
        )
    return flat.reshape((height, width), order="F").astype(bool, copy=False)
