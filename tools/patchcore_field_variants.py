"""Derive controlled output fields from one frozen PatchCore forward pass.

The three variants isolate the effect of PatchCore's anomaly-map rendering
without changing embeddings, nearest-neighbour distances, crop geometry, or
processed pixels.  ``raw_patch`` keeps the native patch-score grid,
``resize_roundtrip`` applies the same nearest-neighbour upsampling used by
Anomalib and then projects back without smoothing, and ``anomalib_default``
uses the installed model's anomaly-map generator before the same projection.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F


RAW_PATCH = "raw_patch"
RESIZE_ROUNDTRIP = "resize_roundtrip"
ANOMALIB_DEFAULT = "anomalib_default"
FIELD_VARIANTS = (RAW_PATCH, RESIZE_ROUNDTRIP, ANOMALIB_DEFAULT)


def derive_patchcore_fields(
    patch_scores: torch.Tensor,
    image_size: tuple[int, int],
    output_size: tuple[int, int],
    anomaly_map_generator: Callable[[torch.Tensor, tuple[int, int]], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return three aligned score fields from identical patch scores.

    Args:
        patch_scores: Native PatchCore distances with shape ``(B, 1, H, W)``.
        image_size: Spatial size used by Anomalib's anomaly-map generator.
        output_size: Registered crop-field size used by the evaluation.
        anomaly_map_generator: Frozen model's standard map generator.

    The raw control is intentionally strict: the native grid must already
    equal the registered output size.  Silently interpolating it would defeat
    the purpose of testing whether the upsample/smooth/downsample path creates
    the crop-coordinate structure.
    """
    if patch_scores.ndim != 4 or patch_scores.shape[1] != 1:
        raise ValueError(
            "patch_scores must have shape (batch, 1, height, width), got "
            f"{tuple(patch_scores.shape)}"
        )
    image_size = (int(image_size[0]), int(image_size[1]))
    output_size = (int(output_size[0]), int(output_size[1]))
    if tuple(patch_scores.shape[-2:]) != output_size:
        raise ValueError(
            "raw patch-score grid does not match the registered output: "
            f"{tuple(patch_scores.shape[-2:])} versus {output_size}"
        )

    raw = patch_scores.float()
    enlarged = F.interpolate(patch_scores, size=image_size)
    resize_roundtrip = F.interpolate(
        enlarged.float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    standard = anomaly_map_generator(patch_scores, image_size)
    standard = F.interpolate(
        standard.float(),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )
    return {
        RAW_PATCH: raw,
        RESIZE_ROUNDTRIP: resize_roundtrip,
        ANOMALIB_DEFAULT: standard,
    }

