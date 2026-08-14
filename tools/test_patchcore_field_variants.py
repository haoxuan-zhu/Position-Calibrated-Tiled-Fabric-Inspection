"""Unit checks for the PatchCore field-rendering controls."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from patchcore_field_variants import (
    ANOMALIB_DEFAULT,
    RAW_PATCH,
    RESIZE_ROUNDTRIP,
    derive_patchcore_fields,
)


class FakeGenerator:
    """Expose a deterministic non-identity standard rendering path."""

    def __call__(
        self, patch_scores: torch.Tensor, image_size: tuple[int, int]
    ) -> torch.Tensor:
        return F.interpolate(patch_scores, size=image_size) + 2.0


def main() -> None:
    patch_scores = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
    fields = derive_patchcore_fields(
        patch_scores,
        image_size=(8, 8),
        output_size=(4, 4),
        anomaly_map_generator=FakeGenerator(),
    )
    assert torch.equal(fields[RAW_PATCH], patch_scores)
    expected_roundtrip = F.interpolate(
        F.interpolate(patch_scores, size=(8, 8)),
        size=(4, 4),
        mode="bilinear",
        align_corners=False,
    )
    assert torch.equal(fields[RESIZE_ROUNDTRIP], expected_roundtrip)
    assert torch.equal(fields[ANOMALIB_DEFAULT], expected_roundtrip + 2.0)

    try:
        derive_patchcore_fields(
            patch_scores,
            image_size=(8, 8),
            output_size=(5, 5),
            anomaly_map_generator=FakeGenerator(),
        )
    except ValueError as error:
        assert "raw patch-score grid" in str(error)
    else:
        raise AssertionError("raw-grid shape mismatch was not rejected")

    print("PATCHCORE_FIELD_VARIANT_TEST_OK")


if __name__ == "__main__":
    main()

