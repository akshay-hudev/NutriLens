import importlib.util
from pathlib import Path

from reconstruction.types import PoseEstimationResult, ReconstructionResult
from services.segmentation_service import SegmentationResult
from utils.image_io import ImageAsset


class NeRFReconstructor:
    """Placeholder adapter for future NeRF training/inference backends."""

    def __init__(self, native_enabled: bool) -> None:
        self.native_enabled = native_enabled

    def reconstruct(
        self,
        assets: list[ImageAsset],
        segments: list[SegmentationResult],
        poses: PoseEstimationResult,
        run_dir: Path,
    ) -> ReconstructionResult:
        available = importlib.util.find_spec("torch") is not None
        warnings = list(poses.warnings)
        if not self.native_enabled or not available:
            warnings.append("NeRF backend is not configured; falling back to Gaussian Splat proxy.")
        else:
            warnings.append(
                "NeRF backend hook is ready, but training/inference integration is external to this demo."
            )

        return ReconstructionResult(
            status="not_configured",
            backend="nerf",
            confidence=0.0,
            image_count=len(assets),
            warnings=warnings,
        )
