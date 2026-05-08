import logging
from dataclasses import dataclass
from pathlib import Path

from config import Config
from reconstruction.colmap import estimate_camera_poses
from reconstruction.gsplat import GaussianSplatReconstructor
from reconstruction.nerf import NeRFReconstructor
from reconstruction.types import PoseEstimationResult, ReconstructionResult
from reconstruction.volume import estimate_volume_cm3
from services.segmentation_service import SegmentationResult
from utils.image_io import ImageAsset


LOGGER = logging.getLogger(__name__)


@dataclass
class ReconstructionPipelineResult:
    poses: PoseEstimationResult
    reconstruction: ReconstructionResult
    volume: dict


class ReconstructionService:
    def __init__(self, config: type[Config]) -> None:
        self.config = config
        self.gsplat = GaussianSplatReconstructor(
            static_dir=config.STATIC_DIR,
            frame_diameter_cm=config.DEFAULT_FRAME_DIAMETER_CM,
            native_enabled=config.ENABLE_GSPLAT,
        )
        self.nerf = NeRFReconstructor(native_enabled=config.ENABLE_NERF)

    def reconstruct(
        self,
        assets: list[ImageAsset],
        segments: list[SegmentationResult],
        run_dir: Path,
    ) -> ReconstructionPipelineResult:
        image_paths = [asset.path for asset in assets]
        poses = estimate_camera_poses(
            image_paths=image_paths,
            run_dir=run_dir,
            colmap_bin=self.config.COLMAP_BIN,
            enabled=self.config.ENABLE_COLMAP,
            timeout_seconds=self.config.RECONSTRUCTION_TIMEOUT_SECONDS,
        )

        reconstruction = self._run_backend(assets, segments, poses, run_dir)
        volume = estimate_volume_cm3(reconstruction)
        return ReconstructionPipelineResult(poses=poses, reconstruction=reconstruction, volume=volume)

    def _run_backend(
        self,
        assets: list[ImageAsset],
        segments: list[SegmentationResult],
        poses: PoseEstimationResult,
        run_dir: Path,
    ) -> ReconstructionResult:
        try:
            if self.config.RECONSTRUCTION_BACKEND == "nerf":
                nerf_result = self.nerf.reconstruct(assets, segments, poses, run_dir)
                if nerf_result.status != "not_configured":
                    return nerf_result

            return self.gsplat.reconstruct(assets, segments, poses, run_dir)
        except Exception as exc:
            LOGGER.exception("Reconstruction backend failed")
            return ReconstructionResult(
                status="failed",
                backend=self.config.RECONSTRUCTION_BACKEND,
                confidence=0.1,
                image_count=len(assets),
                warnings=[f"Reconstruction failed: {exc}. A generic volume prior was used."],
            )
