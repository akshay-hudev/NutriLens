import importlib.util
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.types import PoseEstimationResult, ReconstructionResult
from services.segmentation_service import SegmentationResult
from utils.image_io import ImageAsset, rel_path
from utils.volume_math import clamp, mask_bbox


LOGGER = logging.getLogger(__name__)


class GaussianSplatReconstructor:
    """Lightweight Gaussian-splat proxy with hooks for a native gsplat backend."""

    def __init__(self, static_dir: Path, frame_diameter_cm: float, native_enabled: bool) -> None:
        self.static_dir = Path(static_dir)
        self.frame_diameter_cm = frame_diameter_cm
        self.native_enabled = native_enabled

    def reconstruct(
        self,
        assets: list[ImageAsset],
        segments: list[SegmentationResult],
        poses: PoseEstimationResult,
        run_dir: Path,
    ) -> ReconstructionResult:
        warnings = list(poses.warnings)
        native_available = importlib.util.find_spec("gsplat") is not None
        if self.native_enabled and native_available:
            warnings.append(
                "Native gsplat package detected; this project currently uses the lightweight proxy hook."
            )
        else:
            warnings.append(
                "Native Gaussian Splatting is not configured; using a silhouette-depth Gaussian proxy."
            )

        measurements = self._measure_segments(assets, segments)
        if not measurements:
            return ReconstructionResult(
                status="failed",
                backend="gaussian-splat-proxy",
                confidence=0.08,
                image_count=len(assets),
                warnings=warnings + ["No valid segmentation masks were available for reconstruction."],
            )

        metadata = self._geometry_from_measurements(measurements, len(assets))
        point_cloud_path, point_count = self._write_proxy_point_cloud(
            measurements=measurements,
            metadata=metadata,
            poses=poses,
            run_dir=run_dir,
        )

        mask_confidence = float(np.mean([seg.confidence for seg in segments])) if segments else 0.2
        image_bonus = 0.08 * min(max(len(assets) - 1, 0), 4)
        confidence = clamp(0.18 + image_bonus + 0.42 * mask_confidence + poses.confidence * 0.25, 0.12, 0.72)
        if len(assets) == 1:
            confidence = min(confidence, 0.34)
            status = "single_view_depth_prior"
            warnings.append("Volume is low confidence because only one view constrains the food depth.")
        else:
            status = "sparse_view_proxy_reconstruction"

        artifact_rel = rel_path(point_cloud_path, self.static_dir)
        return ReconstructionResult(
            status=status,
            backend="gaussian-splat-proxy",
            confidence=round(confidence, 2),
            image_count=len(assets),
            point_count=point_count,
            artifact_path=str(point_cloud_path),
            artifact_rel_path=artifact_rel,
            metadata=metadata,
            warnings=warnings,
        )

    def _measure_segments(
        self, assets: list[ImageAsset], segments: list[SegmentationResult]
    ) -> list[dict]:
        measurements = []
        segment_by_image = {seg.image_path: seg for seg in segments}

        for asset in assets:
            segment = segment_by_image.get(asset.path)
            if not segment:
                continue

            mask = np.asarray(Image.open(segment.mask_path).convert("L")) > 127
            bbox = mask_bbox(mask)
            if not bbox:
                continue

            area_ratio = float(mask.mean())
            width_cm = clamp((bbox.width / asset.width) * self.frame_diameter_cm, 3.5, 30.0)
            plane_height_cm = clamp((bbox.height / asset.height) * self.frame_diameter_cm, 3.5, 30.0)
            area_cm2 = clamp(area_ratio * (self.frame_diameter_cm**2), 8.0, 650.0)
            measurements.append(
                {
                    "asset": asset,
                    "segment": segment,
                    "mask": mask,
                    "bbox": bbox,
                    "area_ratio": area_ratio,
                    "width_cm": width_cm,
                    "plane_height_cm": plane_height_cm,
                    "area_cm2": area_cm2,
                }
            )

        return measurements

    def _geometry_from_measurements(self, measurements: list[dict], image_count: int) -> dict:
        widths = np.asarray([m["width_cm"] for m in measurements], dtype=np.float32)
        plane_heights = np.asarray([m["plane_height_cm"] for m in measurements], dtype=np.float32)
        areas = np.asarray([m["area_cm2"] for m in measurements], dtype=np.float32)

        width_cm = float(np.median(widths))
        if image_count > 1:
            depth_cm = float(np.median(plane_heights) * 0.86)
        else:
            depth_cm = width_cm * 0.72

        footprint_area = float(np.median(areas))
        equivalent_radius = math.sqrt(max(footprint_area, 1.0) / math.pi)
        height_cm = clamp(1.1 + equivalent_radius * (0.20 if image_count > 1 else 0.14), 1.2, 8.5)

        if width_cm > 22 or depth_cm > 22:
            height_cm = min(height_cm, 5.5)

        fill_ratio = 0.62 if image_count > 1 else 0.54
        return {
            "width_cm": round(width_cm, 2),
            "depth_cm": round(clamp(depth_cm, 3.5, 30.0), 2),
            "height_cm": round(height_cm, 2),
            "footprint_area_cm2": round(footprint_area, 2),
            "fill_ratio": fill_ratio,
            "scale_assumption": f"Frame width approximates {self.frame_diameter_cm:g} cm around the plate.",
        }

    def _write_proxy_point_cloud(
        self,
        measurements: list[dict],
        metadata: dict,
        poses: PoseEstimationResult,
        run_dir: Path,
    ) -> tuple[Path, int]:
        output_path = run_dir / "reconstruction" / "food_proxy_gaussians.ply"
        points: list[tuple[float, float, float, int, int, int]] = []
        width_cm = float(metadata["width_cm"])
        depth_cm = float(metadata["depth_cm"])
        height_cm = float(metadata["height_cm"])

        pose_by_path = {pose["image_path"]: pose for pose in poses.poses}
        rng = np.random.default_rng(42)

        for measurement in measurements:
            asset = measurement["asset"]
            mask = measurement["mask"]
            ys, xs = np.where(mask)
            if len(xs) == 0:
                continue

            sample_count = min(800, len(xs))
            choice = rng.choice(len(xs), size=sample_count, replace=False)
            pose = pose_by_path.get(asset.path, {})
            angle = math.radians(float(pose.get("azimuth_deg", 0.0)))

            for x_px, y_px in zip(xs[choice], ys[choice]):
                x_norm = (x_px / max(asset.width - 1, 1)) - 0.5
                y_norm = 0.5 - (y_px / max(asset.height - 1, 1))
                radial = min(1.0, math.sqrt((x_norm / 0.5) ** 2 + (y_norm / 0.5) ** 2))
                local_x = x_norm * width_cm
                local_y = y_norm * depth_cm
                local_z = max(0.0, height_cm * (1.0 - 0.65 * radial))
                world_x = local_x * math.cos(angle) - local_y * math.sin(angle)
                world_y = local_x * math.sin(angle) + local_y * math.cos(angle)
                points.append((world_x, world_y, local_z, 34, 111, 84))

        with output_path.open("w", encoding="utf-8") as fh:
            fh.write("ply\nformat ascii 1.0\n")
            fh.write(f"element vertex {len(points)}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
            fh.write("end_header\n")
            for point in points:
                fh.write(
                    f"{point[0]:.4f} {point[1]:.4f} {point[2]:.4f} {point[3]} {point[4]} {point[5]}\n"
                )

        return output_path, len(points)
