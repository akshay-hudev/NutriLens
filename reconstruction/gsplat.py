import importlib.util
import logging
import math
from pathlib import Path

import numpy as np
from PIL import Image

from reconstruction.types import PoseEstimationResult, ReconstructionResult, ScaleEstimationResult
from services.segmentation_service import SegmentationResult
from utils.image_io import ImageAsset, rel_path
from utils.volume_math import clamp, mask_bbox


LOGGER = logging.getLogger(__name__)


class GaussianSplatReconstructor:
    """Lightweight Gaussian-splat proxy with hooks for a native gsplat backend."""

    def __init__(
        self,
        static_dir: Path,
        frame_diameter_cm: float,
        voxel_size_cm: float,
        voxel_max_dim: int,
        colmap_point_limit: int,
        native_enabled: bool,
    ) -> None:
        self.static_dir = Path(static_dir)
        self.frame_diameter_cm = frame_diameter_cm
        self.voxel_size_cm = voxel_size_cm
        self.voxel_max_dim = voxel_max_dim
        self.colmap_point_limit = colmap_point_limit
        self.native_enabled = native_enabled

    def reconstruct(
        self,
        assets: list[ImageAsset],
        segments: list[SegmentationResult],
        poses: PoseEstimationResult,
        scale: ScaleEstimationResult,
        run_dir: Path,
    ) -> ReconstructionResult:
        warnings = list(poses.warnings)
        warnings.extend(scale.warnings)
        native_available = importlib.util.find_spec("gsplat") is not None
        if self.native_enabled and native_available:
            warnings.append(
                "Native gsplat package detected; this project currently uses the lightweight proxy hook."
            )
        else:
            warnings.append(
                "Native Gaussian Splatting is not configured; using a silhouette-depth Gaussian proxy."
            )

        measurements = self._measure_segments(assets, segments, scale)
        if not measurements:
            return ReconstructionResult(
                status="failed",
                backend="gaussian-splat-proxy",
                confidence=0.08,
                image_count=len(assets),
                warnings=warnings + ["No valid segmentation masks were available for reconstruction."],
            )

        metadata = self._geometry_from_measurements(measurements, len(assets))
        metadata["scale_cm_per_px"] = round(scale.scale_cm_per_px, 6)
        metadata["scale_method"] = scale.method
        metadata["scale_confidence"] = round(scale.confidence, 3)
        metadata["scale_relative_uncertainty"] = round(scale.relative_uncertainty, 3)
        metadata["scale_assumption"] = (
            f"{scale.method} ({scale.scale_cm_per_px:.4f} cm/px, conf {scale.confidence:.2f})"
        )
        points = self._build_proxy_points(
            measurements=measurements,
            metadata=metadata,
            poses=poses,
        )
        point_cloud_path, point_count = self._write_proxy_point_cloud(points, run_dir)

        colmap_points = self._load_colmap_points(run_dir)
        if colmap_points:
            colmap_points = self._scale_colmap_points(colmap_points, metadata)
            metadata["colmap_points_used"] = len(colmap_points)
        else:
            metadata["colmap_points_used"] = 0

        voxel_points = points + colmap_points
        voxel_path, voxel_info = self._build_voxel_grid(voxel_points, run_dir)
        if voxel_path:
            voxel_info["grid_rel_path"] = rel_path(voxel_path, self.static_dir)
            voxel_info["grid_path"] = str(voxel_path)
            voxel_info["used_for_volume"] = True
        metadata["voxel"] = voxel_info

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
        self,
        assets: list[ImageAsset],
        segments: list[SegmentationResult],
        scale: ScaleEstimationResult,
    ) -> list[dict]:
        measurements = []
        segment_by_image = {seg.image_path: seg for seg in segments}
        scale_cm_per_px = max(scale.scale_cm_per_px, 1e-6)

        for asset in assets:
            segment = segment_by_image.get(asset.path)
            if not segment:
                continue

            mask = np.asarray(Image.open(segment.mask_path).convert("L")) > 127
            bbox = mask_bbox(mask)
            if not bbox:
                continue

            area_ratio = float(mask.mean())
            frame_width_cm = scale_cm_per_px * asset.width
            frame_height_cm = scale_cm_per_px * asset.height
            width_cm = clamp(bbox.width * scale_cm_per_px, 3.5, 30.0)
            plane_height_cm = clamp(bbox.height * scale_cm_per_px, 3.5, 30.0)
            area_cm2 = clamp(area_ratio * (frame_width_cm * frame_height_cm), 8.0, 650.0)
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

    def _build_proxy_points(
        self,
        measurements: list[dict],
        metadata: dict,
        poses: PoseEstimationResult,
    ) -> list[tuple[float, float, float, int, int, int]]:
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

        return points

    def _write_proxy_point_cloud(
        self, points: list[tuple[float, float, float, int, int, int]], run_dir: Path
    ) -> tuple[Path, int]:
        output_path = run_dir / "reconstruction" / "food_proxy_gaussians.ply"
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

    def _load_colmap_points(self, run_dir: Path) -> list[tuple[float, float, float]]:
        sparse_dir = run_dir / "reconstruction" / "colmap_sparse"
        if not sparse_dir.exists():
            return []

        points_file = None
        for candidate in sparse_dir.rglob("points3D.txt"):
            points_file = candidate
            break
        if points_file is None:
            return []

        points = []
        for raw_line in points_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                points.append((x, y, z))
            except ValueError:
                continue

        if len(points) > self.colmap_point_limit:
            rng = np.random.default_rng(7)
            idx = rng.choice(len(points), size=self.colmap_point_limit, replace=False)
            points = [points[i] for i in idx]

        return points

    def _scale_colmap_points(self, points: list[tuple[float, float, float]], metadata: dict) -> list[
        tuple[float, float, float, int, int, int]
    ]:
        if not points:
            return []

        coords = np.asarray(points, dtype=np.float32)
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        center = (min_xyz + max_xyz) / 2.0
        extent = np.maximum(max_xyz - min_xyz, 1e-6)

        width_cm = float(metadata.get("width_cm", 1.0))
        depth_cm = float(metadata.get("depth_cm", 1.0))
        height_cm = float(metadata.get("height_cm", 1.0))
        scale = np.array([width_cm, depth_cm, height_cm], dtype=np.float32) / extent

        scaled = (coords - center) * scale
        return [(float(x), float(y), float(z), 46, 97, 160) for x, y, z in scaled]

    def _build_voxel_grid(
        self,
        points: list[tuple[float, float, float, int, int, int]],
        run_dir: Path,
    ) -> tuple[Path | None, dict]:
        if not points:
            return None, {}

        coords = np.asarray(points, dtype=np.float32)[:, :3]
        min_xyz = coords.min(axis=0)
        max_xyz = coords.max(axis=0)
        voxel_size = max(self.voxel_size_cm, 0.05)
        dims = np.ceil((max_xyz - min_xyz) / voxel_size).astype(int) + 1
        max_dim = int(dims.max())

        if max_dim > self.voxel_max_dim:
            scale = max_dim / max(self.voxel_max_dim, 1)
            voxel_size = voxel_size * scale
            dims = np.ceil((max_xyz - min_xyz) / voxel_size).astype(int) + 1

        indices = np.floor((coords - min_xyz) / voxel_size).astype(int)
        indices = np.clip(indices, 0, dims - 1)
        indices = np.unique(indices, axis=0)

        occupied = indices.shape[0]
        volume_cm3 = occupied * (voxel_size ** 3)
        fill_ratio = occupied / max(float(dims[0] * dims[1] * dims[2]), 1.0)

        output_path = run_dir / "reconstruction" / "food_proxy_voxels.npz"
        np.savez_compressed(
            output_path,
            indices=indices.astype(np.int32),
            origin=min_xyz.astype(np.float32),
            dims=dims.astype(np.int32),
            voxel_size_cm=np.array([voxel_size], dtype=np.float32),
        )

        info = {
            "occupied": int(occupied),
            "dims": [int(d) for d in dims.tolist()],
            "voxel_size_cm": round(float(voxel_size), 3),
            "fill_ratio": round(float(fill_ratio), 4),
            "volume_cm3": round(float(volume_cm3), 1),
            "note": "Occupancy volume from proxy points; not used for nutrition estimates.",
        }
        return output_path, info
