import logging
import math
import shutil
import subprocess
from pathlib import Path

from reconstruction.types import PoseEstimationResult


LOGGER = logging.getLogger(__name__)


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> list[list[float]]:
    r00 = 1 - 2 * (qy * qy + qz * qz)
    r01 = 2 * (qx * qy - qz * qw)
    r02 = 2 * (qx * qz + qy * qw)
    r10 = 2 * (qx * qy + qz * qw)
    r11 = 1 - 2 * (qx * qx + qz * qz)
    r12 = 2 * (qy * qz - qx * qw)
    r20 = 2 * (qx * qz - qy * qw)
    r21 = 2 * (qy * qz + qx * qw)
    r22 = 1 - 2 * (qx * qx + qy * qy)
    return [[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]]


def _yaw_from_rot(rot: list[list[float]]) -> float:
    return math.degrees(math.atan2(rot[1][0], rot[0][0]))


def _parse_cameras_txt(path: Path) -> dict[int, dict]:
    cameras = {}
    if not path.exists():
        return cameras

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        camera_id = int(parts[0])
        cameras[camera_id] = {
            "model": parts[1],
            "width": int(parts[2]),
            "height": int(parts[3]),
            "params": [float(p) for p in parts[4:]],
        }
    return cameras


def _parse_images_txt(path: Path) -> list[dict]:
    images = []
    if not path.exists():
        return images

    lines = path.read_text(encoding="utf-8").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        idx += 1
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 9:
            continue

        image_id = int(parts[0])
        qw, qx, qy, qz = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        tx, ty, tz = (float(parts[5]), float(parts[6]), float(parts[7]))
        camera_id = int(parts[8])
        name = parts[9] if len(parts) > 9 else f"{image_id}.jpg"

        rot = _quat_to_rot(qw, qx, qy, qz)
        images.append(
            {
                "image_id": image_id,
                "name": name,
                "camera_id": camera_id,
                "rotation": rot,
                "translation": [tx, ty, tz],
                "yaw_deg": _yaw_from_rot(rot),
            }
        )

        # Skip the 2D-3D correspondence line.
        if idx < len(lines):
            idx += 1
    return images


def _parse_points3d_txt(path: Path) -> dict:
    if not path.exists():
        return {"count": 0, "mean_error": None}

    errors = []
    count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        count += 1
        try:
            errors.append(float(parts[7]))
        except ValueError:
            continue
    mean_error = float(sum(errors) / len(errors)) if errors else None
    return {"count": count, "mean_error": mean_error}


def _pose_confidence(registered: int, total: int, point_count: int, mean_error: float | None) -> float:
    image_score = registered / max(total, 1)
    point_score = min(point_count / 2000.0, 1.0)
    error_score = 0.4 if mean_error is None else 1.0 / (1.0 + mean_error)
    confidence = 0.18 + 0.42 * image_score + 0.28 * point_score + 0.12 * error_score
    return _clamp(confidence, 0.12, 0.9)


def estimate_camera_poses(
    image_paths: list[str],
    run_dir: Path,
    colmap_bin: str,
    enabled: bool,
    timeout_seconds: int,
) -> PoseEstimationResult:
    if len(image_paths) == 1:
        return PoseEstimationResult(
            status="single_view",
            method="single-view canonical camera",
            confidence=0.18,
            poses=[
                {
                    "image_path": image_paths[0],
                    "azimuth_deg": 0.0,
                    "elevation_deg": 50.0,
                    "source": "assumed",
                }
            ],
            warnings=["Only one image was uploaded, so camera pose is assumed."],
        )

    if not enabled:
        return _turntable_pose_prior(
            image_paths,
            warnings=["COLMAP is not enabled; using ordered sparse-view pose priors."],
        )

    if shutil.which(colmap_bin) is None:
        return _turntable_pose_prior(
            image_paths,
            warnings=["COLMAP executable was not found; using ordered sparse-view pose priors."],
        )

    try:
        return _run_colmap(image_paths, run_dir, colmap_bin, timeout_seconds)
    except Exception as exc:
        LOGGER.exception("COLMAP pose recovery failed")
        return _turntable_pose_prior(
            image_paths,
            warnings=[f"COLMAP failed: {exc}; using ordered sparse-view pose priors."],
        )


def _run_colmap(
    image_paths: list[str], run_dir: Path, colmap_bin: str, timeout_seconds: int
) -> PoseEstimationResult:
    recon_dir = run_dir / "reconstruction"
    database_path = recon_dir / "colmap.db"
    sparse_dir = recon_dir / "colmap_sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(image_paths[0]).parent

    commands = [
        [
            colmap_bin,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--ImageReader.single_camera",
            "1",
        ],
        [colmap_bin, "exhaustive_matcher", "--database_path", str(database_path)],
        [
            colmap_bin,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
        ],
    ]

    for command in commands:
        completed = subprocess.run(
            command,
            cwd=run_dir,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(completed.stderr.strip() or "COLMAP command failed")

    model_dirs = [path for path in sparse_dir.iterdir() if path.is_dir()]
    model_root = model_dirs[0] if model_dirs else sparse_dir
    _ = _parse_cameras_txt(model_root / "cameras.txt")
    images = _parse_images_txt(model_root / "images.txt")
    points = _parse_points3d_txt(model_root / "points3D.txt")

    if not images:
        return _turntable_pose_prior(
            image_paths,
            warnings=["COLMAP produced no registered images; using ordered sparse-view pose prior."],
        )

    image_lookup = {Path(path).name: path for path in image_paths}
    poses = []
    for image in images:
        image_path = image_lookup.get(image["name"], str(image_dir / image["name"]))
        poses.append(
            {
                "image_path": image_path,
                "azimuth_deg": round(float(image["yaw_deg"]), 2),
                "rotation": image["rotation"],
                "translation": image["translation"],
                "camera_id": image["camera_id"],
                "source": "colmap",
            }
        )

    warnings: list[str] = []
    registered = len(images)
    if registered < len(image_paths):
        warnings.append("COLMAP registered fewer images than uploaded; using partial pose set.")

    mean_error = points["mean_error"]
    if mean_error and mean_error > 2.0:
        warnings.append(f"High COLMAP reprojection error (mean {mean_error:.2f}px).")

    confidence = _pose_confidence(registered, len(image_paths), points["count"], mean_error)
    return PoseEstimationResult(
        status="colmap_completed",
        method="COLMAP SfM (parsed poses)",
        confidence=round(confidence, 2),
        poses=poses,
        warnings=warnings,
        reprojection_error=mean_error,
        point_count=points["count"],
        registered_images=registered,
    )


def _turntable_pose_prior(image_paths: list[str], warnings: list[str]) -> PoseEstimationResult:
    count = len(image_paths)
    span = min(90.0, 18.0 * max(count - 1, 1))
    start = -span / 2.0
    step = span / max(count - 1, 1)
    poses = []

    for index, image_path in enumerate(image_paths):
        poses.append(
            {
                "image_path": image_path,
                "azimuth_deg": round(start + index * step, 2),
                "elevation_deg": 50.0,
                "source": "ordered-view-prior",
            }
        )

    confidence = min(0.52, 0.26 + 0.07 * count)
    return PoseEstimationResult(
        status="pose_prior",
        method="ordered sparse-view pose prior",
        confidence=round(confidence, 2),
        poses=poses,
        warnings=warnings,
        reprojection_error=None,
        point_count=0,
        registered_images=len(image_paths),
    )
