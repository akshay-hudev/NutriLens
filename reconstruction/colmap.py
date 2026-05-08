import logging
import shutil
import subprocess
from pathlib import Path

from reconstruction.types import PoseEstimationResult


LOGGER = logging.getLogger(__name__)


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

    pose_prior = _turntable_pose_prior(image_paths, warnings=[])
    pose_prior.status = "colmap_completed"
    pose_prior.method = "COLMAP SfM with ordered-pose export placeholder"
    pose_prior.confidence = 0.68
    pose_prior.warnings.append(
        "COLMAP sparse model was created; production pose parsing can be added in reconstruction/colmap.py."
    )
    return pose_prior


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
    )
