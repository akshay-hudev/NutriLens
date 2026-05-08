from reconstruction.types import ReconstructionResult
from utils.volume_math import confidence_label, ellipsoid_cap_volume, uncertainty_bounds


def estimate_volume_cm3(reconstruction_result: ReconstructionResult) -> dict:
    metadata = reconstruction_result.metadata or {}
    voxel_meta = metadata.get("voxel", {}) if isinstance(metadata.get("voxel", {}), dict) else {}
    voxel_volume = voxel_meta.get("volume_cm3")
    required = {"width_cm", "depth_cm", "height_cm", "fill_ratio"}

    if voxel_volume:
        volume = float(voxel_volume)
        method = "occupancy voxel volume"
    elif required.issubset(metadata):
        width_cm = float(metadata["width_cm"])
        depth_cm = float(metadata["depth_cm"])
        height_cm = float(metadata["height_cm"])
        fill_ratio = float(metadata["fill_ratio"])
        volume = ellipsoid_cap_volume(width_cm, depth_cm, height_cm, fill_ratio)
        method = "ellipsoid-cap proxy"
    else:
        volume = 250.0
        lower, upper = uncertainty_bounds(volume, 0.7)
        return {
            "volume": round(volume, 1),
            "lower_bound": round(lower, 1),
            "upper_bound": round(upper, 1),
            "confidence": 0.12,
            "confidence_label": "Low",
            "method": "generic fallback volume prior",
        }

    status = reconstruction_result.status
    if "single_view" in status:
        uncertainty = 0.58
    elif "proxy" in status or "prior" in status:
        uncertainty = 0.38
    else:
        uncertainty = 0.24

    scale_uncertainty = float(metadata.get("scale_relative_uncertainty", 0.0))
    uncertainty += min(scale_uncertainty * 3.0, 0.6)

    uncertainty += max(0.0, 0.50 - reconstruction_result.confidence) * 0.25
    lower, upper = uncertainty_bounds(volume, uncertainty)

    confidence = reconstruction_result.confidence
    return {
        "volume": round(volume, 1),
        "lower_bound": round(lower, 1),
        "upper_bound": round(upper, 1),
        "confidence": round(confidence, 2),
        "confidence_label": confidence_label(confidence),
        "method": method,
    }
