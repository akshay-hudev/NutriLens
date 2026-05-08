from reconstruction.types import ReconstructionResult
from utils.volume_math import confidence_label, ellipsoid_cap_volume, uncertainty_bounds


def estimate_volume_cm3(reconstruction_result: ReconstructionResult) -> dict:
    metadata = reconstruction_result.metadata or {}
    required = {"width_cm", "depth_cm", "height_cm", "fill_ratio"}

    if not required.issubset(metadata):
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

    width_cm = float(metadata["width_cm"])
    depth_cm = float(metadata["depth_cm"])
    height_cm = float(metadata["height_cm"])
    fill_ratio = float(metadata["fill_ratio"])
    volume = ellipsoid_cap_volume(width_cm, depth_cm, height_cm, fill_ratio)

    status = reconstruction_result.status
    if "single_view" in status:
        uncertainty = 0.58
    elif "proxy" in status or "prior" in status:
        uncertainty = 0.38
    else:
        uncertainty = 0.24

    uncertainty += max(0.0, 0.50 - reconstruction_result.confidence) * 0.25
    lower, upper = uncertainty_bounds(volume, uncertainty)

    confidence = reconstruction_result.confidence
    return {
        "volume": round(volume, 1),
        "lower_bound": round(lower, 1),
        "upper_bound": round(upper, 1),
        "confidence": round(confidence, 2),
        "confidence_label": confidence_label(confidence),
        "method": "sparse-view 3D geometry estimate",
    }
