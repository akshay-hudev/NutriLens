import math
from dataclasses import dataclass


@dataclass
class BoundingBox:
    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height


def mask_bbox(mask_pixels) -> BoundingBox | None:
    ys, xs = mask_pixels.nonzero()
    if len(xs) == 0 or len(ys) == 0:
        return None

    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    return BoundingBox(
        x=x_min,
        y=y_min,
        width=max(1, x_max - x_min + 1),
        height=max(1, y_max - y_min + 1),
    )


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def confidence_label(confidence: float) -> str:
    if confidence >= 0.72:
        return "High"
    if confidence >= 0.45:
        return "Medium"
    return "Low"


def ellipsoid_cap_volume(width_cm: float, depth_cm: float, height_cm: float, fill_ratio: float) -> float:
    """Approximate an irregular plated portion as a filled ellipsoid cap."""
    ellipsoid = (4.0 / 3.0) * math.pi * (width_cm / 2.0) * (depth_cm / 2.0) * (
        height_cm / 2.0
    )
    return ellipsoid * fill_ratio


def uncertainty_bounds(value: float, relative_uncertainty: float) -> tuple[float, float]:
    lower = value * (1.0 - relative_uncertainty)
    upper = value * (1.0 + relative_uncertainty)
    return max(0.0, lower), max(lower, upper)
