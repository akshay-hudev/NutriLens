from dataclasses import dataclass, field


@dataclass
class PoseEstimationResult:
    status: str
    method: str
    confidence: float
    poses: list[dict]
    warnings: list[str] = field(default_factory=list)
    reprojection_error: float | None = None
    point_count: int = 0
    registered_images: int = 0


@dataclass
class ScaleEstimationResult:
    status: str
    method: str
    scale_cm_per_px: float
    confidence: float
    relative_uncertainty: float
    sources: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ReconstructionResult:
    status: str
    backend: str
    confidence: float
    image_count: int
    point_count: int = 0
    artifact_path: str | None = None
    artifact_rel_path: str | None = None
    metadata: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
