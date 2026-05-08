import importlib.util
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

from reconstruction.types import ScaleEstimationResult
from utils.volume_math import clamp


LOGGER = logging.getLogger(__name__)


@dataclass
class ScaleCandidate:
    method: str
    scale_cm_per_px: float
    confidence: float
    relative_uncertainty: float
    details: dict = field(default_factory=dict)


class ScaleEstimator:
    def __init__(self, config) -> None:
        self.config = config

    def estimate(self, assets, run_dir: Path) -> ScaleEstimationResult:
        candidates: list[ScaleCandidate] = []
        warnings: list[str] = []

        external = self._from_scale_json(run_dir)
        if external:
            candidates.append(external)

        reference = self._from_reference_config()
        if reference:
            candidates.append(reference)

        plate = self._from_plate_detection(assets)
        if plate:
            candidates.append(plate)

        if not candidates:
            candidates.append(self._from_frame_prior(assets))
            warnings.append("Scale derived from frame prior; metric accuracy is limited.")

        return self._fuse_candidates(candidates, warnings)

    def _from_scale_json(self, run_dir: Path) -> ScaleCandidate | None:
        path = run_dir / self.config.SCALE_JSON_NAME
        if not path.exists():
            return None

        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Scale JSON could not be parsed: %s", path)
            return None

        scale = float(data.get("scale_cm_per_px", 0.0))
        if scale <= 0:
            return None

        confidence = float(data.get("confidence", 0.75))
        uncertainty = float(data.get("relative_uncertainty", 0.12))
        return ScaleCandidate(
            method=data.get("method", "external_scale"),
            scale_cm_per_px=scale,
            confidence=clamp(confidence, 0.1, 0.95),
            relative_uncertainty=clamp(uncertainty, 0.05, 0.7),
            details={"source": data.get("source", "scale.json")},
        )

    def _from_reference_config(self) -> ScaleCandidate | None:
        ref_cm = self.config.REFERENCE_DIAMETER_CM
        ref_px = self.config.REFERENCE_DIAMETER_PX
        if not ref_cm or not ref_px:
            return None

        scale = ref_cm / ref_px
        return ScaleCandidate(
            method="reference_object",
            scale_cm_per_px=scale,
            confidence=0.65,
            relative_uncertainty=0.18,
            details={"reference_cm": ref_cm, "reference_px": ref_px},
        )

    def _from_plate_detection(self, assets) -> ScaleCandidate | None:
        if not self.config.ENABLE_PLATE_DETECTION:
            return None
        if not assets:
            return None
        if importlib.util.find_spec("cv2") is None:
            return None

        try:
            import cv2
        except Exception:
            return None

        plate_cm = self.config.DEFAULT_PLATE_DIAMETER_CM
        if plate_cm <= 0:
            return None

        image_path = Path(assets[0].path)
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None

        blur = cv2.medianBlur(image, 5)
        h, w = blur.shape[:2]
        min_dim = min(h, w)
        circles = cv2.HoughCircles(
            blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=int(min_dim * 0.4),
            param1=120,
            param2=40,
            minRadius=int(min_dim * 0.25),
            maxRadius=int(min_dim * 0.6),
        )
        if circles is None:
            return None

        circles = circles[0, :]
        circles = sorted(circles, key=lambda c: c[2], reverse=True)
        x, y, radius = circles[0]
        if radius <= 0:
            return None

        center_offset = math.hypot(x - (w / 2), y - (h / 2)) / max(min_dim, 1)
        confidence = clamp(0.55 - 0.3 * center_offset, 0.25, 0.7)
        uncertainty = clamp(0.22 + 0.25 * center_offset, 0.15, 0.5)
        scale = plate_cm / (2 * radius)

        return ScaleCandidate(
            method="plate_detection",
            scale_cm_per_px=scale,
            confidence=confidence,
            relative_uncertainty=uncertainty,
            details={"plate_cm": plate_cm, "radius_px": float(radius)},
        )

    def _from_frame_prior(self, assets) -> ScaleCandidate:
        if not assets:
            scale = self.config.DEFAULT_FRAME_DIAMETER_CM / 1000.0
        else:
            scale = self.config.DEFAULT_FRAME_DIAMETER_CM / max(assets[0].width, 1)

        return ScaleCandidate(
            method="frame_prior",
            scale_cm_per_px=scale,
            confidence=0.22,
            relative_uncertainty=0.5,
            details={"frame_cm": self.config.DEFAULT_FRAME_DIAMETER_CM},
        )

    def _fuse_candidates(
        self, candidates: list[ScaleCandidate], warnings: list[str]
    ) -> ScaleEstimationResult:
        if len(candidates) == 1:
            candidate = candidates[0]
            return ScaleEstimationResult(
                status="estimated",
                method=candidate.method,
                scale_cm_per_px=candidate.scale_cm_per_px,
                confidence=round(candidate.confidence, 3),
                relative_uncertainty=round(candidate.relative_uncertainty, 3),
                sources=[self._candidate_dict(candidate)],
                warnings=warnings,
            )

        weights = [max(candidate.confidence, 0.05) for candidate in candidates]
        total = sum(weights)
        mean_scale = sum(
            candidate.scale_cm_per_px * weight for candidate, weight in zip(candidates, weights)
        ) / total
        weighted_uncertainty = sum(
            candidate.relative_uncertainty * weight for candidate, weight in zip(candidates, weights)
        ) / total
        scatter = math.sqrt(
            sum(
                weight * (candidate.scale_cm_per_px - mean_scale) ** 2
                for candidate, weight in zip(candidates, weights)
            )
            / total
        )
        scatter_ratio = scatter / max(mean_scale, 1e-6)
        relative_uncertainty = clamp(max(weighted_uncertainty, scatter_ratio), 0.1, 0.7)
        if scatter_ratio > 0.25:
            warnings.append("Scale candidates disagree; uncertainty inflated.")

        confidence = clamp(1.0 - relative_uncertainty / 0.6, 0.12, 0.95)
        return ScaleEstimationResult(
            status="fused",
            method="fused",
            scale_cm_per_px=mean_scale,
            confidence=round(confidence, 3),
            relative_uncertainty=round(relative_uncertainty, 3),
            sources=[self._candidate_dict(candidate) for candidate in candidates],
            warnings=warnings,
        )

    def _candidate_dict(self, candidate: ScaleCandidate) -> dict:
        return {
            "method": candidate.method,
            "scale_cm_per_px": round(candidate.scale_cm_per_px, 6),
            "confidence": round(candidate.confidence, 3),
            "relative_uncertainty": round(candidate.relative_uncertainty, 3),
            "details": candidate.details,
        }
