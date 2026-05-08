import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.metrics import iou, mae, mape, mean, rmse, std


@dataclass
class RunMetrics:
    run_id: str
    method: str
    image_count: int
    volume_pred: float | None
    volume_gt: float | None
    calories_pred: float | None
    calories_gt: float | None
    weight_pred: float | None
    weight_gt: float | None
    volume_abs_error: float | None
    calories_abs_error: float | None
    weight_abs_error: float | None
    segmentation_iou: float | None
    confidence: float | None


@dataclass
class EvaluationSummary:
    count: int
    volume_mape: float
    volume_mae: float
    volume_rmse: float
    calories_mae: float
    calories_rmse: float
    weight_mae: float
    confidence_mean: float
    confidence_std: float


def _load_ground_truth(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"runs": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_mask(path: Path):
    with Image.open(path) as image:
        return np.asarray(image.convert("L")) > 127


def evaluate_runs(runs_root: Path, ground_truth_path: Path) -> tuple[list[RunMetrics], dict[str, EvaluationSummary]]:
    ground_truth = _load_ground_truth(ground_truth_path)
    run_truth = ground_truth.get("runs", {})
    metrics: list[RunMetrics] = []

    for run_dir in sorted(runs_root.glob("*/")):
        result_path = run_dir / "result.json"
        if not result_path.exists():
            continue

        report = json.loads(result_path.read_text(encoding="utf-8"))
        run_id = report.get("run_id", run_dir.name)
        truth = run_truth.get(run_id)
        if not truth:
            continue

        volume_pred = report.get("volume", {}).get("volume")
        calories_pred = report.get("portion", {}).get("calories")
        weight_pred = report.get("portion", {}).get("weight_g")
        confidence = report.get("overall_confidence")
        image_count = report.get("image_count", 0)
        method = report.get("reconstruction", {}).get("backend", "unknown")

        volume_gt = truth.get("volume_cm3")
        calories_gt = truth.get("calories")
        weight_gt = truth.get("weight_g")

        volume_abs_error = None if volume_gt is None or volume_pred is None else abs(volume_pred - volume_gt)
        calories_abs_error = None if calories_gt is None or calories_pred is None else abs(calories_pred - calories_gt)
        weight_abs_error = None if weight_gt is None or weight_pred is None else abs(weight_pred - weight_gt)

        seg_iou = None
        mask_pred_path = truth.get("pred_mask_path")
        mask_gt_path = truth.get("gt_mask_path")
        if mask_pred_path and mask_gt_path:
            try:
                seg_iou = iou(_load_mask(Path(mask_pred_path)), _load_mask(Path(mask_gt_path)))
            except Exception:
                seg_iou = None

        metrics.append(
            RunMetrics(
                run_id=run_id,
                method=method,
                image_count=image_count,
                volume_pred=volume_pred,
                volume_gt=volume_gt,
                calories_pred=calories_pred,
                calories_gt=calories_gt,
                weight_pred=weight_pred,
                weight_gt=weight_gt,
                volume_abs_error=volume_abs_error,
                calories_abs_error=calories_abs_error,
                weight_abs_error=weight_abs_error,
                segmentation_iou=seg_iou,
                confidence=confidence,
            )
        )

    summary: dict[str, EvaluationSummary] = {}
    for method in sorted({m.method for m in metrics}):
        subset = [m for m in metrics if m.method == method]
        volume_gt = [m.volume_gt for m in subset if m.volume_gt is not None]
        volume_pred = [m.volume_pred for m in subset if m.volume_pred is not None]
        calories_gt = [m.calories_gt for m in subset if m.calories_gt is not None]
        calories_pred = [m.calories_pred for m in subset if m.calories_pred is not None]
        weight_gt = [m.weight_gt for m in subset if m.weight_gt is not None]
        weight_pred = [m.weight_pred for m in subset if m.weight_pred is not None]
        confidences = [m.confidence for m in subset if m.confidence is not None]

        summary[method] = EvaluationSummary(
            count=len(subset),
            volume_mape=mape(volume_gt, volume_pred),
            volume_mae=mae([v for v in (m.volume_abs_error for m in subset) if v is not None]),
            volume_rmse=rmse(volume_gt, volume_pred),
            calories_mae=mae([v for v in (m.calories_abs_error for m in subset) if v is not None]),
            calories_rmse=rmse(calories_gt, calories_pred),
            weight_mae=mae([v for v in (m.weight_abs_error for m in subset) if v is not None]),
            confidence_mean=mean(confidences),
            confidence_std=std(confidences),
        )

    return metrics, summary
