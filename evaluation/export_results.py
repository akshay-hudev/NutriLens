import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from evaluation.evaluate import EvaluationSummary, RunMetrics


def export_run_metrics(metrics: Iterable[RunMetrics], output_dir: Path) -> Path:
    metrics = list(metrics)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "per_run.csv"

    if not metrics:
        csv_path.write_text("", encoding="utf-8")
        return csv_path

    fieldnames = list(asdict(metrics[0]).keys())
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in metrics:
            writer.writerow(asdict(item))
    return csv_path


def export_summary(summary: dict[str, EvaluationSummary], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "summary.csv"
    fieldnames = ["method", "count", "volume_mape", "volume_mae", "volume_rmse", "calories_mae", "calories_rmse", "weight_mae", "confidence_mean", "confidence_std"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for method, stats in summary.items():
            row = asdict(stats)
            row["method"] = method
            writer.writerow(row)

    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps({k: asdict(v) for k, v in summary.items()}, indent=2), encoding="utf-8")
    return csv_path
