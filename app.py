import json
import logging
import time
from datetime import datetime
from dataclasses import asdict
from pathlib import Path

from flask import Flask, redirect, render_template, request, session, url_for
from werkzeug.exceptions import RequestEntityTooLarge

from config import Config
from services.classification_service import ClassificationService
from services.description_service import GeminiDescriptionService
from services.nutrition_service import NutritionService
from services.reconstruction_service import ReconstructionService
from services.research_dataset_service import ResearchDatasetService
from services.segmentation_service import SegmentationService
from utils.image_io import (
    asset_from_manifest,
    create_run_dir,
    read_manifest,
    save_uploaded_images,
    write_manifest,
)
from utils.volume_math import confidence_label


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
LOGGER = logging.getLogger(__name__)


app = Flask(__name__, template_folder=str(Config.TEMPLATE_DIR))
Config.init_app(app)

classifier = ClassificationService(Config.MODEL_PATH)
segmenter = SegmentationService(Config.STATIC_DIR, Config.SEGMENTATION_BACKEND, Config)
reconstructor = ReconstructionService(Config)
nutrition_service = NutritionService(Config.NUTRITION_CSV, Config.DEFAULT_DENSITY_G_PER_CM3)
description_service = GeminiDescriptionService(
    Config.GOOGLE_API_KEY, Config.GEMINI_MODEL, Config.ENABLE_GEMINI
)
research_dataset = ResearchDatasetService(Config.BENCHMARK_NAME, Config.BENCHMARK_DATA_ROOT)


@app.route("/")
def index():
    return render_template(
        "index.html",
        model_loaded=classifier.model_configured,
        classifier_status=classifier.status_message,
        max_images=Config.MAX_IMAGES,
        max_image_mb=round(Config.MAX_IMAGE_BYTES / (1024 * 1024)),
        upload_error=None,
    )


@app.route("/upload", methods=["POST"])
def upload():
    files = request.files.getlist("img")
    run_id, run_dir = create_run_dir(Config.RUNS_ROOT)
    assets, warnings = save_uploaded_images(
        files=files,
        run_dir=run_dir,
        static_dir=Config.STATIC_DIR,
        allowed_extensions=Config.ALLOWED_EXTENSIONS,
        max_images=Config.MAX_IMAGES,
        max_image_bytes=Config.MAX_IMAGE_BYTES,
    )

    if not assets:
        return render_template(
            "index.html",
            model_loaded=classifier.model_configured,
            classifier_status=classifier.status_message,
            max_images=Config.MAX_IMAGES,
            max_image_mb=round(Config.MAX_IMAGE_BYTES / (1024 * 1024)),
            upload_error=" ".join(warnings) or "No valid food images were uploaded.",
        )

    write_manifest(run_dir, run_id, assets, warnings)
    session["run_id"] = run_id

    return render_template(
        "recognize.html",
        files=[asdict(asset) for asset in assets],
        count=len(assets),
        run_id=run_id,
        upload_warnings=warnings,
        low_view_warning=len(assets) == 1,
    )


@app.route("/predict")
def predict_latest():
    run_id = session.get("run_id")
    if not run_id:
        return redirect(url_for("index"))
    return redirect(url_for("predict", run_id=run_id))


@app.route("/predict/<run_id>")
def predict(run_id: str):
    run_dir = _run_dir(run_id)
    if not run_dir.exists():
        LOGGER.warning("Requested missing run %s", run_id)
        return redirect(url_for("index"))

    try:
        manifest = read_manifest(run_dir)
        assets = [asset_from_manifest(raw) for raw in manifest["images"]]
    except Exception:
        LOGGER.exception("Could not read run manifest for %s", run_id)
        return redirect(url_for("index"))

    image_paths = [asset.path for asset in assets]
    t_start = time.perf_counter()
    classification = classifier.classify_many(
        image_paths, filename_hints=[asset.original_filename for asset in assets]
    )
    t_classification = time.perf_counter()
    segments = segmenter.segment_many(assets, run_dir)
    t_segmentation = time.perf_counter()
    reconstruction = reconstructor.reconstruct(assets, segments, run_dir)
    t_reconstruction = time.perf_counter()
    portion = nutrition_service.estimate_portion(classification.top_food, reconstruction.volume)
    t_portion = time.perf_counter()
    ai_description = description_service.describe(
        assets[0].path, classification.top_food, portion["calories"]
    )
    t_description = time.perf_counter()

    report = _build_report(
        run_id=run_id,
        manifest=manifest,
        assets=assets,
        segments=segments,
        classification=classification,
        reconstruction_pipeline=reconstruction,
        portion=portion,
        ai_description=ai_description,
        runtime={
            "classification_s": round(t_classification - t_start, 3),
            "segmentation_s": round(t_segmentation - t_classification, 3),
            "reconstruction_s": round(t_reconstruction - t_segmentation, 3),
            "portion_s": round(t_portion - t_reconstruction, 3),
            "description_s": round(t_description - t_portion, 3),
            "total_s": round(t_description - t_start, 3),
        },
    )
    (run_dir / "result.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    return render_template(
        "results.html",
        report=report,
        model_loaded=classification.model_loaded,
        classifier_configured=classifier.model_configured,
    )


@app.route("/reset")
def reset():
    session.pop("run_id", None)
    return redirect(url_for("index"))


@app.errorhandler(RequestEntityTooLarge)
def handle_large_upload(_error):
    return (
        render_template(
            "index.html",
            model_loaded=classifier.model_configured,
            classifier_status=classifier.status_message,
            max_images=Config.MAX_IMAGES,
            max_image_mb=round(Config.MAX_IMAGE_BYTES / (1024 * 1024)),
            upload_error=f"Upload is too large. Use up to {Config.MAX_IMAGES} images under {round(Config.MAX_IMAGE_BYTES / (1024 * 1024))} MB each.",
        ),
        413,
    )


def _build_report(
    run_id: str,
    manifest: dict,
    assets: list,
    segments: list,
    classification,
    reconstruction_pipeline,
    portion: dict,
    ai_description: str | None,
    runtime: dict | None = None,
) -> dict:
    segment_by_image = {segment.image_path: segment for segment in segments}
    images = []
    for asset in assets:
        segment = segment_by_image.get(asset.path)
        images.append(
            {
                "image": asset.rel_path,
                "filename": asset.original_filename,
                "width": asset.width,
                "height": asset.height,
                "mask": segment.mask_rel_path if segment else None,
                "overlay": segment.overlay_rel_path if segment else None,
                "segmentation_method": segment.method if segment else "not available",
                "segmentation_backend": segment.backend if segment else "not available",
                "segmentation_detail": segment.backend_detail if segment else None,
                "segmentation_confidence": segment.confidence if segment else 0,
            }
        )

    avg_seg_confidence = (
        round(sum(segment.confidence for segment in segments) / len(segments), 2) if segments else 0
    )
    avg_mask_ratio = (
        round(sum(segment.mask_area_ratio for segment in segments) / len(segments) * 100, 1)
        if segments
        else 0
    )
    reconstruction = reconstruction_pipeline.reconstruction
    poses = reconstruction_pipeline.poses
    scale = reconstruction_pipeline.scale
    volume = reconstruction_pipeline.volume

    warnings = []
    warnings.extend(manifest.get("upload_warnings", []))
    if classification.warning:
        warnings.append(classification.warning)
    for segment in segments:
        if segment.warning:
            warnings.append(segment.warning)
    warnings.extend(reconstruction.warnings)
    warnings.extend(scale.warnings)

    classification_confidence = max(pred["confidence"] for pred in classification.predictions) / 100.0
    pose_confidence = float(poses.confidence) if poses else 0.2
    scale_confidence = float(scale.confidence) if scale else 0.2
    confidence = min(
        float(volume["confidence"]),
        avg_seg_confidence or 0.2,
        classification_confidence,
        pose_confidence,
        scale_confidence,
    )

    return {
        "run_id": run_id,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "images": images,
        "image_count": len(images),
        "top_food": classification.top_food,
        "predictions": classification.predictions,
        "classification_source": classification.source,
        "classification_warning": classification.warning,
        "segmentation": {
            "method": _join_unique(segment.method for segment in segments),
            "backend": _join_unique(segment.backend for segment in segments),
            "confidence": avg_seg_confidence,
            "confidence_label": confidence_label(avg_seg_confidence),
            "average_mask_area_percent": avg_mask_ratio,
        },
        "poses": {
            "status": poses.status,
            "method": poses.method,
            "confidence": poses.confidence,
            "count": len(poses.poses),
            "reprojection_error": poses.reprojection_error,
            "point_count": poses.point_count,
            "registered_images": poses.registered_images,
        },
        "reconstruction": {
            "status": reconstruction.status,
            "backend": reconstruction.backend,
            "confidence": reconstruction.confidence,
            "confidence_label": confidence_label(reconstruction.confidence),
            "point_count": reconstruction.point_count,
            "artifact": reconstruction.artifact_rel_path,
            "metadata": reconstruction.metadata,
        },
        "volume": volume,
        "scale": {
            "status": scale.status,
            "method": scale.method,
            "scale_cm_per_px": round(scale.scale_cm_per_px, 6),
            "relative_uncertainty": round(scale.relative_uncertainty, 3),
            "confidence": round(scale.confidence, 3),
            "sources": scale.sources,
        },
        "portion": portion,
        "overall_confidence": round(confidence, 2),
        "overall_confidence_label": confidence_label(confidence),
        "confidence_breakdown": {
            "classification": round(classification_confidence, 3),
            "segmentation": round(avg_seg_confidence, 3),
            "pose": round(pose_confidence, 3),
            "scale": round(scale_confidence, 3),
            "reconstruction": round(float(reconstruction.confidence), 3),
            "volume": round(float(volume["confidence"]), 3),
        },
        "runtime": runtime or {},
        "warnings": _dedupe(warnings),
        "ai_description": ai_description,
        "research_dataset": research_dataset.describe(),
    }


def _join_unique(values) -> str:
    unique = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return ", ".join(unique) if unique else "not available"


def _dedupe(values: list[str]) -> list[str]:
    seen = set()
    clean = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        clean.append(value)
    return clean


def _run_dir(run_id: str) -> Path:
    safe_run_id = "".join(ch for ch in run_id if ch.isalnum() or ch in {"-", "_"})
    return Config.RUNS_ROOT / safe_run_id


if __name__ == "__main__":
    import click

    @click.command()
    @click.option("--debug", is_flag=True)
    @click.option("--threaded", is_flag=True)
    @click.argument("HOST", default="127.0.0.1")
    @click.argument("PORT", default=5000, type=int)
    def run(debug, threaded, host, port):
        app.run(host=host, port=port, debug=debug, threaded=threaded)

    run()
