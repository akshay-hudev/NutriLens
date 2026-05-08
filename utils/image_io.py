import json
import uuid
from dataclasses import asdict, dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename


@dataclass
class ImageAsset:
    path: str
    rel_path: str
    filename: str
    original_filename: str
    width: int
    height: int


def allowed_file(filename: str, allowed_extensions: set[str]) -> bool:
    if not filename or "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in allowed_extensions


def create_run_dir(runs_root: Path) -> tuple[str, Path]:
    run_id = uuid.uuid4().hex[:12]
    run_dir = runs_root / run_id
    (run_dir / "images").mkdir(parents=True, exist_ok=False)
    (run_dir / "masks").mkdir(parents=True, exist_ok=True)
    (run_dir / "reconstruction").mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


def _static_rel_path(path: Path, static_dir: Path) -> str:
    return str(path.resolve().relative_to(static_dir.resolve())).replace("\\", "/")


def save_uploaded_images(
    files: Iterable[FileStorage],
    run_dir: Path,
    static_dir: Path,
    allowed_extensions: set[str],
    max_images: int,
    max_image_bytes: int,
) -> tuple[list[ImageAsset], list[str]]:
    assets: list[ImageAsset] = []
    errors: list[str] = []

    selected_files = [f for f in files if f and f.filename]
    if not selected_files:
        return assets, ["Choose at least one food image."]

    if len(selected_files) > max_images:
        errors.append(f"Only the first {max_images} images were used.")
        selected_files = selected_files[:max_images]

    for index, storage in enumerate(selected_files, start=1):
        original = storage.filename or f"image_{index}"
        safe_original = secure_filename(original)

        if not allowed_file(safe_original, allowed_extensions):
            errors.append(f"{original} was skipped because its file type is not supported.")
            continue

        data = storage.read()
        if len(data) > max_image_bytes:
            mb = max_image_bytes / (1024 * 1024)
            errors.append(f"{original} was skipped because it is larger than {mb:.0f} MB.")
            continue

        try:
            with Image.open(BytesIO(data)) as image:
                normalized = ImageOps.exif_transpose(image).convert("RGB")
        except Exception:
            errors.append(f"{original} was skipped because it is not a readable image.")
            continue

        filename = f"view_{index:02d}.jpg"
        output_path = run_dir / "images" / filename
        normalized.save(output_path, format="JPEG", quality=92, optimize=True)

        assets.append(
            ImageAsset(
                path=str(output_path),
                rel_path=_static_rel_path(output_path, static_dir),
                filename=filename,
                original_filename=safe_original or original,
                width=normalized.width,
                height=normalized.height,
            )
        )

    if not assets and not errors:
        errors.append("No valid images were uploaded.")

    return assets, errors


def write_manifest(run_dir: Path, run_id: str, assets: list[ImageAsset], warnings: list[str]) -> None:
    manifest = {
        "run_id": run_id,
        "images": [asdict(asset) for asset in assets],
        "upload_warnings": warnings,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def read_manifest(run_dir: Path) -> dict:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing run manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def asset_from_manifest(raw: dict) -> ImageAsset:
    return ImageAsset(**raw)


def rel_path(path: Path, static_dir: Path) -> str:
    return _static_rel_path(path, static_dir)
