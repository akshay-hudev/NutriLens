import importlib.util
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from utils.image_io import ImageAsset, rel_path
from utils.volume_math import mask_bbox


LOGGER = logging.getLogger(__name__)


@dataclass
class SegmentationResult:
    image_path: str
    mask_path: str
    overlay_path: str
    mask_rel_path: str
    overlay_rel_path: str
    method: str
    confidence: float
    mask_area_ratio: float
    bbox: dict | None
    warning: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SegmentationService:
    def __init__(self, static_dir: Path) -> None:
        self.static_dir = Path(static_dir)

    def segment_many(self, assets: list[ImageAsset], run_dir: Path) -> list[SegmentationResult]:
        results = []
        for index, asset in enumerate(assets, start=1):
            try:
                results.append(self.segment(asset, run_dir, index))
            except Exception as exc:
                LOGGER.exception("Segmentation failed for %s", asset.path)
                results.append(self._central_fallback(asset, run_dir, index, str(exc)))
        return results

    def segment(self, asset: ImageAsset, run_dir: Path, index: int) -> SegmentationResult:
        image = Image.open(asset.path).convert("RGB")

        if importlib.util.find_spec("cv2") is not None:
            try:
                mask = self._grabcut_mask(image)
                if self._valid_mask(mask):
                    return self._save_result(
                        image=image,
                        mask=mask,
                        asset=asset,
                        run_dir=run_dir,
                        index=index,
                        method="grabcut",
                        warning=None,
                    )
            except Exception:
                LOGGER.info("GrabCut segmentation unavailable for %s", asset.path, exc_info=True)

        mask = self._heuristic_mask(image)
        warning = "Using lightweight segmentation because no dedicated food segmenter is configured."
        return self._save_result(
            image=image,
            mask=mask,
            asset=asset,
            run_dir=run_dir,
            index=index,
            method="color-center-prior",
            warning=warning,
        )

    def _grabcut_mask(self, image: Image.Image) -> np.ndarray:
        import cv2

        rgb = np.asarray(image)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        h, w = bgr.shape[:2]
        margin_x = max(1, int(w * 0.06))
        margin_y = max(1, int(h * 0.06))
        rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

        mask = np.zeros((h, w), np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        cv2.grabCut(bgr, mask, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
        binary = np.where((mask == 2) | (mask == 0), 0, 255).astype("uint8")

        kernel = np.ones((5, 5), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        return binary

    def _heuristic_mask(self, image: Image.Image) -> np.ndarray:
        small = image.resize((min(768, image.width), int(image.height * min(768, image.width) / image.width)))
        arr = np.asarray(small, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        border = np.concatenate(
            [
                arr[: max(1, h // 18), :, :].reshape(-1, 3),
                arr[-max(1, h // 18) :, :, :].reshape(-1, 3),
                arr[:, : max(1, w // 18), :].reshape(-1, 3),
                arr[:, -max(1, w // 18) :, :].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(border, axis=0)
        color_distance = np.linalg.norm(arr - background, axis=2)
        color_distance = color_distance / max(float(color_distance.max()), 1e-6)

        max_c = arr.max(axis=2)
        min_c = arr.min(axis=2)
        saturation = max_c - min_c

        yy, xx = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        sigma_x, sigma_y = w * 0.42, h * 0.42
        center_prior = np.exp(-(((xx - cx) ** 2) / (2 * sigma_x**2) + ((yy - cy) ** 2) / (2 * sigma_y**2)))

        score = 0.50 * color_distance + 0.30 * center_prior + 0.20 * saturation
        threshold = max(float(np.percentile(score, 58)), 0.28)
        mask = (score >= threshold).astype("uint8") * 255

        pil_mask = Image.fromarray(mask, mode="L")
        pil_mask = pil_mask.filter(ImageFilter.MedianFilter(size=7))
        pil_mask = pil_mask.filter(ImageFilter.MaxFilter(size=7))
        pil_mask = pil_mask.filter(ImageFilter.MinFilter(size=5))
        pil_mask = pil_mask.resize(image.size, Image.Resampling.BILINEAR)
        return (np.asarray(pil_mask) > 127).astype("uint8") * 255

    def _central_fallback(
        self, asset: ImageAsset, run_dir: Path, index: int, reason: str
    ) -> SegmentationResult:
        image = Image.open(asset.path).convert("RGB")
        h, w = image.height, image.width
        yy, xx = np.mgrid[0:h, 0:w]
        ellipse = (((xx - w / 2) / (w * 0.34)) ** 2 + ((yy - h / 2) / (h * 0.28)) ** 2) <= 1.0
        mask = ellipse.astype("uint8") * 255
        return self._save_result(
            image=image,
            mask=mask,
            asset=asset,
            run_dir=run_dir,
            index=index,
            method="central-ellipse-fallback",
            warning=f"Segmentation fallback used: {reason}",
        )

    def _save_result(
        self,
        image: Image.Image,
        mask: np.ndarray,
        asset: ImageAsset,
        run_dir: Path,
        index: int,
        method: str,
        warning: str | None,
    ) -> SegmentationResult:
        masks_dir = run_dir / "masks"
        mask_path = masks_dir / f"mask_{index:02d}.png"
        overlay_path = masks_dir / f"overlay_{index:02d}.jpg"

        Image.fromarray(mask, mode="L").save(mask_path)
        overlay = self._overlay(image, mask)
        overlay.save(overlay_path, format="JPEG", quality=90, optimize=True)

        mask_bool = mask > 127
        area_ratio = float(mask_bool.mean())
        bbox = mask_bbox(mask_bool)
        confidence = self._confidence(area_ratio, bbox, image.width, image.height, method)

        return SegmentationResult(
            image_path=asset.path,
            mask_path=str(mask_path),
            overlay_path=str(overlay_path),
            mask_rel_path=rel_path(mask_path, self.static_dir),
            overlay_rel_path=rel_path(overlay_path, self.static_dir),
            method=method,
            confidence=confidence,
            mask_area_ratio=round(area_ratio, 4),
            bbox=bbox.__dict__ if bbox else None,
            warning=warning,
        )

    def _overlay(self, image: Image.Image, mask: np.ndarray) -> Image.Image:
        base = image.convert("RGBA")
        color = np.zeros((image.height, image.width, 4), dtype=np.uint8)
        color[..., 0] = 34
        color[..., 1] = 111
        color[..., 2] = 84
        color[..., 3] = np.where(mask > 127, 110, 0).astype(np.uint8)
        overlay = Image.fromarray(color, mode="RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")

    def _valid_mask(self, mask: np.ndarray) -> bool:
        ratio = float((mask > 127).mean())
        return 0.02 <= ratio <= 0.88

    def _confidence(self, area_ratio: float, bbox, width: int, height: int, method: str) -> float:
        if not bbox:
            return 0.1

        bbox_fraction = bbox.area / float(width * height)
        plausible_area = 1.0 - min(abs(area_ratio - 0.24) / 0.24, 1.0)
        plausible_box = 1.0 - min(abs(bbox_fraction - 0.38) / 0.38, 1.0)
        method_bonus = 0.18 if method == "grabcut" else 0.0
        confidence = 0.28 + 0.32 * plausible_area + 0.22 * plausible_box + method_bonus
        return round(max(0.12, min(0.86, confidence)), 2)
