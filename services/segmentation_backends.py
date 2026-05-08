import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter


LOGGER = logging.getLogger(__name__)


@dataclass
class BackendResult:
    mask: np.ndarray
    backend_id: str
    method: str
    warning: str | None = None
    detail: dict | None = None


class SegmentationBackend:
    backend_id: str = ""
    method: str = ""

    def available(self) -> bool:
        return True

    def segment(self, image: Image.Image) -> BackendResult | None:
        raise NotImplementedError


class GrabCutBackend(SegmentationBackend):
    backend_id = "grabcut"
    method = "grabcut"

    def available(self) -> bool:
        return importlib.util.find_spec("cv2") is not None

    def segment(self, image: Image.Image) -> BackendResult | None:
        if not self.available():
            return None

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

        return BackendResult(mask=binary, backend_id=self.backend_id, method=self.method)


class HeuristicBackend(SegmentationBackend):
    backend_id = "heuristic"
    method = "color-center-prior"

    def segment(self, image: Image.Image) -> BackendResult | None:
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
        binary = (np.asarray(pil_mask) > 127).astype("uint8") * 255

        return BackendResult(
            mask=binary,
            backend_id=self.backend_id,
            method=self.method,
            warning="Using lightweight segmentation because no learned segmenter is configured.",
        )


class DeepLabV3Backend(SegmentationBackend):
    backend_id = "deeplabv3"
    method = "deeplabv3"

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self._model = None
        self._preprocess = None

    def available(self) -> bool:
        return importlib.util.find_spec("torch") is not None and importlib.util.find_spec("torchvision") is not None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50

        weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self._preprocess = weights.transforms()
        model = deeplabv3_resnet50(weights=weights)
        model.eval()
        model.to(self.device)
        self._model = model

    def segment(self, image: Image.Image) -> BackendResult | None:
        if not self.available():
            return None

        self._load()
        import torch

        input_tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self._model(input_tensor)["out"][0]
        pred = output.argmax(0).cpu().numpy()
        mask = (pred != 0).astype("uint8") * 255
        mask = Image.fromarray(mask, mode="L").resize(image.size, Image.Resampling.NEAREST)
        mask = (np.asarray(mask) > 127).astype("uint8") * 255

        return BackendResult(
            mask=mask,
            backend_id=self.backend_id,
            method=self.method,
            warning="Generic semantic segmentation; may include non-food regions.",
        )


class U2NetOnnxBackend(SegmentationBackend):
    backend_id = "u2net"
    method = "u2net"

    def __init__(self, model_path: Path | None, input_size: int = 320) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.input_size = input_size
        self._session = None
        self._input_name = None

    def available(self) -> bool:
        return (
            self.model_path is not None
            and self.model_path.exists()
            and importlib.util.find_spec("onnxruntime") is not None
        )

    def _load(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort

        self._session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name

    def segment(self, image: Image.Image) -> BackendResult | None:
        if not self.available():
            return None

        self._load()
        resized = image.resize((self.input_size, self.input_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))[None, ...]

        outputs = self._session.run(None, {self._input_name: arr})
        if not outputs:
            return None

        prob = np.squeeze(outputs[0])
        if prob.ndim == 3:
            prob = prob[0]
        if prob.max() > 1.0 or prob.min() < 0.0:
            prob = 1.0 / (1.0 + np.exp(-prob))

        mask = (prob > 0.5).astype("uint8") * 255
        mask = Image.fromarray(mask, mode="L").resize(image.size, Image.Resampling.BILINEAR)
        mask = (np.asarray(mask) > 127).astype("uint8") * 255

        return BackendResult(
            mask=mask,
            backend_id=self.backend_id,
            method=self.method,
        )


class SamBackend(SegmentationBackend):
    backend_id = "sam"
    method = "sam"

    def __init__(self, checkpoint: Path | None, model_type: str = "vit_b") -> None:
        self.checkpoint = Path(checkpoint) if checkpoint else None
        self.model_type = model_type
        self._generator = None

    def available(self) -> bool:
        return (
            self.checkpoint is not None
            and self.checkpoint.exists()
            and importlib.util.find_spec("segment_anything") is not None
        )

    def _load(self) -> None:
        if self._generator is not None:
            return
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry[self.model_type](checkpoint=str(self.checkpoint))
        self._generator = SamAutomaticMaskGenerator(sam)

    def segment(self, image: Image.Image) -> BackendResult | None:
        if not self.available():
            return None

        self._load()
        masks = self._generator.generate(np.asarray(image))
        if not masks:
            return None

        best = max(masks, key=lambda m: m.get("area", 0))
        mask = best.get("segmentation")
        if mask is None:
            return None

        mask = mask.astype("uint8") * 255
        return BackendResult(
            mask=mask,
            backend_id=self.backend_id,
            method=self.method,
            warning="SAM automatic mask; may include non-food regions.",
        )
