import hashlib
import importlib.util
import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


LOGGER = logging.getLogger(__name__)


FOOD101_LABELS = sorted(
    [
        "apple pie",
        "baby back ribs",
        "baklava",
        "beef carpaccio",
        "beef tartare",
        "beet salad",
        "beignets",
        "bibimbap",
        "bread pudding",
        "breakfast burrito",
        "bruschetta",
        "caesar salad",
        "cannoli",
        "caprese salad",
        "carrot cake",
        "ceviche",
        "cheese plate",
        "cheesecake",
        "chicken curry",
        "chicken quesadilla",
        "chicken wings",
        "chocolate cake",
        "chocolate mousse",
        "churros",
        "clam chowder",
        "club sandwich",
        "crab cakes",
        "creme brulee",
        "croque madame",
        "cup cakes",
        "deviled eggs",
        "donuts",
        "dumplings",
        "edamame",
        "eggs benedict",
        "escargots",
        "falafel",
        "filet mignon",
        "fish and_chips",
        "foie gras",
        "french fries",
        "french onion soup",
        "french toast",
        "fried calamari",
        "fried rice",
        "frozen yogurt",
        "garlic bread",
        "gnocchi",
        "greek salad",
        "grilled cheese sandwich",
        "grilled salmon",
        "guacamole",
        "gyoza",
        "hamburger",
        "hot and sour soup",
        "hot dog",
        "huevos rancheros",
        "hummus",
        "ice cream",
        "lasagna",
        "lobster bisque",
        "lobster roll sandwich",
        "macaroni and cheese",
        "macarons",
        "miso soup",
        "mussels",
        "nachos",
        "omelette",
        "onion rings",
        "oysters",
        "pad thai",
        "paella",
        "pancakes",
        "panna cotta",
        "peking duck",
        "pho",
        "pizza",
        "pork chop",
        "poutine",
        "prime rib",
        "pulled pork sandwich",
        "ramen",
        "ravioli",
        "red velvet cake",
        "risotto",
        "samosa",
        "sashimi",
        "scallops",
        "seaweed salad",
        "shrimp and grits",
        "spaghetti bolognese",
        "spaghetti carbonara",
        "spring rolls",
        "steak",
        "strawberry shortcake",
        "sushi",
        "tacos",
        "octopus balls",
        "tiramisu",
        "tuna tartare",
        "waffles",
    ]
)


@dataclass
class ClassificationResult:
    top_food: str
    predictions: list[dict]
    source: str
    model_loaded: bool
    warning: str | None = None


class ClassificationService:
    def __init__(self, model_path: Path, labels: list[str] | None = None) -> None:
        self.model_path = Path(model_path)
        self.labels = labels or FOOD101_LABELS
        self._model = None
        self._tf = None
        self._load_error: str | None = None

    @property
    def tensorflow_available(self) -> bool:
        return importlib.util.find_spec("tensorflow") is not None

    @property
    def model_configured(self) -> bool:
        return self.model_path.exists() and self.tensorflow_available

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    @property
    def status_message(self) -> str:
        if self.model_loaded:
            return "Food-101 classifier loaded"
        if not self.model_path.exists():
            return "Classifier model file is missing"
        if not self.tensorflow_available:
            return "TensorFlow is not installed"
        if self._load_error:
            return self._load_error
        return "Classifier model is configured and will load during analysis"

    def classify_many(
        self, image_paths: list[str], filename_hints: list[str] | None = None
    ) -> ClassificationResult:
        if self._ensure_model_loaded():
            try:
                predictions = [self._predict_model(path) for path in image_paths]
                mean_pred = np.mean(predictions, axis=0)
                return self._format_prediction(mean_pred, source="Food-101 classifier")
            except Exception as exc:
                LOGGER.exception("Classifier inference failed")
                return self._fallback_prediction(
                    image_paths,
                    filename_hints,
                    warning=f"Classifier inference failed, using deterministic visual fallback: {exc}",
                )

        warning = self.status_message + "; using deterministic low-confidence fallback."
        return self._fallback_prediction(image_paths, filename_hints, warning=warning)

    def _ensure_model_loaded(self) -> bool:
        if self._model is not None:
            return True

        if not self.model_path.exists() or not self.tensorflow_available:
            return False

        try:
            import tensorflow as tf
            from tensorflow.keras.models import load_model

            tf.keras.backend.clear_session()
            self._tf = tf
            self._model = load_model(self.model_path, compile=False)
            LOGGER.info("Loaded Food-101 model from %s", self.model_path)
            return True
        except Exception as exc:
            self._load_error = f"Classifier model could not be loaded: {exc}"
            LOGGER.exception("Could not load classifier model")
            return False

    def _predict_model(self, image_path: str) -> np.ndarray:
        with Image.open(image_path) as image:
            resized = image.convert("RGB").resize((224, 224))
        image_array = np.asarray(resized, dtype=np.float32) / 255.0
        batch = np.expand_dims(image_array, axis=0)
        pred = np.asarray(self._model.predict(batch, verbose=0)[0], dtype=np.float32)

        if pred.size != len(self.labels):
            fixed = np.zeros(len(self.labels), dtype=np.float32)
            fixed[: min(pred.size, len(self.labels))] = pred[: len(self.labels)]
            pred = fixed

        if np.any(np.isnan(pred)) or pred.sum() <= 0:
            raise ValueError("model returned invalid probabilities")

        return pred / pred.sum()

    def _format_prediction(
        self, pred: np.ndarray, source: str, warning: str | None = None
    ) -> ClassificationResult:
        top3_idx = pred.argsort()[-3:][::-1]
        predictions = [
            {"food": self.labels[i], "confidence": round(float(pred[i]) * 100, 1)}
            for i in top3_idx
        ]
        return ClassificationResult(
            top_food=self.labels[int(top3_idx[0])],
            predictions=predictions,
            source=source,
            model_loaded=self.model_loaded,
            warning=warning,
        )

    def _fallback_prediction(
        self, image_paths: list[str], filename_hints: list[str] | None, warning: str
    ) -> ClassificationResult:
        scores = np.full(len(self.labels), 0.001, dtype=np.float32)
        candidate_labels = []
        names = list(filename_hints or []) + image_paths
        for label in self._filename_candidates(names) + self._visual_candidates(image_paths):
            if label not in candidate_labels:
                candidate_labels.append(label)
        candidate_labels = candidate_labels[:3]

        weights = [0.34, 0.23, 0.15]
        for label, weight in zip(candidate_labels, weights):
            if label in self.labels:
                scores[self.labels.index(label)] = weight

        scores = scores / scores.sum()
        return self._format_prediction(scores, source="deterministic visual fallback", warning=warning)

    def _filename_candidates(self, image_paths: list[str]) -> list[str]:
        names = " ".join(Path(path).name.lower().replace("_", " ") for path in image_paths)
        normalized = "".join(ch if ch.isalnum() else " " for ch in names)
        matches = []
        for label in self.labels:
            label_normalized = label.replace("_", " ")
            if label_normalized in normalized:
                matches.append(label)
        return matches[:3]

    def _visual_candidates(self, image_paths: list[str]) -> list[str]:
        means = []
        saturations = []
        brightness = []

        for path in image_paths:
            with Image.open(path) as image:
                small = image.convert("RGB").resize((64, 64))
            arr = np.asarray(small, dtype=np.float32) / 255.0
            means.append(arr.mean(axis=(0, 1)))
            max_c = arr.max(axis=2)
            min_c = arr.min(axis=2)
            saturations.append(float((max_c - min_c).mean()))
            brightness.append(float(arr.mean()))

        mean_rgb = np.mean(means, axis=0)
        sat = float(np.mean(saturations))
        bright = float(np.mean(brightness))
        red, green, blue = [float(v) for v in mean_rgb]

        if green > red * 1.08 and green > blue * 1.05 and sat > 0.12:
            return ["greek salad", "caesar salad", "seaweed salad"]
        if red > green * 1.08 and sat > 0.16:
            return ["pizza", "spaghetti bolognese", "tacos"]
        if bright < 0.36 and red > blue:
            return ["steak", "chocolate cake", "baby back ribs"]
        if bright > 0.72 and sat < 0.12:
            return ["ice cream", "cheesecake", "panna cotta"]
        if red > 0.45 and green > 0.35 and blue < 0.32:
            return ["french fries", "fried rice", "onion rings"]

        signature = hashlib.sha1("|".join(image_paths).encode("utf-8")).hexdigest()
        seed = int(signature[:8], 16)
        fallback_pool = [
            "fried rice",
            "pizza",
            "hamburger",
            "pasta",
            "omelette",
            "sushi",
            "caesar salad",
            "chicken curry",
        ]
        fallback_pool = [name for name in fallback_pool if name in self.labels]
        if not fallback_pool:
            return self.labels[:3]
        start = seed % len(fallback_pool)
        return [fallback_pool[(start + offset) % len(fallback_pool)] for offset in range(3)]


def calories_from_macros(nutrition: list[dict]) -> int:
    by_name = {item["name"]: float(item["value"]) for item in nutrition}
    calories = by_name.get("Protein", 0.0) * 4 + by_name.get("Carbohydrates", 0.0) * 4
    calories += by_name.get("Fat", 0.0) * 9
    if math.isnan(calories):
        return 0
    return round(calories)
