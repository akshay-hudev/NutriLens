import importlib.util
import logging
from pathlib import Path

from PIL import Image


LOGGER = logging.getLogger(__name__)


class GeminiDescriptionService:
    def __init__(self, api_key: str, model_name: str, enabled: bool = True) -> None:
        self.api_key = api_key
        self.model_name = model_name
        self.enabled = enabled
        self._configured = False

    @property
    def available(self) -> bool:
        return (
            self.enabled
            and bool(self.api_key)
            and importlib.util.find_spec("google.generativeai") is not None
        )

    def describe(self, image_path: str, food_label: str, calories: int) -> str | None:
        if not self.available:
            return None

        try:
            import google.generativeai as genai

            if not self._configured:
                genai.configure(api_key=self.api_key)
                self._configured = True

            model = genai.GenerativeModel(self.model_name)
            with Image.open(Path(image_path)) as image:
                response = model.generate_content(
                    [
                        (
                            "In one sentence, describe the visible food. "
                            "Mention that nutrition is estimated from the app's 3D volume pipeline. "
                            f"Predicted class: {food_label}. Estimated calories: {calories} kcal."
                        ),
                        image,
                    ]
                )
            return getattr(response, "text", None)
        except Exception:
            LOGGER.warning("Gemini description failed; continuing without optional text.")
            return None
