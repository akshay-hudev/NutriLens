import csv
import logging
from pathlib import Path

from services.classification_service import FOOD101_LABELS, calories_from_macros


LOGGER = logging.getLogger(__name__)


class NutritionService:
    def __init__(self, nutrition_csv: Path, default_density: float) -> None:
        self.nutrition_csv = Path(nutrition_csv)
        self.default_density = default_density
        self.table = self._load_table()

    def estimate_portion(self, food_label: str, volume: dict) -> dict:
        nutrition_100g = self.get_nutrition_100g(food_label)
        density = self.density_for(food_label)
        weight_g = float(volume["volume"]) * density["value"]
        lower_weight_g = float(volume["lower_bound"]) * density["value"]
        upper_weight_g = float(volume["upper_bound"]) * density["value"]

        per100_calories = calories_from_macros(nutrition_100g)
        portion_calories = per100_calories * weight_g / 100.0
        lower_calories = per100_calories * lower_weight_g / 100.0
        upper_calories = per100_calories * upper_weight_g / 100.0

        return {
            "weight_g": round(weight_g, 1),
            "weight_lower_g": round(lower_weight_g, 1),
            "weight_upper_g": round(upper_weight_g, 1),
            "density_g_per_cm3": density["value"],
            "density_source": density["source"],
            "calories": round(portion_calories),
            "calories_lower": round(lower_calories),
            "calories_upper": round(upper_calories),
            "calories_per_100g": per100_calories,
            "nutrition_per_100g": nutrition_100g,
            "nutrition_per_portion": self._scale_nutrition(nutrition_100g, weight_g / 100.0),
            "nutritionix_url": "https://www.nutritionix.com/food/" + food_label.replace(" ", "-"),
        }

    def get_nutrition_100g(self, food_label: str) -> list[dict]:
        if food_label in self.table:
            return self.table[food_label]

        LOGGER.warning("No nutrition row for %s; falling back to %s", food_label, FOOD101_LABELS[0])
        return self.table.get(FOOD101_LABELS[0], [])

    def density_for(self, food_label: str) -> dict:
        label = food_label.lower().replace("_", " ")
        exact = {
            "pizza": 0.55,
            "french fries": 0.42,
            "onion rings": 0.38,
            "ice cream": 0.62,
            "sushi": 0.88,
            "sashimi": 0.93,
            "guacamole": 0.95,
            "hummus": 0.96,
            "edamame": 0.72,
            "waffles": 0.45,
            "pancakes": 0.48,
            "omelette": 0.70,
        }
        if label in exact:
            return {"value": exact[label], "source": "category density prior"}

        groups = [
            (("soup", "bisque", "chowder", "ramen", "pho"), 1.02, "liquid/broth prior"),
            (("salad", "ceviche"), 0.35, "leafy/loose food prior"),
            (("rice", "paella", "risotto", "bibimbap"), 0.76, "cooked grain prior"),
            (("spaghetti", "gnocchi", "ravioli", "pad thai", "dumplings", "gyoza"), 0.78, "cooked starch prior"),
            (("cake", "pie", "baklava", "tiramisu", "pudding", "mousse", "macaron", "cannoli"), 0.55, "dessert prior"),
            (("steak", "ribs", "pork", "beef", "chicken", "duck", "salmon", "scallops", "shrimp", "oysters", "mussels"), 0.94, "protein prior"),
            (("sandwich", "hamburger", "hot dog", "burrito", "quesadilla", "tacos"), 0.60, "assembled food prior"),
            (("bread", "toast", "beignets", "churros", "donuts"), 0.42, "bread/fried dough prior"),
        ]

        for keywords, value, source in groups:
            if any(keyword in label for keyword in keywords):
                return {"value": value, "source": source}

        return {"value": self.default_density, "source": "default food density prior"}

    def _load_table(self) -> dict[str, list[dict]]:
        table: dict[str, list[dict]] = {}
        if not self.nutrition_csv.exists():
            LOGGER.warning("Nutrition CSV not found at %s", self.nutrition_csv)
            return table

        with self.nutrition_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    name = row["name"].strip()
                    table[name] = [
                        {"name": "Protein", "unit": "g", "value": round(float(row["protein"]), 2)},
                        {
                            "name": "Calcium",
                            "unit": "mg",
                            "value": round(float(row["calcium"]) * 1000, 1),
                        },
                        {"name": "Fat", "unit": "g", "value": round(float(row["fat"]), 2)},
                        {
                            "name": "Carbohydrates",
                            "unit": "g",
                            "value": round(float(row["carbohydrates"]), 2),
                        },
                        {
                            "name": "Vitamins",
                            "unit": "mg",
                            "value": round(float(row["vitamins"]) * 1000, 2),
                        },
                    ]
                except Exception:
                    LOGGER.exception("Skipping malformed nutrition row: %s", row)
        return table

    def _scale_nutrition(self, nutrition: list[dict], scale: float) -> list[dict]:
        return [
            {
                "name": item["name"],
                "unit": item["unit"],
                "value": round(float(item["value"]) * scale, 2),
            }
            for item in nutrition
        ]
