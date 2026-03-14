"""
get_nutrition_data.py
Run this ONCE to regenerate nutrition101.csv from the USDA FoodData Central API.
Your existing nutrition101.csv is fine — only run this if you need fresh data.

Usage:
    python get_nutrition_data.py

Get a free API key at: https://fdc.nal.usda.gov/api-guide.html
Replace USDA_API_KEY below with your key.
"""

import requests
import pandas as pd

USDA_API_KEY = "YOUR_USDA_API_KEY_HERE"

FOOD_NAMES = [
    'apple pie', 'baby back ribs', 'baklava', 'beef carpaccio', 'beef tartare',
    'beet salad', 'beignets', 'bibimbap', 'bread pudding', 'breakfast burrito',
    'bruschetta', 'caesar salad', 'cannoli', 'caprese salad', 'carrot cake',
    'ceviche', 'cheese plate', 'cheesecake', 'chicken curry', 'chicken quesadilla',
    'chicken wings', 'chocolate cake', 'chocolate mousse', 'churros', 'clam chowder',
    'club sandwich', 'crab cakes', 'creme brulee', 'croque madame', 'cup cakes',
    'deviled eggs', 'donuts', 'dumplings', 'edamame', 'eggs benedict',
    'escargots', 'falafel', 'filet mignon', 'fish and chips', 'foie gras',
    'french fries', 'french onion soup', 'french toast', 'fried calamari', 'fried rice',
    'frozen yogurt', 'garlic bread', 'gnocchi', 'greek salad', 'grilled cheese sandwich',
    'grilled salmon', 'guacamole', 'gyoza', 'hamburger', 'hot and sour soup',
    'hot dog', 'huevos rancheros', 'hummus', 'ice cream', 'lasagna',
    'lobster bisque', 'lobster roll sandwich', 'macaroni and cheese', 'macarons', 'miso soup',
    'mussels', 'nachos', 'omelette', 'onion rings', 'oysters',
    'pad thai', 'paella', 'pancakes', 'panna cotta', 'peking duck',
    'pho', 'pizza', 'pork chop', 'poutine', 'prime rib',
    'pulled pork sandwich', 'ramen', 'ravioli', 'red velvet cake', 'risotto',
    'samosa', 'sashimi', 'scallops', 'seaweed salad', 'shrimp and grits',
    'spaghetti bolognese', 'spaghetti carbonara', 'spring rolls', 'steak', 'strawberry shortcake',
    'sushi', 'tacos', 'octopus balls', 'tiramisu', 'tuna tartare', 'waffles',
]


def get_nutrition(food_name):
    url = (
        f"https://api.nal.usda.gov/fdc/v1/foods/search"
        f"?api_key={USDA_API_KEY}&query={food_name}&pageSize=1"
    )
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("foods"):
        print(f"  WARNING: No results for '{food_name}'")
        return None

    food = data["foods"][0]
    nutrients = {n["nutrientNumber"]: n.get("value", 0) for n in food.get("foodNutrients", [])}

    protein  = float(nutrients.get("203", 0))
    calcium  = float(nutrients.get("301", 0)) / 1000
    fat      = float(nutrients.get("204", 0))
    carbs    = float(nutrients.get("205", 0))
    vitamins = (float(nutrients.get("318", 0)) + float(nutrients.get("401", 0))) / 1000

    return {
        "name": food_name,
        "protein": protein,
        "calcium": calcium,
        "fat": fat,
        "carbohydrates": carbs,
        "vitamins": vitamins,
    }


def main():
    rows = []
    for i, name in enumerate(FOOD_NAMES, 1):
        print(f"[{i:03d}/{len(FOOD_NAMES)}] {name}")
        try:
            row = get_nutrition(name)
            if row:
                rows.append(row)
        except Exception as e:
            print(f"  ERROR: {e}")

    # pandas 2.0 compatible (replaces deprecated .append())
    df = pd.DataFrame(rows, columns=["name", "protein", "calcium", "fat", "carbohydrates", "vitamins"])
    df = df.reset_index(drop=True)
    df.to_csv("nutrition101.csv")
    print(f"\nSaved nutrition101.csv with {len(df)} rows")


if __name__ == "__main__":
    main()
