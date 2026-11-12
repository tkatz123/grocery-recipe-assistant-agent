"""Download every recipe from TheMealDB and save a cleaned copy to data/recipes.json.

TheMealDB has no "list everything" endpoint, so we search by first letter (a-z);
each search returns full meal records.
"""

import json
import string
import time
from pathlib import Path

import requests
import truststore


truststore.inject_into_ssl()

BASE = "https://www.themealdb.com/api/json/v1/1"  # "1" = free test/education key
OUT_PATH = Path(__file__).parent / "data" / "recipes.json"


def fetch_all_meals():
    meals = {}
    with requests.Session() as session:
        for letter in string.ascii_lowercase:
            r = session.get(f"{BASE}/search.php", params={"f": letter}, timeout=15)
            r.raise_for_status()
            batch = r.json().get("meals") or []  # "meals" is null when no results
            for m in batch:
                meals[m["idMeal"]] = m  # dedupe by id
            print(f"  {letter}: {len(batch)} meals")
            time.sleep(0.2)  # be polite to the API

        # Safety net: names starting with digits/accents are missed by a-z search.
        # Collect every id via category filters and look up any we don't have yet.
        categories = session.get(f"{BASE}/list.php", params={"c": "list"}, timeout=15).json()["meals"]
        all_ids = set()
        for c in categories:
            r = session.get(f"{BASE}/filter.php", params={"c": c["strCategory"]}, timeout=15)
            r.raise_for_status()
            all_ids |= {m["idMeal"] for m in r.json().get("meals") or []}
            time.sleep(0.2)
        missing = sorted(all_ids - meals.keys())
        for meal_id in missing:
            r = session.get(f"{BASE}/lookup.php", params={"i": meal_id}, timeout=15)
            r.raise_for_status()
            for m in r.json().get("meals") or []:
                meals[m["idMeal"]] = m
                print(f"  + recovered via category: {m['strMeal']}")
            time.sleep(0.2)
    return list(meals.values())


def clean(meal):
    """Collapse strIngredient1..20 / strMeasure1..20 into one list."""
    ingredients = []
    for i in range(1, 21):
        ing = (meal.get(f"strIngredient{i}") or "").strip()
        meas = (meal.get(f"strMeasure{i}") or "").strip()
        if ing:
            ingredients.append({"ingredient": ing, "measure": meas})
    return {
        "id": meal["idMeal"],
        "name": (meal.get("strMeal") or "").strip(),
        "category": meal.get("strCategory") or "",
        "area": meal.get("strArea") or "",
        "instructions": (meal.get("strInstructions") or "").strip(),
        "tags": [t.strip() for t in (meal.get("strTags") or "").split(",") if t.strip()],
        "image": meal.get("strMealThumb") or "",
        "youtube": meal.get("strYoutube") or "",
        "source": meal.get("strSource") or "",
        "ingredients": ingredients,
    }


def main():
    print("Fetching meals from TheMealDB...")
    recipes = sorted((clean(m) for m in fetch_all_meals()), key=lambda r: r["name"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(recipes)} recipes to {OUT_PATH}")


if __name__ == "__main__":
    main()
