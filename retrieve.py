"""Vector top 20 -> ingredient top 5 -> cross-encoder -> full JSON by ID."""
import argparse
import json
import re
import unicodedata

from load_chroma import DB_PATH, COLLECTION, RECIPES_PATH, to_document


def normalize(name):
    # Exact normalized names: no unsafe fuzzy equivalence for dietary constraints.
    name = unicodedata.normalize("NFKC", name).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", name).split())


class RecipeRetriever:
    def __init__(self, recipes_path=RECIPES_PATH, collection=None, reranker=None):
        with open(recipes_path, encoding="utf-8-sig") as f:
            recipes = json.load(f)
        self.recipes = {}
        for recipe in recipes:
            recipe_id = str(recipe["id"])
            if recipe_id in self.recipes:
                raise ValueError(f"Duplicate recipe ID: {recipe_id}")
            self.recipes[recipe_id] = recipe
        if collection is None:
            import chromadb
            # Fail if not loaded, rather than silently creating an empty database.
            collection = chromadb.PersistentClient(path=str(DB_PATH)).get_collection(COLLECTION)
        self.collection = collection
        self.reranker = reranker

    def get_recipe(self, recipe_id):
        """Return the complete JSON record, including measures and instructions."""
        return self.recipes[str(recipe_id)]

    def search(self, query, ingredients=(), where=None):
        if isinstance(ingredients, str):
            raise TypeError("ingredients must be a list of ingredient names")
        available = {normalize(i) for i in ingredients if normalize(i)}
        query = query.strip()
        if not query and not available:
            raise ValueError("Provide a query or at least one ingredient")
        text = query
        if available:
            text += "\nAvailable ingredients: " + ", ".join(sorted(available))
        count = self.collection.count()
        if not count:
            return []
        kwargs = dict(query_texts=[text], n_results=min(20, count),
                      include=["distances"])
        if where:
            kwargs["where"] = where
        result = self.collection.query(**kwargs)
        candidates = []
        for recipe_id, distance in zip(result["ids"][0], result["distances"][0]):
            if recipe_id not in self.recipes:
                raise ValueError(f"Chroma ID {recipe_id} missing from recipes.json; reload the index")
            recipe = self.recipes[recipe_id]
            needed = {normalize(i["ingredient"]) for i in recipe["ingredients"]}
            needed.discard("")
            matched = needed & available
            candidates.append({
                "recipe_id": recipe_id, "vector_distance": float(distance),
                "overlap_count": len(matched),
                "overlap_score": len(matched) / len(available) if available else 0.0,
                "matched_ingredients": sorted(matched),
                "missing_ingredients": sorted(needed - available),
            })
        # Count rewards recipes using more of the user's distinct ingredients.
        # Vector distance breaks ties and handles searches without ingredients.
        candidates.sort(key=lambda c: (-c["overlap_count"], c["vector_distance"]))
        finalists = candidates[:5]
        if not finalists:
            return []
        if self.reranker is None:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [(text, to_document(self.recipes[c["recipe_id"]])) for c in finalists]
        scores = self.reranker.predict(pairs)
        if len(scores) != len(finalists):
            raise ValueError("Cross-encoder returned an unexpected number of scores")
        for candidate, score in zip(finalists, scores):
            candidate["rerank_score"] = float(score)
        finalists.sort(key=lambda c: -c["rerank_score"])
        for candidate in finalists:
            candidate["recipe"] = self.get_recipe(candidate["recipe_id"])
        return finalists


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="")
    parser.add_argument("--ingredients", nargs="*", default=[])
    parser.add_argument("--category")
    parser.add_argument("--area", help="Cuisine, using the existing JSON area field")
    args = parser.parse_args()
    filters = [{k: v} for k, v in (("category", args.category), ("area", args.area)) if v]
    where = {"$and": filters} if len(filters) > 1 else (filters[0] if filters else None)
    results = RecipeRetriever().search(args.query, args.ingredients, where)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
