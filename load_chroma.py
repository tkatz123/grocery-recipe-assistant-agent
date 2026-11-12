"""Load data/recipes.json into a persistent Chroma collection (one document per recipe).

Run fetch_recipes.py first. Re-running is safe: upsert by recipe id avoids duplicates.

Usage:
    python load_chroma.py                       # load, then run sample queries
    python load_chroma.py --query "spicy noodle soup" --n 5
"""

import argparse
import json
from pathlib import Path

import chromadb
import truststore

# Chroma downloads its default embedding model (~80MB) on first use; use the OS
# certificate store so this works behind TLS-inspecting antivirus/proxies.
truststore.inject_into_ssl()

ROOT = Path(__file__).parent
RECIPES_PATH = ROOT / "data" / "recipes.json"
DB_PATH = ROOT / "chroma_db"
COLLECTION = "recipes"


def to_document(r):
    ing_lines = "\n".join(
        f"- {i['measure']} {i['ingredient']}".replace("-  ", "- ") for i in r["ingredients"]
    )
    return (
        f"Recipe: {r['name']}\n"
        f"Category: {r['category']}\n"
        f"Cuisine: {r['area']}\n"
        f"Tags: {', '.join(r['tags'])}\n\n"
        f"Ingredients:\n{ing_lines}\n\n"
        f"Instructions:\n{r['instructions']}"
    )


def to_metadata(r):
    # Chroma metadata values must be str/int/float/bool (no None, no lists)
    return {
        "name": r["name"],
        "category": r["category"],
        "area": r["area"],
        "tags": ",".join(r["tags"]),
        "ingredients": ",".join(i["ingredient"].lower() for i in r["ingredients"]),
        "num_ingredients": len(r["ingredients"]),
        "image": r["image"],
        "youtube": r["youtube"],
        "source": r["source"],
    }


def get_collection():
    client = chromadb.PersistentClient(path=str(DB_PATH))
    return client.get_or_create_collection(COLLECTION)


def load():
    with open(RECIPES_PATH, encoding="utf-8") as f:
        recipes = json.load(f)
    collection = get_collection()
    batch = 100
    for start in range(0, len(recipes), batch):
        chunk = recipes[start : start + batch]
        collection.upsert(
            ids=[r["id"] for r in chunk],
            documents=[to_document(r) for r in chunk],
            metadatas=[to_metadata(r) for r in chunk],
        )
        print(f"  upserted {start + len(chunk)}/{len(recipes)}")
    print(f"Collection '{COLLECTION}' now has {collection.count()} documents")
    return collection


def search(collection, text, n=5, where=None):
    res = collection.query(query_texts=[text], n_results=n, where=where)
    print(f"\nQuery: {text!r}" + (f"  where={where}" if where else ""))
    for meta, dist in zip(res["metadatas"][0], res["distances"][0]):
        print(f"  {dist:.3f}  {meta['name']}  ({meta['category']}, {meta['area']})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", help="run a single query against the existing collection")
    parser.add_argument("--n", type=int, default=5)
    args = parser.parse_args()

    if args.query:
        search(get_collection(), args.query, args.n)
        return

    collection = load()
    search(collection, "something quick with chicken and rice")
    search(collection, "I have eggs, spinach and feta cheese")
    search(collection, "creamy pasta", where={"area": "Italian"})
    search(collection, "chocolate dessert", where={"category": "Dessert"})


if __name__ == "__main__":
    main()
