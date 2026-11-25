import streamlit as st
from langchain.tools import tool, ToolRuntime
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from tools.retrieve import RecipeRetriever, normalize
from tools.krogertool import search_grocery_prices, estimate_selected_basket

retriever = RecipeRetriever()


# ============== Product relevance check ==============
# Kroger's search ranks by keyword, so "beef cutlet" can return a cutlet sauce and
# "mixed grain" can return cereal bars. A similarity score cannot fix this: it would
# also reject real matches whose names share no words (courgettes vs. zucchini).
# So a small LLM judges each product, and non-matches never reach the agent.

class ProductVerdict(BaseModel):
    index: int = Field(description="The product's number from the list")
    is_match: bool = Field(description="True only if the product is the ingredient itself, in a form a home cook would buy to cook with")


class ProductVerdicts(BaseModel):
    verdicts: list[ProductVerdict]


JUDGE_PROMPT = """You check grocery search results for a recipe ingredient.

Ingredient needed: {ingredient}

Products the store returned:
{listing}

For EACH product, decide whether it IS the ingredient, in a form a home cook would buy to use it in a recipe.

A match: the ingredient itself, fresh, frozen, canned, dried or ground. Different brands, package sizes and cuts of the same food are fine. Recipe ingredient names are often British: courgette = zucchini, aubergine = eggplant, coriander leaves = cilantro, minced beef = ground beef, tinned = canned, spring onion = green onion, double cream = heavy cream.

Not a match: a sauce, seasoning, mix or dressing named after the ingredient (steak sauce is not steak); a snack, drink, cereal or ready meal (breakfast bars are not grain); a different food that only shares a word (chili-style diced tomatoes are not red chilli); a flavored version when the plain ingredient is needed (strawberry yogurt is not yogurt); cooked deli meat when raw meat is needed for cooking.

Return a verdict for every product number."""

_judge = ChatOpenAI(
    model="gpt-4.1-mini",
    api_key=st.secrets["openai"]["api_key"],
    temperature=0,
).with_structured_output(ProductVerdicts)


# Every product search_prices has returned, by product_id. estimate_basket accepts only
# these and always uses this stored copy, never the model's, so an invented product
# (seen in testing: "Mixed Grain", product_id 0000000000000, $3.00) or an edited price
# is rejected instead of being totaled.
_returned_products = {}

# The search terms that returned each product, by product_id. estimate_basket uses this
# to refuse a product tagged as a different ingredient than the one it was found for
# (seen in run 5: ground coriander tagged as "turmeric powder").
_product_searches = {}

# Judge verdicts, cached by ingredient and the exact products shown, so a repeated
# search (or a cached Kroger result) does not pay for a second judge call.
_judge_cache = {}


def _matching_products(ingredient, products):
    """Keep only the products the judge says really are the ingredient."""
    if not products:
        return products
    cache_key = (normalize(ingredient), tuple(p['product_id'] for p in products))
    if cache_key in _judge_cache:
        keep = _judge_cache[cache_key]
        return [p for i, p in enumerate(products) if i in keep]
    listing = "\n".join(f"{i}. {p['name']} ({p['size']})" for i, p in enumerate(products))
    try:
        result = _judge.invoke(JUDGE_PROMPT.format(ingredient=ingredient, listing=listing))
    except Exception:
        # If the check itself fails, fall back to the unfiltered list rather than
        # blocking the shopping list; the agent's own product rules still apply.
        return products
    keep = {v.index for v in result.verdicts if v.is_match}
    _judge_cache[cache_key] = keep
    return [p for i, p in enumerate(products) if i in keep]


# ============== Ingredient name matching ==============

# Recipe names are often British; the store and the user use American names. Used so a
# zucchini on the list covers a recipe's "Courgettes" (run 5 reported it missing).
# Whole normalized names only: "coriander" alone is left out because it can mean the
# leaves (cilantro) or the ground seed.
SYNONYMS = {
    "courgette": "zucchini", "courgettes": "zucchini",
    "aubergine": "eggplant", "aubergines": "eggplant",
    "coriander leaves": "cilantro", "fresh coriander": "cilantro",
    "spring onion": "green onion", "spring onions": "green onion",
    "scallion": "green onion", "scallions": "green onion",
    "minced beef": "ground beef", "beef mince": "ground beef",
    "minced pork": "ground pork", "pork mince": "ground pork",
    "minced lamb": "ground lamb", "lamb mince": "ground lamb",
    "double cream": "heavy cream", "single cream": "light cream",
    "chick peas": "chickpeas", "garbanzo beans": "chickpeas",
    "rocket": "arugula", "beetroot": "beets", "swede": "rutabaga",
    "mangetout": "snow peas", "prawns": "shrimp", "king prawns": "shrimp",
    "plain flour": "all purpose flour", "caster sugar": "superfine sugar",
    "icing sugar": "powdered sugar", "tinned tomatoes": "canned tomatoes",
    "tinned tomatos": "canned tomatoes", "chopped tomatoes": "diced tomatoes",
    "red chilli": "red chili", "green chilli": "green chili",
}


def _alt_names(name):
    """A normalized ingredient name plus its American equivalent, if it has one."""
    return {name, SYNONYMS.get(name, name)}


# Words that turn a food into a different product: the user's "chicken" does not cover
# a recipe's "chicken stock", and a "garlic" item does not cover "garlic powder".
DERIVED_WORDS = {
    "stock", "broth", "bouillon", "cube", "cubes", "sauce", "paste", "powder",
    "oil", "vinegar", "juice", "extract", "flakes", "granules", "seasoning",
}

# Descriptive words ignored when checking a product's tag against the search that found
# it, so "ground cumin" is not accepted for a product found by "ground coriander".
DESCRIPTOR_WORDS = {
    "ground", "fresh", "dried", "powder", "chopped", "diced", "sliced", "minced", "whole",
    "large", "small", "red", "green", "yellow", "white", "black", "leaves", "leaf",
    "tinned", "canned", "frozen", "raw", "plain", "sweet", "hot", "smoked", "organic",
    "of", "and",
}


def _content_words(name):
    """The words of a normalized name and its American equivalent, minus descriptors."""
    words = set()
    for alt in _alt_names(name):
        words |= set(alt.split())
    return words - DESCRIPTOR_WORDS


def _tag_fits_product(tag, product):
    """True if a basket item's ingredient tag fits the search(es) that found the product.

    Cheap check first: the tag shares a real word with a search term ("chicken thighs"
    for a product found by "chicken thighs"). Otherwise, for a renamed search such as
    zucchini tagged "Courgettes", the relevance judge decides.
    """
    tag_words = _content_words(tag)
    for term in _product_searches.get(product['product_id'], ()):
        if tag_words & _content_words(term):
            return True
    if _matching_products(tag, [product]):
        _product_searches.setdefault(product['product_id'], set()).add(tag)
        return True
    return False

# Store-bought products whose labels can hide allergens the recipe never names.
LABEL_WORDS = {"paste", "sauce", "pesto", "satay", "dressing", "stock", "bouillon"}


def _label_items(recipe):
    """The recipe ingredients that are store-bought pastes, sauces or stocks."""
    return [item["ingredient"] for item in recipe["ingredients"]
            if set(normalize(item["ingredient"]).split()) & LABEL_WORDS]


@tool
def search_recipes(query: str, ingredients: list[str]) -> list[dict]:
    """Search the recipe database and return up to 5 matching recipes, best match first.

    Use this whenever the user wants recipe ideas: from ingredients they have at home,
    from groceries you suggested buying, or from a description of a dish.

    Args:
        query: A short description of the dish the user wants, for example
            "quick weeknight dinner" or "spicy noodle soup". Pass an empty string ""
            if the user only gave ingredients.
        ingredients: Ingredient names the user has or plans to buy, one per item,
            for example ["chicken", "rice", "garlic"]. Pass an empty list [] if the
            user only described a dish.

    Each result has the recipe_id, name, category, cuisine area, the user's
    ingredients it uses (matched_ingredients), and the ingredients the user still
    needs (missing_ingredients). Instructions are NOT included: call get_recipe with
    the recipe_id when the user picks a recipe or wants step-by-step cooking help.

    check_label lists the recipe's store-bought pastes, sauces and stocks. Their labels
    can hide allergens the recipe does not name: if the user has ANY allergy and you
    recommend this recipe, tell them to check each of these labels for that allergen.
    """

    recipes = []

    results = retriever.search(query, ingredients)

    for result in results:

        recipe_dict = {
            'recipe_id': result['recipe_id'],
            'name': result['recipe']['name'],
            'category': result['recipe']['category'],
            'area': result['recipe']['area'],
            'matched_ingredients': result['matched_ingredients'],
            'missing_ingredients': result['missing_ingredients'],
            # Put the allergen-label reminder next to the recipe itself, where the model
            # reads it (run 5 dropped the curry paste warning when it was only in the prompt).
            'check_label': _label_items(result['recipe'])

        }

        recipes.append(recipe_dict)

    return recipes

@tool
def get_recipe(recipe_id: str) -> dict:
    """Get one full recipe by its recipe_id: every ingredient with its measure, plus the cooking instructions.

    Use this after search_recipes, when the user chooses a recipe, asks what
    quantities they need, or wants to be walked through cooking it step by step.

    Args:
        recipe_id: The recipe_id from a search_recipes result, for example "53085".
    """

    return retriever.get_recipe(recipe_id)

@tool
def search_prices(ingredient: str) -> list[dict]:
    """Look up live grocery products and prices for ONE ingredient at a Kroger store.

    Use this when the user wants to know what something costs, what to buy on a
    budget, or what the missing_ingredients of a recipe would cost. Call it once
    per ingredient. Skip pantry staples (salt, pepper, water, cooking oil) unless
    the user asks for them.

    Args:
        ingredient: One plain ingredient name, for example "chicken breast" or
            "basmati rice". Never a comma-separated list.

    Returns a list of product options with product_id, name, brand, size, price
    (USD, sale price if there is one, None if Kroger lists no price) and sold_by.
    Products that are not actually the ingredient (sauces, snacks, look-alikes)
    are already removed. Pick the option with a sensible package size.
    An empty list means the store has nothing that matches: retry once with a
    plain American name, then tell the user the ingredient was not found.
    Prices are store listed prices and exclude tax.
    """

    products = []

    try:
        results = search_grocery_prices(ingredient)

        for product in results['products']:

            products_dict = {
                'product_id': product['product_id'],
                # Strip trademark symbols: the model garbles them when it copies names back.
                'name': product['name'].replace('®', '').replace('™', '').replace('  ', ' ').strip(),
                'brand': product['brand'],
                'size': product['size'],
                'price': product['price'],
                'sold_by': product['sold_by']
            }

            products.append(products_dict)

        # Drop products that are not actually the ingredient (see _matching_products).
        matches = _matching_products(ingredient, products)
        # Remember every product actually returned, so estimate_basket can refuse
        # products the model made up and always use the real price.
        for p in matches:
            _returned_products[p['product_id']] = p
            _product_searches.setdefault(p['product_id'], set()).add(normalize(ingredient))
        return matches

    except (RuntimeError, ValueError) as e:
        return {'error': str(e)}

# Pantry staples the user is assumed to have. Must match the list in prompts.py.
STAPLES = {
    "salt", "pepper", "black pepper", "ground black pepper", "water",
    "oil", "vegetable oil", "olive oil", "extra virgin olive oil",
    "sunflower oil", "rapeseed oil", "canola oil",
}


def _user_has(ingredient, user_words):
    """True if every word of something the user has appears in the recipe ingredient.

    So a user's "chicken" covers a recipe's "chicken thighs", but not "chicken stock":
    a derived product (stock, sauce, powder, ...) is covered only if the user's own
    item names it too.
    """
    words = set(ingredient.split())
    derived = words & DERIVED_WORDS
    return any(term and term <= words and derived <= term for term in user_words)


def check_coverage(items, recipe_ids, user_has):
    """For each chosen recipe, list the ingredients nothing accounts for.

    An ingredient is covered if it is a pantry staple, something the user has, or
    the "ingredient" tag of a basket item. Returns {recipe name: [uncovered]}.
    """
    in_basket = {normalize(entry.get("ingredient") or "") for entry in items}
    # A tag covers a longer recipe name built from it: "garlic" covers "Garlic Clove".
    basket_words = [set(tag.split()) for tag in in_basket]
    user_words = [set(normalize(i).split()) for i in user_has]
    coverage = {}
    for recipe_id in recipe_ids:
        recipe = retriever.get_recipe(recipe_id)
        uncovered = []
        for item in recipe["ingredients"]:
            name = normalize(item["ingredient"])
            if not name or name in STAPLES:
                continue
            # Check the recipe's own name and its American equivalent (courgettes -> zucchini).
            if any(alt in in_basket or _user_has(alt, basket_words) or _user_has(alt, user_words)
                   for alt in _alt_names(name)):
                continue
            uncovered.append(item["ingredient"])
        coverage[recipe["name"]] = uncovered
    return coverage


@tool
def estimate_basket(items: list[dict], recipe_ids: list[str],
                    user_has: list[str] | None = None, budget: float | None = None) -> dict:
    """Total the shopping list, check the budget, and check every chosen recipe is fully covered.

    ALWAYS use this to total a basket instead of doing the arithmetic yourself.
    Use it after choosing products with search_prices.

    Args:
        items: One entry per product to buy. Every entry names the recipe ingredient it
            is for, copied exactly as it appears in the recipe. The shape depends on the
            product's sold_by:
            sold_by "UNIT":   {"ingredient": "cannellini beans", "product": <product dict copied unchanged from search_prices>, "quantity": 2}
                              quantity is a whole number of packages, at least 1.
            sold_by "WEIGHT": {"ingredient": "chicken thighs", "product": <product dict copied unchanged from search_prices>, "pounds": 1.5}
                              price is per pound; pounds is how much the recipe needs.
        recipe_ids: The recipe_id of every recipe you are recommending, for example ["53143", "52811"].
        user_has: Ingredients the user said they already have, for example ["chicken", "rice"].
            Omit it if they named none.
        budget: The user's budget in US dollars, for example 40.0. Omit it if the
            user gave no budget.

    Returns each line with its subtotal, the estimated_total, within_budget, and
    "uncovered": for each recipe, the ingredients that are not on the list, not a
    pantry staple, and not something the user has. all_covered is true only when
    every recipe is fully covered. If anything is uncovered, price it and call this
    again with the complete list, or tell the user exactly what is missing.
    Weight items are estimates, since real packages vary in weight.
    If you get an error, read it, fix the call, and try again.
    """

    # Only products search_prices really returned, using the stored copy (real name and price).
    checked = []
    for entry in items:
        product = entry.get("product") or {}
        stored = _returned_products.get(product.get("product_id"))
        if stored is None:
            return {'error': f"{product.get('name', 'A product')} was not returned by search_prices. "
                             "Only use products exactly as search_prices returned them. If the store "
                             "has nothing for an ingredient, leave it out and tell the user it was not found."}
        # The product must be for the ingredient it is tagged as, not just any searched one.
        tag = normalize(entry.get("ingredient") or "")
        # A real product_id under a made-up name (seen November 24: "Badia Ground Cumin" sent
        # with the product_id of the Badia Smoked Paprika it had found). Say so directly, or
        # the model cannot connect the error to what it thinks it sent and resends it.
        # Only a name with a word the real product lacks counts; a trimmed name is fine.
        sent_name = product.get("name")
        if sent_name and set(normalize(sent_name).split()) - set(normalize(stored['name']).split()):
            return {'error': f"product_id {stored['product_id']} belongs to {stored['name']}, not {sent_name}. "
                             "Copy each product exactly as search_prices returned it. If no search "
                             f"returned a product for {entry.get('ingredient')}, call search_prices for it."}
        if tag and not _tag_fits_product(tag, stored):
            searched = ", ".join(sorted(_product_searches.get(stored['product_id'], ()))) or "another ingredient"
            return {'error': f"{stored['name']} was found by searching for {searched}, and it is not "
                             f"{entry.get('ingredient')}. Call search_prices for {entry.get('ingredient')} "
                             "and use a product from that search. If the store has none, either leave it "
                             "out, or buy a real substitute tagged as what it actually is (for example "
                             "tag brown rice as 'brown rice'), and tell the user what is missing or "
                             "substituted."}
        checked.append({**entry, "product": stored})

    # Never charge the user for something they said they already have (run 5 priced
    # basmati rice for a user who has rice). Dropped items are reported back.
    user_words = [set(normalize(i).split()) for i in (user_has or [])]
    items, already_have = [], []
    for entry in checked:
        tag = normalize(entry.get("ingredient") or "")
        if tag and any(_user_has(alt, user_words) for alt in _alt_names(tag)):
            already_have.append(entry.get("ingredient"))
        else:
            items.append(entry)

    try:
        result = estimate_selected_basket(items, budget)
        coverage = check_coverage(items, recipe_ids, user_has or [])
    except ValueError as e:
        return {'error': str(e)}
    except KeyError as e:
        return {'error': f"Unknown recipe_id {e}. Use a recipe_id from search_recipes."}

    if already_have:
        result['removed_user_has'] = already_have
        result['removed_note'] = ("These were removed because the user already has them. "
                                  "Do not list or price them.")
    result['uncovered'] = coverage
    result['all_covered'] = not any(coverage.values())
    if not result['all_covered']:
        result['uncovered_hint'] = (
            "Tag each item with the recipe's own ingredient name, even if you searched under "
            "another name (for example tag zucchini as 'Courgettes'). Retag and call again, or "
            "price the missing ingredient. If the store has nothing for it, tell the user.")
    return result

def _load_profile(runtime):

    namespace = ("users", runtime.context.user_id)

    item = runtime.store.get(namespace, "profile")

    if item is not None:
        return item.value
    else:
        return {}

@tool
def get_profile(runtime: ToolRuntime) -> dict:
    """Load what you remember about this user from earlier conversations.

    Call this ONCE at the start of every new conversation, before anything else,
    so you know their allergies, dietary restrictions, usual budget, pantry, likes
    and dislikes. Returns an empty dict if this is a new user.
    """

    return _load_profile(runtime)

@tool
def save_profile(
                 runtime: ToolRuntime,
                 allergies: list[str] | None = None,
                 dietary_restrictions: list[str] | None  = None,
                 budget: float | None  = None,
                 pantry: list[str] | None  = None,
                 likes: list[str] | None  = None,
                 dislikes: list[str] | None  = None
                ) -> dict:
    """Remember lasting facts about the user for future conversations.

    Call this whenever the user states something that will still be true next
    time: an allergy, a dietary restriction, their usual budget, what they keep
    in their pantry, or foods they like or dislike. Do NOT save one-off requests
    ("just this once", "for tonight").

    Pass only the fields that changed; anything you leave out is kept as it is.
    Each list REPLACES the saved list, so always send the complete updated list:
    if the saved allergies are ["peanuts"] and the user adds shellfish, send
    ["peanuts", "shellfish"]. To clear a field, send an empty list [].

    Args:
        allergies: Foods the user is allergic to, for example ["peanuts"].
        dietary_restrictions: For example ["vegetarian"] or ["halal", "gluten-free"].
        budget: Their usual grocery budget in US dollars, for example 50.0.
        pantry: Ingredients they usually have at home, for example ["rice", "garlic"].
        likes: Foods or cuisines they like, for example ["Thai", "spicy food"].
        dislikes: Foods they do not want, for example ["mushrooms"].

    Returns the full saved profile.
    """

    profile = _load_profile(runtime)

    profile_dict = {
        'allergies': allergies,
        'dietary_restrictions': dietary_restrictions,
        'budget': budget,
        'pantry': pantry,
        'likes': likes,
        'dislikes': dislikes
    }

    for key, value in profile_dict.items():

        if value is not None:
            profile[key] = value

    runtime.store.put(("users", runtime.context.user_id), 'profile', profile)

    return profile