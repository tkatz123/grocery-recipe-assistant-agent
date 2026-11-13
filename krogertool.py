"""Live Kroger product prices. Credentials stay outside tool arguments."""
import math
import time
from decimal import Decimal

import requests
import streamlit as st
import truststore

truststore.inject_into_ssl()
BASE = "https://api.kroger.com/v1"


class KrogerClient:
    def __init__(self, client_id, client_secret):
        self.client_id = client_id
        self.client_secret = client_secret
        self.session = requests.Session()
        self.token = None
        self.expires_at = 0
        self.store = None

    def _authenticate(self):
        response = self.session.post(
            f"{BASE}/connect/oauth2/token",
            auth=(self.client_id, self.client_secret),
            data={"grant_type": "client_credentials", "scope": "product.compact"},
            timeout=30,
        )
        if not response.ok:
            raise RuntimeError(f"Kroger authentication failed (HTTP {response.status_code})")
        payload = response.json()
        self.token = payload["access_token"]
        self.expires_at = time.monotonic() + max(0, float(payload["expires_in"]) - 60)

    def _get(self, endpoint, params):
        if not self.token or time.monotonic() >= self.expires_at:
            self._authenticate()
        for attempt in range(2):
            response = self.session.get(
                f"{BASE}/{endpoint}", params=params,
                headers={"Authorization": f"Bearer {self.token}"}, timeout=30,
            )
            if response.status_code == 401 and attempt == 0:
                self._authenticate()
                continue
            if not response.ok:
                raise RuntimeError(f"Kroger {endpoint} request failed (HTTP {response.status_code})")
            return response.json()

    def get_store(self):
        """Use the first real store returned near downtown Cincinnati."""
        if self.store is None:
            locations = self._get("locations", {
                "filter.zipCode.near": "45202", "filter.radiusInMiles": 25,
                "filter.limit": 10,
            }).get("data", [])
            if not locations:
                raise RuntimeError("Kroger returned no stores near the default ZIP code 45202")
            location = locations[0]
            self.store = {"location_id": location["locationId"],
                          "name": location.get("name", ""),
                          "address": location.get("address", {})}
        return self.store

    def search(self, term, limit=5):
        if not isinstance(term, str) or not term.strip():
            raise ValueError("Provide a nonempty product search term")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 50:
            raise ValueError("limit must be an integer between 1 and 50")
        store = self.get_store()
        data = self._get("products", {
            "filter.term": term.strip(), "filter.locationId": store["location_id"],
            "filter.limit": limit,
        }).get("data", [])
        products = []
        for product in data:
            for item in product.get("items", []):
                price = item.get("price") or {}
                # Missing/zero prices are unknown, never free groceries.
                regular = positive_price(price.get("regular"))
                promo = positive_price(price.get("promo"))
                products.append({
                    "product_id": product["productId"], "upc": product.get("upc"),
                    "item_id": item.get("itemId"), "name": product.get("description", ""),
                    "brand": product.get("brand", ""), "size": item.get("size", ""),
                    "regular_price": regular, "promo_price": promo,
                    "price": promo if promo is not None else regular,
                    "inventory": item.get("inventory"),
                    "fulfillment": item.get("fulfillment"),
                    "sold_by": item.get("soldBy"),
                })
        return {"store": store, "query": term, "currency": "USD",
                "products": products, "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "note": "Store-specific listed prices; verify product, package size, sale eligibility, and availability. Taxes and fees excluded."}


def positive_price(value):
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        if math.isfinite(value) and value > 0:
            return float(value)
    return None


@st.cache_resource
def get_client():
    return KrogerClient(st.secrets["kroger"]["client_id"],
                        st.secrets["kroger"]["client_secret"])


def search_grocery_prices(ingredient: str, limit: int = 5) -> dict:
    """AI tool: search live products/prices for one ingredient at a US store.

    Returns options for relevance/package selection; does not automatically
    equate a recipe measure with a purchasable package.
    """
    return get_client().search(ingredient, limit)


def estimate_selected_basket(items: list[dict], budget: float | None = None) -> dict:
    """Total confirmed fixed-price packages: [{product: <option>, quantity: 2}].

    Product options should come from search_grocery_prices. The caller must
    confirm ingredient relevance and package quantity before selecting them.
    """
    if budget is not None and (not math.isfinite(budget) or budget < 0):
        raise ValueError("budget must be a finite nonnegative amount")
    total = Decimal("0")
    lines = []
    for entry in items:
        product, quantity = entry["product"], entry["quantity"]
        if isinstance(quantity, bool) or not isinstance(quantity, int) or quantity < 1:
            raise ValueError("Each quantity must be a positive integer package count")
        if product.get("sold_by") not in (None, "", "UNIT"):
            raise ValueError("Variable-weight products need an explicit weight calculation")
        price = positive_price(product.get("price"))
        if price is None:
            raise ValueError(f"No usable price for {product.get('name', 'product')}")
        subtotal = Decimal(str(price)) * quantity
        total += subtotal
        lines.append({"product": product, "quantity": quantity, "subtotal": float(subtotal)})
    return {"items": lines, "estimated_total": float(total), "currency": "USD",
            "budget": budget, "within_budget": total <= Decimal(str(budget)) if budget is not None else None,
            "note": "Estimate for selected packages only; excludes taxes, fees, and unpriced ingredients."}


# Generic JSON Schema for registration with your agent framework.
SEARCH_GROCERY_PRICES_SCHEMA = {
    "name": "search_grocery_prices",
    "description": "Look up live grocery product options and prices at a US Kroger-family store.",
    "parameters": {"type": "object", "properties": {
        "ingredient": {"type": "string", "description": "Ingredient name, e.g. chicken breasts"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
    }, "required": ["ingredient"], "additionalProperties": False},
}
