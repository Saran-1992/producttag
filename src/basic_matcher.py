"""
Level 1: Basic String Matching Logic
-------------------------------------
Product name and description-il irukka words-ai master tags-oda
compare panni tags assign pannurom (case-insensitive).
"""

import json


def load_data(products_path: str, tags_path: str):
    """Load products and tags from JSON files."""
    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = json.load(f)
    return products, tags


def match_tags(product: dict, tags: list) -> list:
    """
    Level 1: Simple keyword matching.
    Product name + description text-il tag irundha assign panni return pannurom.
    """
    text = (product.get("name", "") + " " + product.get("description", "")).lower()
    matched = [tag for tag in tags if tag.lower() in text]
    return matched


def run_basic_matcher(products_path: str, tags_path: str) -> list:
    """Run Level 1 matching on all products and return results."""
    products, tags = load_data(products_path, tags_path)
    results = []
    for product in products:
        matched_tags = match_tags(product, tags)
        results.append({
            "id": product["id"],
            "name": product["name"],
            "matched_tags": matched_tags,
            "method": "basic_string_match"
        })
    return results
