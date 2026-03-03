"""
Level 2: TF-IDF + Cosine Similarity Matching
----------------------------------------------
scikit-learn TF-IDF vectorizer use panni product text-ai
tags-oda compare panni top matching tags return pannurom.
"""

import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(products_path: str, tags_path: str):
    """Load products and tags from JSON files."""
    with open(products_path, "r", encoding="utf-8") as f:
        products = json.load(f)
    with open(tags_path, "r", encoding="utf-8") as f:
        tags = json.load(f)
    return products, tags


def run_tfidf_matcher(products_path: str, tags_path: str, top_n: int = 3) -> list:
    """
    Level 2: TF-IDF vectorizer use panni product text-ai tags-oda
    cosine similarity calculate panni top_n tags-ai return pannurom.
    """
    products, tags = load_data(products_path, tags_path)

    product_texts = [
        p.get("name", "") + " " + p.get("description", "")
        for p in products
    ]

    vectorizer = TfidfVectorizer()
    all_texts = product_texts + tags
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    product_vecs = tfidf_matrix[: len(products)]
    tag_vecs = tfidf_matrix[len(products):]

    results = []
    for i, product in enumerate(products):
        sims = cosine_similarity(product_vecs[i], tag_vecs).flatten()
        top_indices = np.argsort(sims)[::-1][:top_n]
        matched_tags = [tags[idx] for idx in top_indices if sims[idx] > 0]
        results.append({
            "id": product["id"],
            "name": product["name"],
            "matched_tags": matched_tags,
            "method": "tfidf_cosine_similarity"
        })
    return results
