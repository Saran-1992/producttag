"""
app.py – Flask API for ProductTaggingSystem
Vercel serverless deployment entry point.

Endpoints:
  GET  /api/tag?level=1           → Tag all products (Level 1 or 2)
  GET  /api/products              → List all products
  GET  /api/tags                  → List all master tags
  POST /api/tag/single            → Tag a single product (JSON body)
"""

import os
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRODUCTS_PATH = os.path.join(BASE_DIR, "data", "products.json")
TAGS_PATH = os.path.join(BASE_DIR, "data", "tags.json")


# ── Helper ────────────────────────────────────────────────
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Routes ────────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({
        "app": "Product Tagging System",
        "version": "1.0.0",
        "endpoints": {
            "GET /api/tag?level=1": "Tag all products (1=Basic, 2=TF-IDF)",
            "GET /api/products": "List all products",
            "GET /api/tags": "List all master tags",
            "POST /api/tag/single": "Tag a single product"
        }
    })


@app.route("/api/products", methods=["GET"])
def get_products():
    products = load_json(PRODUCTS_PATH)
    return jsonify({"count": len(products), "products": products})


@app.route("/api/tags", methods=["GET"])
def get_tags():
    tags = load_json(TAGS_PATH)
    return jsonify({"count": len(tags), "tags": tags})


@app.route("/api/tag", methods=["GET"])
def tag_all():
    level = request.args.get("level", "1")

    if level == "1":
        from src.basic_matcher import run_basic_matcher
        results = run_basic_matcher(PRODUCTS_PATH, TAGS_PATH)
    elif level == "2":
        from src.ai_matcher import run_tfidf_matcher
        results = run_tfidf_matcher(PRODUCTS_PATH, TAGS_PATH)
    else:
        return jsonify({"error": "Invalid level. Use 1 or 2."}), 400

    return jsonify({"level": int(level), "results": results})


@app.route("/api/tag/single", methods=["POST"])
def tag_single():
    """
    Tag a single product passed in the request body.
    Body: { "name": "...", "description": "...", "level": 1 }
    """
    data = request.get_json()
    if not data or "name" not in data:
        return jsonify({"error": "Request body must include 'name'."}), 400

    tags = load_json(TAGS_PATH)
    level = data.get("level", 1)
    product = {"id": 0, "name": data["name"], "description": data.get("description", "")}

    if level == 1:
        from src.basic_matcher import match_tags
        matched = match_tags(product, tags)
        method = "basic_string_match"
    elif level == 2:
        # For a single product use basic TF-IDF inline
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np

        text = product["name"] + " " + product["description"]
        vectorizer = TfidfVectorizer()
        matrix = vectorizer.fit_transform([text] + tags)
        sims = cosine_similarity(matrix[0], matrix[1:]).flatten()
        top_indices = np.argsort(sims)[::-1][:3]
        matched = [tags[i] for i in top_indices if sims[i] > 0]
        method = "tfidf_cosine_similarity"
    else:
        return jsonify({"error": "Invalid level. Use 1 or 2."}), 400

    return jsonify({
        "name": product["name"],
        "matched_tags": matched,
        "method": method
    })


if __name__ == "__main__":
    app.run(debug=True)
