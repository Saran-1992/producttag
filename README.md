# 🏷️ Product Tagging System

Automatic product tag assignment using three progressive matching levels — from simple string matching to AI-powered semantic similarity.

---

## 📁 Project Structure

```
ProductTaggingSystem/
├── data/
│   ├── products.json       # Input product list
│   └── tags.json           # Master tags list
├── src/
│   ├── __init__.py
│   ├── basic_matcher.py    # Level 1: String matching
│   └── ai_matcher.py       # Level 2 & 3: TF-IDF / Transformer
├── requirements.txt
├── main.py
└── README.md
```

---

## ⚙️ Setup

```bash
pip install -r requirements.txt
```

For **Level 3** (Semantic Transformer), install the optional dependency:

```bash
pip install sentence-transformers
```

---

## 🚀 How to Run

### Level 1 – Basic String Matching
```bash
python main.py --level 1
```

### Level 2 – TF-IDF + Cosine Similarity
```bash
python main.py --level 2
```

### Level 3 – Semantic Transformer
```bash
python main.py --level 3

---

## 🗂️ Data Files

### `data/products.json`
Each product should have:
```json
{ "id": 1, "name": "Product Name", "description": "Product description text" }
```

### `data/tags.json`
A flat list of tag strings:
```json
["electronics", "smartphone", "footwear", ...]
```

---

## 📦 Dependencies

| Library | Purpose |
|---|---|
| `pandas` | Data manipulation |
| `scikit-learn` | TF-IDF vectorization, cosine similarity |
| `numpy` | Array operations |
| `sentence-transformers` *(optional)* | Level 3 semantic matching |
