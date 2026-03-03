import argparse
import json
import os

PRODUCTS_PATH = os.path.join("data", "products.json")
TAGS_PATH = os.path.join("data", "tags.json")


def print_results(results: list):
    print("\n" + "=" * 55)
    print(f"{'Product Tagging Results':^55}")
    print("=" * 55)
    for r in results:
        print(f"\n[{r['id']}] {r['name']}")
        print(f"  Method : {r['method']}")
        print(f"  Tags   : {', '.join(r['matched_tags']) if r['matched_tags'] else 'No tags found'}")
    print("\n" + "=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Product Tagging System")
    parser.add_argument(
        "--level",
        type=int,
        choices=[1, 2],
        default=1,
        help="Matching level: 1=Basic String Match, 2=TF-IDF Cosine Similarity"
    )
    args = parser.parse_args()

    if args.level == 1:
        from src.basic_matcher import run_basic_matcher
        results = run_basic_matcher(PRODUCTS_PATH, TAGS_PATH)
    elif args.level == 2:
        from src.ai_matcher import run_tfidf_matcher
        results = run_tfidf_matcher(PRODUCTS_PATH, TAGS_PATH)

    print_results(results)


if __name__ == "__main__":
    main()
