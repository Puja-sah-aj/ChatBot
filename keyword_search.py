import json
import re
from typing import List, Dict
from collections import Counter


class KeywordSearch:
    def __init__(self, file_path="chunks.json"):
        print("🔍 Loading keyword search data...")

        with open(file_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        # Preprocess texts
        self.processed_data = []
        for item in self.data:
            tokens = self.tokenize(item["text"])
            self.processed_data.append({
                "url": item["url"],
                "text": item["text"],
                "tokens": tokens
            })

        print(f"✅ Loaded {len(self.processed_data)} documents for keyword search")

    # 🔥 Simple tokenizer
    def tokenize(self, text: str):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    # 🔥 Score based on keyword frequency
    def score(self, query_tokens, doc_tokens):
        doc_counter = Counter(doc_tokens)

        score = 0
        for token in query_tokens:
            score += doc_counter.get(token, 0)

        return score

    # 🔥 Main search function
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        query_tokens = self.tokenize(query)

        scored_results = []

        for item in self.processed_data:
            score = self.score(query_tokens, item["tokens"])

            if score > 0:
                scored_results.append({
                    "url": item["url"],
                    "text": item["text"],
                    "score": score
                })

        # Sort by score descending
        scored_results.sort(key=lambda x: x["score"], reverse=True)

        return scored_results[:top_k]


# 🔥 TEST
if __name__ == "__main__":
    ks = KeywordSearch()

    while True:
        q = input("\nSearch: ")
        if q == "exit":
            break

        results = ks.search(q)

        print("\nTop Results:\n")
        for r in results:
            print("Score:", r["score"])
            print("URL:", r["url"])
            print("Text:", r["text"][:200])
            print("-" * 50)