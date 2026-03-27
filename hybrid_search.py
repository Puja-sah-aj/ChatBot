import numpy as np
from typing import List, Dict


class HybridSearch:
    def __init__(self, vector_store, keyword_search):
        self.vector_store = vector_store
        self.keyword_search = keyword_search

    def normalize_scores(self, scores):
        """Normalize scores between 0 and 1"""
        if not scores:
            return scores

        min_s = min(scores)
        max_s = max(scores)

        if max_s == min_s:
            return [1.0 for _ in scores]

        return [(s - min_s) / (max_s - min_s) for s in scores]

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        # 🔹 Step 1: Vector search
        vector_results = self.vector_store.query(query, top_k=top_k * 2)

        # 🔹 Step 2: Keyword search
        keyword_results = self.keyword_search.search(query, top_k=top_k * 2)

        # 🔹 Step 3: Assign scores
        combined = {}

        # Vector scoring (higher rank = higher score)
        for rank, item in enumerate(vector_results):
            key = item["text"]

            score = 1 / (rank + 1)  # simple rank-based score

            if key not in combined:
                combined[key] = {
                    "url": item["url"],
                    "text": item["text"],
                    "vector_score": score,
                    "keyword_score": 0
                }
            else:
                combined[key]["vector_score"] = score

        # Keyword scoring
        for item in keyword_results:
            key = item["text"]

            score = item.get("score", 1)

            if key not in combined:
                combined[key] = {
                    "url": item["url"],
                    "text": item["text"],
                    "vector_score": 0,
                    "keyword_score": score
                }
            else:
                combined[key]["keyword_score"] = score

        # 🔹 Step 4: Normalize scores
        vector_scores = [v["vector_score"] for v in combined.values()]
        keyword_scores = [v["keyword_score"] for v in combined.values()]

        norm_vector = self.normalize_scores(vector_scores)
        norm_keyword = self.normalize_scores(keyword_scores)

        # 🔹 Step 5: Final scoring
        results = []
        for i, key in enumerate(combined.keys()):
            item = combined[key]

            final_score = (0.7 * norm_vector[i]) + (0.3 * norm_keyword[i])

            results.append({
                "url": item["url"],
                "text": item["text"],
                "score": final_score
            })

        # 🔹 Step 6: Sort & return
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]