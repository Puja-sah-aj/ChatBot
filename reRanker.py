import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


class Reranker:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("🔁 Loading reranker model...")
        torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def normalize(self, emb):
        return torch.nn.functional.normalize(emb, p=2, dim=1)

    def embed(self, texts):
        encoded_input = self.tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])
        embeddings = self.normalize(embeddings)

        return embeddings.cpu().numpy()

    def filter_chunks(self, results):
        filtered = []

        for r in results:
            text = r["text"].lower()

            # remove only real junk
            if any(x in text for x in ["login", "signup", "menu"]):
                continue

            if len(text.strip()) < 40:
                continue

            filtered.append(r)

        return filtered

    def rerank(self, query, results, top_k=3):
        if not results:
            return []

        results = self.filter_chunks(results)
        if not results:
            return []

        texts = [r["text"] for r in results]

        query_emb = self.embed([query])[0]
        doc_embs = self.embed(texts)

        scores = np.dot(doc_embs, query_emb)

        reranked = []
        for i, score in enumerate(scores):
            reranked.append({
                "url": results[i]["url"],
                "text": results[i]["text"],
                "score": float(score)
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)

        return reranked[:top_k]