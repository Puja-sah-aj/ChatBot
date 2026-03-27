# vector_store.py
import json
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Dict


class VectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("Loading embedding model for query...")
        torch.set_num_threads(1)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

        self.index = None
        self.texts: List[str] = []
        self.urls: List[str] = []

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def normalize_l2(self, embeddings: np.ndarray) -> np.ndarray:
        """L2 normalize for cosine similarity via FAISS IndexFlatIP"""
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / np.maximum(norms, 1e-9)

    def embed_texts(self, texts: List[str], batch_size=32) -> np.ndarray:
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            encoded = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt"
            )
            with torch.no_grad():
                output = self.model(**encoded)
            emb = self.mean_pooling(output, encoded["attention_mask"])
            all_embeddings.append(emb.cpu().numpy())
        final = np.concatenate(all_embeddings, axis=0).astype("float32")
        return self.normalize_l2(final)   # ✅ normalize

    def embed_query(self, text: str) -> np.ndarray:
        emb = self.embed_texts([text])
        return emb[0]

    def load_embeddings(self, file_path="embeddings.json"):
        print("Loading embeddings...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        embeddings = []
        for item in data:
            embeddings.append(item["embedding"])
            self.texts.append(item["text"])
            self.urls.append(item["url"])

        embeddings = np.array(embeddings).astype("float32")
        # ✅ Re-normalize on load (safety)
        embeddings = self.normalize_l2(embeddings)

        dimension = embeddings.shape[1]
        # ✅ FIXED: Use IndexFlatIP for cosine similarity (not L2!)
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print(f"✅ FAISS index (cosine) with {len(self.texts)} vectors")

    def query(self, user_query: str, top_k: int = 5) -> List[Dict]:
        if self.index is None or not self.texts:
            return []

        query_vector = self.embed_query(user_query)
        query_vector = np.expand_dims(query_vector, axis=0)

        scores, indices = self.index.search(query_vector, top_k)
        results = []
        for i, idx in enumerate(indices[0]):
            if 0 <= idx < len(self.texts):
                results.append({
                    "text": self.texts[idx],
                    "url": self.urls[idx],
                    "score": float(scores[0][i])
                })
        return results

    def create_embeddings_from_chunks(self, chunks: List[Dict], output_file="embeddings.json"):
        texts = [c["text"] for c in chunks]
        urls = [c["url"] for c in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        all_emb = self.embed_texts(texts, batch_size=32)

        data_to_save = [
            {"text": texts[i], "url": urls[i], "embedding": all_emb[i].tolist()}
            for i in range(len(texts))
        ]

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f)
        print(f"✅ Saved {len(data_to_save)} embeddings to {output_file}")
