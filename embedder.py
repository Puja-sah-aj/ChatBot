import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel


class Embedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        print("🚀 Loading embedding model...")

        # ✅ Fix macOS crash (VERY IMPORTANT)
        torch.set_num_threads(1)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Disable gradients (faster)
        self.model.eval()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    def normalize(self, embeddings):
        """🔥 Normalize embeddings (VERY IMPORTANT for accuracy)"""
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def generate_embeddings_batch(self, texts):
        """🔥 Batch processing (FASTER + STABLE)"""

        encoded_input = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )

        with torch.no_grad():
            model_output = self.model(**encoded_input)

        embeddings = self.mean_pooling(model_output, encoded_input["attention_mask"])

        # ✅ Normalize embeddings
        embeddings = self.normalize(embeddings)

        return embeddings.cpu().numpy()

    def process_chunks(self, chunks, batch_size=32):
        embedded_data = []

        texts = [chunk["text"] for chunk in chunks]

        print(f"🔄 Processing {len(texts)} chunks in batches...")

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_chunks = chunks[i:i + batch_size]

            print(f"Processing batch {i//batch_size + 1}")

            embeddings = self.generate_embeddings_batch(batch_texts)

            for j, emb in enumerate(embeddings):
                embedded_data.append({
                    "url": batch_chunks[j]["url"],
                    "text": batch_chunks[j]["text"],
                    "embedding": emb.tolist()
                })

        return embedded_data


if __name__ == "__main__":
    # Load chunks
    with open("chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    embedder = Embedder()

    embedded_chunks = embedder.process_chunks(chunks, batch_size=32)

    print(f"\n✅ Total Embedded Chunks: {len(embedded_chunks)}")

    # Save embeddings
    with open("embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embedded_chunks, f, indent=2)

    # Preview
    for item in embedded_chunks[:2]:
        print("\nURL:", item["url"])
        print("Text:", item["text"][:100])
        print("Embedding Length:", len(item["embedding"]))