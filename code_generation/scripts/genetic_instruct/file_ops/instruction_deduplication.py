import os
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm
from datasets import load_dataset
import datasets

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from src.shared.llm import LLMClient  # noqa: E402

# Parameters

# Optional: set a custom Hugging Face datasets cache directory.
# If not set, the default is used (usually ~/.cache/huggingface/datasets).
datasets.config.HF_DATASETS_CACHE = "PATH_TO_CUSTOM_CACHE"

OUTPUT_FILE = "final_deduplicated_samples_MiniLM.parquet"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.90
FAISS_K = 50
BATCH_SIZE = 128
USE_LLM_CONFIRMATION = False

class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x, y):
        self.parent[self.find(y)] = self.find(x)

    def get_unique_roots(self):
        return set(self.find(x) for x in range(len(self.parent)))

def main():
    # Load dataset
    dataset = load_dataset("amal-abed/test2",  split="train")
    df = dataset.to_pandas()
    instructions = df["instruction"].tolist()
    print(f"📦 Loaded {len(instructions)} samples.")

    # Generate sentence embeddings
    print("🔍 Generating embeddings...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(instructions, show_progress_bar=True, convert_to_numpy=True)

    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings)

    # FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Optional LLM init
    if USE_LLM_CONFIRMATION:
        llm = LLMClient(
            base_url="http://localhost:8002/v1",
            api_key="no-key-needed",
            model="gemma-3-27b-it",
        )

    # Initialize union-find
    uf = UnionFind(len(embeddings))

    # Batched FAISS search
    print("🔍 Searching for duplicate candidates using FAISS...")
    I = np.zeros((len(embeddings), FAISS_K), dtype=np.int64)
    for start in tqdm(range(0, len(embeddings), BATCH_SIZE), desc="🔄 FAISS Batched Search"):
        end = min(start + BATCH_SIZE, len(embeddings))
        _, neighbors = index.search(embeddings[start:end], k=FAISS_K)
        I[start:end] = neighbors

    batched_prompts = []
    batched_pairs = []

    def flush_llm_batch():
        nonlocal batched_prompts, batched_pairs
        responses = llm.batched_inference(batched_prompts)
        for idx, resp_text in enumerate(responses):
            resp_text = resp_text.strip().lower()
            if resp_text == "true":
                i, j = batched_pairs[idx]
                uf.union(i, j)
        batched_prompts.clear()
        batched_pairs.clear()

    # Process neighbors and union duplicates
    for i, neighbors in tqdm(enumerate(I), total=len(I), desc="🔗 Clustering Similar Items"):
        for j in neighbors[1:]:  # Skip self
            if uf.find(i) == uf.find(j):
                continue
            sim = np.dot(embeddings[i], embeddings[j])
            if sim >= SIMILARITY_THRESHOLD:
                if USE_LLM_CONFIRMATION:
                    prompt = f"""Help me determine if the following two coding problems are the same.
First problem: {instructions[i]}
Second problem: {instructions[j]}
Disregard names and minor changes in word order. If the two problems are functionally the same, respond ONLY with "True" or "False"."""
                    batched_prompts.append(prompt)
                    batched_pairs.append((i, j))
                    if len(batched_prompts) >= BATCH_SIZE:
                        flush_llm_batch()
                else:
                    uf.union(i, j)

    if USE_LLM_CONFIRMATION and batched_prompts:
        flush_llm_batch()

    # Save deduplicated dataset
    unique_roots = uf.get_unique_roots()
    dedup_df = df.iloc[list(unique_roots)].copy()
    dedup_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"✅ Deduplicated {len(instructions)} → {len(dedup_df)} samples into {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

