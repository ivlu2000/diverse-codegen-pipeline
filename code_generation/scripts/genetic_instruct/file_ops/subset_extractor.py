from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import gzip

DATASET = "amal-abed/genetic_instruct_data"
TEXT_COL = "instruction"

SAMPLE_N = 550_000
RNG_SEED = 123

TARGET_ROWS = 10_000
SIM_THRESH_START = 0.70
SIM_THRESH_MIN = 0.60
SIM_THRESH_STEP = 0.05

BATCH = 1024

M = 64
EF_CONSTRUCT = 400
EF_SEARCH = 256
ADD_CHUNK = 100_000

MAX_NEIGHBORS_PER_NODE = 200_000
RANGE_CHUNK = 20_000         

FULL_VERIFY = False
VERIFY_SAMPLE_PAIRS = 50_000

OUTPUT_ROWS = f"similar_rows_{TEXT_COL}_10k_from_550k_mydata.parquet"
OUTPUT_META = "mutual_similarity_meta_550k_to_10k_mydata.json.gz"

dataset = load_dataset(DATASET, split="train")
texts0 = [t.strip() if t is not None else "" for t in dataset[TEXT_COL]]
mask = [len(t) >= 8 for t in texts0]
dataset = dataset.select([i for i, m in enumerate(mask) if m])

total = len(dataset)
sample_n = min(SAMPLE_N, total)
if sample_n < total:
    dataset = dataset.shuffle(seed=RNG_SEED).select(range(sample_n))
texts = dataset[TEXT_COL]
print(f"Loaded {len(texts)} rows after filtering+sampling; columns: {dataset.column_names}")

model = SentenceTransformer("all-mpnet-base-v2")
emb = model.encode(
    texts, batch_size=BATCH, convert_to_numpy=True, show_progress_bar=True, num_workers=4
).astype("float32")
faiss.normalize_L2(emb)
n, d = emb.shape
print(f"Embeddings: n={n}, d={d}")

def build_flat_ip_index(x, chunk=ADD_CHUNK):
    flat = faiss.IndexFlatIP(x.shape[1])
    for start in tqdm(range(0, x.shape[0], chunk), desc="Building FlatIP (chunks)"):
        end = min(start + chunk, x.shape[0])
        flat.add(x[start:end])
    return flat

query_index = build_flat_ip_index(emb)

def range_search_twoarg(index_obj, xq, radius):
    """
    Stable 2-arg FAISS API:
      lims, D, I = index.range_search(xq, radius)
    Returns (lims, labels) as numpy arrays.
    """
    lims, D, I = index_obj.range_search(xq, radius)
    lims = np.asarray(lims, dtype=np.int64)
    labels = np.asarray(I, dtype=np.int64)
    return lims, labels

def build_neighbor_sets(radius: float, chunk_size: int = RANGE_CHUNK):
    """
    For each vector i, collect all j with cosine >= radius (excluding i).
    Uses FlatIP range_search in chunks; returns list[set[int]] of length n.
    """
    print(f"\nBuilding neighbor sets with cosine ≥ {radius:.4f} (FlatIP range_search, chunked)...")
    neighbor_sets = [set() for _ in range(n)]

    for q_start in tqdm(range(0, n, chunk_size), desc="Range search (chunks)"):
        q_end = min(q_start + chunk_size, n)
        xq = emb[q_start:q_end]

        lims, labels = range_search_twoarg(query_index, xq, radius)

        for local_q in range(q_end - q_start):
            global_q = q_start + local_q
            start, end = lims[local_q], lims[local_q + 1]
            neigh = labels[start:end]

            if neigh.size:
                mask = (neigh != global_q) & (neigh >= 0)
                neigh = neigh[mask]

                if neigh.size > MAX_NEIGHBORS_PER_NODE:
                    sims = emb[neigh] @ emb[global_q]
                    keep_idx = np.argsort(-sims)[:MAX_NEIGHBORS_PER_NODE]
                    neigh = neigh[keep_idx]

                neighbor_sets[global_q] = set(map(int, neigh.tolist()))
            else:
                neighbor_sets[global_q] = set()

    return neighbor_sets

def find_clique_of_size(neighbor_sets, size: int):
    degrees = np.array([len(s) for s in neighbor_sets], dtype=np.int32)
    seed_order = np.argsort(-degrees)

    for seed in tqdm(seed_order, desc="Clique seeding (by degree)"):
        if degrees[seed] + 1 < size:
            break
        clique = [int(seed)]
        candidates = set(neighbor_sets[seed])
        cand_sorted = sorted(candidates, key=lambda x: len(neighbor_sets[x]), reverse=True)

        for c in cand_sorted:
            if len(clique) == size:
                return clique
            if len(candidates) < (size - len(clique)):
                break
            if c not in candidates:
                continue
            clique.append(c)
            candidates.intersection_update(neighbor_sets[c])

        if len(clique) >= size:
            return clique
    return None

def verify_all_pairwise(indices, thresh: float, full: bool = FULL_VERIFY, sample_pairs: int = VERIFY_SAMPLE_PAIRS) -> bool:
    if indices is None or len(indices) == 0:
        return False
    k = len(indices)
    X = emb[indices]

    if not full:
        rng = np.random.default_rng(RNG_SEED)
        max_pairs = k * (k - 1) // 2
        m = min(sample_pairs, max_pairs)
        print(f"Verifying by sampling {m:,} random pairs (k={k:,}).")
        for _ in tqdm(range(m), desc="Pairwise sample verify"):
            i = rng.integers(0, k)
            j = rng.integers(0, k - 1)
            if j >= i:
                j += 1
            if float(np.dot(X[i], X[j])) < (thresh - 1e-6):
                return False
        return True

    print("Full verification enabled — may be extremely slow / memory heavy.")
    bs = 2048
    total = (k + bs - 1) // bs
    for i in tqdm(range(0, k, bs), total=total, desc="Verifying pairwise cosines"):
        Xi = X[i:i+bs]
        S = Xi @ X.T
        if np.any(S < (thresh - 1e-6)):
            return False
    return True

selected_indices = None
final_thresh = None

tau = SIM_THRESH_START
while tau >= SIM_THRESH_MIN:
    nbr_sets = build_neighbor_sets(tau, chunk_size=RANGE_CHUNK)

    degs = np.array([len(s) for s in nbr_sets])
    tqdm.write(
        f"τ={tau:.4f} | degree: mean={degs.mean():.1f}, median={np.median(degs):.1f}, "
        f"p90={np.percentile(degs,90):.1f}, max={degs.max():d}"
    )

    clique = find_clique_of_size(nbr_sets, TARGET_ROWS)
    if clique is not None:
        ok = verify_all_pairwise(clique, tau, full=FULL_VERIFY, sample_pairs=VERIFY_SAMPLE_PAIRS)
        if ok:
            selected_indices = clique
            final_thresh = tau
            print(f"\nSUCCESS: Found a {len(clique)}-row mutually-similar set at cosine ≥ {tau:.4f}.")
            break
        else:
            print(f"Verification failed at τ={tau:.4f}; relaxing threshold.")
    else:
        print(f"No clique of size {TARGET_ROWS} at τ={tau:.4f}.")
    tau -= SIM_THRESH_STEP

if selected_indices is None:
    raise SystemExit(
        f"Could not find a {TARGET_ROWS}-row pairwise-similar set at cosine ≥ {SIM_THRESH_MIN:.2f}. "
        f"Try lowering SIM_THRESH_MIN, using a different embedding model, or reducing TARGET_ROWS."
    )


selected_indices = sorted(map(int, selected_indices))
subset = dataset.select(selected_indices)
df_rows = pd.DataFrame({col: subset[col] for col in dataset.column_names})

print("Writing output parquet...")
df_rows.to_parquet(OUTPUT_ROWS, index=False)
print(f"Saved {len(df_rows)} mutually-similar rows to {OUTPUT_ROWS} (pairwise cosine ≥ {final_thresh:.4f}).")

meta = {
    "source_dataset": DATASET,
    "text_col": TEXT_COL,
    "sample_n": int(sample_n),
    "target_rows": TARGET_ROWS,
    "final_threshold": float(final_thresh),
    "sim_thresh_start": SIM_THRESH_START,
    "sim_thresh_min": SIM_THRESH_MIN,
    "sim_thresh_step": SIM_THRESH_STEP,
    "query_index": "FlatIP",
    "range_chunk": RANGE_CHUNK,
    "max_neighbors_per_node": MAX_NEIGHBORS_PER_NODE,
}
with gzip.open(OUTPUT_META, "wt", encoding="utf-8") as f:
    json.dump(meta, f)
print(f"Wrote meta to {OUTPUT_META}")





