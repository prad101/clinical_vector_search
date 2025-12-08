import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer
import faiss
import tenseal as ts
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(CURRENT_DIR)
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

from pipeline.utils import normalize_rows, norm_vec

# Force FAISS single-threaded for mac stability
try:
    faiss.omp_set_num_threads(1)
except:
    pass

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "dataset", "medical_transcriptions.csv")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

QUERIES = [
    "post-operative knee arthroscopy pain management",
    "chest pain with ECG changes",
    "abdominal pain after appendectomy",
    "shortness of breath with asthma history",
    "lumbar spine MRI findings",
    "arthroscopic shoulder repair recovery",
    "ECG abnormalities in heart attack",
    "knee joint effusion",
    "diabetic foot ulcer treatment",
    "post-surgical infection management"
]

TOP_K = 10
SAFE_MODE = False


# ============================================================
# Utility Functions
# ============================================================

def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, (time.time() - t0) * 1000.0


def ndcg_at_k(ranks, k):
    ranks = np.array(ranks)[:k]
    gains = 1.0 / np.log2(np.arange(2, len(ranks) + 2))
    return np.sum(gains * ranks) / np.sum(gains)


def index_agreement(ref, other):
    matches = sum(1 for a, b in zip(ref, other) if a == b)
    return matches / len(ref)


# ============================================================
# NEW METRICS
# ============================================================

def rank_movement(ref_ids, method_ids):
    rank_map = {doc_id: i for i, doc_id in enumerate(ref_ids)}
    diffs = []

    for method_rank, doc in enumerate(method_ids):
        if doc in rank_map:
            diffs.append(abs(method_rank - rank_map[doc]))
        else:
            diffs.append(len(ref_ids))

    return float(np.mean(diffs))


def diversity_score(vecs):
    if len(vecs) <= 1:
        return 0.0
    sims = cosine_similarity(vecs)
    cohesion = np.mean(sims)
    return float(1.0 - cohesion)


def geometry_distortion(base_vecs, dp_index, sample_size=300):
    n = min(len(base_vecs), sample_size)
    idx = np.random.choice(len(base_vecs), n, replace=False)

    diffs = []
    for i in range(n - 1):
        base_i = base_vecs[idx[i]]
        dp_i = dp_index.reconstruct(idx[i])

        for j in range(i + 1, n):
            base_j = base_vecs[idx[j]]
            dp_j = dp_index.reconstruct(idx[j])

            cos_base = float(np.dot(base_i, base_j))
            cos_dp = float(np.dot(dp_i, dp_j))
            diffs.append(abs(cos_base - cos_dp))

    return float(np.mean(diffs))


def query_robustness(query_text, model, qv_orig, embeddings, index, k=10):
    # Drop one random word
    words = query_text.split()
    if len(words) > 4:
        drop_idx = np.random.randint(0, len(words))
        perturbed = words[:drop_idx] + words[drop_idx+1:]
        perturbed_query = " ".join(perturbed)
    else:
        perturbed_query = query_text

    qv_pert = model.encode([perturbed_query])
    qv_pert = normalize_rows(qv_pert.astype(np.float32))

    _, I_orig = index.search(qv_orig, k)
    _, I_pert = index.search(qv_pert, k)

    ids_orig = set(I_orig[0])
    ids_pert = set(I_pert[0])

    return float(len(ids_orig.intersection(ids_pert)) / k)


# ============================================================
# Build embeddings and indexes
# ============================================================

def build_all(data_path, model_name):
    raw = pd.read_csv(data_path)
    raw.columns = [c.lower() for c in raw.columns]

    col_trans = "transcription"
    col_specialty = "medical_specialty"

    pdf = pd.DataFrame({
        "name": raw["name"],
        "gender": raw["gender"],
        "age": raw["age"],
        "city": raw["city"],
        "medical_specialty": raw[col_specialty],
        "transcription": raw[col_trans],
    })

    pdf["text"] = pdf.apply(
        lambda x: f"{x['medical_specialty']}, {x['transcription']}"
        if pd.notnull(x["medical_specialty"]) else x["transcription"],
        axis=1
    )

    pdf = pdf.drop_duplicates(subset=["text"]).reset_index(drop=True)

    model = SentenceTransformer(model_name)
    emb = model.encode(pdf["text"].tolist(), batch_size=32, show_progress_bar=True)
    emb = normalize_rows(np.array(emb, dtype=np.float32))
    pdf["vec"] = list(emb)

    d = emb.shape[1]
    base_index = faiss.IndexFlatIP(d)
    base_index.add(emb)

    # DP vectors
    attr_texts = [f"{n} {g} {a} {c}" for n, g, a, c in zip(pdf["name"], pdf["gender"], pdf["age"], pdf["city"])]
    attr_emb = normalize_rows(model.encode(attr_texts).astype(np.float32))
    noisy = normalize_rows(attr_emb + np.random.normal(0, 0.15, attr_emb.shape).astype(np.float32))

    dp_vecs = normalize_rows(np.hstack([emb * 0.7, noisy * 0.3]).astype(np.float32))
    dp_index = faiss.IndexFlatIP(dp_vecs.shape[1])
    dp_index.add(dp_vecs)

    # RAG
    tokenized = [txt.lower().split() for txt in pdf["text"]]
    bm25 = BM25Okapi(tokenized)

    rag_index = faiss.IndexFlatIP(d)
    rag_index.add(emb)

    return pdf, emb, base_index, dp_index, bm25, rag_index, model


# ============================================================
# Evaluate all
# ============================================================

def evaluate_all(pdf, emb, base_index, dp_index, bm25, rag_index, model):
    results = []

    for query in QUERIES:
        print(f"\nEvaluating: {query}")

        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        # BASELINE
        (_, lat_base) = timed(base_index.search, qv, TOP_K)
        _, I_base = base_index.search(qv, TOP_K)
        base_ids = I_base[0]

        # DP
        dp_dim = dp_index.d
        text_dim = qv.shape[1]
        if dp_dim > text_dim:
            qv_dp = normalize_rows(
                np.hstack([qv * 0.7, np.zeros((1, dp_dim - text_dim))])
            )
        else:
            qv_dp = qv

        (_, lat_dp) = timed(dp_index.search, qv_dp, TOP_K)
        _, I_dp = dp_index.search(qv_dp, TOP_K)
        dp_ids = I_dp[0]

        dp_drift = float(cosine_similarity(qv, qv_dp)[0][0])

        # FHE synthetic
        d_target = 64
        R = np.random.normal(0, 1 / np.sqrt(qv.shape[1]), size=(qv.shape[1], d_target))
        qv_small = normalize_rows(qv @ R)[0]
        qv_small = norm_vec(qv_small.astype(np.float32))

        small_vecs = normalize_rows(np.random.randn(10, d_target).astype(np.float32))

        ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=8192,
                         coeff_mod_bit_sizes=[60, 40, 40, 60])
        ctx.global_scale = 2**40
        ctx.generate_galois_keys()

        enc_q = ts.ckks_vector(ctx, qv_small.tolist())
        fhe_scores = []
        t0 = time.time()
        for v in small_vecs:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0
        fhe_ids = np.argsort(fhe_scores)[::-1][:TOP_K]

        # RAG
        bm_ids = bm25.get_top_n(query.split(), list(range(len(pdf))), TOP_K * 4)
        (_, lat_rag) = timed(rag_index.search, qv, TOP_K * 4)
        _, I_rag = rag_index.search(qv, TOP_K * 4)
        vec_ids = I_rag[0].tolist()

        cand = list(dict.fromkeys(bm_ids + vec_ids))
        cand_vecs = emb[cand]
        sims = cand_vecs @ qv.ravel()

        selected = []
        remain = list(range(len(cand)))

        while len(selected) < TOP_K:
            if len(selected) == 0:
                best = int(np.argmax(sims))
            else:
                selected_vecs = cand_vecs[selected]
                diversity = cosine_similarity(cand_vecs[remain], selected_vecs).max(axis=1)
                score = 0.7 * sims[remain] - 0.3 * diversity
                best = remain[int(np.argmax(score))]
            selected.append(best)
            remain.remove(best)

        rag_ids = [cand[i] for i in selected]

        # NDCG & agreement
        def rank_positions(ref, pred):
            m = {doc: i+1 for i, doc in enumerate(ref)}
            return [m.get(x, 0) for x in pred]

        ranks_dp = rank_positions(base_ids, dp_ids)
        ranks_rag = rank_positions(base_ids, rag_ids)

        ndcg_dp = ndcg_at_k(ranks_dp, TOP_K)
        ndcg_rag = ndcg_at_k(ranks_rag, TOP_K)

        # NEW METRICS
        rms_dp = rank_movement(base_ids, dp_ids)
        rms_rag = rank_movement(base_ids, rag_ids)
        rms_fhe = rank_movement(base_ids, fhe_ids)

        div_base = diversity_score(emb[base_ids])
        div_dp = diversity_score(emb[dp_ids])
        div_rag = diversity_score(emb[rag_ids])

        geom_dp = geometry_distortion(emb, dp_index)

        robustness = query_robustness(query, model, qv, emb, base_index)

        results.append({
            "query": query,
            "baseline_latency": lat_base,
            "dp_latency": lat_dp,
            "fhe_latency": lat_fhe,
            "rag_latency": lat_rag,
            "ndcg_dp": ndcg_dp,
            "ndcg_rag": ndcg_rag,
            "dp_drift": dp_drift,
            "agreement_dp": index_agreement(base_ids, dp_ids),
            "agreement_rag": index_agreement(base_ids, rag_ids),

            # NEW METRICS
            "rms_dp": rms_dp,
            "rms_rag": rms_rag,
            "rms_fhe": rms_fhe,
            "diversity_base": div_base,
            "diversity_dp": div_dp,
            "diversity_rag": div_rag,
            "geometry_dp": geom_dp,
            "query_robustness": robustness,

            "rag_improvement": ndcg_rag - ndcg_dp
        })

    return pd.DataFrame(results)


# ============================================================
# Plots
# ============================================================

def plot_results(df):

    # Rank Movement
    plt.figure(figsize=(8,5))
    plt.plot(df["rms_dp"], marker="o", label="DP")
    plt.plot(df["rms_rag"], marker="o", label="RAG")
    plt.title("Rank Movement Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rank_movement.png"))
    plt.close()

    # Diversity
    plt.figure(figsize=(8,5))
    plt.plot(df["diversity_base"], label="Baseline")
    plt.plot(df["diversity_dp"], label="DP")
    plt.plot(df["diversity_rag"], label="RAG")
    plt.title("Diversity Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "diversity.png"))
    plt.close()

    # Geometry Distortion
    plt.figure(figsize=(8,5))
    plt.plot(df["geometry_dp"], marker="o")
    plt.title("DP Geometry Distortion")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "geometry_distortion.png"))
    plt.close()

    # Query Robustness
    plt.figure(figsize=(8,5))
    plt.plot(df["query_robustness"], marker="o")
    plt.title("Query Robustness Score")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "query_robustness.png"))
    plt.close()


# ============================================================
# Main
# ============================================================

def main():
    print("Building environment...")

    pdf, emb, base_index, dp_index, bm25, rag_index, model = build_all(
        DATA_PATH,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    df = evaluate_all(
        pdf,
        emb,
        base_index,
        dp_index,
        bm25,
        rag_index,
        model
    )

    out = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    df.to_csv(out, index=False)
    print(f"Saved metrics to {out}")

    plot_results(df)
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
