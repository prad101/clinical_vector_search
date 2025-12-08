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

try:
    faiss.mp_set_num_threads(1)
except Exception:
    try:
        faiss.omp_set_num_threads(1)
    except Exception:
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

def timed(fn, *args, **kwargs):
    t0 = time.time()
    out = fn(*args, **kwargs)
    return out, (time.time() - t0) * 1000.0


def ndcg_at_k(ranks, k):
    ranks = np.array(ranks)[:k]
    gains = 1.0 / np.log2(np.arange(2, len(ranks) + 2))
    return float(np.sum(gains * ranks) / np.sum(gains))

def true_ndcg_at_k(pred, gold, k):

    pred = pred[:k]
    gold_set = set(gold[:k])

    # relevance vector
    rel = np.array([1 if pid in gold_set else 0 for pid in pred], dtype=np.float32)

    # DCG
    gains = rel / np.log2(np.arange(2, k + 2))
    dcg = gains.sum()

    # IDCG
    ideal = np.ones(min(k, len(gold_set)), dtype=np.float32)
    ideal_gains = ideal / np.log2(np.arange(2, len(ideal) + 2))
    idcg = ideal_gains.sum()

    return float(dcg / idcg) if idcg > 0 else 0.0


def index_agreement(ref, other):
    """Percentage of exact index matches."""
    ref = list(ref)
    other = list(other)
    matches = sum(1 for a, b in zip(ref, other) if a == b)
    return matches / len(ref)

def rank_movement(ref_ids, method_ids):
    """Mean absolute rank difference comparing method vs baseline."""
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
    cohesion = float(np.mean(sims))
    return float(1.0 - cohesion)


def geometry_distortion(base_vecs, method_vecs, sample_size=300):

    n = min(len(base_vecs), len(method_vecs), sample_size)
    idx = np.random.choice(len(base_vecs), n, replace=False)

    diffs = []
    for i in range(n - 1):
        bi = base_vecs[idx[i]]
        mi = method_vecs[idx[i]]
        for j in range(i + 1, n):
            bj = base_vecs[idx[j]]
            mj = method_vecs[idx[j]]

            cos_base = float(np.dot(bi, bj))
            cos_meth = float(np.dot(mj, mj))  
            diffs.append(abs(cos_base - cos_meth))
    return float(np.mean(diffs))


def query_robustness(search_fn, query_text, k=TOP_K):

    words = query_text.split()
    if len(words) > 4:
        drop_idx = np.random.randint(0, len(words))
        perturbed = words[:drop_idx] + words[drop_idx + 1:]
        pert_query = " ".join(perturbed)
    else:
        pert_query = query_text

    ids_orig = search_fn(query_text)
    ids_pert = search_fn(pert_query)

    if not ids_orig or not ids_pert:
        return 0.0

    overlap = len(set(ids_orig).intersection(set(ids_pert))) / float(k)
    return float(overlap)


def geometry_distortion(base_vecs, method_vecs, sample_size=300):
    n = min(len(base_vecs), len(method_vecs), sample_size)
    idx = np.random.choice(len(base_vecs), n, replace=False)

    diffs = []
    for i in range(n - 1):
        bi = base_vecs[idx[i]]
        mi = method_vecs[idx[i]]
        for j in range(i + 1, n):
            bj = base_vecs[idx[j]]
            mj = method_vecs[idx[j]]
            cos_base = float(np.dot(bi, bj))
            cos_meth = float(np.dot(mi, mj))
            diffs.append(abs(cos_base - cos_meth))
    return float(np.mean(diffs))

def build_all(data_path, model_name):
    print("Loading dataset with pandas (no Spark)...")
    raw = pd.read_csv(data_path)

    raw.columns = [c.lower() for c in raw.columns]

    col_name = "name" if "name" in raw.columns else None
    col_gender = "gender" if "gender" in raw.columns else None
    col_age = "age" if "age" in raw.columns else None
    col_city = "city" if "city" in raw.columns else None
    col_specialty = "medical_specialty" if "medical_specialty" in raw.columns else None
    col_trans = "transcription" if "transcription" in raw.columns else None

    missing_core = [c for c, v in {
        "name": col_name,
        "gender": col_gender,
        "age": col_age,
        "city": col_city,
        "medical_specialty": col_specialty,
        "transcription": col_trans,
    }.items() if v is None]
    if missing_core:
        raise ValueError(f"CSV missing expected MTSamples columns: {missing_core}")

    pdf = pd.DataFrame({
        "name": raw[col_name],
        "gender": raw[col_gender],
        "age": raw[col_age],
        "city": raw[col_city],
        "medical_specialty": raw[col_specialty],
        "transcription": raw[col_trans],
    })

    pdf["text"] = pdf.apply(
        lambda x: f"{x['medical_specialty']}, {x['transcription']}"
        if pd.notnull(x["medical_specialty"])
        else x["transcription"],
        axis=1,
    )
    
    pdf = pdf.drop_duplicates(subset=["text"]).reset_index(drop=True)

    print("Embedding dataset...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        pdf["text"].tolist(),
        batch_size=32,
        show_progress_bar=True,
    )
    embeddings = normalize_rows(np.array(embeddings, dtype=np.float32))
    pdf["vec"] = list(embeddings)

    # Baseline index (FlatIP)
    d = embeddings.shape[1]
    base_index = faiss.IndexFlatIP(d)
    base_index.add(embeddings)

    # Shared attribute embeddings
    attr_texts = [
        f"{n} {g} {a} {c}"
        for n, g, a, c in zip(pdf["name"], pdf["gender"], pdf["age"], pdf["city"])
    ]
    attr_emb = model.encode(attr_texts, batch_size=32, show_progress_bar=False)
    attr_emb = normalize_rows(attr_emb.astype(np.float32))

    sigma = 0.15
    noisy_attr = attr_emb + np.random.normal(0, sigma, attr_emb.shape).astype(np.float32)
    noisy_attr = normalize_rows(noisy_attr)

    # DP (30 percent attribute mix)
    dp_vecs = normalize_rows(
        np.hstack([embeddings * 0.7, noisy_attr * 0.3]).astype(np.float32)
    )
    dp_index = faiss.IndexFlatIP(dp_vecs.shape[1])
    dp_index.add(dp_vecs)

    # Optimal RAG (40 percent mix)
    opt_vecs = normalize_rows(
        np.hstack([embeddings * 0.6, noisy_attr * 0.4]).astype(np.float32)
    )
    opt_index = faiss.IndexFlatIP(opt_vecs.shape[1])
    opt_index.add(opt_vecs)

    # RAG Structures (BM25 + FAISS + MMR)
    tokenized = [t.lower().split() for t in pdf["text"]]
    bm25 = BM25Okapi(tokenized)

    # Use FlatIP instead of HNSW for macOS
    rag_index = faiss.IndexFlatIP(d)
    rag_index.add(embeddings)

    return pdf, embeddings, dp_vecs, opt_vecs, base_index, dp_index, opt_index, bm25, rag_index, model

def evaluate_all(pdf, embeddings, dp_vecs, opt_vecs,
                 base_index, dp_index, opt_index, bm25, rag_index, model):
    results = []

    geom_dp_global = geometry_distortion(embeddings, dp_vecs)
    geom_opt_global = geometry_distortion(embeddings, opt_vecs)

    fhe_sample = min(256, embeddings.shape[0])
    fhe_idx = np.random.choice(embeddings.shape[0], fhe_sample, replace=False)
    d_target_geom = 64
    R_fhe_geom = np.random.normal(
        0, 1 / np.sqrt(embeddings.shape[1]),
        size=(embeddings.shape[1], d_target_geom)
    ).astype(np.float32)
    emb_fhe_geom = normalize_rows((embeddings[fhe_idx] @ R_fhe_geom).astype(np.float32))
    geom_fhe_global = geometry_distortion(embeddings[fhe_idx], emb_fhe_geom)

    for query in QUERIES:
        print(f"\nEvaluating: {query}")

        qv = model.encode([query])
        qv = normalize_rows(qv.astype(np.float32))

        (_, lat_base) = timed(base_index.search, qv, TOP_K)
        _, I_base = base_index.search(qv, TOP_K)
        base_ids = I_base[0]

        text_dim = qv.shape[1]
        dp_dim = dp_index.d
        if dp_dim > text_dim:
            attr_dim = dp_dim - text_dim
            qv_zero_attr = np.zeros((1, attr_dim), dtype=np.float32)
            qv_dp = normalize_rows(np.hstack([qv * 0.7, qv_zero_attr * 0.3]))
        else:
            qv_dp = qv

        (_, lat_dp) = timed(dp_index.search, qv_dp, TOP_K)
        _, I_dp = dp_index.search(qv_dp, TOP_K)
        dp_ids = I_dp[0]
        
        num_docs = embeddings.shape[0]
        fhe_cand_size = min(128, num_docs)
        fhe_cand_ids = list(range(fhe_cand_size))

        d_target = 64
        R = np.random.normal(
            0, 1 / np.sqrt(qv.shape[1]),
            size=(qv.shape[1], d_target)
        ).astype(np.float32)

        emb_fhe = normalize_rows((embeddings[fhe_cand_ids] @ R).astype(np.float32))

        qv_small = qv @ R
        qv_small = normalize_rows(qv_small)[0]
        qv_small = norm_vec(qv_small.astype(np.float32))

        ctx = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 40, 40, 60],
        )
        ctx.global_scale = 2**40
        ctx.generate_galois_keys()

        enc_q = ts.ckks_vector(ctx, qv_small.tolist())

        fhe_scores = []
        t0 = time.time()
        for v in emb_fhe:
            fhe_scores.append(enc_q.dot(v.tolist()).decrypt()[0])
        lat_fhe = (time.time() - t0) * 1000.0

        fhe_order = np.argsort(fhe_scores)[::-1][:TOP_K]
        fhe_ids = [fhe_cand_ids[i] for i in fhe_order]

        bm25_ids = bm25.get_top_n(query.split(), list(range(len(pdf))), TOP_K * 4)

        (_, lat_rag_raw) = timed(rag_index.search, qv, TOP_K * 4)
        _, I_rag = rag_index.search(qv, TOP_K * 4)
        vec_ids = I_rag[0].tolist()

        cand = list(dict.fromkeys(bm25_ids + vec_ids))
        cand_vecs = embeddings[cand]
        sims = cand_vecs @ qv.ravel()

        selected = []
        candidate_ids = list(range(len(cand)))

        while len(selected) < TOP_K:
            if len(selected) == 0:
                best = int(np.argmax(sims))
            else:
                selected_vecs = cand_vecs[selected]
                diversity = cosine_similarity(cand_vecs[candidate_ids], selected_vecs).max(axis=1)
                score = 0.7 * sims[candidate_ids] - 0.3 * diversity
                best_local = int(np.argmax(score))
                best = candidate_ids[best_local]

            selected.append(best)
            candidate_ids.remove(best)

        rag_ids = [cand[i] for i in selected]

        opt_dim = opt_index.d
        if opt_dim > text_dim:
            opt_attr_dim = opt_dim - text_dim
            qv_zero_attr_opt = np.zeros((1, opt_attr_dim), dtype=np.float32)
            qv_opt = normalize_rows(np.hstack([qv * 0.6, qv_zero_attr_opt * 0.4]))
        else:
            qv_opt = qv

        (_, lat_opt_dense) = timed(opt_index.search, qv_opt, TOP_K * 4)
        _, I_opt = opt_index.search(qv_opt, TOP_K * 4)
        opt_vec_ids = I_opt[0].tolist()

        opt_cand = list(dict.fromkeys(bm25_ids + opt_vec_ids))
        opt_cand_vecs = opt_vecs[opt_cand]
        opt_sims = opt_cand_vecs @ qv_opt.ravel()

        opt_selected = []
        opt_candidate_ids = list(range(len(opt_cand)))

        while len(opt_selected) < TOP_K:
            if len(opt_selected) == 0:
                best = int(np.argmax(opt_sims))
            else:
                selected_vecs_opt = opt_cand_vecs[opt_selected]
                diversity_opt = cosine_similarity(opt_cand_vecs[opt_candidate_ids], selected_vecs_opt).max(axis=1)
                score_opt = 0.7 * opt_sims[opt_candidate_ids] - 0.3 * diversity_opt
                best_local = int(np.argmax(score_opt))
                best = opt_candidate_ids[best_local]

            opt_selected.append(best)
            opt_candidate_ids.remove(best)

        opt_ids = [opt_cand[i] for i in opt_selected]
        lat_opt = lat_opt_dense

        def rank_positions(ref, pred):
            mapping = {doc_id: i + 1 for i, doc_id in enumerate(ref)}
            return [mapping.get(x, 0) for x in pred]

        ranks_dp = rank_positions(base_ids, dp_ids)
        ranks_rag = rank_positions(base_ids, rag_ids)
        ranks_opt = rank_positions(base_ids, opt_ids)

        ndcg_dp = true_ndcg_at_k(dp_ids, base_ids, TOP_K)
        ndcg_enhanced = true_ndcg_at_k(rag_ids, base_ids, TOP_K)
        ndcg_optimal = true_ndcg_at_k(optimal_ids, base_ids, TOP_K)

        rag_improvement_enhanced = ndcg_enhanced - ndcg_dp
        rag_improvement_optimal = ndcg_optimal - ndcg_dp

        # Rank movement vs Baseline
        rms_dp = rank_movement(base_ids, dp_ids)
        rms_rag = rank_movement(base_ids, rag_ids)
        rms_opt = rank_movement(base_ids, opt_ids)
        rms_fhe = rank_movement(base_ids, fhe_ids)
        rms_base = 0.0

        # Diversity in baseline embedding space
        div_base = diversity_score(embeddings[base_ids])
        div_dp = diversity_score(embeddings[dp_ids])
        div_rag = diversity_score(embeddings[rag_ids])
        div_opt = diversity_score(embeddings[opt_ids])
        div_fhe = diversity_score(embeddings[fhe_ids])

        # Geometry distortion
        geom_base = 0.0
        geom_rag = 0.0
        geom_dp = geom_dp_global
        geom_opt = geom_opt_global
        geom_fhe = geom_fhe_global

        # Query robustness
        def search_baseline(qtxt):
            q = normalize_rows(model.encode([qtxt]).astype(np.float32))
            _, I = base_index.search(q, TOP_K)
            return I[0].tolist()

        def search_dp(qtxt):
            q = normalize_rows(model.encode([qtxt]).astype(np.float32))
            if dp_index.d > q.shape[1]:
                attr_dim = dp_index.d - q.shape[1]
                qz = np.zeros((1, attr_dim), dtype=np.float32)
                q_dp_loc = normalize_rows(np.hstack([q * 0.7, qz * 0.3]))
            else:
                q_dp_loc = q
            _, I = dp_index.search(q_dp_loc, TOP_K)
            return I[0].tolist()

        def search_rag(qtxt):
            q = normalize_rows(model.encode([qtxt]).astype(np.float32))
            bm_ids_loc = bm25.get_top_n(qtxt.split(), list(range(len(pdf))), TOP_K * 4)
            _, I = rag_index.search(q, TOP_K * 4)
            v_ids = I[0].tolist()
            cand_loc = list(dict.fromkeys(bm_ids_loc + v_ids))
            cvecs = embeddings[cand_loc]
            sims_loc = cvecs @ q.ravel()

            sel = []
            cand_ids_loc = list(range(len(cand_loc)))
            while len(sel) < TOP_K:
                if len(sel) == 0:
                    best = int(np.argmax(sims_loc))
                else:
                    svecs = cvecs[sel]
                    divv = cosine_similarity(cvecs[cand_ids_loc], svecs).max(axis=1)
                    score_loc = 0.7 * sims_loc[cand_ids_loc] - 0.3 * divv
                    best_local = int(np.argmax(score_loc))
                    best = cand_ids_loc[best_local]
                sel.append(best)
                cand_ids_loc.remove(best)
            return [cand_loc[i] for i in sel]

        def search_opt(qtxt):
            q = normalize_rows(model.encode([qtxt]).astype(np.float32))
            if opt_index.d > q.shape[1]:
                attr_dim = opt_index.d - q.shape[1]
                qz = np.zeros((1, attr_dim), dtype=np.float32)
                q_opt_loc = normalize_rows(np.hstack([q * 0.6, qz * 0.4]))
            else:
                q_opt_loc = q

            bm_ids_loc = bm25.get_top_n(qtxt.split(), list(range(len(pdf))), TOP_K * 4)
            _, I = opt_index.search(q_opt_loc, TOP_K * 4)
            v_ids = I[0].tolist()
            cand_loc = list(dict.fromkeys(bm_ids_loc + v_ids))
            cvecs = opt_vecs[cand_loc]
            sims_loc = cvecs @ q_opt_loc.ravel()

            sel = []
            cand_ids_loc = list(range(len(cand_loc)))
            while len(sel) < TOP_K:
                if len(sel) == 0:
                    best = int(np.argmax(sims_loc))
                else:
                    svecs = cvecs[sel]
                    divv = cosine_similarity(cvecs[cand_ids_loc], svecs).max(axis=1)
                    score_loc = 0.7 * sims_loc[cand_ids_loc] - 0.3 * divv
                    best_local = int(np.argmax(score_loc))
                    best = cand_ids_loc[best_local]
                sel.append(best)
                cand_ids_loc.remove(best)
            return [cand_loc[i] for i in sel]

        def search_fhe_plain(qtxt):
            q = normalize_rows(model.encode([qtxt]).astype(np.float32))
            q_small = normalize_rows(q @ R)[0]
            q_small = norm_vec(q_small.astype(np.float32))
            sims_f = emb_fhe @ q_small
            order = np.argsort(sims_f)[::-1][:TOP_K]
            return [fhe_cand_ids[i] for i in order]

        robust_base = query_robustness(search_baseline, query)
        robust_dp = query_robustness(search_dp, query)
        robust_rag = query_robustness(search_rag, query)
        robust_opt = query_robustness(search_opt, query)
        robust_fhe = query_robustness(search_fhe_plain, query)

        results.append({
            "query": query,

            # Latencies
            "latency_baseline": lat_base,
            "latency_dp": lat_dp,
            "latency_fhe": lat_fhe,
            "latency_enhanced_rag": lat_rag_raw,
            "latency_optimal_rag": lat_opt,

            # NDCG
            "ndcg_dp": ndcg_dp,
            "ndcg_enhanced_rag": ndcg_rag,
            "ndcg_optimal_rag": ndcg_opt,

            # RAG improvement
            "rag_improvement_enhanced": rag_improvement_enhanced,
            "rag_improvement_optimal": rag_improvement_optimal,

            # Rank movement
            "rms_baseline": rms_base,
            "rms_dp": rms_dp,
            "rms_enhanced_rag": rms_rag,
            "rms_optimal_rag": rms_opt,
            "rms_fhe": rms_fhe,

            # Diversity
            "diversity_baseline": div_base,
            "diversity_dp": div_dp,
            "diversity_enhanced_rag": div_rag,
            "diversity_optimal_rag": div_opt,
            "diversity_fhe": div_fhe,

            # Geometry distortion
            "geometry_baseline": geom_base,
            "geometry_dp": geom_dp,
            "geometry_enhanced_rag": geom_rag,
            "geometry_optimal_rag": geom_opt,
            "geometry_fhe": geom_fhe,

            # Query robustness
            "robustness_baseline": robust_base,
            "robustness_dp": robust_dp,
            "robustness_enhanced_rag": robust_rag,
            "robustness_optimal_rag": robust_opt,
            "robustness_fhe": robust_fhe,
        })

    return pd.DataFrame(results)

def plot_results(df):
    
    plt.figure(figsize=(9, 5))
    modes = [
        "latency_baseline",
        "latency_dp",
        "latency_fhe",
        "latency_enhanced_rag",
        "latency_optimal_rag"
    ]
    df[modes].mean().plot(kind="bar", log=True)
    plt.xticks(
        ticks=range(len(modes)),
        labels=["Baseline", "DP", "FHE", "Enhanced RAG", "Optimal RAG"],
        rotation=45
    )
    plt.title("Latency (log scale)")
    plt.ylabel("Latency (ms, log)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "latency_log.png"))
    plt.close()

    # RAG improvement over DP
    plt.figure(figsize=(9, 5))
    plt.plot(df["rag_improvement_enhanced"], marker="o", label="Enhanced RAG vs DP")
    plt.plot(df["rag_improvement_optimal"], marker="o", label="Optimal RAG vs DP")
    plt.title("RAG NDCG Improvement Over DP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rag_improvement.png"))
    plt.close()

    # Rank Movement Score
    plt.figure(figsize=(9, 5))
    plt.plot(df["rms_dp"], marker="o", label="DP")
    plt.plot(df["rms_enhanced_rag"], marker="o", label="Enhanced RAG")
    plt.plot(df["rms_optimal_rag"], marker="o", label="Optimal RAG")
    plt.plot(df["rms_fhe"], marker="o", label="FHE")
    plt.title("Rank Movement Score (vs Baseline)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "rank_movement.png"))
    plt.close()

    # Diversity
    plt.figure(figsize=(9, 5))
    plt.plot(df["diversity_baseline"], marker="o", label="Baseline")
    plt.plot(df["diversity_dp"], marker="o", label="DP")
    plt.plot(df["diversity_enhanced_rag"], marker="o", label="Enhanced RAG")
    plt.plot(df["diversity_optimal_rag"], marker="o", label="Optimal RAG")
    plt.plot(df["diversity_fhe"], marker="o", label="FHE")
    plt.title("Diversity Score (higher is more diverse)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "diversity.png"))
    plt.close()

    # Geometry distortion
    plt.figure(figsize=(9, 5))
    geom_means = {
        "Baseline": df["geometry_baseline"].mean(),
        "DP": df["geometry_dp"].mean(),
        "Enhanced RAG": df["geometry_enhanced_rag"].mean(),
        "Optimal RAG": df["geometry_optimal_rag"].mean(),
        "FHE": df["geometry_fhe"].mean(),
    }
    plt.bar(list(geom_means.keys()), list(geom_means.values()))
    plt.title("Geometry Distortion vs Baseline")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "geometry_distortion.png"))
    plt.close()

    # Query robustness
    plt.figure(figsize=(9, 5))
    plt.plot(df["robustness_baseline"], marker="o", label="Baseline")
    plt.plot(df["robustness_dp"], marker="o", label="DP")
    plt.plot(df["robustness_enhanced_rag"], marker="o", label="Enhanced RAG")
    plt.plot(df["robustness_optimal_rag"], marker="o", label="Optimal RAG")
    plt.plot(df["robustness_fhe"], marker="o", label="FHE")
    plt.title("Query Robustness (overlap under perturbation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "query_robustness.png"))
    plt.close()

def main():
    print("Building full environment fresh...")

    pdf, embeddings, dp_vecs, opt_vecs, base_index, dp_index, opt_index, bm25, rag_index, model = build_all(
        DATA_PATH,
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    df = evaluate_all(
        pdf,
        embeddings,
        dp_vecs,
        opt_vecs,
        base_index,
        dp_index,
        opt_index,
        bm25,
        rag_index,
        model
    )

    out = os.path.join(PROJECT_ROOT, "evaluation_results.csv")
    df.to_csv(out, index=False)
    print(f"\nSaved metrics to {out}")

    plot_results(df)
    print(f"Plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
