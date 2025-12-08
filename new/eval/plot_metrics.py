import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "figure.figsize": (9, 5),
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.dpi": 120
})

METHODS = ["Baseline", "DP", "FHE", "Enhanced RAG", "Optimal RAG"]
COLORS = ["tab:blue", "tab:orange", "tab:red", "tab:green", "tab:purple"]

def normalize_only_rank(arr):
    arr = np.array(arr, dtype=float)
    arr[arr == 0] = 0.1
    return arr

def plot_all(csv_path, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    lat_means = [
        df["latency_baseline"].mean(),
        df["latency_dp"].mean(),
        df["latency_fhe"].mean(),
        df["latency_enhanced_rag"].mean(),
        df["latency_optimal_rag"].mean(),
    ]

    plt.figure()
    plt.bar(METHODS, lat_means, color=COLORS)
    plt.yscale("log")
    plt.title("Latency by Method (log scale)")
    plt.ylabel("Latency (ms, log scale)")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency.png"))
    plt.close()

    x = np.arange(len(df))
    labels = [f"Query {i+1}" for i in range(len(df))]

    plt.figure()
    plt.plot(x, df["rag_improvement_enhanced"], marker="o", linewidth=2,
             label="Enhanced RAG vs DP", color="tab:green")
    plt.plot(x, df["rag_improvement_optimal"], marker="o", linewidth=2,
             label="Optimal RAG vs DP", color="tab:purple")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.title("NDCG@K Improvement Over DP")
    plt.ylabel("Δ NDCG")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "delta_ndcg.png"))
    plt.close()

    rms_means = normalize_only_rank([
        df["rms_baseline"].mean(),
        df["rms_dp"].mean(),
        df["rms_fhe"].mean(),
        df["rms_enhanced_rag"].mean(),
        df["rms_optimal_rag"].mean(),
    ])

    plt.figure()
    plt.bar(METHODS, rms_means, color=COLORS)
    plt.title("Rank Movement Score vs Baseline")
    plt.ylabel("Mean absolute rank change")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "rank_movement.png"))
    plt.close()

    div_means = [
        df["diversity_baseline"].mean(),
        df["diversity_dp"].mean(),
        df["diversity_fhe"].mean(),
        df["diversity_enhanced_rag"].mean(),
        df["diversity_optimal_rag"].mean(),
    ]

    plt.figure()
    plt.bar(METHODS, div_means, color=COLORS)
    plt.title("Diversity Score (higher = more diverse)")
    plt.ylabel("Mean diversity")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "diversity.png"))
    plt.close()

    geom_means = [
        df["geometry_baseline"].mean(),
        df["geometry_dp"].mean(),
        df["geometry_fhe"].mean(),
        df["geometry_enhanced_rag"].mean(),
        df["geometry_optimal_rag"].mean(),
    ]

    plt.figure()
    plt.bar(METHODS, geom_means, color=COLORS)
    plt.title("Geometry Distortion vs Baseline")
    plt.ylabel("Mean cosine distance distortion")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "geometry.png"))
    plt.close()

    rob_means = [
        df["robustness_baseline"].mean(),
        df["robustness_dp"].mean(),
        df["robustness_fhe"].mean(),
        df["robustness_enhanced_rag"].mean(),
        df["robustness_optimal_rag"].mean(),
    ]

    plt.figure()
    plt.bar(METHODS, rob_means, color=COLORS)
    plt.title("Query Robustness Under Perturbation")
    plt.ylabel("Mean overlap fraction")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "robustness.png"))
    plt.close()

    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    plot_all("newresults.csv")
