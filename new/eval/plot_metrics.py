import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Professional global styling
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 130
plt.rcParams["axes.titlesize"] = 15
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["xtick.labelsize"] = 10
plt.rcParams["ytick.labelsize"] = 10

METHODS = ["Baseline", "DP", "FHE", "Enhanced RAG", "Optimal RAG"]

COLOR_MAP = {
    "Baseline": "#4C72B0",
    "DP": "#55A868",
    "FHE": "#C44E52",
    "Enhanced RAG": "#8172B2",
    "Optimal RAG": "#CCB974",
}


def load_results(csv_path):
    return pd.read_csv(csv_path)


def barplot(metric_values, title, ylabel, save_path):
    """Reusable high-quality bar plot generator."""
    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=METHODS,
        y=metric_values,
        palette=[COLOR_MAP[m] for m in METHODS]
    )
    plt.title(title, pad=12)
    plt.ylabel(ylabel)
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def lineplot_per_query(df, series_list, labels, title, ylabel, save_path):
    """Plot series over queries with professional aesthetics."""
    plt.figure(figsize=(11, 5))
    x = np.arange(len(df))

    for series, label in zip(series_list, labels):
        plt.plot(
            x,
            df[series],
            marker="o",
            linewidth=2,
            markersize=6,
            label=label
        )

    plt.xticks(x, df["query"], rotation=45, ha="right")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.legend()
    sns.despine()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_all(csv_path, output_dir="plots_professional"):
    os.makedirs(output_dir, exist_ok=True)
    df = load_results(csv_path)

    # ----------------------------------------------------------
    # LATENCY (log scale)
    # ----------------------------------------------------------
    latency_means = [
        df["latency_baseline"].mean(),
        df["latency_dp"].mean(),
        df["latency_fhe"].mean(),
        df["latency_enhanced_rag"].mean(),
        df["latency_optimal_rag"].mean(),
    ]

    plt.figure(figsize=(9, 5))
    sns.barplot(
        x=METHODS,
        y=latency_means,
        palette=[COLOR_MAP[m] for m in METHODS]
    )
    plt.yscale("log")
    plt.title("Latency by Method (Log Scale)", pad=12)
    plt.ylabel("Latency (ms)")
    plt.xticks(rotation=20)
    sns.despine()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "latency_log.png"), bbox_inches="tight")
    plt.close()

    # ----------------------------------------------------------
    # NDCG improvement curves (per query)
    # ----------------------------------------------------------
    lineplot_per_query(
        df,
        series_list=["rag_improvement_enhanced", "rag_improvement_optimal"],
        labels=["Enhanced RAG – DP", "Optimal RAG – DP"],
        title="NDCG@K Improvement Over DP",
        ylabel="Δ NDCG",
        save_path=os.path.join(output_dir, "rag_improvement.png")
    )

    # ----------------------------------------------------------
    # Rank movement
    # ----------------------------------------------------------
    rms_means = [
        df["rms_baseline"].mean(),
        df["rms_dp"].mean(),
        df["rms_fhe"].mean(),
        df["rms_enhanced_rag"].mean(),
        df["rms_optimal_rag"].mean(),
    ]
    barplot(
        rms_means,
        "Rank Movement Score",
        "Mean Absolute Rank Shift",
        os.path.join(output_dir, "rank_movement.png")
    )

    # ----------------------------------------------------------
    # Diversity
    # ----------------------------------------------------------
    diversity_means = [
        df["diversity_baseline"].mean(),
        df["diversity_dp"].mean(),
        df["diversity_fhe"].mean(),
        df["diversity_enhanced_rag"].mean(),
        df["diversity_optimal_rag"].mean(),
    ]
    barplot(
        diversity_means,
        "Diversity Score",
        "Average Pairwise Cosine Distance",
        os.path.join(output_dir, "diversity.png")
    )

    # ----------------------------------------------------------
    # Geometry distortion
    # ----------------------------------------------------------
    geom_means = [
        df["geometry_baseline"].mean(),
        df["geometry_dp"].mean(),
        df["geometry_fhe"].mean(),
        df["geometry_enhanced_rag"].mean(),
        df["geometry_optimal_rag"].mean(),
    ]
    barplot(
        geom_means,
        "Geometry Distortion",
        "Mean Embedding Shift (Cosine Distance)",
        os.path.join(output_dir, "geometry_distortion.png")
    )

    # ----------------------------------------------------------
    # Query Robustness
    # ----------------------------------------------------------
    robust_means = [
        df["robustness_baseline"].mean(),
        df["robustness_dp"].mean(),
        df["robustness_fhe"].mean(),
        df["robustness_enhanced_rag"].mean(),
        df["robustness_optimal_rag"].mean(),
    ]
    barplot(
        robust_means,
        "Query Robustness Under Perturbation",
        "Overlap Fraction",
        os.path.join(output_dir, "query_robustness.png")
    )

    print(f"Professional plots saved to: {output_dir}")
