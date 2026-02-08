import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(style="whitegrid")


def load_results():
    tsp_path = "benchmark_solvers/results/tsp_results.parquet"
    iti_path = "benchmark_solvers/results/itinerary_results.parquet"
    comp_path = "benchmark_solvers/results/comparison_results.parquet"

    if not os.path.exists(tsp_path):
        raise FileNotFoundError("Missing TSP results file.")
    if not os.path.exists(iti_path):
        raise FileNotFoundError("Missing Itinerary results file.")
    if not os.path.exists(comp_path):
        raise FileNotFoundError("Missing comparison results file.")

    tsp = pd.read_parquet(tsp_path)
    iti = pd.read_parquet(iti_path)
    comp = pd.read_parquet(comp_path)

    return tsp, iti, comp


def plot_tsp_metrics(tsp):
    print("\n=== TSP METRICS ===")

    # Rating par solveur
    plt.figure(figsize=(10, 6))
    sns.barplot(data=tsp, x="matrix", y="rating", hue="solver")
    plt.title("TSP — Rating par solveur")
    plt.show()

    # Coût moyen
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=tsp, x="matrix", y="mean_cost", hue="solver", marker="o")
    plt.title("TSP — Coût moyen par solveur")
    plt.show()

    # Distribution des coûts
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=tsp, x="solver", y="mean_cost")
    plt.title("TSP — Distribution des coûts")
    plt.show()


def plot_itinerary_metrics(iti):
    print("\n=== ITINERARY METRICS ===")

    # Score moyen
    plt.figure(figsize=(10, 6))
    sns.barplot(data=iti, x="matrix", y="mean_score", hue="solver")
    plt.title("Itinéraire — Score moyen par solveur")
    plt.show()

    # Diversité
    plt.figure(figsize=(10, 6))
    sns.barplot(data=iti, x="matrix", y="variety", hue="solver")
    plt.title("Itinéraire — Diversité des POIs visités")
    plt.show()

    # Temps moyen
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=iti, x="matrix", y="mean_time", hue="solver", marker="o")
    plt.title("Itinéraire — Temps moyen par solveur")
    plt.show()


def plot_comparison(comp):
    print("\n=== COMPARISON TSP vs ITINERARY ===")

    # Scatter : coût TSP vs score Iti
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=comp, x="mean_cost_tsp", y="best_score_iti", hue="matrix", s=200
    )
    plt.title("Comparaison TSP vs Itinéraire")
    plt.xlabel("Coût TSP (plus bas = mieux)")
    plt.ylabel("Score Itinéraire (plus haut = mieux)")
    plt.show()

    # Barplot solveur gagnant
    plt.figure(figsize=(10, 6))
    sns.barplot(data=comp, x="matrix", y="rating_tsp", color="steelblue", label="TSP")
    sns.barplot(
        data=comp, x="matrix", y="best_score_iti", color="orange", label="Itinéraire"
    )
    plt.title("Comparaison globale TSP vs Itinéraire")
    plt.ylabel("Score / Rating")
    plt.legend()
    plt.show()

    # Ratio score Iti / coût TSP
    if "tsp_vs_iti_score_ratio" in comp.columns:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=comp, x="matrix", y="tsp_vs_iti_score_ratio", marker="o")
        plt.title("Ratio Score Itinéraire / Coût TSP")
        plt.ylabel("Ratio")
        plt.show()


def main():
    tsp, iti, comp = load_results()

    plot_tsp_metrics(tsp)
    plot_itinerary_metrics(iti)
    plot_comparison(comp)

    print("\nVisualisation terminée.")


if __name__ == "__main__":
    main()
