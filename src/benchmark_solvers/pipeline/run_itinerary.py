import os

import pandas as pd

from benchmark_solvers.common.dataset_loader import load_all_datasets
from benchmark_solvers.itinerary.benchmark.itinerary_runner import (
    ItineraryBenchmarkRunner,
)


def run_itinerary():
    print("\n==============================")
    print("   ITINERARY BENCHMARK START  ")
    print("==============================\n")

    # Dossier des résultats
    os.makedirs("results", exist_ok=True)

    # 1) Charger les datasets (POIs + matrices OSRM)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "benchmark_solvers/data")

    print(f"Loading datasets from: {data_dir}")
    datasets = load_all_datasets(data_dir, data_dir)
    print(f"Datasets loaded: {list(datasets.keys())}")

    if not datasets:
        print("No datasets found. Check your data/ folder and filenames.")
        return pd.DataFrame()

    # 2) Lancer le benchmark Itinéraire
    print("\nRunning Itinerary benchmark...")
    runner = ItineraryBenchmarkRunner(datasets, runs=10)
    results = runner.run()

    if not results:
        print("No results produced by the benchmark.")
        return pd.DataFrame()

    # 3) DataFrame des résultats
    df_iti = pd.DataFrame(results)

    # 4) Tri par mean_score
    if "mean_score" in df_iti.columns:
        df_sorted = df_iti.sort_values("mean_score", ascending=False)
    else:
        df_sorted = df_iti
        print("Warning: 'mean_score' column not found in results.")

    print("\n=== BEST ITINERARY SOLUTIONS ===")
    cols = [
        c
        for c in [
            "matrix",
            "solver",
            "mean_score",
            "best_score",
            "mean_time",
            "variety",
        ]
        if c in df_sorted.columns
    ]
    print(df_sorted[cols])

    # 5) Sauvegarde
    out_path = os.path.join("results", "itinerary_results.parquet")
    df_sorted.to_parquet(out_path)
    print(f"\nResults saved to: {out_path}")

    print("\nItinerary benchmark completed.\n")
    return df_sorted


if __name__ == "__main__":
    run_itinerary()
