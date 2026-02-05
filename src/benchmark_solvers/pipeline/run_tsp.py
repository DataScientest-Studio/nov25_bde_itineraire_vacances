import os

import pandas as pd

from benchmark_solvers.common.dataset_loader import load_all_datasets
from benchmark_solvers.tsp.benchmark.tsp_runner import TSPBenchmarkRunner


def run_tsp():
    print("\n==============================")
    print("        TSP BENCHMARK")
    print("==============================\n")

    os.makedirs("results", exist_ok=True)

    # Charger les datasets
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_dir = os.path.join(base_dir, "benchmark_solvers/data")

    print(f"Loading datasets from: {data_dir}")
    datasets = load_all_datasets(data_dir, data_dir)
    print(f"Datasets loaded: {list(datasets.keys())}")

    if not datasets:
        print("No datasets found.")
        return pd.DataFrame()

    # Lancer le benchmark
    print("\nRunning TSP benchmark...")
    runner = TSPBenchmarkRunner(datasets, start=0, runs=10)
    results = runner.run()

    if not results:
        print("No results produced.")
        return pd.DataFrame()

    # DataFrame final
    df_tsp = pd.DataFrame(results)

    # Tri par rating décroissant
    if "rating" in df_tsp.columns:
        df_sorted = df_tsp.sort_values("rating", ascending=False)
    else:
        df_sorted = df_tsp

    print("\n=== BEST TSP SOLUTIONS ===")
    cols = [
        "matrix",
        "size",
        "solver",
        "mean_cost",
        "min_cost",
        "max_cost",
        "mean_time",
        "robustness",
        "efficiency",
        "rating",
    ]
    cols = [c for c in cols if c in df_sorted.columns]

    print(df_sorted[cols])

    # Sauvegarde
    out_path = os.path.join("results", "tsp_results.parquet")
    df_sorted.to_parquet(out_path)
    print(f"\nResults saved to: {out_path}")

    print("\nTSP benchmark completed.\n")
    return df_sorted


if __name__ == "__main__":
    run_tsp()
