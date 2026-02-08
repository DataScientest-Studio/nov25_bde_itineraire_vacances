import os

import pandas as pd

from benchmark_solvers.common.dataset_loader import load_all_datasets
from benchmark_solvers.comparator.compare_tsp_itinerary import build_comparison_table
from benchmark_solvers.itinerary.benchmark.itinerary_runner import (
    ItineraryBenchmarkRunner,
)
from benchmark_solvers.tsp.benchmark.tsp_runner import TSPBenchmarkRunner


def run_all():
    print("\n====================================")
    print("        GLOBAL BENCHMARK RUN")
    print("====================================\n")

    # ------------------------------------------------------------
    # 1) Préparation
    # ------------------------------------------------------------
    os.makedirs("results", exist_ok=True)

    print("=== Loading datasets ===")
    datasets = load_all_datasets("benchmark_solvers/data/", "benchmark_solvers/data/")
    print(f"Datasets loaded: {list(datasets.keys())}")

    if not datasets:
        print("No datasets found.")
        return None

    # ------------------------------------------------------------
    # 2) Benchmark TSP
    # ------------------------------------------------------------
    print("\n=== Running TSP benchmark ===")
    tsp_runner = TSPBenchmarkRunner(datasets, runs=10)
    tsp_results = pd.DataFrame(tsp_runner.run())

    tsp_out = "benchmark_solvers/results/tsp_results.parquet"
    tsp_results.to_parquet(tsp_out)
    print(f"TSP results saved to: {tsp_out}")

    # ------------------------------------------------------------
    # 3) Benchmark Itinéraire
    # ------------------------------------------------------------
    print("\n=== Running Itinerary benchmark ===")
    iti_runner = ItineraryBenchmarkRunner(datasets, runs=10)
    iti_results = pd.DataFrame(iti_runner.run())

    iti_out = "benchmark_solvers/results/itinerary_results.parquet"
    iti_results.to_parquet(iti_out)
    print(f"Itinerary results saved to: {iti_out}")

    # ------------------------------------------------------------
    # 4) Comparaison TSP vs Itinéraire
    # ------------------------------------------------------------
    print("\n=== Building TSP vs Itinerary comparison ===")
    comp_results = build_comparison_table(tsp_results, iti_results)

    comp_out = "benchmark_solvers/results/comparison_results.parquet"
    comp_results.to_parquet(comp_out)
    print(f"Comparison results saved to: {comp_out}")

    # ------------------------------------------------------------
    # 5) Résumé global affiché
    # ------------------------------------------------------------
    print("\n=== GLOBAL BENCHMARK SUMMARY ===")

    display_cols = [
        "matrix",
        "solver_tsp",
        "mean_cost_tsp",
        "rating_tsp",
        "solver_iti",
        "best_score_iti",
        "variety_iti",
    ]

    display_cols = [c for c in display_cols if c in comp_results.columns]

    print(comp_results[display_cols])

    print("\nGlobal benchmark completed.\n")

    return tsp_results, iti_results, comp_results


if __name__ == "__main__":
    run_all()
