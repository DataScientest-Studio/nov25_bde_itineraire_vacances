import pandas as pd
from pathlib import Path

from tsp.nn2opt import NN2OptSolver
from tsp.sa import SA_Solver
from tsp.ga_solver import GASolver

from benchmark.runner import BenchmarkRunner
from loaders.loader import *

from analysis.plots import (
    boxplot_costs,
    boxplot_gaps,
    heatmap_gap,
    plot_pareto,
    stability_plot,
)

from analysis.metrics import (
    compute_best_per_matrix,
    add_gap_column,
    stability_stats,
    pareto_front,
)

PROJECT_ROOT = Path("../..")
OSRM_MATRIX_PATH = PROJECT_ROOT / "data" / "processed"


def main():
    # # 1. Chemins vers tes matrices OSRM (Parquet)
    # paths = [
    #     OSRM_MATRIX_PATH / "df_osrm_dist" / "14_merged_20260109_115825.parquet",
    #     OSRM_MATRIX_PATH / "df_osrm_dist" / "30_merged_20260109_115553.parquet",
    #     OSRM_MATRIX_PATH / "df_osrm_dist" / "58_merged_20260109_115141.parquet",
    #     OSRM_MATRIX_PATH / "df_osrm_dist" / "100_merged_20260109_115720.parquet",
    # ]


    # === 1. Charger matrices + POIs ===
    matrix_paths = [
        OSRM_MATRIX_PATH / "df_osrm_dist" / "14_merged_20260109_115825.parquet",
        OSRM_MATRIX_PATH / "df_osrm_dist" / "30_merged_20260109_115553.parquet",
        OSRM_MATRIX_PATH / "df_osrm_dist" / "58_merged_20260109_115141.parquet",
        OSRM_MATRIX_PATH / "df_osrm_dist" / "100_merged_20260109_115720.parquet",
    ]

    pois_paths = [
        OSRM_MATRIX_PATH / "df_clustered" / "14_merged_20260109_115825.parquet",
        OSRM_MATRIX_PATH / "df_clustered" / "30_merged_20260109_115553.parquet",
        OSRM_MATRIX_PATH / "df_clustered" / "58_merged_20260109_115141.parquet",
        OSRM_MATRIX_PATH / "df_clustered" / "100_merged_20260109_115720.parquet",
    ]


    # 2. Charger les matrices
    #matrices = load_multiple_matrices(paths)
    matrices, pois = load_all_matrices_and_pois(matrix_paths, pois_paths)


    # 3. Solveurs à tester
    solver_classes = [
        NN2OptSolver,
        SA_Solver,
        GASolver,
        # Neo4jSolver,
    ]

    # 4. Lancer le benchmark
    runner = BenchmarkRunner(start=0)
    runner.run_on_multiple_matrices(matrices, solver_classes, repeat=10)

    df = runner.to_dataframe()
    #save df to parquet
    df.to_parquet(PROJECT_ROOT / "data" / "processed" / "results_benchmark.parquet")

    # 5. Calcul du gap
    best_per_matrix = compute_best_per_matrix(df)
    df = add_gap_column(df, best_per_matrix)

    # 6. Statistiques globales
    print("\n=== Stabilité globale ===")
    print(stability_stats(df))

    # 7. Pareto
    pareto = pareto_front(df)
    print("\n=== Front de Pareto ===")
    print(pareto)

    # 8. Visualisations
    boxplot_costs(df, by="solver")
    boxplot_costs(df, by="matrix")
    boxplot_gaps(df, by="solver")
    heatmap_gap(df)
    stability_plot(df)
    plot_pareto(pareto)


if __name__ == "__main__":
    main()