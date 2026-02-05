from typing import List

import numpy as np
import polars as pl

from .evaluate_solver import evaluate_solver_on_cluster


def compare_solvers_on_sizes(
    df_all_pois: pl.DataFrame,
    matrix_dur: np.ndarray,
    matrix_dist: np.ndarray,
    compute_ga,
    compute_nn2o,
    pipeline,
    sizes: List[int],
    runs_per_size: int = 5,
    matrix_id: str = "walk",
) -> pl.DataFrame:
    """
    Compare GA et NN2O pour différentes tailles de clusters.
    Objectif : déterminer le seuil AUTO (à partir de quelle taille GA > NN2O).
    """

    print("\n=== Benchmark seuil AUTO (GA vs NN2O) ===")
    # Sécurité : convertir en liste d'entiers
    sizes = [int(x) for x in sizes if x <= df_all_pois.height]

    print("SIZES USED:", sizes)

    results = []

    for size in sizes:
        print(f"\n  → Taille cluster = {size} POIs")

        # Sélectionner un sous-échantillon de POIs
        df_sample = df_all_pois.sample(size, with_replacement=False)

        for run_id in range(runs_per_size):
            print(f"    • Run {run_id}")

            for solver in ["ga", "nn2o"]:
                res = evaluate_solver_on_cluster(
                    solver=solver,
                    df_cluster=df_sample,
                    matrix_dur=matrix_dur,
                    matrix_dist=matrix_dist,
                    compute_ga=compute_ga,
                    compute_nn2o=compute_nn2o,
                    pipeline=pipeline,
                    matrix_id=matrix_id,
                    run_id=run_id,
                )

                # Ajouter la taille du cluster dans les résultats
                res["cluster_size"] = size

                results.append(res)

    return pl.DataFrame(results)
