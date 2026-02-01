import polars as pl
import numpy as np
from typing import Dict, List
from .evaluate_solver import evaluate_solver_on_cluster


def compare_stability_on_matrices(
    df_all_pois: pl.DataFrame,
    matrices: Dict[str, Dict[str, np.ndarray]],
    compute_ga,
    compute_nn2o,
    pipeline,
    solver: str,
    cluster_sizes: List[int] = [2, 10, 25, 50],
    runs_per_matrix: int = 3,
) -> pl.DataFrame:
    """
    Mesure la stabilité/robustesse d’un solveur (GA ou NN2O)
    sur plusieurs matrices OSRM et plusieurs tailles de clusters.

    Pour chaque matrice (walk, bike, car, walk_perturbed),
    pour chaque taille (2, 10, 25, 50),
    pour chaque répétition,
    on exécute le solveur et on mesure :
        - total_distance
        - total_duration
        - order_signature
        - runtime_ms
    """
    print(f"\n=== Benchmark stabilité {solver.upper()} ===")

    cluster_sizes = [int(x) for x in cluster_sizes if x <= df_all_pois.height]

    print("SIZES USED:", cluster_sizes)
    results = []

    for matrix_id, mats in matrices.items():
        print(f"\n  → Matrice = {matrix_id}")

        matrix_dur = mats["dur"]
        matrix_dist = mats["dist"]

        for size in cluster_sizes:
            # Sous-échantillon de POIs
            df_cluster = df_all_pois.sample(size, with_replacement=False)

            for run_id in range(runs_per_matrix):

                res = evaluate_solver_on_cluster(
                    solver=solver,
                    df_cluster=df_cluster,
                    matrix_dur=matrix_dur,
                    matrix_dist=matrix_dist,
                    compute_ga=compute_ga,
                    compute_nn2o=compute_nn2o,
                    pipeline=pipeline,
                    matrix_id=matrix_id,
                    run_id=run_id,
                )

                # Ajouter la taille du cluster
                res["cluster_size"] = size

                results.append(res)

    return pl.DataFrame(results)