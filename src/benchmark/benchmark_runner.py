from typing import Dict, List

import numpy as np
import polars as pl

from .compare_sizes import compare_solvers_on_sizes
from .compare_stability import compare_stability_on_matrices


def run_benchmark(
    df_all_pois: pl.DataFrame,
    matrices: Dict[str, Dict[str, np.ndarray]],
    compute_ga,
    compute_nn2o,
    pipeline,
    cluster_sizes: List[int] = [2, 10, 25, 50],
    # cluster_sizes: List[int] = [2, 22],
    runs_per_matrix: int = 5,  # 10
    runs_per_size: int = 10,  # 5
):
    """
    Orchestrateur du benchmark complet :
    - Comparaison GA vs NN2O pour déterminer le seuil AUTO (2 → 50 POIs)
    - Stabilité GA sur matrices × tailles × répétitions
    - Stabilité NN2O sur matrices × tailles × répétitions
    """
    print("\n==============================")
    print("   LANCEMENT DU BENCHMARK")
    print("==============================\n")

    # ---------------------------------------------------------
    # 1) Comparaison par tailles (seuil AUTO)
    # ---------------------------------------------------------
    print("→ Benchmark seuil AUTO (GA vs NN2O, tailles 2 → 50)…")

    # sizes_for_auto = list(range(2, 48, 10))
    sizes_for_auto = list(range(2, 20, 2))

    df_size_comp = compare_solvers_on_sizes(
        df_all_pois=df_all_pois,
        matrix_dur=matrices["walk"]["dur"],
        matrix_dist=matrices["walk"]["dist"],
        compute_ga=compute_ga,
        compute_nn2o=compute_nn2o,
        pipeline=pipeline,
        sizes=sizes_for_auto,
        runs_per_size=runs_per_size,
        matrix_id="walk",
    )

    # ---------------------------------------------------------
    # 2) Stabilité GA
    # ---------------------------------------------------------
    print("→ Benchmark stabilité GA (matrices × tailles)…")

    df_stab_ga = compare_stability_on_matrices(
        df_all_pois=df_all_pois,
        matrices=matrices,
        compute_ga=compute_ga,
        compute_nn2o=compute_nn2o,
        pipeline=pipeline,
        solver="ga",
        cluster_sizes=cluster_sizes,
        runs_per_matrix=runs_per_matrix,
    )

    # ---------------------------------------------------------
    # 3) Stabilité NN2O
    # ---------------------------------------------------------
    print("→ Benchmark stabilité NN2O (matrices × tailles)…")

    df_stab_nn2o = compare_stability_on_matrices(
        df_all_pois=df_all_pois,
        matrices=matrices,
        compute_ga=compute_ga,
        compute_nn2o=compute_nn2o,
        pipeline=pipeline,
        solver="nn2o",
        cluster_sizes=cluster_sizes,
        runs_per_matrix=runs_per_matrix,
    )

    # ---------------------------------------------------------
    # 4) Sauvegarde automatique des résultats
    # ---------------------------------------------------------
    from benchmark.benchmark_io import save_benchmark

    results = {
        "size_comparison": df_size_comp,
        "stability_ga": df_stab_ga,
        "stability_nn2o": df_stab_nn2o,
    }

    print("\n→ Sauvegarde des résultats…")
    try:
        save_benchmark(results)
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        pass

    print("\n=== Benchmark terminé ===\n")

    return results
