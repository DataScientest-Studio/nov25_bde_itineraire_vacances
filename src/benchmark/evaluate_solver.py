

import time
import polars as pl
import numpy as np
import hashlib
from typing import Literal, Dict, Any, List


def hash_order(order: List[int]) -> str:
    """Hash simple pour comparer les ordres entre matrices."""
    return hashlib.md5(",".join(map(str, order)).encode()).hexdigest()


def evaluate_solver_on_cluster(
    solver: Literal["ga", "nn2o"],
    df_cluster: pl.DataFrame,
    matrix_dur: np.ndarray,
    matrix_dist: np.ndarray,
    compute_ga,
    compute_nn2o,
    pipeline,
    matrix_id: str,
    run_id: int,
) -> Dict[str, Any]:
    """
    Évalue un solveur (GA ou NN2O) sur un cluster donné + une matrice OSRM.
    Retourne : temps, distance totale, durée totale, signature d'ordre.
    """

    print(f"    → [{solver.upper()}] run {run_id} | matrix={matrix_id} | n={df_cluster.height}")

    start = time.perf_counter()

    # 1. Résultat brut du solveur
    if solver == "ga":
        _, df_route = compute_ga(df_cluster, matrix_dur)
    else:
        _, df_route = compute_nn2o(df_cluster, matrix_dur)

    # Si le solveur échoue
    if df_route is None or df_route.is_empty():
        print(" Solveur a retourné un DF vide")

        runtime_ms = (time.perf_counter() - start) * 1000

        #print(f"      ✓ OK | dist={total_distance:.1f}m | dur={total_duration:.1f}s | {runtime_ms:.1f}ms")
        return {
            "solver": solver,
            "cluster_size": df_cluster.height,
            "matrix_id": matrix_id,
            "run_id": run_id,
            "runtime_ms": runtime_ms,
            "total_distance": None,
            "total_duration": None,
            "order_signature": None,
        }

    # 2. Extraire l’ordre OSRM
    order = df_route.sort("order")["osrm_index"].to_list()
    
    # 3. Enrichissement via ton pipeline
    df_itinerary = pipeline.enrich_itinerary(
        df_day=df_route,
        order=order,
        matrix_durations=matrix_dur,
        matrix_distances=matrix_dist,
    )
    runtime_ms = (time.perf_counter() - start) * 1000

    # 4. Totaux
    total_distance = float(df_itinerary["day_total_distance"].max())
    total_duration = float(df_itinerary["day_total_duration"].max())


    # 5. Signature de l’ordre (pour mesurer la stabilité)
    order_sig = hash_order(order)

    return {
        "solver": solver,
        "cluster_size": df_cluster.height,
        "matrix_id": matrix_id,
        "run_id": run_id,
        "runtime_ms": runtime_ms,
        "total_distance": total_distance,
        "total_duration": total_duration,
        "order_signature": order_sig,
    }
