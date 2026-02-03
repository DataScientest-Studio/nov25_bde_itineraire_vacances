import time

import polars as pl

from benchmark.ga_solver import run_ga_on_cluster

GA_CONFIGS = {
    "fast": {"pop_size": 20, "ngen": 40, "cxpb": 0.9, "mutpb": 0.05},
    "balanced": {"pop_size": 30, "ngen": 60, "cxpb": 0.8, "mutpb": 0.10},
    "premium": {"pop_size": 40, "ngen": 80, "cxpb": 0.8, "mutpb": 0.10},
}


def run_ga_tuning(
    df_clustered: pl.DataFrame,
    matrices: dict,
    matrix_id: str = "walk",
    runs_per_cluster: int = 3,
):
    """
    Tuning GA indépendant du pipeline.
    Utilise matrices[matrix_id]["dur"] pour construire la matrice locale.
    """

    print("\n==============================")
    print("   TUNING GA (runtime + fitness)")
    print("==============================\n")

    cluster_sizes = sorted(df_clustered["cluster_size"].unique().to_list())
    print("Tailles de clusters disponibles :", cluster_sizes)

    results = []

    for size in cluster_sizes:
        print(f"\n→ Cluster size = {size}")

        df_cluster_size = df_clustered.filter(pl.col("cluster_size") == size)
        if df_cluster_size.height == 0:
            continue

        cluster_id = df_cluster_size["cluster_id"][0]
        df_cluster = df_clustered.filter(pl.col("cluster_id") == cluster_id)

        for config_name, config in GA_CONFIGS.items():
            print(f"  → Config GA = {config_name}")

            for run in range(runs_per_cluster):
                start = time.perf_counter()

                best_route_local, fitness = run_ga_on_cluster(
                    df_cluster=df_cluster,
                    matrices=matrices,
                    matrix_id=matrix_id,
                    pop_size=config["pop_size"],
                    ngen=config["ngen"],
                    cxpb=config["cxpb"],
                    mutpb=config["mutpb"],
                )

                runtime_ms = (time.perf_counter() - start) * 1000

                results.append(
                    {
                        "cluster_size": size,
                        "cluster_id": cluster_id,
                        "config": config_name,
                        "run": run,
                        "runtime_ms": runtime_ms,
                        "fitness": fitness,
                        "success": best_route_local is not None,
                    }
                )

                print(f"      ✓ runtime={runtime_ms:.1f}ms | fitness={fitness}")

    print("\n=== Tuning GA terminé ===\n")
    return pl.DataFrame(results)
