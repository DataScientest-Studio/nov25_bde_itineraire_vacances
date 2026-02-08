import pandas as pd


def build_comparison_table(df_tsp, df_iti):
    # 1) Sélection du meilleur solveur TSP
    tsp_best = (
        df_tsp.sort_values("rating", ascending=False)
        .groupby("matrix")
        .head(1)
        .reset_index(drop=True)
    )

    # Renommage explicite des colonnes TSP
    tsp_best = tsp_best.rename(
        columns={
            "solver": "solver_tsp",
            "mean_cost": "mean_cost_tsp",
            "rating": "rating_tsp",
            "best_route": "best_route_tsp",
        }
    )

    # 2) Sélection du meilleur solveur Itinéraire
    iti_best = (
        df_iti.sort_values("best_score", ascending=False)
        .groupby("matrix")
        .head(1)
        .reset_index(drop=True)
    )

    # Renommage explicite des colonnes Itinéraire
    iti_best = iti_best.rename(
        columns={
            "solver": "solver_iti",
            "best_score": "best_score_iti",
            "variety": "variety_iti",
            "best_route": "best_route_iti",
        }
    )

    # 3) Fusion propre
    merged = tsp_best.merge(iti_best, on="matrix")

    # 4) Ratio comparatif
    merged["tsp_vs_iti_score_ratio"] = (
        merged["best_score_iti"] / merged["mean_cost_tsp"]
    )

    return merged
