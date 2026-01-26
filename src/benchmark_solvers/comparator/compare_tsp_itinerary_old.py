import numpy as np
import pandas as pd


def normalize(series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def build_comparison_table(df_tsp: pd.DataFrame, df_iti: pd.DataFrame) -> pd.DataFrame:
    """
    Compare le meilleur solveur TSP et le meilleur solveur Itinéraire
    sur deux axes :
    - logistique (distance + temps)
    - touristique (score + variété + timing + distance raisonnable)
    """

    # Meilleur solveur TSP
    best_tsp = df_tsp.sort_values("rating", ascending=False).iloc[0]

    # Meilleur solveur Itinéraire
    best_iti = df_iti.sort_values("mean_score", ascending=False).iloc[0]

    rows = []

    # Vue logistique
    rows.append({
        "world": "TSP",
        "solver": best_tsp["solver"],
        "type": "logistic",
        "distance": best_tsp["mean_cost"],
        "time": best_tsp["mean_time"],
        "score_tourist": 0.0,
        "variety": 0,
        "lunch_time": None,
    })

    rows.append({
        "world": "Itinerary",
        "solver": best_iti["solver"],
        "type": "logistic",
        "distance": best_iti.get("mean_distance", np.nan),
        "time": best_iti["mean_time"],
        "score_tourist": best_iti["mean_score"],
        "variety": best_iti["variety"],
        #"lunch_time": best_iti["lunch_time"],
    })

    # Vue touristique
    rows.append({
        "world": "TSP",
        "solver": best_tsp["solver"],
        "type": "tourist",
        "distance": best_tsp["mean_cost"],
        "time": best_tsp["mean_time"],
        "score_tourist": 0.0,
        "variety": 0,
        "lunch_time": None,
    })

    rows.append({
        "world": "Itinerary",
        "solver": best_iti["solver"],
        "type": "tourist",
        "distance": best_iti.get("mean_distance", np.nan),
        "time": best_iti["mean_time"],
        "score_tourist": best_iti["mean_score"],
        "variety": best_iti["variety"],
        #"lunch_time": best_iti["lunch_time"],
    })

    df = pd.DataFrame(rows)

    # --- Rating logistique ---
    mask_log = df["type"] == "logistic"
    df_log = df[mask_log].copy()

    df_log["dist_norm"] = 1 - normalize(df_log["distance"])
    df_log["time_norm"] = 1 - normalize(df_log["time"])

    df.loc[mask_log, "rating_logistic"] = (
        0.6 * df_log["dist_norm"] + 0.4 * df_log["time_norm"]
    )

    # --- Rating touristique ---
    mask_tour = df["type"] == "tourist"
    df_tour = df[mask_tour].copy()

    df_tour["score_norm"] = normalize(df_tour["score_tourist"].fillna(0))
    df_tour["variety_norm"] = normalize(df_tour["variety"].fillna(0))

    lt = df_tour["lunch_time"].fillna(0)
    df_tour["lunch_norm"] = np.exp(-(lt - 13) ** 2)

    df_tour["dist_norm"] = 1 - normalize(df_tour["distance"])

    df.loc[mask_tour, "rating_tourist"] = (
        0.4 * df_tour["score_norm"]
        + 0.2 * df_tour["variety_norm"]
        + 0.2 * df_tour["lunch_norm"]
        + 0.2 * df_tour["dist_norm"]
    )

    return df