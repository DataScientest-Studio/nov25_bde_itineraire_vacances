import numpy as np
import pandas as pd


import matplotlib.pyplot as plt
import numpy as np



def normalize(series):
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


def build_comparison_table(df_tsp: pd.DataFrame, df_iti: pd.DataFrame) -> pd.DataFrame:
    # On suppose une même "matrix" (ex: lyon_50) pour comparer
    # et qu’on prend le meilleur solveur de chaque monde
    best_tsp = df_tsp.sort_values("rating", ascending=False).iloc[0]
    best_iti = df_iti.sort_values("mean_score", ascending=False).iloc[0]

    rows = []

    # Vue logistique (distance/temps)
    rows.append({
        "world": "TSP",
        "solver": best_tsp["solver"],
        "type": "logistic",
        "distance": best_tsp["mean_cost"],
        "time": best_tsp["mean_time"],
        "score_tourist": np.nan,
        "variety": np.nan,
        "lunch_time": np.nan,
    })

    rows.append({
        "world": "Itinerary",
        "solver": best_iti["solver"],
        "type": "logistic",
        "distance": best_iti.get("mean_distance", np.nan),
        "time": best_iti["mean_time"],
        "score_tourist": best_iti["mean_score"],
        "variety": best_iti["variety"],
        "lunch_time": best_iti["lunch_time"],
    })

    # Vue touriste (score/variété/timing)
    rows.append({
        "world": "TSP",
        "solver": best_tsp["solver"],
        "type": "tourist",
        "distance": best_tsp["mean_cost"],
        "time": best_tsp["mean_time"],
        "score_tourist": 0.0,  # TSP ne sait pas optimiser ça
        "variety": np.nan,
        "lunch_time": np.nan,
    })

    rows.append({
        "world": "Itinerary",
        "solver": best_iti["solver"],
        "type": "tourist",
        "distance": best_iti.get("mean_distance", np.nan),
        "time": best_iti["mean_time"],
        "score_tourist": best_iti["mean_score"],
        "variety": best_iti["variety"],
        "lunch_time": best_iti["lunch_time"],
    })

    df = pd.DataFrame(rows)

    # Normalisations pour ratings
    # Logistique : distance bas, temps bas
    mask_log = df["type"] == "logistic"
    df_log = df[mask_log].copy()
    df_log["dist_norm"] = 1 - normalize(df_log["distance"])
    df_log["time_norm"] = 1 - normalize(df_log["time"])
    df.loc[mask_log, "rating_logistic"] = (
        0.6 * df_log["dist_norm"] + 0.4 * df_log["time_norm"]
    )

    # Touriste : score haut, variété haute, lunch proche de 13h, distance raisonnable
    mask_tour = df["type"] == "tourist"
    df_tour = df[mask_tour].copy()

    df_tour["score_norm"] = normalize(df_tour["score_tourist"].fillna(0))
    df_tour["variety_norm"] = normalize(df_tour["variety"].fillna(0))

    # lunch_time_score ~ exp(-(lt-13)^2)
    lt = df_tour["lunch_time"].fillna(0)
    df_tour["lunch_norm"] = np.exp(-(lt - 13) ** 2)

    # distance raisonnable : on normalise et on inverse un peu
    df_tour["dist_norm"] = 1 - normalize(df_tour["distance"])

    df.loc[mask_tour, "rating_tourist"] = (
        0.4 * df_tour["score_norm"]
        + 0.2 * df_tour["variety_norm"]
        + 0.2 * df_tour["lunch_norm"]
        + 0.2 * df_tour["dist_norm"]
    )

    return df

def radar_comparison_tourist(df_comp):
    df_tour = df_comp[df_comp["type"] == "tourist"].copy()

    # On suppose 2 lignes : TSP et Itinerary
    metrics = ["distance", "score_tourist", "variety", "lunch_time"]
    labels = ["distance", "score", "variety", "lunch"]

    # Normalisation globale
    for m in metrics:
        df_tour[m + "_norm"] = normalize(df_tour[m].fillna(0))

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for world in ["TSP", "Itinerary"]:
        row = df_tour[df_tour["world"] == world].iloc[0]
        values = [row[m + "_norm"] for m in metrics]
        values = np.concatenate((values, [values[0]]))
        ax.plot(angles, values, label=world, linewidth=2)
        ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title("Comparaison touristique – TSP vs Itinéraire")
    ax.legend()
    return fig


# Vue synthétique
df_view = df_comp[[
    "world", "type", "solver",
    "distance", "time",
    "score_tourist", "variety", "lunch_time",
    "rating_logistic", "rating_tourist",
]]

print(df_view)

def side_by_side_plots(df_tsp, df_iti, matrix_name):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # TSP : distance vs temps
    sub_tsp = df_tsp[df_tsp["matrix"] == matrix_name]
    for _, row in sub_tsp.iterrows():
        axes[0].scatter(row["mean_cost"], row["mean_time"], s=120, label=row["solver"])
    axes[0].set_xlabel("Distance moyenne")
    axes[0].set_ylabel("Temps moyen (s)")
    axes[0].set_title("TSP – Distance vs Temps")
    axes[0].legend()

    # Itinéraire : score vs distance
    sub_iti = df_iti[df_iti["matrix"] == matrix_name]
    for _, row in sub_iti.iterrows():
        axes[1].scatter(row.get("mean_distance", np.nan), row["mean_score"], s=120, label=row["solver"])
    axes[1].set_xlabel("Distance moyenne")
    axes[1].set_ylabel("Score touristique")
    axes[1].set_title("Itinéraire – Score vs Distance")
    axes[1].legend()

    fig.tight_layout()
    return fig

df_tsp = pd.DataFrame(tsp_results)
df_iti = pd.DataFrame(iti_results)

from common.comparator import build_comparison_table, radar_comparison_tourist, side_by_side_plots

df_comp = build_comparison_table(df_tsp, df_iti)
st.subheader("Comparaison TSP vs Itinéraire")
st.dataframe(df_comp)

fig_radar = radar_comparison_tourist(df_comp)
st.pyplot(fig_radar)

fig_side = side_by_side_plots(df_tsp, df_iti, matrix_name="lyon_50")
st.pyplot(fig_side)