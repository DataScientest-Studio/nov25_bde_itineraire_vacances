import matplotlib.pyplot as plt
import numpy as np


def radar_comparison_tourist(df_comp):
    df_tour = df_comp[df_comp["type"] == "tourist"].copy()

    metrics = ["distance", "score_tourist", "variety", "lunch_time"]
    labels = ["distance", "score", "variety", "lunch"]

    # Normalisation
    for m in metrics:
        df_tour[m + "_norm"] = (df_tour[m].fillna(0) - df_tour[m].min()) / (
            df_tour[m].max() - df_tour[m].min() + 1e-9
        )

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


def rating_barplot(df_comp):
    fig, ax = plt.subplots(figsize=(6, 4))

    df = df_comp[df_comp["type"] == "tourist"]

    ax.bar(df["world"], df["rating_tourist"])
    ax.set_title("Rating touristique – TSP vs Itinéraire")
    ax.set_ylabel("Rating (0–1)")
    return fig


def side_by_side_plots(df_tsp, df_iti, matrix_name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # TSP
    sub_tsp = df_tsp[df_tsp["matrix"] == matrix_name]
    for _, row in sub_tsp.iterrows():
        axes[0].scatter(row["mean_cost"], row["mean_time"], s=120, label=row["solver"])
    axes[0].set_title("TSP – Distance vs Temps")
    axes[0].set_xlabel("Distance")
    axes[0].set_ylabel("Temps (s)")
    axes[0].legend()

    # Itinéraire
    sub_iti = df_iti[df_iti["matrix"] == matrix_name]
    for _, row in sub_iti.iterrows():
        axes[1].scatter(
            row.get("mean_distance", np.nan),
            row["mean_score"],
            s=120,
            label=row["solver"],
        )
    axes[1].set_title("Itinéraire – Score vs Distance")
    axes[1].set_xlabel("Distance")
    axes[1].set_ylabel("Score touristique")
    axes[1].legend()

    fig.tight_layout()
    return fig
