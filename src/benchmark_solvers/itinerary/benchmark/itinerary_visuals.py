import folium
import matplotlib.pyplot as plt
import numpy as np


def radar_itinerary(df, solver_name):
    metrics = [
        "mean_score",
        "variety",
        "lunch_time_score",
        "duration_score",
        "distance_score",
    ]
    values = df[df["solver"] == solver_name][metrics].iloc[0].values

    # Normalisation 0-1
    values = (values - values.min()) / (values.max() - values.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f"Radar Chart – {solver_name}")
    return fig


def timeline_itinerary(route, poi_df, scoring):
    import matplotlib.pyplot as plt

    times = []
    labels = []

    current_time = 9  # début de journée
    for poi in route:
        labels.append(poi_df.loc[poi_df.osrm_index == poi, "poi_id"].values[0])
        duration = scoring.activity_duration([poi]) / 60
        times.append(duration)
        current_time += duration

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.barh([0] * len(times), times, left=np.cumsum([0] + times[:-1]))
    ax.set_yticks([])
    ax.set_xticks(np.arange(9, 20))
    ax.set_title("Timeline de la journée")
    return fig


def folium_itinerary(route, poi_df):
    # centre de la carte
    lat = poi_df.latitude.mean()
    lon = poi_df.longitude.mean()
    m = folium.Map(location=[lat, lon], zoom_start=13)

    coords = []
    for poi in route:
        row = poi_df.loc[poi_df.osrm_index == poi].iloc[0]
        coords.append((row.latitude, row.longitude))
        folium.Marker(
            location=(row.latitude, row.longitude),
            popup=row["name"],
            tooltip=row["sub_category"],
        ).add_to(m)

    folium.PolyLine(coords, color="blue", weight=4).add_to(m)
    return m


def score_vs_distance(df, matrix_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    subset = df[df["matrix"] == matrix_name]

    for _, row in subset.iterrows():
        ax.scatter(row["mean_distance"], row["mean_score"], s=120, label=row["solver"])

    ax.set_xlabel("Distance totale (km)")
    ax.set_ylabel("Score touristique")
    ax.set_title(f"Score vs Distance – {matrix_name}")
    ax.legend()
    return fig


def boxplot_scores(runner_results, matrix_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    data = {}
    for r in runner_results:
        if r["matrix"] == matrix_name:
            data.setdefault(r["solver"], []).append(r["mean_score"])

    ax.boxplot(data.values(), labels=data.keys())
    ax.set_title(f"Distribution des scores – {matrix_name}")
    ax.set_ylabel("Score touristique")
    return fig


def variety_barplot(route, poi_df):
    df = poi_df.loc[poi_df.osrm_index.isin(route)]
    counts = df["sub_category"].value_counts()

    fig, ax = plt.subplots(figsize=(6, 4))
    counts.plot(kind="bar", ax=ax)
    ax.set_title("Variété des POIs visités")
    ax.set_ylabel("Nombre de POIs")
    return fig
