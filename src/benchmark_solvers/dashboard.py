import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_folium import st_folium

from loaders.loader import (
    load_all_matrices_and_pois,
    validate_all,
)

from analysis.metrics import (
    compute_best_per_matrix,
    add_gap_column,
    stability_stats,
    pareto_front,
    solver_ranking_by_distance
)

from analysis.plots import (
    boxplot_costs,
    boxplot_gaps,
    heatmap_gap,
    stability_plot,
    plot_pareto,
    radar_chart
)


from map.folium_map import create_route_map


from streamlit_folium import st_folium
from pathlib import Path

OSRM_MATRIX_PATH = Path("../../data/processed")


# ---------------------------------------------------------
# CONFIGURATION : chemins vers matrices + POIs
# ---------------------------------------------------------

matrix_paths = [
        OSRM_MATRIX_PATH / "df_osrm_dist" / "14_merged_20260109_115825.parquet",
        OSRM_MATRIX_PATH / "df_osrm_dist" / "30_merged_20260109_115553.parquet",
        OSRM_MATRIX_PATH / "df_osrm_dist" / "58_merged_20260109_115141.parquet",
        OSRM_MATRIX_PATH / "df_osrm_dist" / "100_merged_20260109_115720.parquet",
    ]

pois_paths = [
        OSRM_MATRIX_PATH / "df_clustered" / "14_merged_20260109_115825.parquet",
        OSRM_MATRIX_PATH / "df_clustered" / "30_merged_20260109_115553.parquet",
        OSRM_MATRIX_PATH / "df_clustered" / "58_merged_20260109_115141.parquet",
        OSRM_MATRIX_PATH / "df_clustered" / "100_merged_20260109_115720.parquet",
    ]

matrices, pois = load_all_matrices_and_pois(matrix_paths, pois_paths)

# ---------------------------------------------------------
# CHARGEMENT DES DONNÉES
# ---------------------------------------------------------
st.sidebar.title("Chargement des données")

with st.sidebar:
    st.write("Chargement des matrices OSRM + POIs…")

matrices, pois = load_all_matrices_and_pois(matrix_paths, pois_paths)
#matrices, pois = load_all_matrices_and_pois(MATRIX_PATHS, POIS_PATHS)
validate_all(matrices, pois)

st.sidebar.success("Matrices + POIs chargés et validés.")


# ---------------------------------------------------------
# CHARGEMENT DES RÉSULTATS
# ---------------------------------------------------------
results_path = st.sidebar.text_input(
    "Fichier résultats (parquet/csv)",
    value="../../data/processed/results_benchmark.parquet",
)

if results_path.endswith(".parquet"):
    df = pd.read_parquet(results_path)
else:
    df = pd.read_csv(results_path)

# Ajouter gap
best_per_matrix = compute_best_per_matrix(df)
df = add_gap_column(df, best_per_matrix)

# ---------------------------------------------------------
# INTERFACE PRINCIPALE
# ---------------------------------------------------------
st.title("Dashboard Benchmark Solveurs TSP / GA / Neo4j")

st.subheader("Aperçu des résultats")
st.dataframe(df.head())
st.write("Distances moyennes par solveur (km)")
st.dataframe(df.groupby("solver")["distance_km"].mean().round(2))


# ---------------------------------------------------------
# VISUALISATIONS
# ---------------------------------------------------------
st.header("Visualisations")


viz = st.multiselect(
    "Choisir les visualisations",
    ["Boxplot coûts",  "Boxplot distances (km)","Boxplot gaps", "Heatmap gaps", "Stabilité", "Pareto", "Radar chart"],
    default=["Boxplot coûts", "Heatmap gaps", "Pareto"],
)


if "Boxplot coûts" in viz:
    st.subheader("Boxplot des coûts par solveur")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="solver", y="cost", ax=ax)
    st.pyplot(fig)

if "Boxplot distances (km)" in viz:
    st.subheader("Boxplot des distances totales (km)")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="solver", y="distance_km", ax=ax)
    st.pyplot(fig)


if "Boxplot gaps" in viz:
    st.subheader("Boxplot des gaps par solveur")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="solver", y="gap", ax=ax)
    st.pyplot(fig)

if "Heatmap gaps" in viz:
    st.subheader("Heatmap des gaps moyens")
    heatmap_gap(df)

if "Stabilité" in viz:
    st.subheader("Stabilité (écart-type)")
    stability_plot(df)

if "Pareto" in viz:
    st.subheader("Front de Pareto")
    pareto = pareto_front(df)
    plot_pareto(pareto)

if "Radar chart" in viz:
    st.subheader("Radar chart (distance / gap / temps / stabilité)")
    fig = radar_chart(df)
    st.pyplot(fig)


st.header("Classement des solveurs par distance (km)")
ranking = solver_ranking_by_distance(df)
st.dataframe(ranking.style.highlight_min("distance_km", color="lightgreen"))


# ---------------------------------------------------------
# CARTE FOLIUM
# ---------------------------------------------------------
st.header("Visualisation cartographique")

matrix_sel = st.selectbox("Choisir une matrice", list(matrices.keys()))
solver_sel = st.selectbox("Choisir un solveur", sorted(df["solver"].unique()))
run_sel = st.number_input("Run", min_value=1, step=1, value=1)

row = df[
    (df["matrix"] == matrix_sel)
    & (df["solver"] == solver_sel)
    & (df["run"] == run_sel)
]


if row.empty:
    st.warning("Aucune route trouvée pour cette combinaison.")
else:
    st.write(f"Distance totale : {row.iloc[0]['distance_km']:.2f} km")
    route = row.iloc[0]["route"]
    pois_df = pois[matrix_sel]

    m = create_route_map(pois_df, route)
    st_folium(m, width=700, height=500)
