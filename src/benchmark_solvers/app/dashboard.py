import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from analysis.metrics import (
    compute_best_per_matrix,
    add_gap_column,
    stability_stats,
    pareto_front,
)
from analysis.plots import (
    boxplot_costs,
    boxplot_gaps,
    heatmap_gap,
    stability_plot,
    plot_pareto,
)

from streamlit_folium import st_folium
from map.folium_map import create_route_map


def load_results(path: str) -> pd.DataFrame:
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    elif path.endswith(".csv"):
        return pd.read_csv(path)
    else:
        raise ValueError("Format non supporté. Utilise .parquet ou .csv")


def main():
    st.title("Benchmark TSP / GA / Neo4j — Dashboard")

    st.sidebar.header("Configuration")
    results_path = st.sidebar.text_input(
        "Chemin vers les résultats (parquet/csv)",
        value="../data/processed/results_benchmark.parquet",
    )

    if not results_path:
        st.warning("Spécifie un fichier de résultats.")
        return

    try:
        df = load_results(results_path)
    except Exception as e:
        st.error(f"Erreur de chargement : {e}")
        return

    st.subheader("Aperçu brut des résultats")
    st.dataframe(df.head())

    # Gap
    best_per_matrix = compute_best_per_matrix(df)
    df = add_gap_column(df, best_per_matrix)

    st.subheader("Statistiques de stabilité par solver")
    stab = stability_stats(df)
    st.dataframe(stab)

    st.subheader("Front de Pareto (gap vs temps)")
    pareto = pareto_front(df)
    st.dataframe(pareto)

    # Choix des visualisations
    viz = st.sidebar.multiselect(
        "Visualisations",
        ["Boxplot coûts (solver)", "Boxplot coûts (matrix)",
         "Boxplot gaps (solver)", "Heatmap gap", "Stabilité", "Pareto"],
        default=["Boxplot coûts (solver)", "Heatmap gap", "Pareto"],
    )

    if "Boxplot coûts (solver)" in viz:
        st.markdown("### Boxplot des coûts par solver")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="solver", y="cost", ax=ax)
        st.pyplot(fig)

    if "Boxplot coûts (matrix)" in viz:
        st.markdown("### Boxplot des coûts par matrice")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="matrix", y="cost", ax=ax)
        st.pyplot(fig)

    if "Boxplot gaps (solver)" in viz:
        st.markdown("### Boxplot des gaps par solver")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x="solver", y="gap", ax=ax)
        st.pyplot(fig)

    if "Heatmap gap" in viz:
        st.markdown("### Heatmap des gaps moyens (solver × matrice)")
        pivot = df.groupby(["solver", "matrix"])["gap"].mean().reset_index()
        pivot = pivot.pivot(index="solver", columns="matrix", values="gap")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis", ax=ax)
        st.pyplot(fig)

    if "Stabilité" in viz:
        st.markdown("### Stabilité (écart-type du coût par solver et matrice)")
        stab_mat = df.groupby(["solver", "matrix"])["cost"].std().reset_index()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=stab_mat, x="matrix", y="cost", hue="solver", ax=ax)
        ax.set_ylabel("Écart-type du coût")
        st.pyplot(fig)

    if "Pareto" in viz:
        st.markdown("### Front de Pareto (gap_mean vs time_mean)")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=pareto,
            x="time_mean",
            y="gap_mean",
            hue="pareto",
            style="pareto",
            s=120,
            palette={True: "red", False: "gray"},
            ax=ax,
        )
        for _, row in pareto.iterrows():
            ax.text(row["time_mean"], row["gap_mean"], row["solver"], fontsize=8)
        ax.set_xlabel("Temps moyen (s)")
        ax.set_ylabel("Gap moyen")
        st.pyplot(fig)


    st.subheader("Visualisation cartographique (une matrice, un solver, un run)")

    # chargement des POIs à partir d'un parquet/csv
    pois_path = st.sidebar.text_input("Chemin POIs (parquet/csv)", value="data/pois.parquet")

    try:
        if pois_path.endswith(".parquet"):
            pois_df = pd.read_parquet(pois_path)
        else:
            pois_df = pd.read_csv(pois_path)
    except Exception as e:
        st.error(f"Erreur de chargement POIs : {e}")
        pois_df = None

    if pois_df is not None:
        matrix_sel = st.selectbox("Matrice", sorted(df["matrix"].unique()))
        solver_sel = st.selectbox("Solver", sorted(df["solver"].unique()))
        run_sel = st.number_input("Run", min_value=1, step=1, value=1)

        row = df[(df["matrix"] == matrix_sel) &
                (df["solver"] == solver_sel) &
                (df["run"] == run_sel)]

        if row.empty:
            st.warning("Aucune route trouvée pour cette combinaison.")
        else:
            route = row.iloc[0]["route"]
            m = create_route_map(pois_df, route, lat_col="lat", lon_col="lon", id_col="id")
            st_folium(m, width=700, height=500)



if __name__ == "__main__":
    main()