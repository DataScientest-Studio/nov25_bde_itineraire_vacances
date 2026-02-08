import streamlit as st
import polars as pl
import pydeck as pdk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

from pipeline import ItineraryPipeline
from features.osrm import OSRMClient

# ---------------------------------------------------------
# CONFIG GLOBALE
# ---------------------------------------------------------

st.set_page_config(
    page_title="Pipeline d’itinéraires – Démo interactive",
    layout="wide"
)

STEPS = [
    "Filtrage",
    "Clustering spatial",
    "Balancing",
    "OSRM (matrices)",
    "TSP",
    "Itinéraires enrichis",
]

NB_STEPS = len(STEPS)


# ---------------------------------------------------------
# INIT SESSION STATE
# ---------------------------------------------------------

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "cached_data" not in st.session_state:
    st.session_state.cached_data = {}


def go_next():
    if st.session_state.current_step < NB_STEPS - 1:
        st.session_state.current_step += 1


def go_prev():
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1


def restart():
    st.session_state.current_step = 0
    st.session_state.cached_data = {}


# ---------------------------------------------------------
# EN-TÊTE
# ---------------------------------------------------------

st.title("🧭 Démo interactive du pipeline d’itinéraires")

st.markdown("""
Cette application n’est **pas** l’interface utilisateur finale,  
mais une **présentation interactive** du pipeline :

1. Filtrage des POIs  
2. Clustering spatial  
3. Balancing  
4. OSRM (matrices de durées & distances)  
5. TSP (ordre optimal)  
6. Itinéraires enrichis (cumul, matin/après-midi, etc.)
""")


# ---------------------------------------------------------
# PARAMÈTRES UTILISATEUR (pour la démo)
# ---------------------------------------------------------

st.sidebar.header("Paramètres de la démo")

commune = st.sidebar.text_input("Commune", "Paris")
nb_days = st.sidebar.slider("Nombre de jours", 1, 7, 3)
min_score = st.sidebar.slider("Score minimum", 0.0, 1.0, 0.9)
mode = st.sidebar.selectbox("Mode", ["walk", "drive"])
anchor_lat = st.sidebar.number_input("Latitude ancrage", value=48.8566)
anchor_lon = st.sidebar.number_input("Longitude ancrage", value=2.3522)

DATA_PATH = Path("../data/processed/merged_20260106_135958.parquet")
pipeline = ItineraryPipeline(DATA_PATH)


# ---------------------------------------------------------
# BARRE DE PROGRESSION & BADGES D’ÉTAPES
# ---------------------------------------------------------

col1, col2 = st.columns([3, 1])

with col1:
    progress = (st.session_state.current_step + 1) / NB_STEPS
    st.progress(progress)

    step_labels = []
    for i, name in enumerate(STEPS):
        if i == st.session_state.current_step:
            step_labels.append(f"**➡️ {i+1}. {name}**")
        elif i < st.session_state.current_step:
            step_labels.append(f"✅ {i+1}. {name}")
        else:
            step_labels.append(f"{i+1}. {name}")

    st.markdown(" · ".join(step_labels))

with col2:
    st.write("")
    st.write("")
    st.button("🔁 Rejouer depuis le début", on_click=restart, use_container_width=True)


# ---------------------------------------------------------
# FONCTIONS UTILITAIRES POUR CHAQUE ÉTAPE
# ---------------------------------------------------------

def get_filtered():
    if "filtered" not in st.session_state.cached_data:
        filtered_lf = pipeline._filter_pois(
            commune,
            ["Culture & Musées", "Patrimoine & Monuments",
             "Gastronomie & Restauration", "Bien-être & Santé"],
            min_score,
        )
        st.session_state.cached_data["filtered"] = filtered_lf.collect()
    return st.session_state.cached_data["filtered"]


def get_clustered():
    if "clustered" not in st.session_state.cached_data:
        filtered = get_filtered()
        clustered_lf = pipeline._cluster_pois(
            filtered.lazy(), nb_days, anchor_lat, anchor_lon
        )
        st.session_state.cached_data["clustered"] = clustered_lf.collect()
    return st.session_state.cached_data["clustered"]


def get_balanced():
    if "balanced" not in st.session_state.cached_data:
        clustered = get_clustered()
        balanced_lf = pipeline._balance_pois(
            clustered.lazy(), nb_days, mode
        )
        st.session_state.cached_data["balanced"] = balanced_lf.collect()
    return st.session_state.cached_data["balanced"]


def get_osrm_matrices_for_day0():
    if "osrm_day0" not in st.session_state.cached_data:
        balanced = get_balanced()
        df_day0 = balanced.filter(pl.col("day") == 0)

        if df_day0.height == 0:
            st.session_state.cached_data["osrm_day0"] = None
        else:
            coords = [(anchor_lat, anchor_lon)] + list(
                zip(df_day0["latitude"], df_day0["longitude"])
            )
            client = OSRMClient()
            data = client.table(coords, annotations="duration,distance")
            st.session_state.cached_data["osrm_day0"] = (df_day0, data)
    return st.session_state.cached_data["osrm_day0"]


def get_itineraries():
    if "itineraries" not in st.session_state.cached_data:
        itineraries = pipeline.run(
            commune=commune,
            main_categories=[
                "Culture & Musées",
                "Patrimoine & Monuments",
                "Gastronomie & Restauration",
                "Bien-être & Santé",
            ],
            min_score=min_score,
            nb_days=nb_days,
            mode=mode,
            anchor_lat=anchor_lat,
            anchor_lon=anchor_lon,
            verbose=False,
        )
        st.session_state.cached_data["itineraries"] = itineraries
    return st.session_state.cached_data["itineraries"]


# ---------------------------------------------------------
# AFFICHAGE DE L’ÉTAPE COURANTE
# ---------------------------------------------------------

step = st.session_state.current_step

if step == 0:
    # -------------------- FILTRAGE -------------------------
    st.header("1️⃣ Filtrage des POIs")

    st.markdown("""
**Objectif :**  
Ne garder que les POIs pertinents pour la commune et le score demandés.
""")

    filtered = get_filtered()

    col_t, col_m = st.columns([2, 3])

    with col_t:
        st.subheader("Table des POIs filtrés")
        st.dataframe(filtered.head(50))
        st.caption(f"{filtered.height} POIs après filtrage")

    with col_m:
        st.subheader("Carte des POIs filtrés")
        if filtered.height > 0:
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=filtered["latitude"].mean(),
                    longitude=filtered["longitude"].mean(),
                    zoom=11,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=filtered.to_pandas(),
                        get_position=["longitude", "latitude"],
                        get_color=[0, 120, 220, 160],
                        get_radius=50,
                    )
                ],
            ))
        else:
            st.info("Aucun POI après filtrage.")

elif step == 1:
    # ----------------- CLUSTERING SPATIAL ------------------
    st.header("2️⃣ Clustering spatial (H3 + KMeans)")

    st.markdown("""
**Objectif :**  
Regrouper les POIs en journées cohérentes spatialement (1 cluster = 1 jour).
""")

    clustered = get_clustered()

    col_t, col_m = st.columns([2, 3])

    with col_t:
        st.subheader("Table clusterisée")
        st.dataframe(clustered.select(["nom_du_poi", "latitude", "longitude", "day"]).head(50))
        st.caption("Colonne `day` = jour attribué au POI")

    with col_m:
        st.subheader("Carte des clusters")
        if clustered.height > 0:
            pdf = clustered.to_pandas()
            pdf["color"] = pdf["day"].astype(int) * 40 + 50
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=clustered["latitude"].mean(),
                    longitude=clustered["longitude"].mean(),
                    zoom=11,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=pdf,
                        get_position=["longitude", "latitude"],
                        get_color="[color, 100, 200, 180]",
                        get_radius=60,
                    )
                ],
            ))
        else:
            st.info("Aucun POI à clusteriser.")

elif step == 2:
    # --------------------- BALANCING -----------------------
    st.header("3️⃣ Balancing (équilibrage par jour)")

    st.markdown("""
**Objectif :**  
Répartir les POIs pour que chaque journée soit **équilibrée** et **réaliste**.
""")

    balanced = get_balanced()

    col_t, col_g = st.columns([2, 3])

    with col_t:
        st.subheader("Table équilibrée")
        st.dataframe(balanced.select(["nom_du_poi", "day"]).head(50))

    with col_g:
        st.subheader("Nombre de POIs par jour")
        if balanced.height > 0:
            counts = balanced.group_by("day").count().to_pandas()
            st.bar_chart(data=counts, x="day", y="count")
        else:
            st.info("Aucun POI à équilibrer.")

elif step == 3:
    # ------------------- OSRM MATRICES ---------------------
    st.header("4️⃣ OSRM – Matrices de durées & distances")

    st.markdown("""
**Objectif :**  
Calculer les durées et distances entre tous les points d'une journée (matrice NxN).
""")

    res = get_osrm_matrices_for_day0()

    if res is None:
        st.info("Pas de POIs pour le jour 0, impossible de montrer la matrice.")
    else:
        df_day0, data = res
        dur = np.array(data["durations"])
        dist = np.array(data["distances"])

        col_d, col_s = st.columns([2, 2])

        with col_d:
            st.subheader("Heatmap des durées (jour 1)")
            fig, ax = plt.subplots()
            sns.heatmap(dur, ax=ax)
            st.pyplot(fig)

        with col_s:
            st.subheader("Heatmap des distances (jour 1)")
            fig2, ax2 = plt.subplots()
            sns.heatmap(dist, ax=ax2)
            st.pyplot(fig2)

        st.caption("Les indices correspondent à : [ancrage] + liste des POIs du jour 1.")

elif step == 4:
    # ------------------------ TSP --------------------------
    st.header("5️⃣ TSP – Ordre optimal de visite")

    st.markdown("""
**Objectif :**  
Trouver l'ordre de visite qui minimise le temps de trajet pour chaque jour.
""")

    itineraries = get_itineraries()

    tabs = st.tabs([f"Jour {i+1}" for i in range(nb_days)])

    for i, tab in enumerate(tabs):
        with tab:
            key = f"day_{i+1}"
            df = itineraries.get(key, pl.DataFrame())
            if df.height == 0:
                st.info("Aucun POI pour ce jour.")
                continue

            st.subheader(f"Itinéraire optimisé – Jour {i+1}")
            st.dataframe(df.select(["order", "name", "period", "cum_total_duration"]).head(50))

            st.subheader("Carte de l’itinéraire")
            pdf = df.to_pandas()
            path = list(zip(pdf["longitude"], pdf["latitude"]))

            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=df["latitude"].mean(),
                    longitude=df["longitude"].mean(),
                    zoom=12,
                ),
                layers=[
                    pdk.Layer(
                        "PathLayer",
                        data=[{"path": path, "color": [255, 0, 0]}],
                        get_path="path",
                        get_color="color",
                        width_scale=20,
                        width_min_pixels=3,
                    ),
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=pdf,
                        get_position=["longitude", "latitude"],
                        get_color=[0, 150, 0, 180],
                        get_radius=80,
                    )
                ],
            ))

elif step == 5:
    # ---------------- ITINÉRAIRES ENRICHIS -----------------
    st.header("6️⃣ Itinéraires enrichis pour le front")

    st.markdown("""
**Objectif :**  
Fournir au front des itinéraires **riches et prêts à afficher** :
- ordre
- distances & durées
- temps de visite
- cumul
- matin / après-midi
- totaux journée
""")

    itineraries = get_itineraries()

    for day_idx in range(nb_days):
        key = f"day_{day_idx+1}"
        df = itineraries.get(key, pl.DataFrame())

        st.subheader(f"Jour {day_idx+1}")

        if df.height == 0:
            st.info("Aucun POI pour ce jour.")
            continue

        col_t, col_stats = st.columns([3, 2])

        with col_t:
            st.dataframe(
                df.select([
                    "order", "name", "period",
                    "distance_from_prev", "duration_from_prev",
                    "visit_time", "cum_total_duration"
                ]).head(50)
            )

        with col_stats:
            st.markdown("**Résumé de la journée**")
            total_distance = df["day_total_distance"][0]
            total_duration = df["day_total_duration"][0]

            st.metric("Distance totale (m)", f"{int(total_distance):,}".replace(",", " "))
            st.metric("Durée totale (trajets + visites, s)", f"{int(total_duration):,}".replace(",", " "))

            st.markdown("**Répartition matin / après-midi**")
            counts = df.group_by("period").count().to_pandas()
            st.bar_chart(data=counts, x="period", y="count")


# ---------------------------------------------------------
# NAVIGATION BAS DE PAGE
# ---------------------------------------------------------

st.markdown("---")
col_prev, col_info, col_next = st.columns([1, 4, 1])

with col_prev:
    st.button("⬅️ Étape précédente", on_click=go_prev, use_container_width=True)

with col_info:
    st.markdown(
        f"Étape **{st.session_state.current_step+1}/{NB_STEPS}** – **{STEPS[st.session_state.current_step]}**",
        unsafe_allow_html=True
    )

with col_next:
    st.button("➡️ Étape suivante", on_click=go_next, use_container_width=True)