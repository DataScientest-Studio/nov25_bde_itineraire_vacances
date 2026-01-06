import streamlit as st
import polars as pl
import pydeck as pdk
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import altair as alt


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

MAIN_CATEGORY = ["Culture & Musées","Patrimoine & Monuments",
                "Gastronomie & Restauration","Bien-être & Santé", "Hébergement","Sports & Loisirs","Santé & Urgences",
                "Famille & Enfants","Culture & Musées","Transports",
                "Shopping & Artisanat","Nature & Paysages","Patrimoine & Monuments",
                "Services & Mobilité","Commerce & Shopping","Camping & Plein Air","Commodités",
                "Transports touristiques","Loisirs & Clubs","Événements & Traditions","Information Touristique"
            ]


NB_STEPS = len(STEPS)

DATA_PATH = Path("../data/processed/merged_20260106_135958.parquet")
pipeline = ItineraryPipeline(DATA_PATH)

# ---------------------------------------------------------
# INIT SESSION STATE
# ---------------------------------------------------------

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "cached_data" not in st.session_state:
    st.session_state.cached_data = {}

if "last_params" not in st.session_state:
    st.session_state.last_params = {}

if "auto_refresh" not in st.session_state:
    st.session_state.auto_refresh = True


def go_next():
    if st.session_state.current_step < NB_STEPS - 1:
        st.session_state.current_step += 1


def go_prev():
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1


def restart():
    st.session_state.current_step = 0
    st.session_state.cached_data = {}


def invalidate_cache():
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
min_score = st.sidebar.slider("Score minimum", 0.9, 0.92, 0.9995)
mode = st.sidebar.selectbox("Mode", ["walk", "drive"])
anchor_lat = st.sidebar.number_input("Latitude ancrage", value=48.8566)
anchor_lon = st.sidebar.number_input("Longitude ancrage", value=2.3522)

st.sidebar.markdown("---")
auto_refresh = st.sidebar.checkbox(
    "Auto-refresh (recalcul dès qu’un paramètre change)",
    value=st.session_state.auto_refresh
)
st.session_state.auto_refresh = auto_refresh

# Paramètres courants
params = {
    "commune": commune,
    "nb_days": nb_days,
    "min_score": min_score,
    "mode": mode,
    "anchor_lat": anchor_lat,
    "anchor_lon": anchor_lon,
}

# Détection des changements de paramètres
if params != st.session_state.last_params:
    if st.session_state.auto_refresh:
        invalidate_cache()
    st.session_state.last_params = params.copy()

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

def get_filtered(force_recompute: bool = False):
    if force_recompute or "filtered" not in st.session_state.cached_data:
        filtered_lf = pipeline._filter_pois(
            commune,
            MAIN_CATEGORY,
            min_score,
        )

        # filtrer sur itineraire True
        filtered_lf = filtered_lf.filter(pl.col("itineraire") ==  True)

        st.session_state.cached_data["filtered"] = filtered_lf.collect()
    return st.session_state.cached_data["filtered"]


def get_clustered(force_recompute: bool = False):
    if force_recompute or "clustered" not in st.session_state.cached_data:
        filtered = get_filtered(force_recompute=False)
        clustered_lf = pipeline._cluster_pois(
            filtered.lazy(), nb_days, anchor_lat, anchor_lon
        )
        st.session_state.cached_data["clustered"] = clustered_lf.collect()
    return st.session_state.cached_data["clustered"]


def get_balanced(force_recompute: bool = False):
    if force_recompute or "balanced" not in st.session_state.cached_data:
        clustered = get_clustered(force_recompute=False)
        balanced_lf = pipeline._balance_pois(
            clustered.lazy(), nb_days, mode
        )
        st.session_state.cached_data["balanced"] = balanced_lf.collect()
    return st.session_state.cached_data["balanced"]


def get_osrm_matrices_for_day0(force_recompute: bool = False):
    if force_recompute or "osrm_day0" not in st.session_state.cached_data:
        balanced = get_balanced(force_recompute=False)
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


def get_itineraries(force_recompute: bool = False):
    if force_recompute or "itineraries" not in st.session_state.cached_data:
        itineraries = pipeline.run(
            commune=commune,
            main_categories=
            MAIN_CATEGORY,
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
# Fonction utilitaire pour construire kes segments
# ---------------------------------------------------------

def build_paths(df):
    paths = []
    for i in range(1, df.height):
        paths.append({
            "from": [df["longitude"][i-1], df["latitude"][i-1]],
            "to":   [df["longitude"][i],   df["latitude"][i]],
            "order": int(df["visit_order"][i])
        })
    return paths

def build_paths_from_itinerary(df_it: pl.DataFrame):
    paths = []
    # On suppose que df_it est déjà dans l'ordre de visite
    for i in range(1, df_it.height):
        paths.append({
            "from": [df_it["longitude"][i-1], df_it["latitude"][i-1]],
            "to":   [df_it["longitude"][i],   df_it["latitude"][i]],
            "order": int(df_it["visit_order"][i]),
            "type": df_it["type"][i],
        })
    return paths


# ---------------------------------------------------------
# OUTILS DE VISUALISATION
# ---------------------------------------------------------

def pydeck_layer_points(df, color=[0, 120, 220, 160], radius=50):
    return pdk.Layer(
        "ScatterplotLayer",
        data=df.to_pandas(),
        get_position=["longitude", "latitude"],
        get_color=color,
        get_radius=radius,
        pickable=True,
    )


def pydeck_layer_points_by_day(df):
    pdf = df.to_pandas()
    if "day" not in pdf.columns:
        return pydeck_layer_points(df)

    pdf["color_r"] = (pdf["day"].astype(int) * 70) % 255
    pdf["color_g"] = 100
    pdf["color_b"] = 220
    pdf["color_a"] = 180

    return pdk.Layer(
        "ScatterplotLayer",
        data=pdf,
        get_position=["longitude", "latitude"],
        get_color="[color_r, color_g, color_b, color_a]",
        get_radius=60,
        pickable=True,
    )


def pydeck_layer_h3_cells(df, h3_col="h3_cell"):
    if h3_col not in df.columns:
        return None

    import h3
    import pandas as pd

    pdf = df.select(["latitude", "longitude", h3_col]).to_pandas()

    def h3_to_polygon(h):
        boundary = h3.h3_to_geo_boundary(h, geo_json=True)
        return [[lon, lat] for lat, lon in boundary]

    pdf["polygon"] = pdf[h3_col].apply(h3_to_polygon)
    pdf["color_r"] = 200
    pdf["color_g"] = 50
    pdf["color_b"] = 50
    pdf["color_a"] = 60

    return pdk.Layer(
        "PolygonLayer",
        data=pdf,
        get_polygon="polygon",
        get_fill_color="[color_r, color_g, color_b, color_a]",
        stroked=True,
        get_line_color=[200, 50, 50],
        line_width_min_pixels=1,
        pickable=True,
    )

def pydeck_layer_icons(df):
    return pdk.Layer(
        "IconLayer",
        df.to_pandas(),
        get_icon="icon_data",
        get_size=4,
        size_scale=10,
        get_position=["longitude", "latitude"],
        pickable=True,
    )


# ---------------------------------------------------------
# AFFICHAGE DE L’ÉTAPE COURANTE
# ---------------------------------------------------------

step = st.session_state.current_step

# Bouton de recalcul de l’étape affiché en haut de la page
st.markdown("### Contrôle de l’étape")
recompute_col1, recompute_col2 = st.columns([1, 3])
with recompute_col1:
    if st.button("🔄 Recalculer cette étape", use_container_width=True):
        if step == 0:
            st.session_state.cached_data.pop("filtered", None)
        elif step == 1:
            st.session_state.cached_data.pop("clustered", None)
        elif step == 2:
            st.session_state.cached_data.pop("balanced", None)

with recompute_col2:
    st.caption(
        "Ce bouton ne touche que les données de l’étape courante. "
        "L’auto-refresh, lui, invalide tout le cache quand un paramètre change."
    )

st.markdown("---")

if step == 0:
    # -------------------- FILTRAGE -------------------------
    st.header("1️⃣ Filtrage des POIs")

    st.markdown("""
**Objectif :**  
Ne garder que les POIs pertinents pour la commune et le score demandés.
""")

    filtered = get_filtered(force_recompute=False)

    col_t, col_m = st.columns([2, 3])

    with col_t:
        st.subheader("Table des POIs filtrés")
        st.dataframe(filtered.head(50))
        st.caption(f"{filtered.height} POIs après filtrage")

    with col_m:
        st.subheader("Carte des POIs filtrés")
        if filtered.height > 0:
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=filtered["latitude"].mean(),
                        longitude=filtered["longitude"].mean(),
                        zoom=11,
                    ),
                    tooltip={"text": "{nom_du_poi}\n{main_category} / {sub_category}"},
                    layers=[pydeck_layer_points(filtered)],
                )
            )
        else:
            st.info("Aucun POI après filtrage.")
    
    if "main_category" in filtered.columns:
        st.subheader("Répartition par catégorie")
        counts = (
            filtered
            .group_by("main_category")
            .len()
            .sort("len", descending=True)
            .to_pandas()
        )
        st.bar_chart(data=counts, x="main_category", y="len")

    st.header("📊 Analyse des scores normalisés")

    score_cols = [
        "density_commune_norm",
        "diversity_commune_norm",
        "popularity_norm",
        "proximity_commune_norm",
        "category_weight_norm",
        "opening_score_norm",
        "final_score"
    ]

    available_scores = [c for c in score_cols if c in df.columns]

    for col in available_scores:
        fig = px.histogram(df.to_pandas(), x=col, nbins=30, title=f"Distribution de {col}")
        st.plotly_chart(fig, use_container_width=True)


elif step == 1:
    # ----------------- CLUSTERING SPATIAL ------------------
    st.header("2️⃣ Clustering spatial (H3 + KMeans)")

    st.markdown("""
**Objectif :**  
Regrouper les POIs en journées cohérentes spatialement (1 cluster = 1 jour).
""")

    clustered = get_clustered(force_recompute=False)

    # Inspecteur de clusters
    st.subheader("Inspecteur de clusters")

    if clustered.height > 0:
        days = sorted(clustered["day"].unique().to_list())
        selected_day = st.selectbox("Jour à afficher", days, index=0)

        df_day = clustered.filter(pl.col("day") == selected_day)

        if "cluster_id" in clustered.columns:
            clusters = sorted(clustered["cluster_id"].unique().to_list())
            selected_cluster = st.selectbox(
                "Cluster (si disponible)",
                ["Tous"] + [str(c) for c in clusters],
                index=0
            )
            if selected_cluster != "Tous":
                df_day = df_day.filter(pl.col("cluster_id") == int(selected_cluster))
        else:
            selected_cluster = None

        col_t, col_m = st.columns([2, 3])

        with col_t:
            st.subheader("Table clusterisée (jour sélectionné)")
            st.dataframe(
                df_day.select(
                    [c for c in ["nom_du_poi", "latitude", "longitude", "day", "cluster_id"]
                     if c in df_day.columns]
                ).head(100)
            )
            st.caption(f"{df_day.height} POIs pour le jour {selected_day}")

        with col_m:
            st.subheader("Carte des clusters (par jour)")
            layer_points = pydeck_layer_points_by_day(df_day)

            # Mode debug H3
            debug_h3 = st.checkbox("Mode debug H3 (afficher les cellules H3)", value=False)
            layers = [layer_points]
            if debug_h3:
                h3_layer = pydeck_layer_h3_cells(df_day, h3_col="h3_cell")
                if h3_layer is not None:
                    layers.append(h3_layer)
                else:
                    st.info("Pas de colonne h3_cell disponible pour debug H3.")

            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=df_day["latitude"].mean(),
                        longitude=df_day["longitude"].mean(),
                        zoom=11,
                    ),
                    tooltip={"text": "{nom_du_poi}\n{main_category} / {sub_category}"},
                    layers=layers,
                )
            )

    else:
        st.info("Aucun POI à clusteriser.")

elif step == 2:
    # --------------------- BALANCING -----------------------
    st.header("3️⃣ Balancing (équilibrage par jour)")

    st.markdown("""
**Objectif :**  
Répartir les POIs pour que chaque journée soit **équilibrée** et **réaliste**.
""")

    balanced = get_balanced(force_recompute=False)

    # --- Carte de tous les POIs balanced ---
    st.subheader("Carte de tous les POIs équilibrés")
    st.caption(f"{balanced.height} POIs équilibrés")

    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=balanced["latitude"].mean(),
                longitude=balanced["longitude"].mean(),
                zoom=12,
            ),
            tooltip={"text": "{nom_du_poi}\nJour {day}\n{main_category} / {sub_category}"},
            layers=[pydeck_layer_points_by_day(balanced)],
        )
    )



    col_t, col_g = st.columns([2, 3])

    with col_t:
        st.subheader("Table équilibrée")
        st.dataframe(balanced.select(["nom_du_poi", "day"]).head(50))

        if balanced.height > 0:
            st.subheader("Carte par jour")
            days = sorted(balanced["day"].unique().to_list())
            selected_day_bal = st.selectbox("Jour à afficher (balancing)", days, index=0)
            df_day_bal = balanced.filter(pl.col("day") == selected_day_bal)

            # --- Carte ---
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=df_day_bal["latitude"].mean(),
                        longitude=df_day_bal["longitude"].mean(),
                        zoom=12,
                    ),
                    tooltip={"text": "{nom_du_poi}\n{main_category} / {sub_category}"},
                    layers=[pydeck_layer_points_by_day(df_day_bal)],
                )
            )

    with col_g:
        st.subheader("Nombre de POIs par jour")
        if balanced.height > 0:
            counts = balanced.group_by("day").count().to_pandas()
            st.bar_chart(data=counts, x="day", y="count")
        else:
            st.info("Aucun POI à équilibrer.")
        
        # Bar chart par catégorie ---
        st.subheader("Répartition par catégorie (jour sélectionné)")
        counts = (
            df_day_bal
            .group_by("main_category")
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
        )

        # st.bar_chart(
        #     data=counts.to_pandas(),  # st.bar_chart attend un DataFrame pandas
        #     x="main_category",
        #     y="count"
        # )
        pie = (
            alt.Chart(counts)
            .mark_arc()
            .encode(
                theta=alt.Theta("count:Q", stack=True),
                color=alt.Color("main_category:N"),
                tooltip=["main_category", "count"]
            )
        )

        st.altair_chart(pie, use_container_width=True)

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
# **Objectif :**  
# Fournir au front des itinéraires **riches et prêts à afficher** :
# - ordre
# - distances & durées
# - temps de visite
# - cumul
# - matin / après-midi
# - totaux journée
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


    st.subheader("Itinéraire du jour")

    df_it = itineraries[f"day_{selected_day}"]

    df_it = itineraries[f"day_{selected_day}"]

    if df_it.height > 0:
        df_pd = df_it.to_pandas()

        # --- Construire les segments ---
        paths = build_paths(df_it)

        line_layer = pdk.Layer(
            "LineLayer",
            paths,
            get_source_position="from",
            get_target_position="to",
            get_color=[30, 144, 255],
            get_width=4,
            pickable=True,
        )

        icon_layer = pdk.Layer(
            "IconLayer",
            df_pd,
            get_icon="icon_data",
            get_size=4,
            size_scale=10,
            get_position=["longitude", "latitude"],
            pickable=True,
        )

        st.pydeck_chart(
            pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=df_pd["latitude"].mean(),
                    longitude=df_pd["longitude"].mean(),
                    zoom=12,
                ),
                layers=[line_layer, icon_layer],
                tooltip={"text": "{nom_du_poi}\nArrivée: {arrival_time}"},
            )
        )



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