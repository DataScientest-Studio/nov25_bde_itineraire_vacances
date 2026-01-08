import asyncio
import json
import os
import time
from pathlib import Path

import polars as pl
import pydeck as pdk
import streamlit as st
import plotly.express as px

# -------------------------------------------------------------------
# CONFIG VISUEL / PAGE
# -------------------------------------------------------------------

st.set_page_config(
    page_title="Itinéraires OSRM – Pipeline",
    page_icon="🧭",
    layout="wide",
)

# -------------------------------------------------------------------
# IMPORTS MÉTIER (à adapter à ton projet)
# -------------------------------------------------------------------

from src.features.pipeline import ItineraryPipeline
from src.features.osrm import OSRMClientAsync
from src.features.post_clustering import build_osrm_matrices_async
from src.features.itinerary_optimizer import ItineraryOptimizer
from src.features.spatial_clustering import SpatialClusterer


# -------------------------------------
# OSRM
# -------------------------------------
osrm_client = OSRMClientAsync()

# ---------------------------------------
# MAIN CATEGROY
# ---------------------------------------
MAIN_CATEGORY = ["Culture & Musées","Patrimoine & Monuments",
                "Gastronomie & Restauration","Bien-être & Santé", "Hébergement","Sports & Loisirs","Santé & Urgences",
                "Famille & Enfants","Culture & Musées","Transports",
                "Shopping & Artisanat","Nature & Paysages","Patrimoine & Monuments",
                "Services & Mobilité","Commerce & Shopping","Camping & Plein Air","Commodités",
                "Transports touristiques","Loisirs & Clubs","Événements & Traditions","Information Touristique"
            ]

# ---------------------------------------
# PARAMETRES PYDECK
# -------------------------------------
DAY_COLORS = [
    [255, 0, 0],      # Jour 0 - rouge
    [0, 150, 255],    # Jour 1 - bleu
    [0, 200, 100],    # Jour 2 - vert
    [255, 165, 0],    # Jour 3 - orange
    [150, 0, 255],    # Jour 4 - violet
    [255, 105, 180],  # Jour 5 - rose
]

def get_day_color(day: int) -> list:
    return DAY_COLORS[day % len(DAY_COLORS)]


# -------------------------------------------------------------------
# CONSTANTES / CHEMINS
# -------------------------------------------------------------------
POIS_PATH = Path("data/processed/merged_20260108_174125.parquet")
CACHE_DIR = Path("cache_osrm")
CACHE_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------------
# INITIALISATION SESSION STATE
# -------------------------------------------------------------------

if "cached_data" not in st.session_state:
    st.session_state.cached_data = {}

if "current_step" not in st.session_state:
    st.session_state.current_step = 0

if "profiling" not in st.session_state:
    st.session_state.profiling = {}

# -------------------------------------------------------------------
# OUTILS CACHE MÉMOIRE + DISQUE
# -------------------------------------------------------------------

def cache_get(key):
    return st.session_state.cached_data.get(key)

def cache_set(key, value):
    st.session_state.cached_data[key] = value

def cache_clear(key):
    st.session_state.cached_data.pop(key, None)

def disk_cache_path(key, ext="json"):
    return CACHE_DIR / f"{key}.{ext}"

def disk_cache_load(key, ext="json"):
    path = disk_cache_path(key, ext)
    if not path.exists():
        return None
    if ext == "json":
        return json.loads(path.read_text(encoding="utf-8"))
    if ext == "parquet":
        return pl.read_parquet(path)
    return None

def disk_cache_save(key, value, ext="json"):
    path = disk_cache_path(key, ext)
    if ext == "json":
        path.write_text(json.dumps(value), encoding="utf-8")
    elif ext == "parquet":
        value.write_parquet(path)

# -------------------------------------------------------------------
# PROFILING
# -------------------------------------------------------------------

def profile_step(name):
    """Décorateur pour profiler une fonction de step."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            t0 = time.time()
            result = func(*args, **kwargs)
            dt = time.time() - t0
            st.session_state.profiling[name] = dt
            return result
        return wrapper
    return decorator

def show_profiling_summary():
    if not st.session_state.profiling:
        st.info("Aucune mesure de temps pour l'instant.")
        return
    st.subheader("⏱ Profiling du pipeline")
    data = [
        {"étape": name, "temps_sec": round(dt, 3)}
        for name, dt in st.session_state.profiling.items()
    ]
    df_prof = pl.DataFrame(data)
    st.dataframe(df_prof)


# -------------------------------------------------------------------
# LAYERS PYDECK
# -------------------------------------------------------------------

def pydeck_layer_points(df: pl.DataFrame, color=(0, 128, 255)):
    df_pd = df.select(["longitude", "latitude", "nom_du_poi", "main_category"]).to_pandas()
    return pdk.Layer(
        "ScatterplotLayer",
        data=df_pd,
        get_position=["longitude", "latitude"],
        get_radius=40,
        get_color=list(color),
        pickable=True,
    )

def pydeck_layer_points_by_day(df: pl.DataFrame):
    df_pd = df.select(["longitude", "latitude", "nom_du_poi", "main_category", "day"]).to_pandas()
    return pdk.Layer(
        "ScatterplotLayer",
        data=df_pd,
        get_position=["longitude", "latitude"],
        get_radius=40,
        get_color="[cluster_id * 40 % 255, 100, 200]",
        pickable=True,
    )

def render_premium_map(
    df_itinerary: pl.DataFrame,
    routes_geojson: dict,
    selected_day: int | str = "Tous"
):
    """
    df_itinerary : DataFrame Polars avec au minimum:
        - day
        - latitude
        - longitude
        - nom_du_poi
        - main_category
        - final_score
        - visit_order
    routes_geojson : dict {day: geojson LineString}
    selected_day : int ou "Tous"
    """

    df_itinerary_pd = df_itinerary.to_pandas()

    layers = []

    # --- 1) Choix des jours à afficher ---
    if selected_day == "Tous":
        days_to_show = sorted(routes_geojson.keys())
    else:
        days_to_show = [selected_day]

    # --- 2) Layers routes OSRM ---
    for day in days_to_show:
        if day not in routes_geojson:
            continue

        route_geojson = routes_geojson[day]
        color = get_day_color(day)

        layers.append(
            pdk.Layer(
                "PathLayer",
                data=[{
                    "path": route_geojson["coordinates"],
                    "day": day,
                }],
                get_path="path",
                get_color=color,
                width_scale=3,
                width_min_pixels=3,
                pickable=False,
            )
        )

    # --- 3) Layers POIs (restaurants vs autres) ---
    df_pois_all_days = df_itinerary_pd[df_itinerary_pd["cluster_id"].isin(days_to_show)]

    # flag restaurant
    if "main_category" in df_pois_all_days.columns:
        is_resto = df_pois_all_days["main_category"].str.contains(
            "rest", case=False, na=False
        )
    else:
        is_resto = [False] * len(df_pois_all_days)

    df_restos = df_pois_all_days[is_resto]
    df_others = df_pois_all_days[~is_resto]

    # autres POIs
    if not df_others.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_others,
                get_position=["longitude", "latitude"],
                get_radius=40,
                get_fill_color=[80, 80, 200, 180],
                pickable=True,
            )
        )

    # restaurants (plus gros, autre couleur)
    if not df_restos.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=df_restos,
                get_position=["longitude", "latitude"],
                get_radius=60,
                get_fill_color=[220, 60, 60, 220],
                pickable=True,
            )
        )

    # --- 4) Tooltip riche ---
    tooltip = {
        "html": """
        <b>{nom_du_poi}</b><br/>
        Jour : {day} – Ordre : {visit_order}<br/>
        Catégorie : {main_category}<br/>
        Score : {final_score}
        """,
        "style": {
            "backgroundColor": "#1f2a3c",
            "color": "white",
            "fontSize": "12px",
        },
    }

    # --- 5) Vue centrée (on prend le premier point applicable) ---
    if not df_pois_all_days.empty:
        lat0 = df_pois_all_days["latitude"].iloc[0]
        lon0 = df_pois_all_days["longitude"].iloc[0]
    else:
        # fallback (Lyon)
        lat0, lon0 = 45.75, 4.85

    view_state = pdk.ViewState(
        latitude=lat0,
        longitude=lon0,
        zoom=12,
        pitch=45,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip=tooltip,
    )

    return deck

# -------------------------------------------------------------------
# INITIALISATION PIPELINE
# -------------------------------------------------------------------

pipeline = ItineraryPipeline(pois_path=POIS_PATH)

# -------------------------------------------------------------------
# PARAMÈTRES DANS LA SIDEBAR
# -------------------------------------------------------------------

st.sidebar.title("⚙️ Paramètres")

st.sidebar.markdown("### Zone / Scoring")
commune = st.sidebar.text_input("Commune", value="Paris")
main_categories = st.sidebar.multiselect(
    "Catégories principales",
    options=MAIN_CATEGORY,
    default=MAIN_CATEGORY,
)
min_score = st.sidebar.slider("Score minimum", 0.0, 1.0, 0.3, 0.05)

st.sidebar.markdown("### Journées & Ancre")
nb_days = st.sidebar.slider("Nombre de jours", 1, 7, 3)
anchor_lat = st.sidebar.number_input("latitude ancre", value=48.8566, format="%.6f")
anchor_lon = st.sidebar.number_input("longitude ancre", value=2.3522, format="%.6f")

st.sidebar.markdown("### OSRM / Debug / Profiling")
osrm_mode = st.sidebar.selectbox("Mode de déplacement", ["walk", "car"], index=0)
debug_osrm = st.sidebar.checkbox("Mode debug OSRM", value=False)
show_prof = st.sidebar.checkbox("Afficher le profiling", value=True)

st.sidebar.markdown("---")
step_labels = [
    "0 - Filtrage",
    "1 - Clustering",
    "2 - POIs OSRM-ready",
    "3 - Matrices OSRM",
    "4 - Itinéraire optimisé",
    "5 - Routes OSRM",
]
st.session_state.current_step = st.sidebar.radio(
    "Étape du pipeline",
    list(range(len(step_labels))),
    format_func=lambda i: step_labels[i],
    index=st.session_state.current_step,
)

# Auto-refresh : si les paramètres clés changent, on invalide le cache global
# (tu peux raffiner selongitude tes besoins)
def clear_all_cache():
    st.session_state.cached_data = {}
    st.session_state.profiling = {}

if st.sidebar.button("♻️ Réinitialiser tout le cache"):
    clear_all_cache()
    st.rerun()

# -------------------------------------------------------------------
# FONCTIONS GET_* (avec cache mémoire + disque)
# -------------------------------------------------------------------

@profile_step("0_filtrage")
def get_filtered(force_recompute=False):
    if not force_recompute:
        cached = cache_get("filtered")
        if cached is not None:
            return cached

    lf_filtered = pipeline._filter_pois(
        commune=commune,
        main_categories=main_categories,
        min_score=min_score,
    )
    df = lf_filtered.collect()
    cache_set("filtered", df)
    return df

@profile_step("1_clustering")
def get_clustered(force_recompute=False):
    if not force_recompute:
        cached = cache_get("clustered")
        if cached is not None:
            return cached

    filtered = get_filtered()
    # si ton _cluster_pois attend un LazyFrame, adapte ici
    df = pipeline._cluster_pois(
        filtered_lf=filtered.lazy(),
        nb_days=nb_days,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
    )
    df = df.collect()
    cache_set("clustered", df)
    return df

@profile_step("2_osrm_ready")
def get_osrm_ready_pois(force_recompute=False):
    if not force_recompute:
        cached = cache_get("osrm_ready")
        if cached is not None:
            return cached

    df_clustered = get_clustered()
    df_osrm_ready = pipeline._build_osrm_ready_pois(
        df_prepared=df_clustered,
        mode=osrm_mode,
        max_pois_per_cluster=40,
        min_score=0.2,
        target_restaurants=2,
        restaurant_category="Gastronomie & Restauration",
    )

    df_osrm_ready = df_osrm_ready.join(
        df_clustered.select([
            "poi_id",
            "nom_du_poi",
        ]),
        on="poi_id",
        how="left"
    )

    cache_set("osrm_ready", df_osrm_ready)
    return df_osrm_ready

@profile_step("3_osrm_matrices")
def get_osrm_matrices(force_recompute=False):
    if not force_recompute:
        cached = cache_get("osrm_matrices")
        if cached is not None:
            return cached

    df_osrm_ready = get_osrm_ready_pois()

    # cache disque pour les matrices (clé basée sur nombre de points)
    key = f"matrices_{df_osrm_ready.height}"

    # Tentative de chargement depuis disque
    df_clustered = disk_cache_load(f"{key}_clustered", ext="parquet")
    df_osrm_dist = disk_cache_load(f"{key}_dist", ext="parquet")
    df_osrm_dur = disk_cache_load(f"{key}_dur", ext="parquet")

    if df_clustered is not None and df_osrm_dist is not None and df_osrm_dur is not None:
        cache_set("osrm_matrices", (df_clustered, df_osrm_dist, df_osrm_dur))
        return df_clustered, df_osrm_dist, df_osrm_dur

    # Sinon : calcul OSRM
    df_clustered, df_osrm_dist, df_osrm_dur = asyncio.run(
        build_osrm_matrices_async(df_osrm_ready, osrm_client)
    )

    # Sauvegarde disque
    disk_cache_save(f"{key}_clustered", df_clustered, ext="parquet")
    disk_cache_save(f"{key}_dist", df_osrm_dist, ext="parquet")
    disk_cache_save(f"{key}_dur", df_osrm_dur, ext="parquet")

    cache_set("osrm_matrices", (df_clustered, df_osrm_dist, df_osrm_dur))
    return df_clustered, df_osrm_dist, df_osrm_dur

@profile_step("4_itinerary")
def get_itinerary(force_recompute=False):
    if not force_recompute:
        cached = cache_get("itinerary")
        if cached is not None:
            return cached

    df_clustered, df_osrm_dist, df_osrm_dur = get_osrm_matrices()

    optimizer = ItineraryOptimizer.from_list_matrix(
        df_pois=df_clustered,
        matrix=df_osrm_dur.to_numpy(),
        metric="duration",
    )

    df_itinerary = optimizer.solve_all_days()

    cache_set("optimizer", optimizer)
    cache_set("itinerary", df_itinerary)
    return df_itinerary

@profile_step("5_routes_osrm")
def get_osrm_routes(force_recompute=False):
    if not force_recompute:
        cached = cache_get("osrm_routes")
        if cached is not None:
            return cached

    optimizer = cache_get("optimizer")
    df_itinerary = cache_get("itinerary")

    if optimizer is None or df_itinerary is None:
        get_itinerary()
        optimizer = cache_get("optimizer")
        df_itinerary = cache_get("itinerary")

    osrm_client = OSRMClientAsync()
    osrm_client.debug = debug_osrm

    routes_geojson = asyncio.run(
        optimizer.build_geojson_all_days_async(df_itinerary, osrm_client)
    )

    cache_set("osrm_routes", routes_geojson)
    return routes_geojson

# -------------------------------------------------------------------
# UI : BOUTON RECOMPUTE ÉTAPE COURANTE
# -------------------------------------------------------------------

step = st.session_state.current_step

st.markdown("### Contrôle de l’étape")
recompute_col1, recompute_col2 = st.columns([1, 3])

with recompute_col1:
    if st.button("🔄 Recalculer cette étape", use_container_width=True):
        if step == 0:
            cache_clear("filtered")
        elif step == 1:
            cache_clear("clustered")
        elif step == 2:
            cache_clear("osrm_ready")
        elif step == 3:
            cache_clear("osrm_matrices")
        elif step == 4:
            cache_clear("itinerary")
            cache_clear("optimizer")
        elif step == 5:
            cache_clear("osrm_routes")
        st.rerun()

with recompute_col2:
    st.caption(
        "Ce bouton ne touche que les données de l’étape courante. "
        "Le bouton de réinitialisation dans la sidebar invalide tout le cache."
    )

st.markdown("---")

# -------------------------------------------------------------------
# ÉTAPES 0 & 1 
# -------------------------------------------------------------------
if step == 0:
    st.header("1️⃣ Filtrage des POIs")

    filtered = get_filtered()

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
        "final_score",
    ]
    available_scores = [c for c in score_cols if c in filtered.columns]
    for col in available_scores:
        fig = px.histogram(filtered.to_pandas(), x=col, nbins=30, title=f"Distribution de {col}")
        st.plotly_chart(fig, use_container_width=True)

elif step == 1:
    st.header("2️⃣ Clustering spatial (H3 + KMeans)")

    clustered = get_clustered()

    st.subheader("Inspecteur de clusters")

    if clustered.height > 0:
        days = sorted(clustered["day"].unique().to_list())
        selected_day = st.selectbox("Jour à afficher", days, index=0)

        df_day = clustered.filter(pl.col("day") == selected_day)

        col_t, col_m = st.columns([2, 3])

        with col_t:
            st.subheader("Table clusterisée (jour sélectionné)")
            st.dataframe(
                df_day.select(
                    [
                        c
                        for c in ["nom_du_poi", "latitude", "longitude", "day", "cluster_id"]
                        if c in df_day.columns
                    ]
                ).head(100)
            )
            st.caption(f"{df_day.height} POIs pour le jour {selected_day}")

        with col_m:
            st.subheader("Carte des clusters (par jour)")
            layer_points = pydeck_layer_points_by_day(df_day)

            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=df_day["latitude"].mean(),
                        longitude=df_day["longitude"].mean(),
                        zoom=11,
                    ),
                    tooltip={"text": "{nom_du_poi}\n{main_category} / {sub_category}"},
                    layers=[layer_points],
                )
            )
    else:
        st.info("Aucun POI à clusteriser.")

# -------------------------------------------------------------------
# ÉTAPE 2 : POIs OSRM-ready
# -------------------------------------------------------------------

elif step == 2:
    st.header("3️⃣ POIs sélectionnés pour OSRM")

    df_osrm_ready = get_osrm_ready_pois()

    col_t, col_m = st.columns([2, 3])

    with col_t:
        st.subheader("Table des POIs OSRM-ready")
        st.dataframe(df_osrm_ready)
        st.caption(f"{df_osrm_ready.height} POIs utilisés pour construire la matrice OSRM")

    with col_m:
        st.subheader("Carte des POIs OSRM-ready")
        if df_osrm_ready.height > 0:
            st.pydeck_chart(
                pdk.Deck(
                    initial_view_state=pdk.ViewState(
                        latitude=df_osrm_ready["latitude"].mean(),
                        longitude=df_osrm_ready["longitude"].mean(),
                        zoom=12,
                    ),
                    tooltip={"text": "{nom_du_poi}\n{main_category}"},
                    layers=[pydeck_layer_points(df_osrm_ready, color=(0, 200, 100))],
                )
            )
        else:
            st.info("Aucun POI sélectionné pour OSRM.")

# -------------------------------------------------------------------
# ÉTAPE 3 : Matrices OSRM
# -------------------------------------------------------------------

elif step == 3:
    st.header("4️⃣ Matrices OSRM (durations / distances)")

    df_clustered, df_osrm_dist, df_osrm_dur = get_osrm_matrices()

    st.subheader("Durations (en secondes)")
    st.dataframe(df_osrm_dur)

    st.subheader("Distances (en mètres)")
    st.dataframe(df_osrm_dist)

    st.caption("Matrices calculées via OSRM (async + chunking).")

# -------------------------------------------------------------------
# ÉTAPE 4 : Itinéraire optimisé
# -------------------------------------------------------------------

elif step == 4:
    st.header("5️⃣ Itinéraire optimisé (TSP)")

    df_itinerary = get_itinerary()

    st.subheader("Table complète")
    st.dataframe(df_itinerary.sort(["cluster_id", "visit_order"]))

    st.subheader("Résumé par jour")
    summary = (
        df_itinerary
        .group_by("cluster_id")
        .agg(
            [
                pl.col("cum_cost").max().alias("distance_totale"),
                #pl.col("cum_duration").max().alias("duree_totale"),
                pl.len().alias("nb_pois"),
            ]
        )
    )
    st.dataframe(summary)

# -------------------------------------------------------------------
# ÉTAPE 5 : Routes OSRM (GeoJSON)
# -------------------------------------------------------------------

elif step == 5:
    st.header("6️⃣ Routes OSRM (GeoJSON)")

    #df_itinerary = get_itinerary()

    routes_geojson = get_osrm_routes()
    df_itinerary = cache_get("itinerary")


    # selected_day = st.selectbox(
    #     "Choisir un jour",
    #     sorted(routes_geojson.keys())
    # )

    # route = routes_geojson[selected_day]


    # st.subheader("Carte de l’itinéraire")

    #df_day = df_itinerary.filter(pl.col("cluster_id") == selected_day)
    #df_day_pd = df_day.to_pandas()

    # poi_layer = pdk.Layer(
    #     "ScatterplotLayer",
    #     data=df_day_pd,
    #     get_position=["longitude", "latitude"],
    #     get_color=[0, 0, 255],
    #     get_radius=40,
    #     pickable=True,
    # )

    # route_layer = pdk.Layer(
    #     "PathLayer",
    #     data=[{"path": route["coordinates"]}],
    #     get_path="path",
    #     get_color=[255, 0, 0],
    #     width_scale=3,
    #     width_min_pixels=2,
    # )

    # view_state = pdk.ViewState(
    #     latitude=df_day_pd["latitude"].mean(),
    #     longitude=df_day_pd["longitude"].mean(),
    #     zoom=13,
    # )

    # st.pydeck_chart(
    #     pdk.Deck(
    #         layers=[route_layer, poi_layer],
    #         initial_view_state=view_state,
    #         tooltip={"text": "{nom_du_poi}\nOrdre: {visit_order}"},
    #     )
    # )

    # Choix du jour : "Tous" ou un jour précis
    all_days = sorted(routes_geojson.keys())
    options = ["Tous"] + all_days

    selected_day = st.selectbox(
        "Jour à afficher",
        options,
        format_func=lambda d: f"Jour {d}" if d != "Tous" else "Tous les jours"
    )


    deck = render_premium_map(df_itinerary, routes_geojson, selected_day)
    st.pydeck_chart(deck)


    st.subheader("Détails du parcours")
    st.dataframe(df_itinerary.sort("visit_order"))

# -------------------------------------------------------------------
# PROFILING (EN BAS DE PAGE)
# -------------------------------------------------------------------

if show_prof:
    st.markdown("---")
    show_profiling_summary()
