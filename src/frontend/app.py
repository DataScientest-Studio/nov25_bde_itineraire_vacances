import streamlit as st
import polars as pl
from geopy.geocoders import Nominatim
from math import radians, sin, cos, asin, sqrt
import pydeck as pdk
from pathlib import Path

DATA_DIR = Path("data")
POIS_PATH = DATA_DIR / "processed" / "merged_20260101_234939.parquet"

# --------------------------------------------------
# Utils
# --------------------------------------------------

def haversine(lat1, lon1, lat2, lon2):
    """
    Distance en km entre deux points (lat/lon) via la formule de Haversine.
    """
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    """
    Geocoding d'une adresse via Nominatim.
    """
    if not address:
        return None
    geolocator = Nominatim(user_agent="poi_itinerary_app")
    loc = geolocator.geocode(address)
    if loc is None:
        return None
    return loc.latitude, loc.longitude

@st.cache_data
def load_pois() -> pl.DataFrame:
    """
    Chargement de ton dataset POIs.
    Ici tu branches ton LazyFrame puis .collect().
    """
    lf = pl.read_parquet(POIS_PATH)
    return lf

# --------------------------------------------------
# App
# --------------------------------------------------

st.set_page_config(
    page_title="Itinéraires de POIs",
    layout="wide",
)

st.title("Itinéraires de POIs optimisés par jour (MVP)")
st.caption("Filtre les POIs, choisis un point de départ, et visualise ceux dans un certain rayon.")

# Chargement des données
df_pois = load_pois()

# --------------------------------------------------
# SIDEBAR - Filtres
# --------------------------------------------------

st.sidebar.header("Filtres")

# 1. Région
regions = sorted(df_pois["region"].unique().to_list())
region = st.sidebar.selectbox("Région", regions)

df_region = df_pois.filter(pl.col("region") == region)

# 2. Ville
villes = sorted(df_region["commune"].unique().to_list())
ville = st.sidebar.selectbox("Ville", villes)

df_city = df_region.filter(pl.col("commune") == ville)

# 3. Point de départ
st.sidebar.subheader("Point de départ")

start_mode = st.sidebar.radio(
    "Définir le point de départ par :",
    ["Adresse (Nominatim)", "Centre de la ville"],
    help="On peut ajouter plus tard le clic sur carte."
)

start_lat, start_lon = None, None

if start_mode == "Adresse (Nominatim)":
    address = st.sidebar.text_input("Adresse de départ")
    if address:
        res = geocode_address(address)
        if res is None:
            st.sidebar.error("Adresse introuvable. Vérifie l'orthographe.")
        else:
            start_lat, start_lon = res
else:
    # Point de départ = barycentre des POIs de la ville
    start_lat = df_city["latitude"].mean()
    start_lon = df_city["longitude"].mean()

# 4. Mode de transport et rayon
st.sidebar.subheader("Déplacement")

transport = st.sidebar.selectbox(
    "Mode de transport",
    ["À pied", "Vélo", "Voiture"]
)

default_radius = {
    "À pied": 3,
    "Vélo": 10,
    "Voiture": 30,
}[transport]

radius_km = st.sidebar.slider(
    "Rayon autour du point de départ (km)",
    min_value=1,
    max_value=100,
    value=default_radius,
)

# 5. Catégories
st.sidebar.subheader("Catégories")

main_cats = sorted(df_city["main_category"].unique().to_list())
selected_main = st.sidebar.multiselect(
    "Main categories",
    options=main_cats,
    default=main_cats
)

df_filtered = df_city.filter(pl.col("main_category").is_in(selected_main))

sub_cats = sorted(df_filtered["sub_category"].unique().to_list())
selected_sub = st.sidebar.multiselect(
    "Sub categories",
    options=sub_cats,
    default=sub_cats
)

df_filtered = df_filtered.filter(pl.col("sub_category").is_in(selected_sub))

# --------------------------------------------------
# MAIN - Calcul du rayon + affichage
# --------------------------------------------------

if (start_lat is None) or (start_lon is None):
    st.warning("Définis un point de départ pour voir les POIs dans le rayon.")
    st.stop()

# 1. Calcul des distances
df_radius = df_filtered.with_columns([
    pl.struct(["latitude", "longitude"]).map_elements(
        lambda s: haversine(start_lat, start_lon, s["latitude"], s["longitude"])
    ).alias("distance_km")
])

df_radius = df_radius.filter(pl.col("distance_km") <= radius_km).sort("distance_km")

# 2. KPIs
nb_pois = df_radius.height

kpi1, kpi2, kpi3 = st.columns(3)
kpi1.metric("Nombre de POIs dans le rayon", nb_pois)
kpi2.metric("Rayon (km)", radius_km)
kpi3.metric("Ville sélectionnée", ville)

st.markdown("---")

# 3. Carte PyDeck
st.subheader("Carte des POIs dans le rayon")

if nb_pois == 0:
    st.info("Aucun POI trouvé dans ce rayon avec ces filtres.")
else:
    data_map = df_radius.select(
        ["nom_du_poi", "latitude", "longitude", "main_category", "sub_category", "distance_km"]
    ).to_pandas()

    data_map["color"] = [[0, 200, 0, 220]] * len(data_map)

    layers = [
        # POIs
        pdk.Layer(
            "ScatterplotLayer",
            data=data_map,
            get_position="[longitude, latitude]",
            get_radius=50,
            get_fill_color="color",
            get_line_color=[0, 80, 0],
            line_width_min_pixels=1,
            pickable=True,
        ),
        # Point de départ
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"latitude": start_lat, "longitude": start_lon}],
            get_position="[longitude, latitude]",
            get_radius=80,
            get_fill_color=[255, 0, 0, 230],
            get_line_color=[120, 0, 0],
            line_width_min_pixels=2,
        ),
    ]

    view_state = pdk.ViewState(
        latitude=start_lat,
        longitude=start_lon,
        zoom=13,
        pitch=45,
    )

    deck = pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={
            "text": "{nom_du_poi}\n{main_category} / {sub_category}\nDistance: {distance_km} km"
        },
    )

    st.pydeck_chart(deck)

# 4. Tableau des POIs
st.subheader("Liste des POIs dans le rayon")

if nb_pois > 0:
    st.dataframe(
        df_radius.select(
            ["nom_du_poi", "main_category", "sub_category", "distance_km"]
        ).to_pandas()
    )