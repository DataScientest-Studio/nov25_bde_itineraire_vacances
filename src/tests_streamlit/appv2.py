import streamlit as st
import polars as pl
from math import radians, sin, cos, asin, sqrt
import pydeck as pdk
import folium
from streamlit_folium import st_folium
from pathlib import Path

DATA_DIR = Path("data")
POIS_PATH = DATA_DIR / "processed" / "merged_20260101_234939.parquet"



def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

@st.cache_data
def load_pois() -> pl.DataFrame:
    """
    Chargement de ton dataset POIs.
    Ici tu branches ton LazyFrame puis .collect().
    """
    lf = pl.read_parquet(POIS_PATH)
    return lf

st.sidebar.header("Filtres")

df_pois = load_pois()

# Région
regions = sorted(df_pois["region"].unique().to_list())
region = st.sidebar.selectbox("Région", regions)
df_region = df_pois.filter(pl.col("region") == region)

# Ville
villes = sorted(df_region["commune"].unique().to_list())
ville = st.sidebar.selectbox("Ville", villes)
df_city = df_region.filter(pl.col("commune") == ville)

# Mode de transport
transport = st.sidebar.selectbox("Mode de transport", ["À pied", "Vélo", "Voiture"])
default_radius = {"À pied": 3, "Vélo": 10, "Voiture": 30}[transport]

radius_km = st.sidebar.slider("Rayon (km)", 1, 100, default_radius)

# Catégories
main_cats = sorted(df_city["main_category"].unique().to_list())
selected_main = st.sidebar.multiselect("Main categories", main_cats, main_cats)

df_filtered = df_city.filter(pl.col("main_category").is_in(selected_main))

sub_cats = sorted(df_filtered["sub_category"].unique().to_list())
selected_sub = st.sidebar.multiselect("Sub categories", sub_cats, sub_cats)

df_filtered = df_filtered.filter(pl.col("sub_category").is_in(selected_sub))

# FOLIUM
st.subheader("Choisis ton point de départ")

# Centre de la carte = barycentre de la ville
center_lat = df_city["latitude"].mean()
center_lon = df_city["longitude"].mean()

m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
m.add_child(folium.LatLngPopup())  # permet de cliquer et récupérer lat/lon

map_data = st_folium(m, height=500, width=700)

start_lat, start_lon = None, None

if map_data and map_data["last_clicked"]:
    start_lat = map_data["last_clicked"]["lat"]
    start_lon = map_data["last_clicked"]["lng"]
    st.success(f"Point sélectionné : {start_lat}, {start_lon}")
else:
    st.info("Clique sur la carte pour définir le point de départ.")
    st.stop()

# Filtrage
df_radius = df_filtered.with_columns([
    pl.struct(["latitude", "longitude"]).map_elements(
        lambda s: haversine(start_lat, start_lon, s["latitude"], s["longitude"])
    ).alias("distance_km")
])

df_radius = df_radius.filter(pl.col("distance_km") <= radius_km).sort("distance_km")
nb_pois = df_radius.height

# KPIs

k1, k2, k3 = st.columns(3)
k1.metric("POIs trouvés", nb_pois)
k2.metric("Rayon (km)", radius_km)
k3.metric("Ville", ville)

# PyDeck
#st.subheader("Carte des POIs dans le rayon")

if nb_pois == 0:
    st.warning("Aucun POI trouvé dans ce rayon.")
else:
    data_map = df_radius.select(
        ["nom_du_poi", "latitude", "longitude", "main_category", "sub_category", "distance_km"]
    ).to_pandas()

    data_map["color"] = [[0, 200, 0, 220]] * len(data_map)

    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            data=data_map,
            get_position="[longitude, latitude]",
            get_radius=120,
            get_fill_color="color",
            pickable=True,
        ),
        pdk.Layer(
            "ScatterplotLayer",
            data=[{"latitude": start_lat, "longitude": start_lon}],
            get_position="[longitude, latitude]",
            get_radius=200,
            get_fill_color=[255, 0, 0, 230],
        ),
    ]

    view_state = pdk.ViewState(
        latitude=start_lat,
        longitude=start_lon,
        zoom=13,
        pitch=45,
    )

    st.pydeck_chart(pdk.Deck(
        layers=layers,
        initial_view_state=view_state,
        tooltip={"text": "{nom_du_poi}\n{main_category} / {sub_category}\n{distance_km} km"},
    ))


# Pois table
st.subheader("Liste des POIs dans le rayon")
st.dataframe(
    df_radius.select(["nom_du_poi", "main_category", "sub_category", "distance_km"]).to_pandas()
) 