import streamlit as st
import polars as pl
from geopy.geocoders import Nominatim
from math import radians, sin, cos, asin, sqrt
import pydeck as pdk
from pathlib import Path

DATA_DIR = Path("data")
INPUT_DIR = DATA_DIR / "processed"
input_path = INPUT_DIR / "merged_20260101_232420.parquet"

# --- utils distance ---
def haversine(lat1, lon1, lat2, lon2):
    # distance en km
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

# --- geocoding ---
@st.cache_data(show_spinner=False)
def geocode_address(address: str):
    geolocator = Nominatim(user_agent="poi_itinerary_app")
    loc = geolocator.geocode(address)
    if loc is None:
        return None
    return loc.latitude, loc.longitude

# --- chargement dataset (adapter à ton pipeline) ---
@st.cache_data
def load_pois():
    lf = pl.read_parquet(input_path)
    return lf.collect()

df_pois = load_pois()

st.sidebar.title("Filtres")
# 1. Région / Ville
regions = sorted(df_pois["region_name"].unique().to_list())
region = st.sidebar.selectbox("Région", regions)

df_region = df_pois.filter(pl.col("region_name") == region)

villes = sorted(df_region["city_name"].unique().to_list())
ville = st.sidebar.selectbox("Ville", villes)

df_city = df_region.filter(pl.col("city_name") == ville)

# 2. Point de départ
st.sidebar.subheader("Point de départ")

start_mode = st.sidebar.radio(
    "Comment définir le point de départ ?",
    ["Adresse", "Clique sur la carte"]
)

start_lat, start_lon = None, None

if start_mode == "Adresse":
    address = st.sidebar.text_input("Adresse de départ")
    if address:
        res = geocode_address(address)
        if res is None:
            st.sidebar.error("Adresse introuvable")
        else:
            start_lat, start_lon = res
else:
    st.write("Clique sur la carte pour choisir le point de départ.")
    # on affiche une carte centrée sur la ville, puis on récupère le clic
    mid_lat = df_city["lat"].mean()
    mid_lon = df_city["lon"].mean()

    initial_view_state = pdk.ViewState(
        latitude=mid_lat,
        longitude=mid_lon,
        zoom=12
    )

    layer_pois = pdk.Layer(
        "ScatterplotLayer",
        data=df_city.to_pandas(),  # pydeck veut du pandas
        get_position="[lon, lat]",
        get_radius=50,
        get_fill_color="[0, 0, 200, 160]",
        pickable=True,
    )

    r = pdk.Deck(
        layers=[layer_pois],
        initial_view_state=initial_view_state,
        map_style="mapbox://styles/mapbox/light-v9",
        tooltip={"text": "{name}"}
    )

    map_click = st.pydeck_chart(r)

    # ici il faut brancher la logique de récupération du clic
    # selon ton setup (Streamlit >=1.37, events, etc.)
    # par exemple via st.session_state ou une lib type st_clickable_map

# 3. Mode de transport / rayon
st.sidebar.subheader("Mode de transport et rayon")

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

# 4. Catégories
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

# 5. Filtrage spatial par rayon (si point de départ défini)
if start_lat is not None and start_lon is not None:
    df_filtered = df_filtered.with_columns([
        pl.struct(["lat", "lon"]).map_elements(
            lambda s: haversine(start_lat, start_lon, s["lat"], s["lon"])
        ).alias("distance_km")
    ])

    df_radius = df_filtered.filter(pl.col("distance_km") <= radius_km).sort("distance_km")

    st.subheader("POIs dans le rayon sélectionné")
    st.dataframe(df_radius.select(["name", "main_category", "sub_category", "distance_km"]).to_pandas())

    # Carte des POIs + point de départ
    if not df_radius.is_empty():
        data_map = df_radius.to_pandas()
        data_map["color"] = [ [0, 150, 0, 180] ] * len(data_map)

        layers = [
            pdk.Layer(
                "ScatterplotLayer",
                data=data_map,
                get_position="[lon, lat]",
                get_radius=60,
                get_fill_color="color",
                pickable=True,
            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=[{"lat": start_lat, "lon": start_lon}],
                get_position="[lon, lat]",
                get_radius=80,
                get_fill_color="[200, 0, 0, 200]",
            )
        ]

        view_state = pdk.ViewState(
            latitude=start_lat,
            longitude=start_lon,
            zoom=13
        )

        st.pydeck_chart(pdk.Deck(
            layers=layers,
            initial_view_state=view_state,
            tooltip={"text": "{name}"}
        ))
else:
    st.info("Définis un point de départ pour voir les POIs dans le rayon.")