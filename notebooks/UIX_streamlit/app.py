# app.py

#Niveau 1 — Data POIs : df_gov, df_view / Colonnes : lat, lon, type, rating, etc.
#Niveau 2 — Zones / quartiers : zones, top_zones / Colonnes : zone_id, lat, lon, label, final_score_quart_reco
#Niveau 3 — Streamlit state st.session_state["anchor"] Contient : {"lat": ...,"lon": ...,"display_name": ...}

# ======================================================================================================================
# IMPORTS
# ======================================================================================================================
import math
import numpy as np
import pandas as pd
import streamlit as st
import requests
from typing import List, Dict, Optional
from auth import login_widget, logout_button

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
from collections import Counter
from datetime import datetime


# ======================================================================================================================
# PAGE CONFIG
# ======================================================================================================================
st.set_page_config(page_title="Vivez au coeur de l'expérience", layout="wide")



# ======================================================================================================================
# AUTH
# ======================================================================================================================
if not login_widget():
    st.title("🧳 Vivez au coeur de l'expérience")
    st.info("Connecte-toi dans la barre latérale pour accéder à l'application.")
    st.stop()

with st.sidebar:
    st.caption(f"Connecté : {st.session_state.username}")
logout_button()




# ======================================================================================================================
# PATHS
# ======================================================================================================================
DT_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances\df_gov_clean_normux_fast.parquet"
AB_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances\df_airbnb_clean_normux_fast.parquet"
TA_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances\df_trip_idf_clean_keep_norm_fast.parquet"



# ======================================================================================================================
# box” d’adresse claire + un toggle “j’ai déjà un hébergement"
# ======================================================================================================================
@st.cache_data(ttl=24 * 3600)
def geocode_nominatim(query: str, limit: int = 5) -> List[Dict]:
    """Geocode an address with Nominatim (OpenStreetMap)."""
    if not query or len(query.strip()) < 3:
        return []
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": query, "format": "json", "limit": limit, "addressdetails": 1}
    headers = {"User-Agent": "streamlit-itinerary-app/1.0 (contact: your@email)"}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def format_nominatim_item(it: Dict) -> str:
    return it.get("display_name", "Adresse inconnue")

def set_anchor_from_choice(choice_label: str, options_map: Dict[str, Dict]):
    it = options_map.get(choice_label)
    if not it:
        return
    st.session_state["anchor"] = {
        "display_name": it.get("display_name"),
        "lat": float(it.get("lat")),
        "lon": float(it.get("lon")),
        "raw": it,
    }



# ======================================================================================================================
# COUNTRIES / CITIES
# ======================================================================================================================
@st.cache_data(ttl=3600)
def get_geo_country_maps():
    """
    Retourne:
      - country_names: liste triée de noms de pays
      - name_to_code: mapping "France" -> "FR" etc.
    """
    try:
        import geonamescache
        gc = geonamescache.GeonamesCache()
        countries = gc.get_countries()  # dict keyed by code
        name_to_code = {}
        country_names = []
        for code, info in countries.items():
            name = info.get("name")
            if name:
                country_names.append(name)
                name_to_code[name] = code
        country_names = sorted(list(set(country_names)))
        return country_names, name_to_code
    except Exception:
        # fallback minimal
        return ["France"], {"France": "FR"}


@st.cache_data(ttl=3600)
def get_cities_list(country_code: str, min_population=50000):
    """
    Liste de villes par pays (via geonamescache), filtrée par population.
    """
    try:
        import geonamescache
        gc = geonamescache.GeonamesCache()
        cities = gc.get_cities()
        out = []
        for c in cities.values():
            if c.get("countrycode") == country_code and int(c.get("population") or 0) >= min_population:
                out.append(c.get("name"))
        return sorted(list(set(out)))
    except Exception:
        if country_code == "FR":
            return ["Paris", "Lyon", "Marseille", "Nice"]
        return []


# ======================================================================================================================
# DATA LOADING (simple + align columns)
# ======================================================================================================================
@st.cache_data(ttl=3600)
def load_df() -> pd.DataFrame:
    df_dt = pd.read_parquet(DT_PATH)
    df_ab = pd.read_parquet(AB_PATH)
    df_ta = pd.read_parquet(TA_PATH)
 

    # -------------------------
    # Schéma commun + types
    # -------------------------
    for d in (df_dt, df_ab, df_ta):

        # name obligatoire (pour tri, labels, affichage)
        if "source_id" not in d.columns:
            d["source_id"] = ""

        if "source" not in d.columns:
            d["source"] = ""

        if "type" not in d.columns:
            d["type"] = ""
        
        if "raw_type" not in d.columns:
            d["raw_type"] = ""
        
        if "url" not in d.columns:
            d["url"] = ""
        
        if "lat" not in d.columns:
            d["lat"] = ""
        
        if "lon" not in d.columns:
            d["lon"] = ""

        if "name" not in d.columns:
            d["name"] = ""
        
        if "address" not in d.columns:
            d["address"] = ""
        
        if "postal_code" not in d.columns:
            d["postal_code"] = ""

        if "city" not in d.columns:
            d["city"] = ""

        if "region" not in d.columns:
            d["region"] = ""
        
        if "country" not in d.columns:
            d["country"] = "France"
        
        if "snippet" not in d.columns:
            d["snippet"] = ""

        if "rating" not in d.columns:
            d["rating"] = ""

        if "review_count" not in d.columns:
            d["review_count"] = ""
        
        if "price_level" not in d.columns:
            d["price_level"] = ""

        if "max_people" not in d.columns:
            d["max_people"] = np.nan

        if "distance_km" not in d.columns:
            d["distance_km"] = np.nan

        # types numériques safe
        d["lat"] = pd.to_numeric(d.get("lat"), errors="coerce")
        d["lon"] = pd.to_numeric(d.get("lon"), errors="coerce")
        d["rating"] = pd.to_numeric(d.get("rating"), errors="coerce")
        d["review_count"] = pd.to_numeric(d.get("review_count"), errors="coerce").fillna(0).astype(int)
        d["price_level"] = pd.to_numeric(d.get("price_level"), errors="coerce")
        

    # -------------------------
    # Concat final
    # -------------------------
    df = pd.concat([df_dt, df_ab, df_ta], ignore_index=True)
    return df


df = load_df()

# Listes pour UI
discover_types = sorted(df["type"].dropna().astype(str).unique().tolist())
reco_types_options = ["Hotel", "Airbnb", "Restaurant"]


# ======================================================================================================================
# SESSION STATE
# ======================================================================================================================
def init_state():
    st.session_state.setdefault("itinerary_discovery", {})
    st.session_state.setdefault("itinerary_reco", {})
    st.session_state.setdefault("types_discovery", [])
    st.session_state.setdefault("types_reco", [])
    st.session_state.setdefault("mode", "none")
    st.session_state.setdefault("last_mode", "none")
    st.session_state.setdefault("history", {})
    st.session_state.setdefault("itinerary_all", {})   # dict username -> list

    # ADRESSE ANCRAGE (ANCHOR) 
    st.session_state.setdefault("page", "anchor")  # "anchor" | "explore"
    st.session_state.setdefault("anchor_step", 1)          # 1/2/3
    st.session_state.setdefault("has_anchor_choice", None) # "yes" / "no"
    st.session_state.setdefault("anchor", None)            # {"display_name","lat","lon",...}
    st.session_state.setdefault("anchor_query", "")        # adresse composée (debug)
    st.session_state.setdefault("anchor_results", [])      # résultats Nominatim

    st.session_state.setdefault("page", "anchor")  # "anchor" | "zones" | "explore"
    st.session_state.setdefault("selected_zone", None)      # zone sélectionnée (dict)
    st.session_state.setdefault("anchor_source", None)      # "address" | "zone_reco"


init_state()



# ======================================================================================================================
# LOGIN
# ======================================================================================================================

def current_user():
    return st.session_state.get("username", "anonymous")

def get_user_list(store_key: str):
    u = current_user()
    store = st.session_state[store_key]
    store.setdefault(u, [])
    return store[u]


# ======================================================================================================================
# UTILS
# ======================================================================================================================
def stars(rating) -> str:
    if pd.isna(rating):
        return "—"
    r = float(rating)
    full = max(0, min(5, int(r)))
    return "⭐" * full + "✩" * (5 - full)


def add_distance_km(df: pd.DataFrame, ref_lat: float, ref_lon: float) -> pd.DataFrame:
    """
    Ajoute/écrase la colonne distance_km (approx) entre (lat, lon) et (ref_lat, ref_lon).
    Formule equirectangulaire (rapide, suffisante pour une ville/région).
    """
    if df is None or df.empty:
        return df

    if "lat" not in df.columns or "lon" not in df.columns:
        return df

    out = df.copy()
    lat = pd.to_numeric(out["lat"], errors="coerce")
    lon = pd.to_numeric(out["lon"], errors="coerce")

    # km par degré
    km_per_deg_lat = 110.574
    km_per_deg_lon = 111.320 * np.cos(np.radians(ref_lat))

    dlat = (lat - ref_lat) * km_per_deg_lat
    dlon = (lon - ref_lon) * km_per_deg_lon

    out["distance_km"] = np.sqrt(dlat**2 + dlon**2)
    return out


def render_folium_cluster_map(df_map: pd.DataFrame, center=(48.85, 2.35), zoom=11, max_points=5000, anchor=None):
    # IMPORTANT PERF : limite les points sur la carte
    df_map = df_map.dropna(subset=["lat", "lon"]).head(max_points)

    m = folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")
    # OK8 - Marqueur du lieu d'ancrage (toujours visible)
    if anchor and anchor.get("lat") is not None and anchor.get("lon") is not None:
        a_lat = float(anchor["lat"])
        a_lon = float(anchor["lon"])
        a_label = anchor.get("display_name", "Hébergement")

        folium.Marker(
            location=[a_lat, a_lon],
            tooltip="📍 Point d’ancrage",
            popup=folium.Popup(f"<b>📍 Point d’ancrage</b><br/>{a_label}", max_width=420),
            icon=folium.Icon(color="red", icon="info-sign"),
        ).add_to(m)
    
    cluster = MarkerCluster().add_to(m)

    for _, row in df_map.iterrows():
        name = str(row.get("name", "")).strip()
        cat = str(row.get("type", "")).strip()
        addr = str(row.get("address", "")).strip()
        cp = str(row.get("postal_code", "")).strip()
        city = str(row.get("city", "")).strip()
        cp_city = (cp + " " + city).strip()

        rating = row.get("rating")
        reviews = row.get("review_count", 0)

        # OK7 - Distance affichée dans le popup
        dist_km = row.get("distance_km")
        dist_html = ""
        try:
            if pd.notna(dist_km):
                dist_html = f'<div style="color:#555; margin-top:4px;"><b>Distance :</b> {float(dist_km):.2f} km</div>'
        except Exception:
            dist_html = ""

        url = str(row.get("url", "")).strip()
        link_html = ""
        if url and url.lower() != "nan":
            link_html = f'<div style="margin-top:8px;"><a href="{url}" target="_blank">🌐 Voir le site</a></div>'

        popup_html = f"""
        <div style="font-family: Arial; width: 290px;">
            <div style="font-size:14px; font-weight:700;">{name}</div>
            <div style="color:#555; margin-top:4px;"><b>Type :</b> {cat}</div>
            <div style="color:#555; margin-top:4px;"><b>Adresse :</b> {addr}</div>
            <div style="color:#555; margin-top:4px;"><b>Ville :</b> {cp_city}</div>
            <div style="color:#555; margin-top:4px;"><b>Note :</b> {rating if pd.notna(rating) else "—"} / 5</div>
            <div style="color:#555; margin-top:4px;"><b>Avis :</b> {int(reviews)}</div>
            {dist_html}
            {link_html}
        </div>
        """

        folium.Marker(
            location=[float(row["lat"]), float(row["lon"])],
            popup=folium.Popup(popup_html, max_width=380),
            tooltip=name if name else None,
        ).add_to(cluster)

    return m


# ======================================================================================================================
# DISTANCE (Haversine)
# ======================================================================================================================
def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


# ======================================================================================================================
# FILTERS + SCORE + SORT
# ======================================================================================================================
def filter_location(df_in: pd.DataFrame, city: str | None) -> pd.DataFrame:
    if not city:
        return df_in
    # comparaison insensible à la casse + safe si NaN
    c = city.strip().lower()
    return df_in[df_in["city"].astype(str).str.strip().str.lower() == c]

# -----------------
# ZONES RECOMMANDÉES (DEMO)
# ----------------
def compute_zone_id(lat_s: pd.Series, lon_s: pd.Series, cell_size_km: float = 1.0) -> pd.Series:
    """Grille simple (MVP) : regroupe les POIs en zones ~ carrés."""
    cell_deg = cell_size_km / 111.0  # approx 1° lat ~ 111 km
    lat_id = np.floor(pd.to_numeric(lat_s, errors="coerce") / cell_deg).astype("Int64")
    lon_id = np.floor(pd.to_numeric(lon_s, errors="coerce") / cell_deg).astype("Int64")
    return lat_id.astype(str) + "_" + lon_id.astype(str)

def normalize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    smin, smax = float(s.min()), float(s.max())
    if smax == smin:
        return s * 0.0
    return (s - smin) / (smax - smin)

@st.cache_data(show_spinner=False)
def compute_reco_quartiers(df_city: pd.DataFrame, cell_km: float = 1.0, top_n: int = 12) -> pd.DataFrame:
    """Retourne un DF de zones recommandées avec final_score_quart_reco + lat/lon centroïde + label."""
    df = df_city.dropna(subset=["lat", "lon", "type"]).copy()
    if df.empty:
        return pd.DataFrame(columns=["zone_id","lat","lon","label","final_score_quart_reco"])

    df["zone_id"] = compute_zone_id(df["lat"], df["lon"], cell_size_km=cell_km)

    zones = df.groupby(["zone_id", "type"]).size().unstack(fill_value=0)

    # Scores "bruts"
    zones["housing_score"]    = zones.get("Hotel", 0) + zones.get("Airbnb", 0)
    zones["restaurant_score"] = zones.get("Food", 0) + zones.get("Restaurant", 0)
    zones["shopping_score"]   = zones.get("Shopping", 0)
    zones["culture_score"]    = zones.get("Museum", 0) + zones.get("Monument", 0)
    zones["life_score"]       = zones.get("Evenement", 0) + zones.get("Cinema", 0) + zones.get("Healthcare", 0)

    # Normalisation
    zones["housing_n"]    = normalize(zones["housing_score"])
    zones["restaurant_n"] = normalize(zones["restaurant_score"])
    zones["shopping_n"]   = normalize(zones["shopping_score"])
    zones["culture_n"]    = normalize(zones["culture_score"])
    zones["life_n"]       = normalize(zones["life_score"])

    # ✅ Score final "quartier recommandé"
    zones["final_score_quart_reco"] = (
        1.2 * zones["housing_n"]
      + 1.0 * zones["culture_n"]
      + 0.8 * zones["life_n"]
      + 0.6 * zones["restaurant_n"]
      + 0.6 * zones["shopping_n"]
    )

    # Centroïdes des zones
    centroids = df.groupby("zone_id").agg(lat=("lat","mean"), lon=("lon","mean"))

    zones = zones.merge(centroids, left_index=True, right_index=True)

    # Label UX simple
    def label_zone(r):
        if r["housing_score"] >= r["culture_score"]:
            return "Idéal pour se loger"
        elif r["culture_score"] >= r["life_score"]:
            return "Quartier culturel"
        else:
            return "Quartier vivant"

    zones["label"] = zones.apply(label_zone, axis=1)

    out = (
        zones.sort_values("final_score_quart_reco", ascending=False)
            .head(top_n)
            .reset_index()
            .loc[:, ["zone_id","lat","lon","label","final_score_quart_reco"]]
    )
    return out

def render_quartiers_reco_demo(df_all: pd.DataFrame):
    """Affiche la carte + boutons de sélection quartier (démo)."""
    st.markdown("### 🧭 Quartiers recommandés (démo)")

    c1, c2, c3 = st.columns([1,1,1])
    with c1:
        cell_km = st.slider("Taille de zone (km)", 0.5, 2.0, 1.0, 0.25, key="demo_cell_km")
    with c2:
        top_n = st.slider("Nombre de quartiers", 5, 25, 12, 1, key="demo_top_n")
    with c3:
        show_score = st.checkbox("Afficher le score", value=True, key="demo_show_score")

    # Ville (pour démo)
    city_demo = st.selectbox("Ville pour la recommandation", ["Paris","Lyon","Marseille","Nice"], index=0, key="demo_city")
    df_city = filter_location(df_all, city_demo)

    zones_df = compute_reco_quartiers(df_city, cell_km=cell_km, top_n=top_n)

    if zones_df.empty:
        st.warning("Aucune zone trouvée (vérifie que ta colonne 'city' est bien remplie).")
        return

    # Carte
    m = folium.Map(location=[zones_df["lat"].mean(), zones_df["lon"].mean()], zoom_start=11, tiles="CartoDB positron")
    cluster = MarkerCluster().add_to(m)

    for _, r in zones_df.iterrows():
        popup = f"<b>{r['label']}</b>"
        if show_score:
            popup += f"<br/>Score : {float(r['final_score_quart_reco']):.2f}"

        folium.Marker(
            [float(r["lat"]), float(r["lon"])],
            tooltip=r["label"],
            popup=popup,
            icon=folium.Icon(color="green", icon="home", prefix="fa"),
        ).add_to(cluster)

    st_folium(m, width=None, height=520)

    # Boutons (liste)
    st.markdown("#### Choisir un quartier")
    for i, r in zones_df.iterrows():
        colA, colB = st.columns([4,1])
        with colA:
            txt = f"🏡 **{r['label']}** — zone `{r['zone_id']}`"
            if show_score:
                txt += f" — score **{float(r['final_score_quart_reco']):.2f}**"
            st.write(txt)
        with colB:
            if st.button("Choisir", key=f"choose_zone_{i}"):
                # ✅ Pour la démo, on peut créer une ancre provisoire
                st.session_state["anchor"] = {
                    "lat": float(r["lat"]),
                    "lon": float(r["lon"]),
                    "display_name": f"Point d'ancrage (quartier recommandé) — {r['label']}",
                }
                st.session_state["page"] = "explore"
                st.rerun()
#-------------------------
#FIN
#------------------------

def apply_discovery_filters(df_in: pd.DataFrame, types_selected) -> pd.DataFrame:
    if not types_selected:
        return df_in
    return df_in[df_in["type"].isin(types_selected)]


def compute_reco_score(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()

    rating = pd.to_numeric(out["rating"], errors="coerce").fillna(0.0)
    rating_norm = (rating / 5.0).clip(0, 1)

    reviews = pd.to_numeric(out["review_count"], errors="coerce").fillna(0.0)
    reviews_log = reviews.apply(lambda x: math.log1p(max(x, 0)))
    denom = max(1.0, float(reviews_log.max()))
    reviews_norm = (reviews_log / denom).clip(0, 1)

    dist = pd.to_numeric(out["distance_km"], errors="coerce").fillna(9999.0).clip(lower=0.1)
    inv = 1.0 / dist
    dist_norm = (inv / float(inv.max())).clip(0, 1)

    budget = pd.to_numeric(out["price_level"], errors="coerce").fillna(3.0).clip(1, 3)
    budget_norm = ((3.0 - budget) / 2.0).clip(0, 1)

    out["reco_score"] = (0.50 * rating_norm) + (0.20 * reviews_norm) + (0.20 * dist_norm) + (0.10 * budget_norm)
    return out


def apply_reco_filters(
    df_in: pd.DataFrame,
    types_selected,
    min_rating: float,
    min_reviews: int,
    budget_level: int,
    n_people: int,
) -> pd.DataFrame:
    out = df_in.copy()

    if types_selected:
        out = out[out["type"].isin(types_selected)]

    out = out[pd.to_numeric(out["rating"], errors="coerce").fillna(0) >= float(min_rating)]
    out = out[pd.to_numeric(out["review_count"], errors="coerce").fillna(0).astype(int) >= int(min_reviews)]
    out = out[pd.to_numeric(out["price_level"], errors="coerce").fillna(9999) <= int(budget_level)]
    out = out[(out["max_people"].isna()) | (out["max_people"] >= int(n_people))]

    return compute_reco_score(out)


def sort_reco(df_in: pd.DataFrame, sort_field: str, sort_order: str) -> pd.DataFrame:
    if df_in.empty:
        return df_in

    ascending = (sort_order == "⬆️")
    field_map = {
        "Score": "reco_score",
        "Note": "rating",
        "Avis": "review_count",
        "Distance": "distance_km",
        "Budget": "price_level",
    }
    col = field_map.get(sort_field, "reco_score")
    tie = [c for c in ["reco_score", "rating", "review_count"] if c in df_in.columns and c != col]
    return df_in.sort_values([col] + tie, ascending=[ascending] + [False] * len(tie), na_position="last")


# ======================================================================================================================
# CALLBACKS mode exclusif
# ======================================================================================================================
def on_change_discovery():
    st.session_state.last_mode = "discover"
    st.session_state.types_reco = []


def on_change_reco():
    st.session_state.last_mode = "reco"
    st.session_state.types_discovery = []



# ======================================================================================================================
# ANCHOR
# ======================================================================================================================
def anchor_step_label():
    step = int(st.session_state.get("anchor_step", 1))
    st.caption(f"Étape {step} / 3")

def compose_address_fields(numero: str, rue: str, cp: str, ville: str, pays: str) -> str:
    parts = [
        f"{(numero or '').strip()} {(rue or '').strip()}".strip(),
        f"{(cp or '').strip()} {(ville or '').strip()}".strip(),
        (pays or "").strip(),
    ]
    return ", ".join([p for p in parts if p])

def save_anchor(it: dict, user_query: str = ""):
    """
    Enregistre l'adresse d'ancrage (hébergement) dans la session.
    it = un résultat Nominatim.
    """
    lat = it.get("lat")
    lon = it.get("lon")

    # Nominatim renvoie souvent des strings -> on sécurise
    try:
        lat_f = float(lat) if lat is not None else None
        lon_f = float(lon) if lon is not None else None
    except Exception:
        lat_f, lon_f = None, None

    st.session_state["anchor"] = {
        "display_name": it.get("display_name", ""),
        "lat": lat_f,
        "lon": lon_f,
        "raw": it,
        "query": user_query,
    }

    # Marque le flow comme terminé
    st.session_state["anchor_step"] = 3

def render_anchor_section():
    anchor_step_label()
    st.write("")  # un peu d’air

    step = int(st.session_state.get("anchor_step", 1))

    # -----------------------------
    # ÉTAPE 1 : Oui / Non (cards)
    # -----------------------------
    if step == 1:
        st.markdown("### As-tu déjà une adresse de point d'ancrage ?")

        left_pad, center, right_pad = st.columns([1, 2, 1])
        with left_pad:
            # Bouton OUI
            st.button(
                "Oui",
                use_container_width=True,
                on_click=set_anchor_choice,
                args=("yes",),
            )
            st.write("")  # petit espace

            # Bouton NON (aligné sous Oui)
            st.button(
                "Non",
                use_container_width=True,
                help="On te proposera les meilleures zones où dormir",
                on_click=set_anchor_choice,
                args=("no",),
            )

        st.caption("Si oui : on optimise l’itinéraire depuis cette adresse. Si non : on recommandera des zones où dormir.")
        return

    # -----------------------------
    # ÉTAPE 2 : Formulaire adresse
    # -----------------------------
    if step == 2:
        st.markdown("### Renseigne l’adresse")

        # Boutons navigation haut
        nav1, nav2 = st.columns([1, 3])
        with nav1:
            if st.button("↩️ Retour", key="back_zones_to_anchor"):
                st.session_state["anchor_step"] = 1
                return

        left_pad, center, right_pad = st.columns([1, 2, 1])
        with left_pad:
            with st.container(border=True):
                with st.form("anchor_address_form", clear_on_submit=False):
                    numero = st.text_input("Numéro", placeholder="10")
                    rue = st.text_input("Rue", placeholder="Rue de Rivoli")
                    cp = st.text_input("Code postal", placeholder="75004")
                    ville = st.text_input("Ville", placeholder="Paris")
                    pays = st.text_input("Pays", value="France")
                    submitted = st.form_submit_button("🔎 Rechercher", type="primary")
                    st.caption("Astuce : si la recherche échoue, essaye sans numéro.")

        if submitted:
            query = ", ".join(
                p for p in [
                    f"{numero} {rue}".strip(),
                    f"{cp} {ville}".strip(),
                    pays
                ] if p and p.strip()
            )
            st.session_state["anchor_query"] = query

            if len(query.strip()) < 8:
                st.warning("Adresse trop courte. Mets au moins Rue + Ville (et idéalement Code postal).")
                st.session_state["anchor_results"] = []
            else:
                try:
                    results = geocode_nominatim(query, limit=6)
                    st.session_state["anchor_results"] = results

                    if not results:
                        st.error("Adresse introuvable 😕")

                        # Plan A : relance simplifiée (sans numéro + sans arrondissement)
                        simplified = ", ".join([p for p in [
                            rue.strip() if rue else "",
                            f"{cp.strip()} {ville.strip()}".strip(),
                            pays.strip() if pays else ""
                        ] if p])

                        s1, s2 = st.columns([1, 1])
                        with s1:
                            if simplified and simplified != query:
                                if st.button("↻ Réessayer sans numéro/arrondissement"):
                                    results2 = geocode_nominatim(simplified, limit=6)
                                    st.session_state["anchor_query"] = simplified
                                    st.session_state["anchor_results"] = results2
                        with s2:
                            if st.button("❌ Je n’ai pas d’adresse (finalement)"):
                                set_anchor_choice("no")
                                return
                                st.session_state["anchor_step"] = 3
                                st.session_state["anchor"] = None

                        st.markdown("**Ou cherche une zone approximative** (quartier, gare, centre-ville)")
                        approx = st.text_input("Zone approximative", placeholder="Gare de Lyon, Montmartre, centre-ville…", key="anchor_approx")
                        if st.button("📍 Rechercher une zone", disabled=not approx.strip()):
                            results3 = geocode_nominatim(approx, limit=6)
                            st.session_state["anchor_query"] = approx
                            st.session_state["anchor_results"] = results3

                except Exception as e:
                    st.error(f"Erreur de recherche d’adresse : {e}")

        # Résultats + confirmation (toujours dans un flux vertical)
        results = st.session_state.get("anchor_results", [])
        if results:
            # --- Résultats alignés (même colonne que le formulaire) ---
            left_pad, center, right_pad = st.columns([1, 2, 1])

            # ✅ Construire options et mapping ICI (avant usage)
            options = [format_choice(it) for it in results]
            label_to_item = {format_choice(it): it for it in results}

            with left_pad:
                with st.container(border=True):
                    st.markdown("### Résultats")
                    st.caption("Choisis l’adresse correcte")

                    selected_label = st.selectbox(
                        "Adresse trouvée",
                        options,
                        index=0,
                        label_visibility="collapsed"
                    )

                    it = label_to_item[selected_label]

                    st.info(it.get("display_name", ""))

                    st.write("")

                    # ✅ Boutons verticaux alignés à gauche
                    if st.button(
                        "✅ Confirmer cette adresse",
                        use_container_width=True,
                        type="primary"
                    ):
                        save_anchor(it, st.session_state.get("anchor_query", ""))
                        return

                    st.write("")

                    if st.button(
                        "🧹 Effacer les résultats",
                        use_container_width=True
                    ):
                        st.session_state["anchor_results"] = []
                        st.session_state["anchor"] = None
                        return
        return


    # -----------------------------
    # ÉTAPE 3 : Résumé + carte
    # -----------------------------
    content, spacer = st.columns([2, 3])
    with content:
        st.markdown("### Confirmation")

        choice = st.session_state.get("has_anchor_choice")

        # --- Cas : pas d'adresse (choix "no") ---
        if choice == "no":
            st.success("Tu n’as pas d’adresse ✅")
            st.caption("Voici des quartiers recommandés pour servir de point d’ancrage (démo).")

            render_quartiers_reco_demo(df)

            if st.button("↩️ Modifier"):
                st.session_state["anchor_step"] = 1
            return

        # --- Cas : l'utilisateur a choisi "yes" mais aucune adresse enregistrée ---
        anchor = st.session_state.get("anchor")
        if not anchor:
            st.warning("Aucune adresse enregistrée.")
            if st.button("↩️ Revenir"):
                st.session_state["anchor_step"] = 1
            return

        # --- Cas : adresse enregistrée ---
        st.success("Adresse enregistrée ✅")
        st.write(anchor["display_name"])

        # ✅ Actions verticales (alignées)
        if st.button("✏️ Modifier l’adresse", use_container_width=True):
            st.session_state["anchor_step"] = 2
            st.session_state["anchor_results"] = []
            return

        st.caption("Tu pourras toujours la changer plus tard.")

        st.markdown("### Carte")
        render_anchor_map(anchor["lat"], anchor["lon"], label="Hébergement")

        st.write("")

        if st.button("➡️ Passer aux recommandations", type="primary", use_container_width=True):
            st.session_state["anchor_step"] = 3
            st.session_state["page"] = "explore"
            st.rerun()


def set_anchor_choice(choice: str):
    # choice = "yes" ou "no"
    st.session_state["has_anchor_choice"] = choice
    st.session_state["anchor_results"] = []
    st.session_state["anchor_query"] = ""

    if choice == "yes":
        st.session_state["anchor_step"] = 2
    else:
        st.session_state["anchor_step"] = 3
        st.session_state["anchor"] = None


def format_choice(it: dict) -> str:
    """
    Formate un résultat Nominatim pour affichage dans un selectbox.
    """
    addr = it.get("address", {}) or {}

    house = addr.get("house_number", "")
    road = addr.get("road", "")
    postcode = addr.get("postcode", "")
    city = (
        addr.get("city")
        or addr.get("town")
        or addr.get("village")
        or ""
    )
    country = addr.get("country", "")

    # Ligne principale
    main = " ".join([p for p in [house, road] if p]).strip()
    if not main:
        main = it.get("display_name", "")

    # Détails
    details = " ".join([p for p in [postcode, city] if p]).strip()

    return f"📍 {main} — {details} · {country}"

def render_anchor_map(lat: float, lon: float, label: str = "Hébergement"):
    if lat is None or lon is None:
        st.warning("Coordonnées invalides, impossible d’afficher la carte.")
        return

    m = folium.Map(location=[lat, lon], zoom_start=15)
    folium.Marker([lat, lon], popup=label, tooltip=label).add_to(m)
    st_folium(m, width=700, height=420)

#OK
def render_zones_section():
    st.markdown("## 🧭 Quartiers recommandés (démo)")

    # ✅ bouton retour vers l'étape “adresse”
    if st.button("↩️ Retour (adresse)"):
        st.session_state["page"] = "anchor"
        st.rerun()

    # (tes sliders + ta carte + ton tableau top_zones ici)

    # ✅ état de sélection
    if "selected_zone" not in st.session_state:
        st.session_state["selected_zone"] = None

    st.markdown("### Choisir un quartier")

    city_reco = st.session_state.get("reco_city", "Paris")
    grid_km = float(st.session_state.get("grid_km", 1.0))
    n_zones = int(st.session_state.get("n_zones", 25))
    show_score = bool(st.session_state.get("show_zone_score", True))

    # ===============================
    # BUILD top_zones (OBLIGATOIRE)
    # ===============================

    # dataframe source (adapte le nom si besoin)
    df_city = df[df["city"].fillna("").str.lower() == city_reco.lower()].copy()

    if df_city.empty:
        st.warning(f"Aucune donnée pour la ville : {city_reco}")
        st.stop()

    # --- création zone_id via grille ---
    lat_step = grid_km / 111.0
    lon_step = grid_km / 75.0

    df_city["lat_bin"] = (df_city["lat"] / lat_step).round(0).astype(int)
    df_city["lon_bin"] = (df_city["lon"] / lon_step).round(0).astype(int)
    df_city["zone_id"] = df_city["lat_bin"].astype(str) + "_" + df_city["lon_bin"].astype(str)

    zones = df_city.groupby(["zone_id", "type"]).size().unstack(fill_value=0)

    zones["housing_score"] = zones.get("Hotel", 0)
    zones["restaurant_score"] = zones.get("Food", 0)
    zones["shopping_score"] = zones.get("Shopping", 0)
    zones["culture_score"] = zones.get("Museum", 0) + zones.get("Monument", 0)
    zones["life_score"] = zones.get("Evenement", 0) + zones.get("Cinema", 0) + zones.get("Healthcare", 0)

    def normalize(s):
        if s.max() == s.min():
            return s * 0
        return (s - s.min()) / (s.max() - s.min())

    zones["housing_n"] = normalize(zones["housing_score"])
    zones["restaurant_n"] = normalize(zones["restaurant_score"])
    zones["shopping_n"] = normalize(zones["shopping_score"])
    zones["culture_n"] = normalize(zones["culture_score"])
    zones["life_n"] = normalize(zones["life_score"])

    zones["final_score_quart_reco"] = (
        2.0 * zones["housing_n"]
        + 1.0 * zones["culture_n"]
        + 0.8 * zones["life_n"]
        + 0.6 * zones["restaurant_n"]
    )

    def label_zone(r):
        if r["housing_n"] >= r[["culture_n","life_n","restaurant_n","shopping_n"]].max():
            return "Idéal pour se loger"
        if r["culture_n"] >= r[["life_n","restaurant_n","shopping_n"]].max():
            return "Quartier culturel"
        if r["life_n"] >= r[["restaurant_n","shopping_n"]].max():
            return "Quartier vivant"
        return "Quartier pratique"

    zones = zones.reset_index()
    zones["label"] = zones.apply(label_zone, axis=1)

    top_zones = (
        zones[["zone_id", "final_score_quart_reco", "label"]]
        .sort_values("final_score_quart_reco", ascending=False)
        .head(n_zones)
        .reset_index(drop=True)
    )
    #---------------------------------------

    for i, r in top_zones.iterrows():
        c1, c2 = st.columns([6, 1])
        with c1:
            st.write(f"🏠 {r['label']} — zone **{r['zone_id']}** — score {r['final_score_quart_reco']:.2f}")
        with c2:
            if st.button("Choisir", key=f"choose_zone_{i}"):
                st.session_state["selected_zone"] = {
                    "zone_id": r["zone_id"],
                    "label": r["label"],
                    "score": float(r["final_score_quart_reco"]),
                    "lat": float(r["lat_center"]),   # adapte au nom de tes colonnes
                    "lon": float(r["lon_center"]),
                }

    # ✅ bouton de validation (évite de “sauter” trop vite)
    sel = st.session_state.get("selected_zone")
    if sel:
        st.success(f"Quartier sélectionné : {sel['label']} (zone {sel['zone_id']})")
        if st.button("✅ Utiliser ce quartier comme point d’ancrage", key="btn_validate_zone", type="primary"):
            st.session_state["anchor"] = {
                "display_name": f"Point d’ancrage (quartier recommandé) — {sel['label']}",
                "lat": sel["lat"],
                "lon": sel["lon"],
                "raw": sel,
            }
            st.session_state["anchor_source"] = "zone_reco"
            st.session_state["page"] = "explore"
            st.rerun()


# OK1
def render_explore_section():
    if st.session_state.get("page") != "explore":
        return 
    st.markdown("## 📍 Point d’ancrage")

    # retour vers quartiers si l'ancre vient des zones
    if st.session_state.get("anchor_source") == "zone_reco":
        if st.button("↩️ Retour aux quartiers recommandés", key="btn_back_to_anchor"):
            st.session_state["page"] = "zones"
            st.rerun()

    anchor = st.session_state.get("anchor")

    if anchor:
        display_name = anchor.get("display_name", "")
        lat = anchor.get("lat")
        lon = anchor.get("lon")

        st.markdown(
            f"""
            <div style="
                background-color:#f6f8fa;
                padding:12px 16px;
                border-radius:10px;
                margin-bottom:12px;
                border-left:4px solid #4CAF50;">
                <span style="color:#555;">{display_name}</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    if st.button("✏️ Modifier le lieu d’ancrage", key="btn_edit_anchor_explore"):
        st.session_state["page"] = "anchor"
        st.session_state["anchor_step"] = 2
        st.rerun()

# ======================================================================================================================
# UI - HEADER
# ======================================================================================================================
st.title("🧳 Vivez au cœur de l’expérience")



# ======================================================================================================================
# UI - SIDEBAR
# ======================================================================================================================
anchor_ready = int(st.session_state.get("anchor_step", 1)) >= 3
with st.sidebar:
    st.header("🔎 Filtres")

    # ✅ Pays : vraie liste
    country_names, name_to_code = get_geo_country_maps()
    default_idx = country_names.index("France") if "France" in country_names else 0
    country = st.selectbox(
        "Pays",
        options=country_names,
        index=default_idx,
        disabled=not anchor_ready
    )

    # Ville dépend du pays
    country_code = name_to_code.get(country, "FR")
    city = st.selectbox(
        "Ville",
        options=get_cities_list(country_code, min_population=50000),
        index=None,
        placeholder="Choose option",
        disabled=not anchor_ready
    )
    
    st.multiselect(
        "Type de lieu (Découvert)",
        options=discover_types,
        key="types_discovery",
        placeholder="Choose option",
        on_change=on_change_discovery,
        disabled=not anchor_ready
    )

    st.multiselect(
        "Type de lieu (Recommandation)",
        options=reco_types_options,
        key="types_reco",
        placeholder="Choose option",
        on_change=on_change_reco,
        disabled=not anchor_ready
    )

    # ✅ Mode : reco uniquement si au moins 1 type reco choisi
    st.session_state.mode = (
        "reco"
        if st.session_state.types_reco
        else ("discover" if st.session_state.types_discovery else "none")
    )

    # OK5
    radius_km = st.slider(
        "Rayon autour de point d'ancrage (km)",
        0.5, 50.0, 10.0, 0.5,
        disabled=not anchor_ready
    )
    st.session_state["radius_km"] = radius_km

    # ✅ Sliders uniquement si "Reco" activé (au moins 1 type)
    if st.session_state.mode == "reco":
        n_people = st.slider("Nombre de personnes 👥", 1, 10, 1, 1)
        duration = st.slider("Durée du séjour (jours)", 1, 14, 3)

        budget_level = st.slider("Budget (≤)", 1, 3, 2, 1, help="1=Petit • 2=Moyen • 3=Confort")
        min_rating = st.slider("Note minimum ⭐", 0.0, 5.0, 4.0, 0.1)

        max_reviews_val = int(pd.to_numeric(df["review_count"], errors="coerce").fillna(0).max())
        max_reviews_val = max(max_reviews_val, 10)
        min_reviews = st.slider("Avis minimum 💬", 0, max_reviews_val, 0, 1)

        c1, c2 = st.columns([3, 2])
        with c1:
            sort_field = st.selectbox("Trier par", ["Score", "Note", "Avis", "Distance", "Budget"], index=0)
        with c2:
            sort_order = st.radio("Sens", ["⬇️", "⬆️"], horizontal=True, index=0)

        st.button("🔍 Rechercher", type="primary")
    else:
        # message UX au lieu d'afficher des sliders inutiles
        st.info("Pour afficher les filtres **Recommandation**, sélectionne au moins **1 type de lieu (Reco)**.")


# ======================================================================================================================
# UI - MAIN
# ======================================================================================================================
page = st.session_state.get("page", "anchor")

if page == "anchor":
    render_anchor_section()
    st.stop()

if page == "zones":
    render_zones_section()
    st.stop()

# sinon explore
render_explore_section()

mode = st.session_state.mode
if mode == "none":  
    st.stop()


df_view = filter_location(df, city=city)
if df_view.empty:
    st.warning("Aucune donnée pour cette ville.")
    st.stop()

#OK2
# Référence distance : centre de la ville (moyenne des points filtrés)

#ref_lat = float(df_view["lat"].mean())
#ref_lon = float(df_view["lon"].mean())
#df_view = add_distance_km(df_view, ref_lat, ref_lon)

anchor = st.session_state.get("anchor")

if anchor and anchor.get("lat") is not None and anchor.get("lon") is not None:
    ref_lat = float(anchor["lat"])
    ref_lon = float(anchor["lon"])
else:
    ref_lat = float(df_view["lat"].mean())
    ref_lon = float(df_view["lon"].mean())

df_view = add_distance_km(df_view, ref_lat, ref_lon)
#OK9
radius_km = float(st.session_state.get("radius_km", 10.0))
df_view = df_view[df_view["distance_km"] <= radius_km]

# ======================================================================================================================
# UI - MODE DÉCOUVERT
# ======================================================================================================================

if mode == "discover":
    discovered = apply_discovery_filters(df_view, st.session_state.types_discovery).sort_values(["type", "name"])

    st.subheader("🔎 Découvert")
    st.caption("Carte unique – sans recommandations")

    if discovered.empty:
        st.warning("Aucun résultat.")
        st.stop()

    # ✅ max_points=5000
    m = render_folium_cluster_map(discovered, center=(ref_lat, ref_lon), zoom=12, max_points=5000, anchor=st.session_state.get("anchor"))
    st_folium(m, width=None, height=550)

    st.markdown("### ➕ Ajouter un POI à l’itinéraire (Découvert)")
    df_opts = discovered.copy()
    df_opts["label"] = df_opts["name"].fillna("") + " — " + df_opts["type"].astype(str)

    selected_label = st.selectbox("POI", options=df_opts["label"].tolist(), index=0)
    selected_row = df_opts[df_opts["label"] == selected_label].iloc[0].to_dict()

    if st.button("➕ Ajouter", type="primary"):
        it = get_user_list("itinerary_all") 
        selected_row["origin"] = "discover"  # optionnel, utile
        key = (selected_row.get("source_id"), selected_row.get("type"))
        existing = {(p.get("source_id"), p.get("type")) for p in it}

        if key not in existing:
            it.append(selected_row)
            st.success("Ajouté.")
        else:
            st.info("Déjà dans l’itinéraire.")

    st.markdown("### 🧾 Itinéraire (Découvert)")
    it = get_user_list("itinerary_all")
    
    if not it:
        st.info("Ajoute des POI depuis la carte.")
    else:
        for i, p in enumerate(it, start=1):
            st.write(f"{i}. **{p.get('name','')}** — {p.get('type','')}")
        st.download_button(
            "⬇️ Export CSV",
            pd.DataFrame(it).to_csv(index=False).encode("utf-8"),
            "itinerary_discovery.csv",
            "text/csv",
        )


# ======================================================================================================================
# UI - MODE RECO
# ======================================================================================================================
else:
    # ✅ ici, types_reco est forcément non vide (car mode=reco)
    selected_types = st.session_state.types_reco

    recommended = apply_reco_filters(
        df_view,
        selected_types,
        min_rating=min_rating,
        min_reviews=min_reviews,
        budget_level=budget_level,
        n_people=n_people,
    )
    recommended = sort_reco(recommended, sort_field, sort_order)
    recommended = add_distance_km(recommended, ref_lat, ref_lon)
    recommended = recommended[recommended["distance_km"] <= radius_km]

    tabs = st.tabs(["🗺️ Carte", "🧭 Itinéraire"])

    with tabs[0]:
        st.subheader("🏆 Top recommandations")
        #st.write(f"Résultats : **{len(recommended)}**")
        if recommended.empty:
            st.warning("Aucune recommandation. Ajuste les filtres.")
        else:
            # ✅ max_points=5000
            max_items = st.slider("Nombre max de résultats à afficher", 10, 200, 50, 10)
            recommended_display = recommended.nlargest(max_items, "reco_score")
            m = render_folium_cluster_map(recommended_display, center=(ref_lat, ref_lon), zoom=12, max_points=max_items, anchor=st.session_state.get("anchor"))
            st_folium(m, width=None, height=550)            

            for idx, row in recommended.head(50).iterrows():
                with st.container(border=True):
                    left, right = st.columns([4, 1])
                    with left:
                        st.markdown(f"### {row.get('name','—')}")
                        st.caption(f"{str(row.get('postal_code','')).strip()} {str(row.get('city','')).strip()}".strip())
                        st.write(
                            f"{row.get('type','—')} · ⭐ {stars(row.get('rating'))} ({row.get('rating','—')}) · "
                            f"💬 {int(row.get('review_count',0))} avis · Score **{row.get('reco_score',0):.2f}**"
                        )
                        st.write(f"📍 {float(row.get('distance_km',0)):.2f} km · 💶 Budget {row.get('price_level','—')}")
                        url = row.get("url", "")
                        if pd.notna(url) and str(url).strip() != "":
                            st.markdown(f"[🌐 Voir le site]({str(url).strip()})")
                    with right:
                        if st.button("Ajouter +", key=f"add_reco_{idx}"):
                            it = get_user_list("itinerary_all")

                            item = row.to_dict()
                            item["origin"] = "reco"

                            key = (item.get("source_id"), item.get("type"))
                            existing = {(p.get("source_id"), p.get("type")) for p in it}

                            if key not in existing:
                                it.append(item)
                                st.success("Ajouté")
                            else:
                                st.info("Déjà dans l’itinéraire")                             
                                      

    with tabs[1]:
        u = current_user()
        st.subheader("🧭 Itinéraire (Reco)")
        it = get_user_list("itinerary_all")
        

        if not it:
            st.info("Ton itinéraire est vide.")
        else:
            buckets = [[] for _ in range(duration)]
            for i, place in enumerate(it):
                buckets[i % duration].append(place)

            for d in range(duration):
                st.markdown(f"### Jour {d+1}")
                for p in buckets[d]:
                    st.write(f"- **{p.get('name','—')}** ({p.get('type','—')})")
            
            u = current_user()
            st.session_state["history"].setdefault(u, [])

            if st.button("💾 Sauvegarder cet itinéraire"):
                snapshot = {
                    "created_at": pd.Timestamp.now().isoformat(),
                    "items": get_user_list("itinerary_all").copy(),
                }
                st.session_state["history"][u].append(snapshot)
                st.success("Itinéraire sauvegardé ✅")

            st.download_button(
                "⬇️ Exporter (CSV)",
                pd.DataFrame(it).to_csv(index=False).encode("utf-8"),
                "itinerary_reco.csv",
                "text/csv",
            )

        # ============================
        # 📁 HISTORIQUE DES ITINÉRAIRES
        # ============================
        st.markdown("## 📁 Historique")

        st.session_state["history"].setdefault(u, [])
        hist = st.session_state["history"][u]

        if not hist:
            st.info("Aucun itinéraire sauvegardé.")
        else:
            # hist = liste des snapshots pour l'utilisateur u
            for i, snap in enumerate(list(reversed(hist)), start=1):
                with st.container(border=True):
                    items = snap.get("items", []) or []

                    # ✅ si counts_by_type n’existe pas (anciens snaps), on le calcule
                    counts_by_type = snap.get("counts_by_type")
                    if not counts_by_type:
                        counts_by_type = Counter(
                            str(x.get("type", "—")) for x in items if isinstance(x, dict)
                        )
                        counts_by_type = dict(counts_by_type)

                    # ✅ pareil pour origin
                    counts_by_origin = snap.get("counts_by_origin")
                    if not counts_by_origin:
                        counts_by_origin = Counter(
                            str(x.get("origin", "—")) for x in items if isinstance(x, dict)
                        )
                        counts_by_origin = dict(counts_by_origin)

                    # --- texte types (top 4) ---
                    top_types = sorted(counts_by_type.items(), key=lambda x: x[1], reverse=True)
                    types_txt = " • ".join([f"{k}: {v}" for k, v in top_types[:4]])

                    # --- texte origin ---
                    orig_txt = " • ".join([f"{k}: {v}" for k, v in counts_by_origin.items()])

                    # --- ligne principale ---
                    line = f"🗓️ {snap.get('created_at','—')} — {len(items)} lieux"
                    if types_txt:
                        line += f" • {types_txt}"
                    st.write(line)

                    if orig_txt:
                        st.caption(orig_txt)

                    # --- boutons Charger / Supprimer ---
                    c1, c2 = st.columns([1, 1])
                    with c1:
                        if st.button("Charger", key=f"load_{i}"):
                            st.session_state["itinerary_all"][u] = items
                            st.success("Chargé ✅")
                            st.rerun()

                    with c2:
                        if st.button("🗑️ Supprimer", key=f"del_{i}"):
                            # on supprime dans la liste originale hist (pas celle reversed)
                            idx_in_hist = len(hist) - i
                            del hist[idx_in_hist]
                            st.session_state["history"][u] = hist
                            st.success("Supprimé ✅")
                            st.rerun()