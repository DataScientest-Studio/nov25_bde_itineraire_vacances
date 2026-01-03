# app.py
# ======================================================================================================================
# IMPORTS
# ======================================================================================================================
import math
import numpy as np
import pandas as pd
import streamlit as st

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium


# ======================================================================================================================
# PAGE CONFIG
# ======================================================================================================================
st.set_page_config(page_title="Itinéraire de Vacances", layout="wide")


# ======================================================================================================================
# PATHS
# ======================================================================================================================
DT_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances\df_gov_clean_normux_fast.parquet"
AB_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances\df_airbnb_clean_normux_fast.parquet"
TA_PATH = r"C:\Users\DELL\Downloads\ItineraireVacances\df_trip_idf_clean_keep_norm_fast.parquet"


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
    st.session_state.setdefault("itinerary_discovery", [])
    st.session_state.setdefault("itinerary_reco", [])
    st.session_state.setdefault("types_discovery", [])
    st.session_state.setdefault("types_reco", [])
    st.session_state.setdefault("mode", "none")
    st.session_state.setdefault("last_mode", "none")


init_state()


# ======================================================================================================================
# UTILS
# ======================================================================================================================
def stars(rating) -> str:
    if pd.isna(rating):
        return "—"
    r = float(rating)
    full = max(0, min(5, int(r)))
    return "⭐" * full + "✩" * (5 - full)


def render_folium_cluster_map(df_map: pd.DataFrame, center=(48.85, 2.35), zoom=11, max_points=5000):
    # IMPORTANT PERF : limite les points sur la carte
    df_map = df_map.dropna(subset=["lat", "lon"]).head(max_points)

    m = folium.Map(location=list(center), zoom_start=zoom, tiles="CartoDB positron")
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

        popup_html = f"""
        <div style="font-family: Arial; width: 290px;">
            <div style="font-size:14px; font-weight:700;">{name}</div>
            <div style="color:#555; margin-top:4px;"><b>Type :</b> {cat}</div>
            <div style="color:#555; margin-top:4px;"><b>Adresse :</b> {addr}</div>
            <div style="color:#555; margin-top:4px;"><b>Ville :</b> {cp_city}</div>
            <div style="color:#555; margin-top:4px;"><b>Note :</b> {rating if pd.notna(rating) else "—"} / 5</div>
            <div style="color:#555; margin-top:4px;"><b>Avis :</b> {int(reviews)}</div>
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
def add_distance_km(df_in: pd.DataFrame, ref_lat: float, ref_lon: float) -> pd.DataFrame:
    out = df_in.copy()
    R = 6371.0

    lat1 = np.radians(out["lat"].astype(float))
    lon1 = np.radians(out["lon"].astype(float))
    lat2 = np.radians(ref_lat)
    lon2 = np.radians(ref_lon)

    dlat = lat1 - lat2
    dlon = lon1 - lon2

    a = np.sin(dlat / 2) ** 2 + np.cos(lat2) * np.cos(lat1) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    out["distance_km"] = R * c
    return out


# ======================================================================================================================
# FILTERS + SCORE + SORT
# ======================================================================================================================
def filter_location(df_in: pd.DataFrame, city: str | None) -> pd.DataFrame:
    if not city:
        return df_in
    # comparaison insensible à la casse + safe si NaN
    c = city.strip().lower()
    return df_in[df_in["city"].astype(str).str.strip().str.lower() == c]


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
    max_distance_km: float,
    budget_level: int,
    n_people: int,
) -> pd.DataFrame:
    out = df_in.copy()

    if types_selected:
        out = out[out["type"].isin(types_selected)]

    out = out[pd.to_numeric(out["rating"], errors="coerce").fillna(0) >= float(min_rating)]
    out = out[pd.to_numeric(out["review_count"], errors="coerce").fillna(0).astype(int) >= int(min_reviews)]
    out = out[pd.to_numeric(out["distance_km"], errors="coerce").fillna(9999) <= float(max_distance_km)]
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
# UI - HEADER
# ======================================================================================================================
st.title("🧳 Vivez au cœur de l’expérience")
st.caption("Découvert = carte simple • Recommandation = score + filtres + tri")


# ======================================================================================================================
# UI - SIDEBAR
# ======================================================================================================================
with st.sidebar:
    st.header("🔎 Filtres")

    # ✅ Pays : vraie liste
    country_names, name_to_code = get_geo_country_maps()
    default_idx = country_names.index("France") if "France" in country_names else 0
    country = st.selectbox("Pays", options=country_names, index=default_idx)

    # Ville dépend du pays
    country_code = name_to_code.get(country, "FR")
    city = st.selectbox(
        "Ville",
        options=get_cities_list(country_code, min_population=50000),
        index=None,
        placeholder="Choose option",
    )
    
    st.multiselect(
        "Type de lieu (Découvert)",
        options=discover_types,
        key="types_discovery",
        placeholder="Choose option",
        on_change=on_change_discovery,
    )

    st.multiselect(
        "Type de lieu (Recommandation)",
        options=reco_types_options,
        key="types_reco",
        placeholder="Choose option",
        on_change=on_change_reco,
    )

    # ✅ Mode : reco uniquement si au moins 1 type reco choisi
    st.session_state.mode = (
        "reco"
        if st.session_state.types_reco
        else ("discover" if st.session_state.types_discovery else "none")
    )

    # ✅ Sliders uniquement si "Reco" activé (au moins 1 type)
    if st.session_state.mode == "reco":
        n_people = st.slider("Nombre de personnes 👥", 1, 10, 1, 1)
        duration = st.slider("Durée du séjour (jours)", 1, 14, 3)

        budget_level = st.slider("Budget (≤)", 1, 3, 2, 1, help="1=Petit • 2=Moyen • 3=Confort")
        max_distance = st.slider("Distance max (km)", 0.5, 50.0, 10.0, 0.5)
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
mode = st.session_state.mode
if mode == "none":
    st.info("1) Choisis une **Ville**\n\n2) Sélectionne un mode : **Découvert** ou **Recommandation**.")
    st.stop()

df_view = filter_location(df, city=city)

if df_view.empty:
    st.warning("Aucune donnée pour cette ville.")
    st.stop()

# Référence distance : centre de la ville (moyenne des points filtrés)
ref_lat = float(df_view["lat"].mean())
ref_lon = float(df_view["lon"].mean())
df_view = add_distance_km(df_view, ref_lat, ref_lon)


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
    m = render_folium_cluster_map(discovered, center=(ref_lat, ref_lon), zoom=12, max_points=5000)
    st_folium(m, width=None, height=550)

    st.markdown("### ➕ Ajouter un POI à l’itinéraire (Découvert)")
    df_opts = discovered.copy()
    df_opts["label"] = df_opts["name"].fillna("") + " — " + df_opts["type"].astype(str)

    selected_label = st.selectbox("POI", options=df_opts["label"].tolist(), index=0)
    selected_row = df_opts[df_opts["label"] == selected_label].iloc[0].to_dict()

    if st.button("➕ Ajouter", type="primary"):
        key = (selected_row.get("source_id"), selected_row.get("type"))
        existing = {(p.get("source_id"), p.get("type")) for p in st.session_state.itinerary_discovery}
        if key not in existing:
            st.session_state.itinerary_discovery.append(selected_row)
            st.success("Ajouté.")

    st.markdown("### 🧾 Itinéraire (Découvert)")
    it = st.session_state.itinerary_discovery
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
        max_distance_km=max_distance,
        budget_level=budget_level,
        n_people=n_people,
    )
    recommended = sort_reco(recommended, sort_field, sort_order)

    tabs = st.tabs(["🗺️ Carte", "📋 Liste", "🧭 Itinéraire"])

    with tabs[0]:
        st.subheader("✨ Recommandation")
        if recommended.empty:
            st.warning("Aucune recommandation. Ajuste les filtres.")
        else:
            # ✅ max_points=5000
            m = render_folium_cluster_map(recommended, center=(ref_lat, ref_lon), zoom=12, max_points=5000)
            st_folium(m, width=None, height=550)

            st.markdown("### 🏆 Top recommandations")
            top = recommended.head(3).to_dict(orient="records")
            cols = st.columns(3)
            for i, item in enumerate(top):
                with cols[i]:
                    st.markdown(f"**{item.get('name','—')}**")
                    st.write(item.get("type", "—"))
                    st.write(f"⭐ {stars(item.get('rating'))} ({item.get('rating','—')})")
                    st.write(f"💬 {int(item.get('review_count',0))} avis")
                    st.write(f"📍 {float(item.get('distance_km',0)):.2f} km")
                    st.write(f"💶 Budget: {int(item.get('price_level',3)) if pd.notna(item.get('price_level')) else '—'}")
                    st.write(f"Score: **{float(item.get('reco_score',0)):.2f}**")
                    url = item.get("url")
                    if pd.notna(url) and url != "":
                        st.link_button("🌐 Voir l’annonce", url)
                    if st.button("Ajouter +", key=f"add_top_{i}"):
                        st.session_state.itinerary_reco.append(item)
                        st.success("Ajouté")

    with tabs[1]:
        st.subheader("📋 Liste (Reco)")
        st.write(f"Résultats : **{len(recommended)}**")

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
                        st.session_state.itinerary_reco.append(row.to_dict())
                        st.success("Ajouté")

    with tabs[2]:
        st.subheader("🧭 Itinéraire (Reco)")
        it = st.session_state.itinerary_reco

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

            st.download_button(
                "⬇️ Exporter (CSV)",
                pd.DataFrame(it).to_csv(index=False).encode("utf-8"),
                "itinerary_reco.csv",
                "text/csv",
            )
