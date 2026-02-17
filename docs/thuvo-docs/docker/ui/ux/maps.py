# ======================================================================================================================
# PAGE CONFIG ---> ---> docker/ui/ui/maps.py
# ======================================================================================================================

import streamlit as st
from infra.imports import folium, MarkerCluster, pd, st_folium


# Cache la carte pour éviter de la reconstruire à chaque interaction Streamlit
@st.cache_resource(show_spinner=False)
def build_day_map(df_jour, anchor_lat, anchor_lon):
    """
    (RESP) UI rendering : mini-carte par jour (col_map)
    Construit une carte Leaflet (Folium) dédiée à une journée d’itinéraire.
    Paramètres
    ----------
    df_jour : DataFrame
        Sous-ensemble des POI pour un jour donné (déjà filtré en amont).
        Doit contenir au minimum : latitude, longitude, name, type.
    anchor_lat : float
        Latitude du point d’ancrage (hôtel / départ).
    anchor_lon : float
        Longitude du point d’ancrage.
    Retour
    ------
    folium.Map
        Carte prête à être affichée via st_folium().
    """
    # Carte centrée sur l’ancre
    m = folium.Map(
        location=[anchor_lat, anchor_lon],
        zoom_start=13,
        tiles="cartodbpositron",)

    # Marker du point d’ancrage
    folium.Marker(
        [anchor_lat, anchor_lon],
        popup="Point d'ancrage",
        icon=folium.Icon(color="red", icon="home"),).add_to(m)

    # Markers du point d’ancrage & des POI du jour
    # adapte les noms de colonnes: lat/lon vs latitude/longitude
    for _, row in df_jour.iterrows():
        lat = row.get("lat", row.get("latitude"))
        lon = row.get("lon", row.get("longitude"))
        if lat is None or lon is None:
            continue
        folium.Marker(
            [lat, lon],
            popup=str(row.get("name", "POI")),
            icon=folium.Icon(color="blue", icon="info-sign"),).add_to(m)
    return m




def render_discovery_map(df_map: pd.DataFrame, anchor_lat: float, anchor_lon: float):
    """
    (RESP) UI/UX pure : rendu carte, clustering, popup HTML, couleur par type
    gestion pts vides + ancre optionnelle + LayerControl
    color_by_type simple et lisible
    Affiche une carte Folium avec clustering des points de l'itinéraire.
    - Centre la carte sur l'ancre si dispo, sinon sur la moyenne des points
    - Ajoute un marqueur "home" pour l'ancre
    - Ajoute un MarkerCluster pour éviter la surcharge visuelle
    - Couleur des points selon le type (central / satellite / restaurant)
    - Popup HTML enrichi (jour/slot/type/catégorie/score/adresse + lien)
    """
    # --- Robustesse colonnes geo ---
    if df_map is None or df_map.empty:
        st.info("Aucun point à afficher sur la carte.")
        return

    # Harmoniser noms possibles
    if "lat" not in df_map.columns and "latitude" in df_map.columns:
        df_map = df_map.rename(columns={"latitude": "lat"})
    if "lon" not in df_map.columns and "longitude" in df_map.columns:
        df_map = df_map.rename(columns={"longitude": "lon"})

    # Extraire depuis geometry/geom si dispo
    if ("lat" not in df_map.columns or "lon" not in df_map.columns):
        geom_col = None
        for c in ["geom", "geometry"]:
            if c in df_map.columns:
                geom_col = c
                break

        if geom_col is not None:
            try:
                # shapely Point : x=lon, y=lat
                df_map["lon"] = df_map[geom_col].apply(lambda g: getattr(g, "x", None))
                df_map["lat"] = df_map[geom_col].apply(lambda g: getattr(g, "y", None))
            except Exception:
                pass

    # Si toujours pas lat/lon → pas de crash
    if "lat" not in df_map.columns or "lon" not in df_map.columns:
        st.warning("Impossible d’afficher la carte : colonnes lat/lon absentes.")
        st.caption(f"Colonnes disponibles: {list(df_map.columns)}")
        return

    pts = df_map.dropna(subset=["lat", "lon"]).copy()
    if pts.empty:
        st.warning("Aucun point géolocalisé à afficher sur la carte.")
        return

    # Centre de la carte : ancre si disponible, sinon barycentre des points
    center = (
        [anchor_lat, anchor_lon]
        if (anchor_lat is not None and anchor_lon is not None)
        else [pts["lat"].mean(), pts["lon"].mean()])
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # Marqueur d'ancrage (point de départ)
    if anchor_lat is not None and anchor_lon is not None:
        folium.Marker(
            location=[anchor_lat, anchor_lon],
            tooltip="📍 Point d’ancrage",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),).add_to(m)

    cluster = MarkerCluster(name="POI Cluster").add_to(m)

    def color_by_type(t) -> str:
        # t peut être None, NaN (float), nombre, etc.
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = ""
        else:
            t = str(t)
        t = t.lower()
        if "central" in t:
            return "red"
        if "satellite" in t:
            return "orange"
        if "restaurant" in t:
            return "green"
        return "cadetblue"

    for _, r in pts.iterrows():
        name = r.get("name", "")
        cat = r.get("main_category", "")
        cat_type = r.get("type_principal", "")
        kind = r.get("type", "")
        address = r.get("address", "")
        url = r.get("url", "")
        price_level = r.get("price_level", "")
        price_range = r.get("price_range", "")
        price = r.get("price", "")

        # Popup HTML compact et lisible
        popup_html = f"""
        <div style="font-family: Arial; width: 280px;">
          <div style="font-size: 14px; font-weight: 700; margin-bottom: 6px;">{name}</div>
          <div style="font-size: 12px; margin-bottom: 6px;">
            <b>Catégorie :</b> {cat}<br>
            <b>Type principal :</b> {cat_type}<br>
            <b>Type :</b> {kind}<br>
            <b>Adresse :</b> {address}<br>
            <b>Niveau de budget :</b> {price_level}<br>
            <b>Prix min-max :</b> {price_range}<br>
            <b>Prix moyen :</b> {price}<br>
          </div>
        """
        # Lien externe si disponible
        if isinstance(url, str) and url.strip():
            popup_html += f'<div style="font-size:12px;"><a href="{url}" target="_blank">🔗 Visiter le site</a></div>'
        popup_html += "</div>"

        c = color_by_type(kind)
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.85,
            tooltip=f"{kind} — {name}",
            popup=folium.Popup(popup_html, max_width=320),).add_to(cluster)

    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=520)



def render_prime_map(itinerary_df: pd.DataFrame, anchor_lat: float, anchor_lon: float):
    """
    (RESP) UI/UX pure : rendu carte, clustering, popup HTML, couleur par type
    gestion pts vides + ancre optionnelle + LayerControl
    color_by_type simple et lisible
    Affiche une carte Folium avec clustering des points de l'itinéraire.
    - Centre la carte sur l'ancre si dispo, sinon sur la moyenne des points
    - Ajoute un marqueur "home" pour l'ancre
    - Ajoute un MarkerCluster pour éviter la surcharge visuelle
    - Couleur des points selon le type (central / satellite / restaurant)
    - Popup HTML enrichi (jour/slot/type/catégorie/score/adresse + lien)
    """
    pts = itinerary_df.dropna(subset=["lat", "lon"]).copy()
    if pts.empty:
        st.warning("Aucun point géolocalisé à afficher sur la carte.")
        return

    # Centre de la carte : ancre si disponible, sinon barycentre des points
    center = (
        [anchor_lat, anchor_lon]
        if (anchor_lat is not None and anchor_lon is not None)
        else [pts["lat"].mean(), pts["lon"].mean()])
    m = folium.Map(location=center, zoom_start=12, tiles="cartodbpositron")

    # Marqueur d'ancrage (point de départ)
    if anchor_lat is not None and anchor_lon is not None:
        folium.Marker(
            location=[anchor_lat, anchor_lon],
            tooltip="📍 Point d’ancrage",
            icon=folium.Icon(color="red", icon="home", prefix="fa"),).add_to(m)

    cluster = MarkerCluster(name="POI Cluster").add_to(m)

    def color_by_type(t) -> str:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            t = ""
        else:
            t = str(t)
        t = t.lower()
        if "central" in t:
            return "red"
        if "satellite" in t:
            return "orange"
        if "restaurant" in t:
            return "green"
        return "cadetblue"

    for _, r in pts.iterrows():
        name = r.get("name", "")
        jour = r.get("jour", "")
        slot = r.get("slot", "")
        cat = r.get("main_category", "")
        cat_type = r.get("type_principal", "")
        kind = r.get("type", "")
        address = r.get("address", "")
        url = r.get("url", "")
        price_level = r.get("price_level", "")
        price_range = r.get("price_range", "")
        price = r.get("price", "")

        # Popup HTML compact et lisible
        popup_html = f"""
        <div style="font-family: Arial; width: 280px;">
          <div style="font-size: 14px; font-weight: 700; margin-bottom: 6px;">{name}</div>
          <div style="font-size: 12px; margin-bottom: 6px;">
            <b>Jour :</b> {jour}<br>
            <b>Slot :</b> {slot}<br>
            <b>Catégorie :</b> {cat}<br>
            <b>Type_principal :</b> {cat_type}<br>
            <b>Type :</b> {kind}<br>
            <b>Adresse :</b> {address}<br>
            <b>Niveau de budget :</b> {price_level}<br>
            <b>Prix min-max :</b> {price_range}<br>
            <b>Prix moyen :</b> {price}<br>
          </div>
        """
        # Lien externe si disponible
        if isinstance(url, str) and url.strip():
            popup_html += f'<div style="font-size:12px;"><a href="{url}" target="_blank">🔗 Visiter le site</a></div>'
        popup_html += "</div>"

        c = color_by_type(kind)
        folium.CircleMarker(
            location=[float(r["lat"]), float(r["lon"])],
            radius=6,
            color=c,
            fill=True,
            fill_color=c,
            fill_opacity=0.85,
            tooltip=f"{kind} — {name}",
            popup=folium.Popup(popup_html, max_width=320),).add_to(cluster)

    folium.LayerControl().add_to(m)
    st_folium(m, width=None, height=520)
