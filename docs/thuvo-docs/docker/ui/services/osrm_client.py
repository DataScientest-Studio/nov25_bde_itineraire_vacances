# ======================================================================================================================
# UI HELPERS
# ---> docker/ui/services/osrm_client.py
# ======================================================================================================================

import streamlit as st
from infra.imports import requests


OSRM_BASE = "https://router.project-osrm.org"
def _osrm_route_safe(lon1, lat1, lon2, lat2, profile: str, timeout: int = 10):
    """
    Appel OSRM safe:
    - retourne (distance_m, duration_s) ou (None, None) si échec
    """
    if lon1 is None or lat1 is None or lon2 is None or lat2 is None:
        return None, None

    # Stabilise le cache Streamlit (évite 48.8566 vs 48.8566001)
    lon1, lat1, lon2, lat2 = map(lambda x: round(float(x), 6), [lon1, lat1, lon2, lat2])

    try:
        url = f"{OSRM_BASE}/route/v1/{profile}/{lon1},{lat1};{lon2},{lat2}"
        params = {"overview": "false", "steps": "false", "alternatives": "false"}
        r = requests.get(url, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok" or not data.get("routes"):
            return None, None
        route0 = data["routes"][0]
        return route0.get("distance"), route0.get("duration")
    except Exception:
        return None, None


@st.cache_data(show_spinner=False)
def osrm_leg(lat1, lon1, lat2, lon2, profile="walking"):
    """
    (RESP) Service externe (OSRM) : enrichissement UX "temps/distance"
    Retourne (distance_m, duration_s) entre 2 points via OSRM.
    OSRM profile réel : "walking" ou "driving"
    """
    return _osrm_route_safe(lon1, lat1, lon2, lat2, profile=profile, timeout=10)


@st.cache_data(show_spinner=False)
def osrm_walk_minutes_cached(lon1, lat1, lon2, lat2, timeout=15):
    """
    (RESP) Service externe (OSRM) : Calcule le temps à pied OSRM (minutes) entre deux points.
    OSRM attend lon,lat
    """
    dist, dur = _osrm_route_safe(lon1, lat1, lon2, lat2, profile="walking", timeout=timeout)
    if dur is None:
        return None
    return int(round(dur / 60))


def walk_minutes_anchor_to_central(df_itin, anchor_lon, anchor_lat, lon_col="lon", lat_col="lat", type_col="type"):
    """
    (RESP) UX helper : afficher un "temps depuis ancre vers central"
    (DÉP) Itinéraire DF doit contenir lat/lon et une colonne type (ou fallback 1ère ligne)
    Calcule ancrage -> POI central (ou 1ère ligne si POI central introuvable).
    """
    if df_itin is None or df_itin.empty:
        return None
    if anchor_lon is None or anchor_lat is None:
        return None

    central = None
    # Trouver la ligne "POI central" (sinon fallback = première ligne)
    if type_col in df_itin.columns:
        mask = df_itin[type_col].astype(str).str.lower().eq("poi central")
        if mask.any():
            central = df_itin[mask].head(1)
    if central is None or central.empty:
        central = df_itin.head(1)
    try:
        lon2 = float(central.iloc[0][lon_col])
        lat2 = float(central.iloc[0][lat_col])
    except Exception:
        return None

    return osrm_walk_minutes_cached(anchor_lon, anchor_lat, lon2, lat2)


def add_osrm_walk_drive(d):
    """
    (RESP) UX helper : enrichit le tableau journalier avec temps/distance entre slots
    Ajoute sur le df du jour : walk_min_from_prev, walk_km_from_prev
    Nécessite colonnes 'lat' et 'lon'. df déjà trié par slot.
    """
    d = d.copy()

    walk_dist_m = [0.0]
    walk_dur_s = [0.0]
    drive_dist_m = [0.0]
    drive_dur_s = [0.0]

    for i in range(1, len(d)):
        lat1, lon1 = float(d.iloc[i - 1]["lat"]), float(d.iloc[i - 1]["lon"])
        lat2, lon2 = float(d.iloc[i]["lat"]), float(d.iloc[i]["lon"])

        wd, wt = osrm_leg(lat1, lon1, lat2, lon2, profile="walking")
        dd, dt = osrm_leg(lat1, lon1, lat2, lon2, profile="driving")

        walk_dist_m.append(float(wd) if wd is not None else 0.0)
        walk_dur_s.append(float(wt) if wt is not None else 0.0)
        drive_dist_m.append(float(dd) if dd is not None else 0.0)
        drive_dur_s.append(float(dt) if dt is not None else 0.0)

    d["walk_min_from_prev"] = [int(round(x / 60)) for x in walk_dur_s]
    d["drive_min_from_prev"] = [int(round(x / 60)) for x in drive_dur_s]

    d["walk_km_from_prev"] = [round(x / 1000, 2) for x in walk_dist_m]
    d["drive_km_from_prev"] = [round(x / 1000, 2) for x in drive_dist_m]

    total_walk_min = int(round(sum(walk_dur_s) / 60))
    total_drive_min = int(round(sum(drive_dur_s) / 60))

    return d, total_walk_min, total_drive_min