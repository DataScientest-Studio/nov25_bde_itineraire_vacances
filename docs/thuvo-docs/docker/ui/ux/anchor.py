# ======================================================================================================================
# PRIME ENGINE — utilitaires ---> docker/ui/ux/anchor.py
# ======================================================================================================================

import streamlit as st
from infra.imports import pd
from services.geocode_client import geocode_address


def build_anchor_address(prefix: str = "prime") -> str:
    """
    Construit l'adresse texte à envoyer au géocodeur (Nominatim)
    à partir du formulaire (prime ou déco).
    """
    # (RESP) Bridge UX -> moteur : lit session_state (inputs) pour produire une string
    # prefix paramétrable ("prime" / "deco") => c'est exactement la bonne direction
    num = (st.session_state.get(f"{prefix}_anchor_numero") or "").strip()
    rue = (st.session_state.get(f"{prefix}_anchor_rue") or "").strip()
    cp  = (st.session_state.get(f"{prefix}_anchor_cp") or "").strip()
    vil = (st.session_state.get(f"{prefix}_anchor_ville") or "").strip()

    parts = [p for p in [num, rue, cp, vil] if p]
    addr = " ".join(parts).strip()
    if not addr:
        return ""
    return addr



def get_anchor_latlon(df_dt_context: pd.DataFrame, prefix: str = "prime"):
    """
    (RESP) Bridge UX/moteur : calcule ancre + écrit dans session_state
    Récupère les coordonnées (lat, lon) du point d’ancrage.
    Étapes
    ------
    1) Construit l'adresse depuis le formulaire.
    2) Tente un géocodage Nominatim.
    3) Stocke la réponse brute dans session_state pour debug.
    4) Si échec, fallback : moyenne des lat/lon disponibles dans df_dt_context.
    Paramètres
    ----------
    df_dt_context : pd.DataFrame
        DataFrame de POI de contexte (DataTourisme), contenant 'lat' et 'lon'.
    Retour
    ------
    (lat, lon) : (float | None, float | None)
    """
    addr = build_anchor_address(prefix)
    wanted_cp = (st.session_state.get(f"{prefix}_anchor_cp") or "").strip()
    wanted_city = (st.session_state.get(f"{prefix}_anchor_ville") or "").strip()

    lat, lon, raw = geocode_address(addr, wanted_cp=wanted_cp, wanted_city=wanted_city)

    # debug raw stocké par prefix (prime_anchor_geocode_raw / deco_anchor_geocode_raw)
    st.session_state[f"{prefix}_anchor_geocode_raw"] = raw

    # (fallback) si géocode fail -> barycentre df_dt_context
    if (lat is None or lon is None):
        tmp = df_dt_context.dropna(subset=["lat", "lon"])
        if len(tmp) > 0:
            lat = float(tmp["lat"].astype(float).mean())
            lon = float(tmp["lon"].astype(float).mean())

    # stocker dans des clés globales pour le reste de l'app
    if lat is not None and lon is not None:
        st.session_state[f"{prefix}_anchor_lat"] = float(lat)
        st.session_state[f"{prefix}_anchor_lon"] = float(lon)
        st.session_state[f"{prefix}_anchor"] = (float(lat), float(lon))

        # Optionnel: clés globales si le reste du code en dépend encore
        st.session_state["anchor_lat"] = float(lat)
        st.session_state["anchor_lon"] = float(lon)

        # clé par mode (source de vérité par prefix)
        st.session_state[f"{prefix}_anchor"] = (float(lat), float(lon))

        # compat historique si ton code l’utilise ailleurs
        if prefix == "prime":
            st.session_state["prime_anchor"] = (float(lat), float(lon))

    return lat, lon



def get_max_km(prefix: str) -> float:
    return float(st.session_state.get(f"{prefix}_distance_max_km", 3.0))
