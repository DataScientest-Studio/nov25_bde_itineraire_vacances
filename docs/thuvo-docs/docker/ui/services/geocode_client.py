# ======================================================================================================================
# PRIME ENGINE — utilitaires (distance, géocodage, anneaux, règles)
# ---> docker/ui/services/geocode_client.py
# ======================================================================================================================

import streamlit as st
from infra.imports import requests


def pick_best_candidate(results: list[dict], wanted_cp: str, wanted_city: str):
    """
    Choisit le "meilleur" résultat parmi ceux retournés par Nominatim.
    Stratégie (heuristique)
    -----------------------
    On score chaque résultat selon :
    - correspondance code postal (fort)
    - correspondance ville (moyen)
    - présence d'un numéro de rue (bonus)
    - type "house" / "residential" (petit bonus)
    Paramètres
    ----------
    results : list[dict]
        Liste de candidats Nominatim (JSON).
    wanted_cp : str
        Code postal entré par l’utilisateur.
    wanted_city : str
        Ville entrée par l’utilisateur.
    Retour
    ------
    dict | None
        Le candidat le mieux scoré, ou None si liste vide.
    """
    # (RESP) Moteur/heuristique interne : améliore la qualité du géocodage UX
    # pondération CP > ville > house_number => logique
    wanted_cp = (wanted_cp or "").strip()
    wanted_city = (wanted_city or "").strip().lower()

    if not results:
        return None

    def score(it: dict) -> int:
        addr = it.get("address", {}) or {}
        cp = str(addr.get("postcode") or "")
        city = str(
            addr.get("city")
            or addr.get("town")
            or addr.get("village")
            or "").lower()
        house = str(addr.get("house_number") or "")
        s = 0
        if wanted_cp and cp.startswith(wanted_cp):
            s += 10
        if wanted_city and wanted_city in city:
            s += 5
        if house:
            s += 4
        if it.get("type") in ("house", "residential"):
            s += 2
        return s
    return max(results, key=score)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def geocode_address(address: str, limit: int = 5, wanted_cp: str = "", wanted_city: str = ""):
    """
    Géocode une adresse via Nominatim (OpenStreetMap).
    Cache
    -----
    st.cache_data(ttl=24h) évite de re-taper l’API à chaque rerun Streamlit.
    Paramètres
    ----------
    address : str
        Adresse textuelle à géocoder.
    limit : int
        Nombre max de résultats retournés par Nominatim.
    Retour
    ------
    (lat, lon, raw)
    - lat : float | None
    - lon : float | None
    - raw : dict | list | None
    """
    # (RESP) Service externe (Nominatim) + cache 24h
    # Cache long: évite de spammer Nominatim pendant tests/rerun
    # (NOTE) le param prefix sert uniquement à lire cp/ville dans session_state pour pick_best_candidate

    # Protection contre les adresses trop courtes/vides
    if not address or len(address.strip()) < 3:
        return None, None, None

    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": address, "format": "json", "limit": limit, "addressdetails": 1}
    headers = {"User-Agent": "streamlit-itinerary-app/1.0 (contact: vominhngocthu@gmail.com)"}

    try:
        r = requests.get(url, params=params, headers=headers, timeout=10)
        r.raise_for_status()
        results = r.json()
    except Exception:
        return None, None, None

    if not results:
        return None, None, []

    # On choisit le meilleur candidat selon CP + ville (heuristique)
    best = pick_best_candidate(results, wanted_cp=wanted_cp, wanted_city=wanted_city)
    if not best:
        return None, None, results
    
    try:
        lat = float(best["lat"])
        lon = float(best["lon"])
    except Exception:
        return None, None, results

    return lat, lon, best
