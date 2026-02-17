# ======================================================================================================================
# STATE UX HELPERS ---> docker/ui/ux/state.py
# ======================================================================================================================

import streamlit as st

DEFAULTS = {
    # GLOBAL (partagé Prime/Déco)
    "DEBUG": False,
    "anchor_lat": None,
    "anchor_lon": None,
    "dt_ctx": None,
    "ta_ctx": None,
    "ab_ctx": None,

    # Anchor géographique
    "anchor_region": "",
    "anchor_department": "",

    # Adresse précise de l'ancrage (géocodée ensuite)
    "anchor_numero": "",
    "anchor_rue": "",
    "anchor_cp": "",
    "anchor_ville": "",

    # Paramètres de séjour
    "nb_personnes": 2,
    "budget": "Moyen",
    "duree_sejour": 1,
    "rythme": "Normal",

    # Filtres de catégories
    "main_categories": [],

    # Logique Incontournables vs Découverte
    "mode_selection": "Mix",
    "ratio_incontournables": 40,

    # Préférences gastronomiques
    "food_continents": [],
    "food_diets": [],

    # Contraintes de distance
    "distance_max_km": 3.0,

    # Sorties / résultats Prime
    "prime_generate": False,
    "prime_itinerary": None,
    "prime_anchor": None,
    "prime_pois_out": None,

    # ----------------------------
    # MODE DECOUVERTE (state séparé)
    # ----------------------------
    "deco_anchor_numero": "",
    "deco_anchor_rue": "",
    "deco_anchor_cp": "",
    "deco_anchor_ville": "",
    "deco_anchor_region": "",
    "deco_anchor_department": "",

    "deco_nb_personnes": 2,
    "deco_duree_sejour": 1,
    "deco_budget": "Moyen",
    "deco_rythme": "Normal",

    "deco_main_categories": [],
    "deco_mode_selection": "Mix",

    "deco_food_continents": [],
    "deco_food_diets": [],

    "deco_distance_max_km": 3.0,
    "deco_visible_types": [],

    # sorties Découverte
    "discover_generate": False,
    "df_visible_pois": None,

    # DECO — shortlist UX
    "pois_added": [],
    "pois_added_ids": set(),}


def _clone_default(v):
    # évite de partager les mêmes objets mutables entre sessions/reruns
    if isinstance(v, (list, dict, set)):
        return v.copy()
    return v


def init_state():
    """
    Initialise le session_state Streamlit avec les valeurs par défaut.
    RESPONSABILITÉ : "Init idempotent" : crée les clés manquantes, sans écraser celles existantes.
    IMPACT UX : Empêche les KeyError + stabilise l'UX (widgets, conditions, affichage).
    """
    # Pattern classique streamlit
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = _clone_default(v)


def reset_state():
    """
    RESPONSABILITÉ : Reset global (toutes les clés connues reviennent au défaut).
    IMPACT UX : Repartir de zéro quand l'utilisateur clique "Réinitialiser".
    RISQUE UX : Reset global efface à la fois Prime et Déco (et potentiellement des choses utiles comme pois_added)
    """
    for k, v in DEFAULTS.items():
        st.session_state[k] = _clone_default(v)


def anchor_form_is_valid(prefix: str = "prime") -> bool:
    """
    RESPONSABILITÉ (UX) : Validation "safe" du formulaire d'ancrage pour un mode donné.
    DÉPENDANCES : Lit : {prefix}_anchor_rue / {prefix}_anchor_cp / {prefix}_anchor_ville
    Exemples :
      - prefix="prime" -> prime_anchor_*
      - prefix="deco"  -> deco_anchor_*
    """
    rue = (st.session_state.get(f"{prefix}_anchor_rue") or "").strip()
    cp = (st.session_state.get(f"{prefix}_anchor_cp") or "").strip()
    ville = (st.session_state.get(f"{prefix}_anchor_ville") or "").strip()
    return bool(rue) and bool(cp) and bool(ville)


def anchor_ready_for_geocode(prefix: str = "prime") -> bool:
    """
    RESPONSABILITÉ (UX) : Condition "minimale" pour autoriser géocodage / génération.
    DÉPENDANCES : Lit : {prefix}_anchor_rue / {prefix}_anchor_cp / {prefix}_anchor_ville
    Ici : on réutilise la validation du formulaire (même règle).
    """
    return anchor_form_is_valid(prefix)


def sync_global_from_prefix(prefix: str = "prime") -> None:
    """
    Pont de compat: recopie les champs {prefix}_* vers les clés globales non préfixées.
    À appeler juste avant d'utiliser du code legacy qui lit anchor_* / budget / etc.
    """
    # paramètres anchor
    st.session_state["anchor_rue"] = st.session_state.get(f"{prefix}_anchor_rue", "")
    st.session_state["anchor_cp"] = st.session_state.get(f"{prefix}_anchor_cp", "")
    st.session_state["anchor_ville"] = st.session_state.get(f"{prefix}_anchor_ville", "")
    st.session_state["anchor_region"] = st.session_state.get(f"{prefix}_anchor_region", "")
    st.session_state["anchor_department"] = st.session_state.get(f"{prefix}_anchor_department", "")

    # paramètres séjour
    st.session_state["nb_personnes"] = st.session_state.get(f"{prefix}_nb_personnes", 2)
    st.session_state["duree_sejour"] = st.session_state.get(f"{prefix}_duree_sejour", 1)
    st.session_state["budget"] = st.session_state.get(f"{prefix}_budget", "Moyen")
    st.session_state["rythme"] = st.session_state.get(f"{prefix}_rythme", "Normal")

    # préférences / filtres
    st.session_state["main_categories"] = st.session_state.get(f"{prefix}_main_categories", [])
    st.session_state["mode_selection"] = st.session_state.get(f"{prefix}_mode_selection", "Mix")
    st.session_state["food_continents"] = st.session_state.get(f"{prefix}_food_continents", [])
    st.session_state["food_diets"] = st.session_state.get(f"{prefix}_food_diets", [])
    st.session_state["distance_max_km"] = st.session_state.get(f"{prefix}_distance_max_km", 3.0)
