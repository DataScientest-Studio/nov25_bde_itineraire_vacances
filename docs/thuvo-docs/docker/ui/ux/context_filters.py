# ======================================================================================================================
# FILTER CONTEXT ---> docker/ui/ux/context_filters.py
# ======================================================================================================================

import streamlit as st
from infra.imports import pd
from data.normalization_admin import norm_txt as _norm_txt, to_str_series as _to_str_series


def filter_context_admin(df: pd.DataFrame, prefix: str = "prime") -> pd.DataFrame:
    """
    (RESP) Bridge UX -> moteur : dépend de st.session_state (anchor_region / anchor_department)
    Filtre un DataFrame par contexte administratif (région / département)
    en se basant sur les choix de l'utilisateur dans la sidebar (session_state).
    (DÉPENDANCES) nécessite _to_str_series et _norm_txt déjà définies plus haut
    Objectif:
    - Réduire le "contexte" de recherche (POI / restos / etc.) à une zone pertinente
    - Comparaison robuste grâce à _to_str_series + _norm_txt (accents/espaces/casse)
    """
    d = df.copy()

    region = (st.session_state.get(f"{prefix}_anchor_region") or st.session_state.get("anchor_region") or "").strip()
    dep = (st.session_state.get(f"{prefix}_anchor_department") or st.session_state.get("anchor_department") or "").strip()

    # Filtre région si la valeur existe et que la colonne est disponible
    if region and "region" in d.columns:
        region_norm = _norm_txt(region)
        d = d[_to_str_series(d["region"]).apply(_norm_txt) == region_norm]

    # Filtre département si la valeur existe et que la colonne est disponible
    if dep and "department" in d.columns:
        dep_norm = _norm_txt(dep)
        d = d[_to_str_series(d["department"]).apply(_norm_txt) == dep_norm]

    return d
