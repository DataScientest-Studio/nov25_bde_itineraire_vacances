# ======================================================================================================================
# FILTER CONTEXT # ---> docker/ui/ux/restaurants.py
# ======================================================================================================================

import streamlit as st
from infra.imports import pd, np

from core.geo_utils import safe_haversine_km
from data.normalization_admin import norm_txt as _norm_txt, to_str_series as _to_str_series


def filter_restaurants(df_ta_context: pd.DataFrame, anchor_lat: float, anchor_lon: float, prefix: str = "prime",) -> pd.DataFrame:
    """
    Prépare / filtre les restaurants (TripAdvisor) autour d'un point d'ancrage.
    1) Garde uniquement les restos géolocalisés
    2) Calcule la distance à l'ancre
    3) Calcule un score restaurant :
       - si reco_score existe : on l'utilise
       - sinon : fallback rating + log(1 + review_count)
    4) Ajuste le score selon le budget via price_level (si dispo)
    5) Applique les filtres cuisine_continent et régimes (food_diets)
       - continent : filtre "soft" (si ça vide tout, on ignore)
       - diets : filtre "hard"
    """
    # (RESP) Moteur + UX : applique préférences (continent/diets/budget) + scoring restos
    # (UX) prefix = cohérence prime vs déco (bonne pratique)
    # (DÉPENDANCES) safe_haversine_km, numpy, pandas
    # (INTRUSION UI) st.info(...) à l’intérieur (c’est UI, pas moteur)

    # --- Récupération des filtres UI (COHÉRENTS avec prefix) ---
    selected_conts = st.session_state.get(f"{prefix}_food_continents") or []
    diets_selected = st.session_state.get(f"{prefix}_food_diets") or []
    budget = st.session_state.get(f"{prefix}_budget", "Moyen")

    dbg = st.session_state.get("_dbg_prime")
    if dbg is not None:
        dbg.write("— filter_restaurants() START —")
        dbg.write(f"budget key read = {prefix}_budget -> {budget}")
        dbg.write(f"selected_conts = {selected_conts} | diets = {diets_selected}")
        dbg.write(f"rows input df_ta_context = {len(df_ta_context)}")

    # --- Base : géoloc ---
    r = df_ta_context.copy()
    r = r.dropna(subset=["lat", "lon"])

    # --- Distance au point d'ancrage ---
    # (on garde safe_haversine_km, mais apply peut être lourd)
    r["distance_to_anchor_km"] = r.apply(
        lambda x: safe_haversine_km(anchor_lat, anchor_lon, x["lat"], x["lon"]),
        axis=1,)
    r = r.dropna(subset=["distance_to_anchor_km"])

    if dbg is not None:
        dbg.write(f"rows after geo+distance = {len(r)}")

    if anchor_lat is None or anchor_lon is None:
        return r.iloc[0:0]  # df vide

    # --- Score restaurant (priorité à un score pré-calculé) ---
    # (RESP) score resto : reco_score sinon fallback rating + log1p(review_count)
    if "reco_score" in r.columns:
        r["resto_score"] = pd.to_numeric(r["reco_score"], errors="coerce").fillna(0.0)
    else:
        rr = pd.to_numeric(r.get("rating"), errors="coerce").fillna(0.0)
        rc = pd.to_numeric(r.get("review_count"), errors="coerce").fillna(0.0)
        r["resto_score"] = rr + np.log1p(rc)

    # --- Ajustement BUDGET via price_level (TripAdvisor) ---
    # (RESP) budget via price_level (soft bonus)
    # --- Ajustement BUDGET via price_level (TripAdvisor) ---
    if "price_level" in r.columns:
        # 1) convertir price_level texte -> échelle interne
        pl = r["price_level"].astype(str).str.strip().str.lower()

        PRICE_MAP = {
            "eco": 1.0,
            "normal": 2.0,
            "confort": 3.0,
            "premium": 4.0,}
        price = pl.map(PRICE_MAP)  # float avec NaN si inconnu

        # 2) normaliser le choix UI budget -> clé de budget_ranges
        b = str(budget).strip().lower()
        BUDGET_UI_MAP = {
            "éco": "eco",
            "normal": "normal",
            "confort": "confort",
            "premium": "premium",}
        bkey = BUDGET_UI_MAP.get(b, "normal")

        # 3) plages sur 1..4 (eco=1, normal=2, confort=3, premium=4)
        budget_ranges = {
            "eco": (1.0, 1.0),
            "normal": (2.0, 2.0),
            "confort": (3.0, 3.0),
            "premium": (4.0, 4.0),}
        low, high = budget_ranges.get(bkey, (2.0, 2.0))

        # DEBUG utile
        if dbg is not None:
            dbg.write(f"budget(UI)={budget} -> bkey={bkey} -> range=({low},{high})")
            dbg.write(f"price_level unique raw={sorted(pl.dropna().unique().tolist())[:20]}")
            dbg.write(f"price numeric sample={price.dropna().head(10).tolist()}")

        # distance à la plage (0 si dedans) ; NaN => dist_to_range = NaN
        dist_to_range = np.where(
            price < low, low - price,
            np.where(price > high, price - high, 0.0))

        # bonus décroissant quand on s'éloigne ; NaN => 0
        price_bonus = 1.0 - (dist_to_range / 2.0)
        price_bonus = np.clip(price_bonus, a_min=0.0, a_max=None)
        price_bonus = pd.Series(price_bonus, index=r.index).fillna(0.0)

        # pondération finale
        r["resto_score"] = r["resto_score"] + 0.5 * price_bonus

    # --- Filtre par continent culinaire (soft) ---
    # (RESP) filtre continent (soft)
    # (INTRUSION UI) st.info(...) si filtre vide -> OK en UX, mais c’est UI dans une fonction moteur
    if selected_conts and "cuisine_continent" in r.columns:
        cont = _to_str_series(r["cuisine_continent"]).apply(_norm_txt)
        selected_norm = {_norm_txt(x) for x in selected_conts}

        r_filtered = r[cont.isin(selected_norm)].copy()

        if not r_filtered.empty:
            r = r_filtered
        else:
            st.info("Aucun resto TripAdvisor ne correspond au continent sélectionné → filtre continent ignoré.")

    # --- Filtres par régimes (hard) ---
    diets = set(diets_selected)
    if diets:
        if "Sans gluten" in diets and "gluten_free" in r.columns:
            r = r[r["gluten_free"] == True]

        if "Halal" in diets and "is_halal" in r.columns:
            r = r[r["is_halal"] == True]

        if "Casher" in diets and "is_kosher" in r.columns:
            r = r[r["is_kosher"] == True]

        if "Vegan" in diets and "vegan_options" in r.columns:
            r = r[r["vegan_options"] == True]

        if "Végétarien" in diets and "vegetarian_friendly" in r.columns:
            r = r[r["vegetarian_friendly"] == True]

    return r



def fallback_restaurants_from_dt(df_dt_context: pd.DataFrame, anchor_lat: float, anchor_lon: float) -> pd.DataFrame:
    """
    Fallback : si TripAdvisor ne renvoie aucun restaurant, on extrait des "restaurants"
    depuis DataTourisme en filtrant les catégories Gastronomie.
    Attention :
    - Ce fallback est moins fiable (pas d'avis / notes comparables)
    - On recycle score_prime si dispo pour avoir un classement minimal
    """
    # (RESP) fallback moteur : reconstitue un set de restos depuis DT
    # (ROBUSTESSE) supporte 2 schémas (main_category_compressed / type_principal)
    d = df_dt_context.dropna(subset=["lat", "lon"]).copy()

    # Filtre "Gastronomie" côté DataTourisme (selon la colonne disponible)
    if "main_category_compressed" in d.columns:
        d["_mc"] = _to_str_series(d["main_category_compressed"]).apply(_norm_txt)
        d = d[d["_mc"].str.contains(_norm_txt("Gastronomie"), na=False)].copy()
        d.drop(columns=["_mc"], inplace=True, errors="ignore")

    elif "type_principal" in d.columns:
        d["_tp"] = _to_str_series(d["type_principal"]).apply(_norm_txt)
        d = d[d["_tp"].str.contains(_norm_txt("Gastronomie"), na=False)].copy()
        d.drop(columns=["_tp"], inplace=True, errors="ignore")

    if d.empty:
        return d

    # Score resto minimal : on réutilise score_prime si existant
    if "score_prime" in d.columns:
        d["resto_score"] = pd.to_numeric(d["score_prime"], errors="coerce").fillna(0.0)
    else:
        d["resto_score"] = 0.0

    # Distance à l'ancre (utile pour choose_restaurant_near)
    d["distance_to_anchor_km"] = d.apply(
        lambda x: safe_haversine_km(anchor_lat, anchor_lon, x["lat"], x["lon"]),
        axis=1)
    d = d.dropna(subset=["distance_to_anchor_km"])

    return d