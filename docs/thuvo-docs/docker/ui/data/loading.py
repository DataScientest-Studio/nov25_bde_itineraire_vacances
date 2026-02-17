# ======================================================================================================================
# DATA LOADING --> docker/ui/data/loading.py
# ======================================================================================================================

import streamlit as st
from infra.imports import pd, np, os, requests


# -------------------------------------------------------------------------------------------------
# 1) Helpers API
# -------------------------------------------------------------------------------------------------
def _api_base_url() -> str:
    return os.getenv("API_BASE_URL", "").strip().rstrip("/")


def _api_enabled() -> bool:
    return bool(_api_base_url())


def _api_get_json(path: str, params: dict | None = None) -> dict:
    base = _api_base_url()
    if not base:
        raise RuntimeError("API_BASE_URL non défini (API disabled).")

    url = f"{base}{path}"
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    return r.json()


# -------------------------------------------------------------------------------------------------
# 2) Normalisation minimale commune (anti-KeyError + types stables)
# -------------------------------------------------------------------------------------------------
def _normalize_min_schema(d: pd.DataFrame) -> pd.DataFrame:
    # colonnes minimales attendues dans l'UI
    for col, default in [
        ("source_id", ""),      # Identifiant propre à la source
        ("source", ""),         # Nom de la source (dt, ab, ta, ...)
        ("type", ""),           # Type d'entité (poi, restaurant, lodging, ...)
        ("url", ""),            # Lien externe
        ("lat", np.nan),        # Latitude
        ("lon", np.nan),        # Longitude
        ("name", ""),           # Nom affiché
        ("address", ""),        # Adresse textuelle
        ("postal_code", ""),    # Code postal
        ("city", ""),           # Ville
        ("country", "France"),  # Pays (France par défaut)
        ("snippet", ""),        # Description courte / extrait
        ("rating", np.nan),     # Note moyenne
        ("review_count", 0),    # Nombre d'avis
    ]:
        if col not in d.columns:
            d[col] = default

    # Typage coords
    d["lat"] = pd.to_numeric(d.get("lat"), errors="coerce")
    d["lon"] = pd.to_numeric(d.get("lon"), errors="coerce")

    # Typage note
    d["rating"] = pd.to_numeric(d.get("rating"), errors="coerce")

    # Typage nb avis
    d["review_count"] = (
        pd.to_numeric(d.get("review_count"), errors="coerce")
        .fillna(0)
        .astype(int)
    )

    # Nettoyage texte minimal
    for txt_col in ["name", "address", "city", "postal_code"]:
        if txt_col in d.columns:
            d[txt_col] = (
                d[txt_col]
                .astype("string")
                .fillna("")
                .str.strip()
            )

    return d


# -------------------------------------------------------------------------------------------------
# 3) Loaders (API ou parquet)
# -------------------------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner="Chargement des données…")
def load_dfs(dt_path: str, ab_path: str, ta_path: str, api_limit: int = 5000):
    """
    Charge et prépare les DataFrames principaux de l'application Streamlit.

    Mode API (recommandé en Docker):
      - si API_BASE_URL est défini, charge via endpoints /ctx/dt, /ctx/ab, /ctx/ta
      - api_limit limite le volume ramené (évite de charger 400k lignes au refresh)

    Mode local (fallback):
      - sinon, charge les parquets via dt_path/ab_path/ta_path

    Retour:
      (df_dt, df_ab, df_ta) normalisés minimalement
    """
    # -----------------------------
    # A) Mode API
    # -----------------------------
    if _api_enabled():
        try:
            dt = _api_get_json("/ctx/dt", params={"limit": api_limit})
            ta = _api_get_json("/ctx/ta", params={"limit": api_limit})

            # Airbnb parfois non dispo : si endpoint absent, on renvoie DF vide normalisé
            try:
                ab = _api_get_json("/ctx/ab", params={"limit": api_limit})
                df_ab = pd.DataFrame(ab.get("rows", []))
            except Exception:
                df_ab = pd.DataFrame()

            df_dt = pd.DataFrame(dt.get("rows", []))
            df_ta = pd.DataFrame(ta.get("rows", []))

            df_dt = _normalize_min_schema(df_dt)
            df_ab = _normalize_min_schema(df_ab)
            df_ta = _normalize_min_schema(df_ta)

            return df_dt, df_ab, df_ta

        except Exception as e:
            import os
            if os.getenv("DEBUG_UI", "0") == "1":
                st.warning(
                    "API indisponible — fallback parquet local actif."
                )
            # on continue en mode parquet

    # -----------------------------
    # B) Mode parquet (fallback)
    # -----------------------------
    try:
        df_dt = pd.read_parquet(dt_path)
        df_ab = pd.read_parquet(ab_path)
        df_ta = pd.read_parquet(ta_path)
    except Exception as e:
        raise RuntimeError(
            "Impossible de charger les fichiers parquet.\n"
            f"DT_PATH={dt_path}\nAB_PATH={ab_path}\nTA_PATH={ta_path}\n"
            f"Erreur: {type(e).__name__}: {e}"
        )

    df_dt = _normalize_min_schema(df_dt)
    df_ab = _normalize_min_schema(df_ab)
    df_ta = _normalize_min_schema(df_ta)

    return df_dt, df_ab, df_ta


def load_context_dfs(paths, api_limit: int = 5000):
    """
    Wrapper pratique: prend DataPaths (config.paths) et renvoie (df_dt, df_ab, df_ta).
    Priorité: variables d'environnement (Docker) -> fallback sur paths.* (local/dev).
    """
    dt_path = os.getenv("DT_PATH") or getattr(paths, "dt_path", "")
    ab_path = os.getenv("AB_PATH") or getattr(paths, "ab_path", "")
    ta_path = os.getenv("TA_PATH") or getattr(paths, "ta_path", "")

    return load_dfs(dt_path, ab_path, ta_path, api_limit=api_limit)
