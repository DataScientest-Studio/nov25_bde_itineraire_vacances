from typing import Any
import zipfile
import json
from collections import Counter
from datetime import datetime
import pandas as pd
from typing import Any
import ast
import re
from pathlib import Path
import numpy as np


# =====================================================================================
# HELPERS GÉNÉRIQUES (robustes notebook → script)
# =====================================================================================

def as_list(x: Any) -> list:
    """
    Normalise une valeur potentiellement hétérogène en liste.
    Cas Datatourisme fréquents :
    - None        → []
    - list        → list
    - dict / str  → [x]
    Objectif :
    - éviter les bugs quand un champ est tantôt un dict, tantôt une liste
    - simplifier les boucles for (for item in as_list(...))
    """
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def list_object_files(zip_path: str) -> list[str]:
    """
    Retourne la liste des fichiers POI Datatourisme contenus dans l'archive ZIP.
    Règle Datatourisme :
    - les POI sont stockés dans le dossier 'objects/'
    - chaque POI correspond à un fichier JSON individuel
    Cette fonction permet :
    - un point unique de vérité pour le listing des POI
    - d’éviter les incohérences entre blocs (notebook copié → script)
    """
    with zipfile.ZipFile(zip_path) as z:
        return [
            n for n in z.namelist()
            if n.startswith("objects/") and n.endswith(".json")]


# ================================================================================================
# E) Extract
# ================================================================================================
def extract_datatourisme(path: str) -> pd.DataFrame:
    """
    Hypothèse :
    - le flux Datatourisme brut (ZIP + JSON) a déjà été ingéré
    - un parquet "clean brut" existe sur disque
    - cette fonction ne fait QUE charger cet artefact

    Charge un snapshot Datatourisme déjà ingéré (parquet/csv).
    Le parsing du flux brut est hors périmètre de ce pipeline.
    est réalisé dans un pipeline d’ingestion séparé.
    """
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Format non supporté: {path}")



# ================================================================================================
# C) CHECK data-safe (structure / volume / types / géoloc / fraîcheur)
# ================================================================================================


#-----------------------------------------
# C.1) Check volume réel de POI¶
#-----------------------------------------
"""
Identification des fichiers POI dans le flux Datatourisme
pour détecter toute anomalie de volume avant de lancer des traitements plus coûteux.
Objectif :
- Lister uniquement les objets métiers (Points of Interest)
- Évaluer rapidement le volume réel de données à traiter
- Disposer d’un indicateur simple pour détecter une anomalie de flux (ex : flux vide ou incomplet)
"""
zip_path = r"C:\Users\DELL\data\datatourisme\snapshots\2026-01-25_flux250126_complete.zip"

# Récupération de la liste complète des fichiers présents dans l’archive
# Les POI sont stockés dans le dossier "objects/" et chaque POI correspond
object_files = list_object_files(zip_path)

# Affichage du nombre total de POI détectés dans le flux
# aux runs précédents pour détecter toute variation anormale
print("Nombre total de POI :", len(object_files))



#-----------------------------------------
# C.2) Check répartition par type (@type)¶
#-----------------------------------------
"""
Objectif : chaque Point of Interest est décrit par un ou plusieurs types. Avant toute transformation ou scoring,
il est crucial de vérifier que la répartition des types est cohérente avec les attentes métier.
- Vérifier que les grands types Datatourisme sont bien présents
- Détecter rapidement une anomalie métier (ex: Restaurants absents)
- Faire un sanity check avant transformation / scoring
Approche :
- Lecture directe depuis le ZIP (sans extraction)
- Échantillonnage (20 000 POI suffisent pour ce check)
- Comptage des types avec Counter
"""
zip_path = r"C:\Users\DELL\data\datatourisme\snapshots\2026-01-25_flux250126_complete.zip"
type_counter = Counter()

with zipfile.ZipFile(zip_path) as z:
    
    # On parcourt un échantillon (rapide et suffisant)
    for name in object_files[:20_000]:
        
        # Lecture du JSON directement depuis l'archive
        data = json.loads(z.read(name))
        
        # Le champ @type peut être une string ou une liste
        types = data.get("@type", [])
        
        if isinstance(types, str):
            types = [types]
            
        # Mise à jour du compteur
        type_counter.update(types)

# Affichage des types les plus fréquents
type_counter.most_common(10)



#-----------------------------------------
# C.3) Check blog-ready
#-----------------------------------------
"""
Inspection de la structure d’un POI Datatourisme : Cette inspection permet de constater que certains champs,
comme isLocatedAt, peuvent varier en structure (liste ou objet) selon les sources de données amont.
Objectif :
- Explorer le schéma réel d’un objet Datatourisme
- Identifier les champs disponibles au niveau racine
- Comprendre la structure des champs imbriqués (ex: isLocatedAt)
Cette étape est essentielle pour :
- adapter les règles de qualité
- écrire un code robuste face aux variations de schéma
- éviter les hypothèses incorrectes sur la structure des données
"""

zip_path = r"C:\Users\DELL\data\datatourisme\snapshots\2026-01-25_flux250126_complete.zip"

with zipfile.ZipFile(zip_path) as z:

    # Sélection d’un POI représentatif (premier fichier du flux)
    # Un seul objet suffit pour explorer la structure générale
    first = object_files[0]

    # Lecture du fichier JSON directement depuis l’archive
    data = json.loads(z.read(first))

# Affichage des premières clés au niveau racine
# Permet d’identifier rapidement les champs principaux du modèle Datatourisme
print("Keys top-level:", list(data.keys())[:30])

# Affichage du ou des types métier associés au POI
# (@type peut contenir plusieurs valeurs)
print("\n@type:", data.get("@type"))

# Vérification du type Python du champ isLocatedAt
# (dict, list ou None selon les sources amont)
print("\nisLocatedAt type:", type(data.get("isLocatedAt")))

# Aperçu partiel du contenu de isLocatedAt
# Utile pour comprendre l’imbrication réelle sans surcharger l’affichage
print("\nisLocatedAt (preview):", str(data.get("isLocatedAt"))[:800])



#-----------------------------------------
# C.4) Check GEO (lat/long)¶
#-----------------------------------------
"""
CHECK DATA-SAFE — Géolocalisation (latitude/longitude) peuvent être positionnées à différents niveaux de la hiérarchie isLocatedAt.
il est donc nécessaire de détecter et normaliser ces structures avant toute exploitation géographique.
Objectif :
- Vérifier la proportion de POI géocodés (cartographiables)
- Détecter si un flux a un problème de géométrie (beaucoup de POI sans coords)
Approche :
- Lecture directe depuis le ZIP (sans extraction)
- Échantillonnage (10 000 POI suffisent)
- Comptage GEO OK vs GEO manquant
"""
zip_path = r"C:\Users\DELL\data\datatourisme\snapshots\2026-01-25_flux250126_complete.zip"

geo_ok = 0
geo_missing = 0

# Taille de l’échantillon
N = 10_000

with zipfile.ZipFile(zip_path) as z:

    # Parcours d’un échantillon de POI
    for name in object_files[:N]:

        # Lecture du POI directement depuis l’archive ZIP
        data = json.loads(z.read(name))

        geo_found = False

        # Parcours des localisations associées au POI
        # as_list() sécurise le cas dict / list / None
        for loc in as_list(data.get("isLocatedAt")):

            # Cas 1 — Coordonnées directement sous isLocatedAt
            geo = loc.get("schema:geo")
            if isinstance(geo, dict):
                lat = geo.get("schema:latitude")
                lon = geo.get("schema:longitude")
                if lat is not None and lon is not None:
                    geo_found = True
                    break

            # Cas 2 — Coordonnées imbriquées dans schema:address
            for addr in as_list(loc.get("schema:address")):
                geo = addr.get("schema:geo")
                if isinstance(geo, dict):
                    lat = geo.get("schema:latitude")
                    lon = geo.get("schema:longitude")
                    if lat is not None and lon is not None:
                        geo_found = True
                        break
            if geo_found:
                break

        # Mise à jour des compteurs qualité
        if geo_found:
            geo_ok += 1
        else:
            geo_missing += 1

# Résumé du check GEO
total = geo_ok + geo_missing
print(f"GEO OK : {geo_ok}/{total} ({geo_ok / total:.1%})")
print(f"GEO manquant : {geo_missing}/{total} ({geo_missing / total:.1%})")



#-----------------------------------------
# C.5) Check Fraîcheur du flux
#-----------------------------------------
"""
CHECK DATA-SAFE — Fraîcheur des données Datatourisme :
La vérification de la fraîcheur des données confirme que le flux Datatourisme consommé correspond bien au dernier run du fournisseur.
La présence d’une date de mise à jour récente (lastUpdateDatatourisme) sur l’ensemble de l’échantillon analysé permet de valider
la fiabilité temporelle du flux avant toute transformation ou exploitation métier.
Objectif :
- Vérifier que le flux est récent
- Confirmer la cohérence avec l'heure de notification
"""
zip_path = r"C:\Users\DELL\data\datatourisme\snapshots\2026-01-25_flux250126_complete.zip"
# Liste des dates de mise à jour collectées
dates = []

# Taille de l’échantillon
# Un sous-ensemble est suffisant pour valider la fraîcheur globale du flux
N = 5_000  # échantillon suffisant

with zipfile.ZipFile(zip_path) as z:

    # Parcours d’un échantillon de POI
    for name in object_files[:N]:

        # Lecture du POI directement depuis l’archive ZIP
        data = json.loads(z.read(name))

        # Extraction de la date de dernière mise à jour Datatourisme
        d = data.get("lastUpdateDatatourisme")
        
        if d:
            # Conversion de la date ISO 8601 en objet datetime Python
            dates.append(datetime.fromisoformat(d.replace("Z", "")))

# Résumé du check de fraîcheur
print("Date min :", min(dates))
print("Date max :", max(dates))
print("Nombre de POI avec date :", len(dates))



# ================================================================================================
# T) Transform (normalisation)
# ================================================================================================

# ---------------------------------------------
# T.1) Construire DataFrame avec les colonnes
# nécessaires depuis ZIP (streaming)¶
# ----------------------------------------------
"""
Objectif :
- Lire les POI Datatourisme directement depuis le ZIP (sans extraire)
- Extraire uniquement les champs utiles (via un mapping de chemins)
- Gérer les variations de schéma (dict vs list) de façon "data-safe"
- Produire un DataFrame typé et ordonné
"""

# ---------------------------------------------
# 0) Paramètres
# ---------------------------------------------
zip_path = r"C:\Users\DELL\data\datatourisme\snapshots\2026-01-25_flux250126_complete.zip"

RENAME_MAP = {
    # Identité / typologie
    "@id": "poi_id",
    "@type": "poi_types",
    "rdfs:label.fr": "label_fr",
    "rdfs:label.en": "label_en",

    # Localisation administrative
    "isLocatedAt.schema:address.schema:postalCode": "postal_code",
    "isLocatedAt.schema:address.hasAddressCity.insee": "city_insee",
    "isLocatedAt.schema:address.hasAddressCity.isPartOfDepartment.insee": "dept_insee",
    "isLocatedAt.schema:address.hasAddressCity.isPartOfDepartment.isPartOfRegion.insee": "region_insee",

    # Coordonnées géographiques
    "isLocatedAt.schema:geo.schema:latitude": "latitude",
    "isLocatedAt.schema:geo.schema:longitude": "longitude",

    # Capacité / jauge
    "allowedPersons": "allowed_persons",

    # Avis / rating (attention : sur Datatourisme c’est souvent “étoiles officielles” pour hébergement)
    "hasReview.hasReviewValue.schema:ratingValue": "rating_value",

    # Itinéraires / randonnées
    "tourDistance": "tour_distance_m",
    "duration": "duration_min",
    "hasPracticeCondition.duration": "practice_duration_min",
    "durationDays": "duration_days",
    "hasPracticeCondition.durationDays": "practice_duration_days",
    "positiveCumulDifference": "positive_elevation_gain_m",
    "negativeCumulDifference": "negative_elevation_loss_m",

    # Prix
    "offers.schema:priceSpecification.schema:price": "price",
    "offers.schema:priceSpecification.schema:minPrice": "min_price",
    "offers.schema:priceSpecification.schema:maxPrice": "max_price",

    # Style architectural
    "hasArchitecturalStyle.rdfs:label.fr": "architectural_style_fr",
    "hasArchitecturalStyle.rdfs:label.en": "architectural_style_en",

    # Description courte
    "hasDescription.shortDescription.fr": "short_desc_fr",
    "hasDescription.shortDescription.en": "short_desc_en",

    # Reviews / labels
    "hasReview.hasReviewValue.rdfs:label.fr": "review_value_label_fr",
    "hasReview.hasReviewValue.rdfs:label.en": "review_value_label_en",
    "hasReview.hasReviewValue.isCompliantWith": "review_compliant_with",

    # Thèmes
    "hasTheme.rdfs:label.fr": "theme_fr",
    "hasTheme.rdfs:label.en": "theme_en",

    # Adresse
    "isLocatedAt.schema:address.schema:addressLocality": "address_locality",
    "isLocatedAt.schema:address.schema:streetAddress": "street_address",

    # Ville / département / région / pays (labels)
    "isLocatedAt.schema:address.hasAddressCity.rdfs:label.fr": "city_label_fr",
    "isLocatedAt.schema:address.hasAddressCity.isPartOfDepartment.rdfs:label.fr": "dept_label_fr",
    "isLocatedAt.schema:address.hasAddressCity.isPartOfDepartment.isPartOfRegion.rdfs:label.fr": "region_label_fr",
    "isLocatedAt.schema:address.hasAddressCity.isPartOfDepartment.isPartOfRegion.isPartOfCountry.rdfs:label.fr": "country_label_fr",

    # Horaires d'ouverture (souvent des listes dans la réalité)
    "isLocatedAt.schema:openingHoursSpecification.schema:opens": "opens_time",
    "isLocatedAt.schema:openingHoursSpecification.schema:closes": "closes_time",
    "isLocatedAt.schema:openingHoursSpecification.schema:validFrom": "hours_valid_from",
    "isLocatedAt.schema:openingHoursSpecification.schema:validThrough": "hours_valid_through",

    # Updates (dates)
    "lastUpdateDatatourisme": "last_update_datatourisme",

    # Dates (événements / périodes)
    "schema:startDate": "start_date",
    "schema:endDate": "end_date",

    # Practice condition (difficulté, locomotion)
    "hasPracticeCondition.hasDifficultyLevel.rdfs:label.fr": "difficulty_level_fr",
    "hasPracticeCondition.hasLocomotionMode.rdfs:label.fr": "locomotion_mode_fr",

    # Tour type
    "hasTourType.rdfs:label.fr": "tour_type_fr",

    # Contacts / médias
    "hasContact.foaf:homepage": "contact_homepage",
    "hasRepresentation.ebucore:hasRelatedResource.ebucore:locator": "media_resource_url",
    "hasMainRepresentation.ebucore:hasRelatedResource.ebucore:locator": "main_media_url",}


# 1) Lister les fichiers POI dans le ZIP (1 POI = 1 JSON dans objects/)
# ---------------------------------------------
def list_object_files(zip_path: str) -> list[str]:
    with zipfile.ZipFile(zip_path) as z:
        return [n for n in z.namelist() if n.startswith("objects/") and n.endswith(".json")]

object_files = list_object_files(zip_path)
print("Nombre de POI dans le zip :", len(object_files))


# 2) Extraction "data-safe" : get_by_path robuste dict/list
# ------------------------
def _first(x: Any) -> Any:
    """
    Normalisation minimale :
    - None -> None
    - list -> premier élément (si non vide)
    - dict/str/int -> inchangé
    """
    if x is None:
        return None
    return x[0] if isinstance(x, list) and len(x) > 0 else x


def get_by_path(obj: Any, path: str, sep: str = ".") -> Any:
    """
    Récupère une valeur dans un JSON à partir d'un chemin "a.b.c".
    Robustesse Datatourisme :
    - si un niveau est une liste, on prend le 1er élément (comportement simple et stable)
    - si un niveau est un dict, on accède à la clé
    - sinon (string/int), chemin impossible -> None
    NB : ce choix “premier élément” est volontairement simple (rapide),
         mais peut perdre de l’info sur les champs multi-valués (thèmes, horaires, offres, etc.).
    """
    cur = obj
    for key in path.split(sep):
        cur = _first(cur)
        if cur is None:
            return None
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return _first(cur)


# 3) Conversion ZIP -> DataFrame (on ne garde que les colonnes mappées)
# ----------------------------
def datatourisme_zip_to_df(
    zip_path: str,
    rename_map: dict[str, str],
    object_files: list[str] | None = None,
    limit: int | None = None,
    column_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Lit le ZIP Datatourisme et renvoie un DataFrame :
    - 1 ligne = 1 POI
    - colonnes = celles de rename_map (renommées)
    - extraction robuste (dict/list) via get_by_path
    - conversions de types "safe"
    - réorganisation des colonnes (si column_order fourni)
    """
    wanted_paths = list(rename_map.keys())
    rows: list[dict[str, Any]] = []

    with zipfile.ZipFile(zip_path) as z:
        files = object_files if object_files is not None else [
            n for n in z.namelist() if n.startswith("objects/") and n.endswith(".json")]
        if limit is not None:
            files = files[:limit]

        for name in files:
            data = json.loads(z.read(name))
            row = {rename_map[p]: get_by_path(data, p) for p in wanted_paths}
            rows.append(row)

    df = pd.DataFrame(rows)

    # Typage "safe"
    # IMPORTANT : postal_code en int fait perdre les zéros (01300 -> 1300).
    # On le garde en string pour l'UX / affichage.
    if "postal_code" in df.columns:
        df["postal_code"] = df["postal_code"].astype("string")

    # Codes INSEE : tu peux les garder en Int64 si tu n’as pas besoin du padding,
    # sinon mets-les aussi en string (recommandé pour l’UX).
    for c in ["city_insee", "dept_insee", "region_insee"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Float-like
    float_cols = [
        "latitude", "longitude",
        "allowed_persons", "rating_value",
        "tour_distance_m", "duration_min", "practice_duration_min",
        "duration_days", "practice_duration_days",
        "positive_elevation_gain_m", "negative_elevation_loss_m",
        "price", "min_price", "max_price",]
    for c in float_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Réorganisation des colonnes
    if column_order is not None:
        existing = [c for c in column_order if c in df.columns]
        rest = [c for c in df.columns if c not in existing]
        df = df[existing + rest]

    return df


# 4) Ordre recommandé des colonnes (UI-friendly)
# ---------------------
ORDERED_COLS = [
    # A. Identité / clés
    "poi_id", "poi_types",

    # B. Contenu
    "label_fr", "label_en", "short_desc_fr", "short_desc_en",

    # C. Coordonnées
    "latitude", "longitude",

    # D. Admin
    "country_label_fr",
    "region_insee", "region_label_fr",
    "dept_insee", "dept_label_fr",
    "city_insee", "city_label_fr",
    "postal_code",

    # E. Adresse
    "address_locality", "street_address",

    # F. Tags
    "theme_fr", "theme_en", "architectural_style_fr", "architectural_style_en",

    # G. Médias
    "main_media_url", "media_resource_url", "contact_homepage",

    # I. Reviews / labels
    "rating_value", "review_value_label_fr", "review_value_label_en", "review_compliant_with",

    # J. Horaires
    "opens_time", "closes_time",
    "hours_valid_from", "hours_valid_through",

    # L. Itinéraires / pratique
    "allowed_persons",
    "difficulty_level_fr", "locomotion_mode_fr", "tour_type_fr",
    "tour_distance_m",
    "duration_min", "practice_duration_min",
    "duration_days", "practice_duration_days",
    "positive_elevation_gain_m", "negative_elevation_loss_m",

    # M. Dates & fraîcheur
    "start_date", "end_date", "last_update_datatourisme",]


# 5) Exécution (construction du df brut Datatourisme)
# ------------------------------
df_dt = datatourisme_zip_to_df(
    zip_path=zip_path,
    rename_map=RENAME_MAP,
    object_files=object_files,
    limit=None,
    column_order=ORDERED_COLS,)

print("df_dt shape:", df_dt.shape)
print(df_dt.head(3))



# ----------------------------------------------------------------------------------------
# T.2) Nettoyer + compléter + transformer les valeurs
# ----------------------------------------------------------------------------------------

# --------------------------------------------
# T.2.1) Détecter et supprimer les doublons¶
# --------------------------------------------

# Bornes géographiques – France métropolitaine
FR_METRO_BOUNDS = {
    "lat_min": 41.0,
    "lat_max": 51.6,
    "lon_min": -5.5,
    "lon_max": 10.0,}


def clean_geo_fr_metro(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    bounds: dict = FR_METRO_BOUNDS,
    drop_zero_zero: bool = True,
) -> pd.DataFrame:
    """
    Nettoie la géolocalisation des POI pour la France métropolitaine.
    - Cast numeric
    - Drop NaN coords
    - Drop (0,0) optionnel
    - Filtre bornes FR métro
    """
    out = df.copy()

    # types robustes
    out[lat_col] = pd.to_numeric(out[lat_col], errors="coerce")
    out[lon_col] = pd.to_numeric(out[lon_col], errors="coerce")

    # coords requises
    out = out.dropna(subset=[lat_col, lon_col])

    # drop (0,0)
    if drop_zero_zero:
        out = out.loc[~((out[lat_col] == 0) & (out[lon_col] == 0))].copy()

    # bornes FR métro
    out = out.loc[
        out[lat_col].between(bounds["lat_min"], bounds["lat_max"], inclusive="both")
        & out[lon_col].between(bounds["lon_min"], bounds["lon_max"], inclusive="both")
    ].copy()

    return out


def deduplicate_keep_best(
    df: pd.DataFrame,
    key_cols: list[str],
    last_update_col: str = "last_update_datatourisme",
    quality_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Déduplique en gardant la meilleure ligne par groupe :
    - priorité : plus récent (si last_update_col existe)
    - puis : plus riche (nb champs non nuls)
    """
    out = df.copy()

    # colonnes de qualité (richesse)
    if quality_cols is None:
        excluded = set(key_cols + [last_update_col])
        quality_cols = [c for c in out.columns if c not in excluded]

    out["_quality"] = out[quality_cols].notna().sum(axis=1)

    # si la colonne de fraîcheur existe, on l’utilise
    if last_update_col in out.columns:
        out["_last_update_dt"] = pd.to_datetime(out[last_update_col], errors="coerce", utc=True)
        out = out.sort_values(["_last_update_dt", "_quality"], ascending=[False, False])
    else:
        # fallback : uniquement richesse
        out = out.sort_values(["_quality"], ascending=[False])

    out = out.drop_duplicates(subset=key_cols, keep="first")
    out = out.drop(columns=["_last_update_dt", "_quality"], errors="ignore")
    return out


def dedup_datatourisme_fr_metro(
    df: pd.DataFrame,
    strict: bool = True,
    coord_round: int = 6,
    quality_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Pipeline :
    1) clean_geo_fr_metro
    2) dédup sur poi_id
    3) option strict : dédup sur (label_fr + lat/lon arrondis + poi_types)
    """
    report: dict[str, dict] = {}

    def _report(tag: str, d: pd.DataFrame):
        report[tag] = {
            "rows": int(len(d)),
            "dup_poi_id": int(d.duplicated(["poi_id"]).sum()) if "poi_id" in d.columns else None,}


    # 1) GEO clean
    d1 = clean_geo_fr_metro(df)
    _report("after_geo_clean", d1)

    # colonnes "richesse" (on ne garde que celles qui existent)
    if quality_cols is None:
        quality_cols = [
            "label_fr", "short_desc_fr",
            "main_media_url", "media_resource_url",
            "theme_fr",
            "street_address", "address_locality",
            "contact_homepage",]
        quality_cols = [c for c in quality_cols if c in d1.columns]


    # 2) dédup SAFE sur poi_id (si dispo)
    if "poi_id" in d1.columns:
        d2 = deduplicate_keep_best(
            d1,
            key_cols=["poi_id"],
            last_update_col="last_update_datatourisme",
            quality_cols=quality_cols,)
    else:
        d2 = d1.copy()

    _report("after_dedup_poi_id", d2)


    # 3) dédup STRICT optionnel
    if strict:
        d2 = d2.copy()

        # si label_fr manquant, on ne peut pas faire strict correctement
        if "label_fr" not in d2.columns:
            _report("after_dedup_strict_skipped_no_label", d2)
            return d2, report

        d2["_lat_r"] = d2["latitude"].round(coord_round)
        d2["_lon_r"] = d2["longitude"].round(coord_round)

        strict_keys = ["label_fr", "_lat_r", "_lon_r"]
        if "poi_types" in d2.columns:
            strict_keys.append("poi_types")

        d3 = deduplicate_keep_best(
            d2,
            key_cols=strict_keys,
            last_update_col="last_update_datatourisme",
            quality_cols=quality_cols,
        ).drop(columns=["_lat_r", "_lon_r"], errors="ignore")

        _report("after_dedup_strict", d3)
        return d3, report

    return d2, report

# Exécution : UNE SEULE FOIS
# (choisis strict=True ou strict=False)
# --------------------------------------
df_clean, rep = dedup_datatourisme_fr_metro(df_dt, strict=True, coord_round=6)
print(rep)



# --------------------------------------------
# T.2.2) Créer colonne is_resto (robuste)
# --------------------------------------------

def _norm_type(t: str) -> str:
    """
    Normalise un type Datatourisme / Schema.org.
    - lowercase
    - suppression du préfixe 'schema:'
    - strip espaces
    """
    return str(t).lower().replace("schema:", "").strip()


# Types considérés comme restaurants
RESTAURANT_TYPES = {"restaurant", "fastfoodrestaurant", "foodestablishment"}


def _norm_label(s: str) -> str:
    """
    Normalise un label :
    - lowercase
    - strip
    - espaces multiples -> 1 espace
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


# Préfix forts : si le label commence par ça, c’est très probablement un resto
RESTO_LABEL_PREFIXES = (
    "restaurant", "resto", "pizzeria", "crêperie", "creperie",
    "bar-restaurant", "bar restaurant", "café-restaurant",
    "cafe-restaurant", "cafe restaurant", "tacos", "snack",
    "kebab", "grill",)

# Mots-clés (moins stricts)
RESTO_LABEL_KEYWORDS = (
    "restaurant", "pizzeria", "bistrot", "snack",
    "kebab", "tacos", "grill", "burger", "sushi",)


def compute_is_resto_from_types(poi_types) -> bool:
    """
    Détecte un restaurant via la colonne poi_types.
    poi_types peut être :
    - string (un type)
    - liste (plusieurs types)
    - NaN / None
    Règle : True si au moins un type normalisé ∈ RESTAURANT_TYPES
    """
    if poi_types is None or (isinstance(poi_types, float) and pd.isna(poi_types)):
        return False

    # poi_types peut être list ou str
    if isinstance(poi_types, list):
        types = poi_types
    else:
        types = [poi_types]

    for t in types:
        if t is None or (isinstance(t, float) and pd.isna(t)):
            continue
        if _norm_type(t) in RESTAURANT_TYPES:
            return True
    return False


def compute_is_resto_from_label(label_fr: str) -> bool:
    """
    Détecte un restaurant via le label_fr (heuristique).
    - True si label commence par un préfix fort
    - ou si le label contient un mot-clé (match mot entier)
    """
    s = _norm_label(label_fr)
    if not s:
        return False

    # 1) Préfix forts
    if s.startswith(RESTO_LABEL_PREFIXES):
        return True

    # 2) Mots-clés (mot entier)
    for kw in RESTO_LABEL_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", s):
            return True

    return False

# Création des colonnes
df_clean["is_resto_type"] = df_clean["poi_types"].apply(compute_is_resto_from_types)
df_clean["is_resto_label"] = df_clean["label_fr"].apply(compute_is_resto_from_label)
df_clean["is_resto"] = df_clean["is_resto_type"] | df_clean["is_resto_label"]



# --------------------------------------------
# T.2.3) Créer colonnes label flags (UI signals)
# --------------------------------------------

def add_label_flags(df: pd.DataFrame, label_col: str = "review_value_label_fr") -> pd.DataFrame:
    """
    Crée des colonnes booléennes de flags à partir de `label_col`.
    - True : le label appartient à une famille
    - False : sinon (ou si label absent / NaN)
    """
    LABEL_GROUPS = {
        "is_label_incontournable": {
            "Patrimoine Mondial UNESCO", "Monument Historique", "Grand Site de France", "Musée de France",
            "Architecture contemporaine remarquable", "Entreprise du Patrimoine Vivant, EPV",
            "Maison des Illustres", "Jardin remarquable", "Jardin protégé Monument Historique",
            "Plus Beaux Villages de France", "Petites cités de caractère", "Ville ou Pays d'art et d'histoire",
            "Site protégé", "Parc Naturel National", "Réserve naturelle nationale",},
        "is_label_famille": {"Famille plus"},
        "is_label_handicap": {
            "Tourisme & Handicap auditif", "Tourisme & Handicap mental",
            "Tourisme & Handicap visuel", "Tourisme & Handicap moteur",},
        "is_label_hebergement": {
            "5 étoiles", "4 étoiles", "3 étoiles", "3 épis / Confort",
            "4 épis / Premium", "5 épis / Luxury ", "3 Clés", "4 Clés", "5 Clés",
            "Gîtes de France", "Chambre d'hôtes référence", "Hébergement Pêche",
            "Hôtel Elégance", "Hôtel Essentiel", "Auberge de Village", "Hôtel Cosy",
            "Camping Qualité",},
        "is_label_gastronomie": {
            "Maître Restaurateur", "Gault&Millau", "Sélection Michelin",
            "Restaurant Gourmand", "Restaurateur de Qualité", "Restaurant de Terroir",
            "Bottin Gourmand", "Restaurant Savoureux", "Food Index for Good",
            "Le Fooding", "Maître Cuisinier de France", "Table gastronomique",
            "Membre du Conservatoire Grand Sud des Cuisines de Terroir",
            "Table de terroir", "Tables et auberges de France", "Ecotable",
            "Sites Remarquables du Goût", "Bistrot de Pays", "Membre Gourméditerrannée",
            "Vélo & Fromage", "3 étoiles Michelin",},
        "is_label_artisanat": {
            "Vignobles & Découvertes", "Agriculture Biologique", "Bienvenue à la ferme",
            "Vignerons Indépendants de France", "Accueil Paysan", "Agriculture raisonnée",
            "Artisans Militants de la Qualité", "Domaine du conservatoire de l'espace littoral",
            "Teritoria",},
        "is_label_randonnee": {
            "Plan Départemental des Itinéraires de Promenade et de Randonnée, PDIPR",
            "Itinéraire de promenade et de randonnée, PR",
            "Fédération Française de Randonnée",
            "Itinéraire de sentier de grande randonnée, GR",
            "Itinéraire de sentier de grande randonnée de pays GRP",},
        "is_label_green": {
            "4 fleurs", "3 fleurs", "5 fleurs", "Parc Naturel Régional", "Qualité Tourisme",
            "Pavillon bleu", "Espace Naturel Sensible", "Ecolabel européen",
            "Valeurs Parc Naturel Régional", "Valeur", "Démarche Tourisme Responsable",
            "Station Verte", "Esprit parc national",
            "Balade à roulette, BR", "Clé Vacances", "Véloroute", "Destination d'excellence",
            "Zone naturelle d'intérêt écologique, faunistique et floristique, ZNIEFF",
            "Réserve naturelle régionale", "Plus Beaux Détours de France", "Site VTT-FFC",
            "Natura 2000", "City Break Confort", "Aire Naturelle", "Relais & châteaux",
            "Fleurs de Soleil", "Eco Jardin", "ISO 20121", "City Break Premium",
            "Accueil chemins de Compostelle en France", "Grande Traversée VTT-FFC",},}

    out = df.copy()

    # sécurité : si la colonne n’existe pas, on met tout à False
    if label_col not in out.columns:
        for col_name in LABEL_GROUPS:
            out[col_name] = False
        return out

    # Application des flags
    for col_name, labels in LABEL_GROUPS.items():
        out[col_name] = out[label_col].isin(labels).fillna(False)

    return out

# exécution
df_clean = add_label_flags(df_clean, label_col="review_value_label_fr")



# --------------------------------------------
# T.2.4) Créer colonne is_etoile et colonne price_level¶
# --------------------------------------------
"""
Traitement de la colonne `rating_value` (DataTourisme)
Dans DataTourisme, la colonne `rating_value` correspond au nombre
d’étoiles officielles des hôtels (classement administratif),
et non à une note de satisfaction utilisateur.
Problèmes à adresser :
- Forte proportion de valeurs manquantes (~86 %)
- Variable numérique mais sémantiquement catégorielle
- Non comparable à des ratings type TripAdvisor ou Google
Objectif :
Transformer `rating_value` en deux variables métiers simples,
robustes et exploitables dans le scoring et l’UX :
1. `is_etoile` : indique si l’hôtel est classé (au moins 1 étoile)
2. `budget` : niveau de gamme estimé (eco / normal / premium)
"""

# 1. Création de la colonne `is_etoile`
# Logique métier :
# - True  → hôtel classé (rating_value >= 1)
# - False → non classé ou information manquante
# On utilise une approche vectorisée, robuste aux NaN.
df_clean["is_etoile"] = df_clean["rating_value"].ge(1).fillna(False)


# 2. Création de la colonne `budget` (3 niveaux)
# Hypothèses métier (simples et défendables) :
# - NaN, 1★, 2★ → "eco"      (non classé ou petit budget)
# - 3★          → "normal"   (milieu de gamme)
# - 4★, 5★      → "premium"  (haut de gamme)
# Cette variable ne représente PAS un niveau de satisfaction,
# mais un positionnement tarifaire / de gamme.

df_clean["price_level"] = np.select([df_clean["rating_value"].isna() | (df_clean["rating_value"] <= 2),
                                      df_clean["rating_value"] == 3,
                                      df_clean["rating_value"] >= 4],
                                     ["eco", "normal", "premium"],
                                     default="eco")



# --------------------------------------------
# T.2.5)  Transformer Colonne poi_types en main_category¶
# --------------------------------------------

# ----------------------------------------------------------
# T.2.5.a) MAIN_CAT DES POIs (version optimisée + stable)
# ----------------------------------------------------------
"""
Objectif (Prime) :
- Chaque POI peut avoir plusieurs types (poi_types_clean = liste de labels nettoyés).
- Pour alimenter le scoring Prime, on veut une seule main_category par POI (déterministe).
- La stratégie est de convertir les types en catégories candidates, puis choisir la catégorie
  ayant le poids le plus élevé (CAT_WEIGHT).

Étapes du bloc :
1) CAT_WEIGHT : dictionnaire de pondération des main_category. Plus le poids est élevé, plus la catégorie
   est considérée "structurante" dans le scoring Prime. En cas de multi-type, la catégorie la plus lourde l’emporte.

2) TYPE_TO_CAT : Mapping fin entre chaque poi_type nettoyé (minuscule) et une main_category.
   Important : les clés doivent être exactement au format de poi_types_clean (lowercase),
   sinon les correspondances échoueront silencieusement.

3) pick_main_category_by_weight() :
Fonction cœur du choix de main_category pour un POI multi-type.

4) Application au DataFrame :
   - main_category : catégorie finale (celle retenue au poids max)
   - main_cat_weight : poids de la catégorie retenue (prêt pour le calcul Prime)
   - main_cat_candidates : liste des catégories candidates triées (debug/contrôle qualité)

Notes / bonnes pratiques :
- Ajuster IGNORE_TYPES si certains types sont trop génériques (ex: localbusiness) et polluent la catégorisation.
- Compléter TYPE_TO_CAT au fur et à mesure que de nouveaux types apparaissent.
- Contrôler la qualité en inspectant main_cat_candidates pour les POI ambigus (multi-catégories).
"""

import re
import pandas as pd

# 0) GARDE-FOU : construire poi_types_clean (liste de types sans préfixe)
# ---------------------------------------------------------------------

_prefix_re = re.compile(r"^[a-z0-9_]+:", flags=re.IGNORECASE)

def strip_prefix(t: str) -> str:
    """
    Normalise un type :
    - lower
    - strip
    - supprime n'importe quel préfixe "xxx:" (schema:, olo:, etc.)
    Exemple: "olo:OrderedList" -> "orderedlist"
    """
    t = str(t).strip().lower()
    t = _prefix_re.sub("", t)
    return t

def to_types_clean(x) -> list[str]:
    """
    Convertit poi_types (string ou liste) -> liste de strings normalisées.
    """
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []

    # cas liste
    if isinstance(x, list):
        out = []
        for v in x:
            sv = str(v).strip()
            if sv:
                out.append(strip_prefix(sv))
        return out

    # cas string
    s = str(x).strip()
    if not s:
        return []
    return [strip_prefix(s)]

# créer la colonne si absente
if "poi_types_clean" not in df_clean.columns:
    df_clean["poi_types_clean"] = df_clean["poi_types"].apply(to_types_clean)


# 1) Poids des main_category
# -------------------------
CAT_WEIGHT = {
    "Culture & Musées": 0.9,
    "Patrimoine & Monuments": 0.8,
    "Nature & Paysages": 0.7,
    "Gastronomie & Restauration": 0.6,
    "Événements & Spectacles & Exposition": 0.5,
    "Loisirs & Activités familiales": 0.4,
    "Shopping & Artisanat": 0.3,
    "Bien-être & Santé": 0.2,
    "Services & Pratique": 0.1,
    "Itinéraires & Circuits": 0.0,
    "Hébergement": 0.0,}


# 2) Mapping type -> main_category
# -------------------------------
TYPE_TO_CAT = {}

# Culture & Musées
for t in ["culturalsite"]:
    TYPE_TO_CAT[t] = "Culture & Musées"

# Patrimoine & Monuments
for t in [
    "archeologicalsite","abbey","basilica","cathedral","chapel","church","cloister","convent",
    "calvary","castle","citadel","bastide","aqueduct","bridge","collegiate","commanderie",
    "chartreuse","bishopric","cityheritage","house","civilcemetery","buddhisttemple"]:
    TYPE_TO_CAT[t] = "Patrimoine & Monuments"

# Gastronomie & Restauration
for t in ["foodestablishment","fastfoodrestaurant","cafeorcoffeeshop","bakery","coveredmarket"]:
    TYPE_TO_CAT[t] = "Gastronomie & Restauration"

# Hébergement
for t in ["accommodation","apartment"]:
    TYPE_TO_CAT[t] = "Hébergement"

# Événements & Spectacles & Exposition
for t in ["event","businessevent","circusplace", "auditorium"]:
    TYPE_TO_CAT[t] = "Événements & Spectacles & Exposition"

# Loisirs & Activités familiales
for t in [
    "library","cinematheque","educationaltrail",
    "amusementpark","adventurepark","bowlingalley","minigolf","golfcourse","climbingwall",
    "gymnasium","frontonbelotacourt","casino","movietheater","activityprovider","nauticalcentre",
    "marina","launchingramp","downhillskiresort","crosscountryskiresort","downhillskirun",
    "crosscountryskitrail","dogsleddingtrail","aquarium","product","convenientservice","civicstructure"]:
    TYPE_TO_CAT[t] = "Loisirs & Activités familiales"

# Nature & Paysages
for t in ["park","placeofinterest","landform", "arena"]:
    TYPE_TO_CAT[t] = "Nature & Paysages"

# Shopping & Artisanat
for t in ["localbusiness", "businessplace"]:
    TYPE_TO_CAT[t] = "Shopping & Artisanat"

# Bien-être & Santé
for t in ["balneotherapycentre","hammam"]:
    TYPE_TO_CAT[t] = "Bien-être & Santé"

# Services & Pratique
for t in [
    "touristinformationcenter","airport","airfield","busstop","busstation",
    "equipmentrentalshop","equipmentrepairshop","multipurposeroomorcommunityroom"
]:
    TYPE_TO_CAT[t] = "Services & Pratique"

# Itinéraires & Circuits (clé sans préfixe)
TYPE_TO_CAT["orderedlist"] = "Itinéraires & Circuits"


# 3) Choix main_category par poids
# -------------------------------
IGNORE_TYPES = set()

def pick_main_category_by_weight(types_clean, type_to_cat=TYPE_TO_CAT, cat_weight=CAT_WEIGHT):
    """
    Retour: (best_cat, best_weight, candidates_sorted)
    """
    if not isinstance(types_clean, (list, tuple, set)):
        return ("Autre", 0.0, [])

    best_cat, best_w = None, -1.0
    candidates = {}

    for t in types_clean:
        if not isinstance(t, str):
            continue
        t = t.strip().lower()
        if not t or t in IGNORE_TYPES:
            continue

        cat = type_to_cat.get(t)
        if not cat:
            continue

        w = cat_weight.get(cat, 0.0)
        candidates[cat] = max(candidates.get(cat, 0.0), w)

        if w > best_w:
            best_w, best_cat = w, cat

    if best_cat is None:
        return ("Autre", 0.0, [])

    candidates_sorted = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
    return (best_cat, best_w, candidates_sorted)


# 4) Application (calculer en une seule fois)
# ----------------------------------------
tmp = df_clean["poi_types_clean"].apply(pick_main_category_by_weight)

df_clean["main_category"] = tmp.apply(lambda x: x[0])
df_clean["main_cat_weight"] = tmp.apply(lambda x: x[1])
df_clean["main_cat_candidates"] = tmp.apply(lambda x: x[2])

# Flag Prime+ (itinéraires)
df_clean["is_prime_plus"] = df_clean["poi_types_clean"].apply(
    lambda xs: "orderedlist" in xs if isinstance(xs, list) else False)



# ----------------------------------------------------------
# T.2.5.b) FORMAT DES POIs (bugfix cache + stable)
# ----------------------------------------------------------
"""
FORMAT DES POIs
Le "format" décrit la nature de l'interaction utilisateur avec un POI, indépendamment de sa catégorie thématique.
- lieu        : POI visitable / contemplatif (sightseeing)
- expérience  : POI impliquant une activité ou une action
- besoin      : POI utilitaire ou fonctionnel (restaurant, hébergement)
- parcours    : objet macro (itinéraires longs, hors Prime, réservé pour Prime+)
"""

# 1) Mapping type -> format
TYPE_TO_FORMAT = {}

for t in [
    "culturalsite", "archeologicalsite", "abbey", "basilica", "cathedral", "chapel", "church",
    "cloister", "convent", "calvary", "castle", "citadel", "bastide", "aqueduct", "bridge",
    "collegiate", "commanderie", "chartreuse", "bishopric", "cityheritage", "house",
    "civilcemetery", "buddhisttemple", "park", "placeofinterest", "landform"]:
    TYPE_TO_FORMAT[t] = "lieu"

for t in [
    "event", "businessevent", "arena", "circusplace", "library", "cinematheque", "auditorium", "educationaltrail",
    "amusementpark", "adventurepark", "bowlingalley", "minigolf", "golfcourse", "climbingwall",
    "gymnasium", "frontonbelotacourt", "casino", "movietheater", "activityprovider", "nauticalcentre",
    "marina", "launchingramp", "downhillskiresort", "crosscountryskiresort", "downhillskirun",
    "crosscountryskitrail", "dogsleddingtrail", "aquarium", "businessplace",
    "localbusiness", "equipmentrentalshop", "equipmentrepairshop"]:
    TYPE_TO_FORMAT[t] = "expérience"

for t in [
    "balneotherapycentre", "hammam", "touristinformationcenter", "convenientservice", "civicstructure",
    "airport", "airfield", "busstop", "busstation", "accommodation", "apartment", "multipurposeroomorcommunityroom",
    "foodestablishment", "fastfoodrestaurant", "cafeorcoffeeshop", "bakery", "coveredmarket", "product"]:
    TYPE_TO_FORMAT[t] = "besoin"

TYPE_TO_FORMAT["orderedlist"] = "parcours"

FORMAT_WEIGHT = {"lieu": 0.09, "expérience": 0.06, "besoin": 0.03, "parcours": 0.0}


# 2) Calcul format_label à partir de poi_types_clean (liste)
def compute_format_label_from_clean(types_clean) -> str:
    """
    Calcule le format dominant à partir de poi_types_clean (liste de types normalisés).
    Règle :
    - vote majoritaire des formats mappés
    - tie-break par priorité : lieu > expérience > besoin > parcours
    - fallback : "lieu"
    """
    if not isinstance(types_clean, (list, tuple, set)) or len(types_clean) == 0:
        return "lieu"

    formats = []
    for t in types_clean:
        if not isinstance(t, str):
            continue
        f = TYPE_TO_FORMAT.get(t)
        if f:
            formats.append(f)

    if not formats:
        return "lieu"

    counts = Counter(formats)
    top_n = max(counts.values())
    top_formats = [k for k, v in counts.items() if v == top_n]

    priority = {"lieu": 3, "expérience": 2, "besoin": 1, "parcours": 0}
    top_formats.sort(key=lambda k: priority.get(k, 0), reverse=True)
    return top_formats[0]


# 3) Cache (clé = tuple(types_clean))
_format_cache = {}

def compute_format_label_cached(types_clean) -> str:
    key = tuple(types_clean) if isinstance(types_clean, list) else tuple()
    if key in _format_cache:
        return _format_cache[key]
    val = compute_format_label_from_clean(types_clean)
    _format_cache[key] = val
    return val


# 4) Application au DF
_format_cache = {}  # reset avant recalcul
df_clean["format_label"] = df_clean["poi_types_clean"].apply(compute_format_label_cached)
df_clean["format_weight"] = df_clean["format_label"].map(FORMAT_WEIGHT).fillna(0.0)



# ----------------------------------------------------------
# T.2.5.c) TEMPO DES POIs (version corrigée + stable + notebook-proof)
# ----------------------------------------------------------
"""
TEMPO DES POIs
Le "tempo" décrit le rythme de visite induit par un POI dans une journée.
- lent       : POI long / immersif / contemplatif
- normal     : POI standard
- dynamique  : POI rapide / fluide / de passage
- zen        : itinéraires (macro-objets)
Le tempo est utilisé dans le scoring Prime pour moduler la densité d’un itinéraire journalier.
"""

# 1) Mapping type -> tempo
TYPE_TO_TEMPO = {}

for t in [
    "culturalsite", "archeologicalsite", "abbey", "basilica", "cathedral", "chapel", "church",
    "cloister", "convent", "calvary", "castle", "citadel", "bastide", "aqueduct", "bridge",
    "collegiate", "commanderie", "chartreuse", "bishopric", "cityheritage", "house",
    "civilcemetery", "buddhisttemple", "park", "placeofinterest", "landform"]:
    TYPE_TO_TEMPO[t] = "lent"

for t in [
    "balneotherapycentre", "hammam", "touristinformationcenter", "convenientservice", "civicstructure",
    "airport", "airfield", "busstop", "busstation", "accommodation", "apartment", "multipurposeroomorcommunityroom",
    "foodestablishment", "fastfoodrestaurant", "cafeorcoffeeshop", "bakery", "coveredmarket", "product"]:
    TYPE_TO_TEMPO[t] = "normal"

for t in [
    "event", "businessevent", "arena", "circusplace", "library", "cinematheque", "auditorium", "educationaltrail",
    "amusementpark", "adventurepark", "bowlingalley", "minigolf", "golfcourse", "climbingwall",
    "gymnasium", "frontonbelotacourt", "casino", "movietheater", "activityprovider", "nauticalcentre",
    "marina", "launchingramp", "downhillskiresort", "crosscountryskiresort", "downhillskirun",
    "crosscountryskitrail", "dogsleddingtrail", "aquarium", "businessplace",
    "localbusiness", "equipmentrentalshop", "equipmentrepairshop"]:
    TYPE_TO_TEMPO[t] = "dynamique"

TYPE_TO_TEMPO["orderedlist"] = "zen"

TEMPO_WEIGHT = {
    "lent": 0.09,
    "normal": 0.06,
    "dynamique": 0.03,
    "zen": 0.0,}


# 2) Calcul tempo_label depuis poi_types_clean (liste)
def compute_tempo_label_from_clean(types_clean) -> str:
    """
    Calcule le tempo dominant à partir de poi_types_clean.
    - vote majoritaire des tempos mappés
    - tie-break par priorité : lent > normal > dynamique > zen
    - fallback : "normal"
    """
    if not isinstance(types_clean, (list, tuple, set)) or len(types_clean) == 0:
        return "normal"

    tempos = []
    for t in types_clean:
        if not isinstance(t, str):
            continue
        tempo = TYPE_TO_TEMPO.get(t)
        if tempo:
            tempos.append(tempo)

    if not tempos:
        return "normal"

    counts = Counter(tempos)
    top_n = max(counts.values())
    top_tempos = [k for k, v in counts.items() if v == top_n]

    priority = {"lent": 3, "normal": 2, "dynamique": 1, "zen": 0}
    top_tempos.sort(key=lambda k: priority.get(k, 0), reverse=True)
    return top_tempos[0]


# 3) Cache (clé = tuple(types_clean))
_tempo_cache = {}

def compute_tempo_label_cached(types_clean) -> str:
    key = tuple(types_clean) if isinstance(types_clean, list) else tuple()
    if key in _tempo_cache:
        return _tempo_cache[key]
    val = compute_tempo_label_from_clean(types_clean)
    _tempo_cache[key] = val
    return val


# 4) Application au DF
_tempo_cache = {}  # reset
df_clean["tempo_label"] = df_clean["poi_types_clean"].apply(compute_tempo_label_cached)

# fallback “normal” = 0.06 (pas 0.6)
df_clean["tempo_weight"] = df_clean["tempo_label"].map(TEMPO_WEIGHT).fillna(0.06)



# --------------------------------------------
# T.2.5.d)  Calculer score_prime
# --------------------------------------------
"""
Le score Prime final repose sur la catégorie principale comme facteur dominant.
Le format et le tempo sont des modulateurs légers qui ajustent :
- la nature de l’expérience (format)
- le rythme de la journée (tempo)
Formule :
FINAL_SCORE = main_cat_weight * (1 + format_weight + tempo_weight)
Aucune pondération liée aux labels (incontournable, green, etc.)
n’intervient ici afin de préserver la neutralité du moteur Prime.
"""

# Sécurisation des valeurs manquantes
for c in ["main_cat_weight", "format_weight", "tempo_weight"]:
    if c in df_clean.columns:
        df_clean[c] = pd.to_numeric(df_clean[c], errors="coerce").fillna(0.0)
    else:
        df_clean[c] = 0.0

# Calcul vectorisé (rapide et stable)
df_clean["score_prime"] = df_clean["main_cat_weight"] * (
    1 + df_clean["format_weight"] + df_clean["tempo_weight"])




# ----------------------------------------------------------------------------------------
# T.3) DataFrame Final
# ----------------------------------------------------------------------------------------

# -----------------------------------
# T.3.1)  Normalisation les colonnes¶
# ------------------------------------
"""
Normalisation des colonnes
- Renommage orienté produit (UI-first).
- Typage et downcast systématiques (float32, Int*, bool) afin de réduire l’empreinte mémoire et la taille des fichiers parquet.
- Nettoyage minimal (lat / lon requis).
- Conversion des identifiants et codes (ex: postal_code) en formats stables et cohérents pour l’affichage.
"""

def safe_str(s: pd.Series) -> pd.Series:
    """String propre : strip, NA conservés"""
    return s.astype("string").str.strip().replace("", pd.NA)

def to_float32(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("float32")

def to_int32_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int32")

def to_int16_nullable(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int16")

def list_to_str(s: pd.Series) -> pd.Series:
    """liste → 'a|b|c' ; NA si pas liste"""
    return s.apply(
        lambda x: "|".join(map(str, x)) if isinstance(x, list) and len(x) > 0 else pd.NA
    ).astype("string")

def normalize_datatourisme(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise l'ensemble des colonnes DataTourisme :
    - types cohérents
    - strings nettoyées
    - floats downcastés
    - identifiants stables
    Aucun filtrage, aucun split.
    """
    df = df.copy()

    # Identifiants / labels
    if "poi_id" in df.columns:
        df["poi_id"] = df["poi_id"].astype("string")

    for c in ["label_fr", "label_en", "short_desc_fr", "short_desc_en"]:
        if c in df.columns:
            df[c] = safe_str(df[c])
        else:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")

    # Géolocalisation
    for c in ["latitude", "longitude"]:
        df[c] = to_float32(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="float32")

    # Localisation administrative
    for c in ["country_label_fr", "region_label_fr", "dept_label_fr", "city_label_fr"]:
        df[c] = safe_str(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    for c, fn in [("region_insee", to_int32_nullable), ("dept_insee", to_int16_nullable), ("city_insee", to_int32_nullable)]:
        df[c] = fn(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="Int32")

    # postal_code : garder en string pour préserver les zéros
    if "postal_code" in df.columns:
        df["postal_code"] = safe_str(df["postal_code"])
        # option : padding 5 chiffres si tu veux (décommente)
        # df["postal_code"] = df["postal_code"].str.zfill(5)
    else:
        df["postal_code"] = pd.Series(pd.NA, index=df.index, dtype="string")

    # Adresse / contact
    for c in ["address_locality", "street_address", "contact_homepage"]:
        df[c] = safe_str(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    # Médias
    for c in ["main_media_url", "media_resource_url"]:
        df[c] = safe_str(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    # Rating / capacité
    df["rating_value"] = to_float32(df["rating_value"]) if "rating_value" in df.columns else pd.Series(pd.NA, index=df.index, dtype="float32")
    df["allowed_persons"] = to_int16_nullable(df["allowed_persons"]) if "allowed_persons" in df.columns else pd.Series(pd.NA, index=df.index, dtype="Int16")

    # Prix
    for c in ["price", "min_price", "max_price"]:
        df[c] = to_float32(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="float32")
    df["price_level"] = safe_str(df["price_level"]) if "price_level" in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    # Catégorisation
    for c in ["main_category", "format_label", "tempo_label"]:
        df[c] = safe_str(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    for c in ["main_cat_weight", "format_weight", "tempo_weight", "score_prime"]:
        df[c] = to_float32(df[c]) if c in df.columns else pd.Series(0.0, index=df.index, dtype="float32")

    if "main_cat_candidates" in df.columns:
        df["main_cat_candidates"] = list_to_str(df["main_cat_candidates"])
    else:
        df["main_cat_candidates"] = pd.Series(pd.NA, index=df.index, dtype="string")

    # Booléens : safe (crée la colonne si absente)
    BOOL_COLS = [
        "is_resto_type", "is_resto_label", "is_resto",
        "is_label_incontournable", "is_label_famille", "is_label_handicap",
        "is_label_hebergement", "is_label_gastronomie", "is_label_artisanat",
        "is_label_randonnee", "is_label_green",
        "is_etoile", "is_prime_plus",]
    for c in BOOL_COLS:
        if c not in df.columns:
            df[c] = False
        else:
            df[c] = df[c].fillna(False).astype(bool)

    # Colonnes excursion (safe)
    for c in ["difficulty_level_fr", "locomotion_mode_fr", "tour_type_fr"]:
        df[c] = safe_str(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    for c in [
        "tour_distance_m", "duration_min", "practice_duration_min",
        "duration_days", "practice_duration_days",
        "positive_elevation_gain_m", "negative_elevation_loss_m",]:
        df[c] = to_float32(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="float32")

    for c in ["start_date", "end_date", "hours_valid_from", "hours_valid_through", "last_update_datatourisme"]:
        df[c] = safe_str(df[c]) if c in df.columns else pd.Series(pd.NA, index=df.index, dtype="string")

    return df


df_norm = normalize_datatourisme(df_clean)
df_norm.info(memory_usage="deep")



# --------------------------------------------
# T.3.2)  Renommage + Split le dataframe en 2 :
# prim_classique & prime_excursion¶
# --------------------------------------------
"""
OBJECTIF :
Ce bloc prépare le dataset de base destiné à l’interface utilisateur
de l’application (Streamlit), à partir du dataframe DataTourisme
préalablement normalisé (`df_norm`).

1) Construction d’un dataframe UI unifié
   - Sélection et renommage explicites des colonnes nécessaires à l’affichage,
     au filtrage et aux calculs futurs (distance, scoring).
   - Ajout de colonnes fonctionnelles absentes de la source :
       • `source_id` et `source` pour l’identification et la traçabilité
       • `review_count` et `distance_km` comme placeholders produits
   - Normalisation des formats pour l’UX (strings propres, types numériques stables).
   - Aucune logique métier Prime / Excursion n’est encore appliquée à ce stade.

2) Split fonctionnel du dataset par lignes
   - Le dataframe UI est séparé en deux sous-ensembles selon la catégorie principale :
       • Prime classique :
         toutes les lignes dont `main_category_compressed` ≠ "Itinéraires & Circuits"
       • Prime excursion :
         uniquement les lignes dont `main_category_compressed` = "Itinéraires & Circuits"
   - Ce split par lignes permet de distinguer clairement les usages
     sans dupliquer les calculs ni les transformations.

3) Enregistrement des dataframes en parquet compressé
"""

# 1) Construire le dataframe UI (renommage + colonnes nécessaires)
UI_RENAME_MAP = {
    # ids / source
    "poi_id": "source_id",

    # géoloc
    "latitude": "lat",
    "longitude": "lon",

    # contenu
    "label_fr": "name",
    "contact_homepage": "url",

    # adresse
    "street_address": "address",
    "city_label_fr": "city",
    "region_label_fr": "region",
    "dept_label_fr": "departement",
    "country_label_fr": "country",

    # catégories
    "main_category": "main_category_compressed",
    "review_compliant_with": "sub_category",
    "poi_types_clean": "type_principal",
    "short_desc_fr": "snippet",
    "short_desc_en": "snippet_en",
    "rating_value": "rating",
    "allowed_persons": "max_people",}

df_ui = df_norm.rename(columns=UI_RENAME_MAP).copy()

# colonnes produit / placeholders
df_ui["source"] = "datatourisme"
df_ui["review_count"] = pd.Series(pd.NA, index=df_ui.index, dtype="Int32")
df_ui["distance_km"] = pd.Series(np.nan, index=df_ui.index, dtype="float32")

# sécuriser main_category_compressed (évite soucis de NA)
if "main_category_compressed" not in df_ui.columns:
    df_ui["main_category_compressed"] = pd.NA
else:
    df_ui["main_category_compressed"] = df_ui["main_category_compressed"].astype("string")

# type_principal : si c’est une liste -> string "a|b|c" (plus stable pour parquet/UI)
if "type_principal" in df_ui.columns:
    df_ui["type_principal"] = df_ui["type_principal"].apply(
        lambda x: "|".join(map(str, x)) if isinstance(x, list) else (str(x) if pd.notna(x) else pd.NA)
    ).astype("string")
else:
    df_ui["type_principal"] = pd.Series(pd.NA, index=df_ui.index, dtype="string")


# 2) Splitter en 2 dataframes (Prime classique vs Excursion)
EXCURSION_CAT = "Itinéraires & Circuits"

df_prime_classique = df_ui[df_ui["main_category_compressed"].fillna("") != EXCURSION_CAT].copy()
df_prime_excursion = df_ui[df_ui["main_category_compressed"].fillna("") == EXCURSION_CAT].copy()


# 3) Export parquet compressé (dossiers créés si besoin)
out_prime_classique = Path(r"C:\Users\DELL\Downloads\ItineraireVacances3\df_prime_classique.parquet")
out_prime_excursion = Path(r"C:\Users\DELL\Downloads\ItineraireVacances3\df_prime_excursion.parquet")


def save_parquet_safe(df: pd.DataFrame, path: Path):
    """
    Enregistre un dataframe en parquet compressé.
    Priorité à zstd (niveau 10), fallback automatique vers snappy.
    Crée le dossier parent si besoin.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        df.to_parquet(path, index=False, compression="zstd", compression_level=10)
        print("saved with zstd level 10:", str(path))
    except TypeError:
        df.to_parquet(path, index=False, compression="zstd")
        print("saved with zstd (default level):", str(path))
    except Exception:
        df.to_parquet(path, index=False, compression="snappy")
        print("saved with snappy:", str(path))

save_parquet_safe(df_prime_classique, out_prime_classique)
save_parquet_safe(df_prime_excursion, out_prime_excursion)

str(out_prime_classique), str(out_prime_excursion)



# =========================================================================
# checkpoint de validation
# =========================================================================
if __name__ == "__main__":

    print("=== PIPELINE START ===")

    print("df_norm :", df_norm.shape)
    print("df_prime_classique :", df_prime_classique.shape)
    print("df_prime_excursion :", df_prime_excursion.shape)

    print("\nPreview prime classique:")
    print(df_prime_classique.head(3))

    print("\nPreview prime excursion:")
    print(df_prime_excursion.head(3))

    print("\n=== PIPELINE END ===")

