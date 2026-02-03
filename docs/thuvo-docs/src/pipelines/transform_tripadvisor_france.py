import os
import re

import numpy as np
import pandas as pd
import pyarrow

# ===================================================================================
# T.1) Charger les données TripAdvisor
# ===================================================================================

# lire CSV_PATH (première fois)
CSV_PATH = "df_tripadvisor_france.csv"
df_trip_eu = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8", low_memory=False)

# Convertir en Parquet --> Fichier beaucoup plus léger --> Lecture rapide
PARQUET_PATH = "df_tripadvisor_france.parquet"
df_trip_eu.to_parquet(PARQUET_PATH, compression="snappy")

# Libère la mémoire et recharger le parquet
del df_trip_eu
df_trip_eu = pd.read_parquet("df_tripadvisor_france.parquet")

# créer un DataFrame filtré avec country = "France" ET region = "Ile-de-France"
df_trip = df_trip_eu[df_trip_eu["country"] == "France"]

print(df_trip.shape)
print(df_trip.head(3))

# ===================================================================================
# T.2) Nettoyer + compléter
# ===================================================================================

df_trip_clean = df_trip.copy()


# 1) Supprimer les lignes sans coordonnées (si les colonnes existent)
df_trip_clean = df_trip_clean.dropna(subset=["latitude", "longitude"]).copy()


# 2) avg_rating : indicateur missing + colonne filled (médiane si possible, sinon valeur neutre)
df_trip_clean["avg_rating_missing"] = (df_trip_clean["avg_rating"].isna()).astype(int)
median_rating = df_trip_clean["avg_rating"].median()
df_trip_clean["avg_rating_filled"] = df_trip_clean["avg_rating"].fillna(median_rating)


# 3) total_reviews_count : indicateur missing + filled=0 (en numérique)
df_trip_clean["total_reviews_count_missing"] = (
    df_trip_clean["total_reviews_count"].isna()
).astype(int)
df_trip_clean["total_reviews_count_filled"] = df_trip_clean[
    "total_reviews_count"
].fillna(0)


# 4) Hypothèse raisonnable : un restaurant non revendiqué n’a probablement pas fait la démarche
df_trip_clean["claimed"] = df_trip_clean["claimed"].fillna("Unclaimed")


# ===================================================================================
# T.3) Transformer (robuste)
# ===================================================================================


# ----------------------------------------------------------------------------
# T.3.1) Colonne "awards" -> binaire 0/1
# ----------------------------------------------------------------------------
df_trip_clean["awards_binary"] = (df_trip_clean["awards"].notna()).astype(int)


# ----------------------------------------------------------------------------
# T.3.2) excellent / very_good / average -> proportions (%)
# ----------------------------------------------------------------------------
# compléter les valeurs manquantes des colonnes "excellent", "very_good", "average" = 0
df_trip_clean[["excellent", "very_good", "average"]] = df_trip_clean[
    ["excellent", "very_good", "average"]
].fillna(0)

# les transformer en proportions (%) en basant sur la colonne "total_reviews_count"
df_trip_clean["excellent_pct"] = (
    (df_trip_clean["excellent"] / df_trip_clean["total_reviews_count"]) * 100
).round(2)
df_trip_clean["very_good_pct"] = (
    (df_trip_clean["very_good"] / df_trip_clean["total_reviews_count"]) * 100
).round(2)
df_trip_clean["average_pct"] = (
    (df_trip_clean["average"] / df_trip_clean["total_reviews_count"]) * 100
).round(2)


# ----------------------------------------------------------------------------
# T.3.3) food / service / value -> score moyen partiel + fiabilité + pondération
# ----------------------------------------------------------------------------
# Créer le score moyen partiel en ignorant automatiquement les NaN + caculant la moyenne sur 1, 2 ou 3 colonnes + retournant NaN seulement si les 3 sont manquantes
df_trip_clean["partial_quality_score"] = (
    (df_trip_clean[["food", "service", "value"]].mean(axis=1, skipna=True))
).round(2)

# Ajouter un indicateur de fiabilité (3=très fiable grace aux valeurs présentes sur les 3 colonnes, 2=fiable, 1=faible, 0=aucune info)
df_trip_clean["partial_quality_count"] = (
    df_trip_clean[["food", "service", "value"]].notna().sum(axis=1)
)

# Flager score partiel incomplet (inférieur à 3)
df_trip_clean["partial_quality_missing"] = (
    df_trip_clean["partial_quality_count"] < 3
).astype(int)

# Score pondéré par la fiabilité
df_trip_clean["partial_quality_score_weighted"] = (
    df_trip_clean["partial_quality_score"]
    * (df_trip_clean["partial_quality_count"] / 3)
).round(2)


# ----------------------------------------------------------------------------
# T.3.4) open_days_per_week & working_shifts_per_week -> score d’accessibilité (robuste)
# ----------------------------------------------------------------------------

# Créer un indicateurs sur les valeurs manquantes
df_trip_clean["open_days_missing"] = (
    df_trip_clean["open_days_per_week"].isna().astype(int)
)
df_trip_clean["working_shifts_missing"] = (
    df_trip_clean["working_shifts_per_week"].isna().astype(int)
)

# Normaliser les deux variables (0 → 1)
df_trip_clean["open_days_norm"] = ((df_trip_clean["open_days_per_week"] / 7)).round(2)
df_trip_clean["working_shifts_norm"] = (
    (df_trip_clean["working_shifts_per_week"] / 14)
).round(2)

# Score d’accessibilité partiel
df_trip_clean["accessibility_score"] = (
    (df_trip_clean[["open_days_norm", "working_shifts_norm"]].mean(axis=1, skipna=True))
).round(2)

# Fiabilité du score : 2=très fiable, 1=partiel, 0=aucune info
df_trip_clean["accessibility_count"] = (
    df_trip_clean[["open_days_norm", "working_shifts_norm"]].notna().sum(axis=1)
)

# Flag “accessibilité incomplète”
df_trip_clean["accessibility_missing"] = (
    df_trip_clean["accessibility_count"] < 2
).astype(int)

# Mise à l’échelle finale
df_trip_clean["accessibility_score_5"] = (
    df_trip_clean["accessibility_score"] * 5
).round(2)


# ----------------------------------------------------------------------------
# T.3.5) special_diets -> is_halal / is_kosher (robuste)
# ----------------------------------------------------------------------------
# Création de la colonne booléenne halal
df_trip_clean["is_halal"] = df_trip_clean["special_diets"].str.contains(
    "Halal", case=False, na=False
)
# Vérification
df_trip_clean["is_halal"].value_counts()

# Création de la colonne booléenne kosher
df_trip_clean["is_kosher"] = df_trip_clean["special_diets"].str.contains(
    "kosher", case=False, na=False
)

# Vérification
print(df_trip_clean["is_kosher"].value_counts())


# ----------------------------------------------------------------------------
# T.3.6) vegetarian_friendly / vegan_options / gluten_free -> bool + diet_level (UX)
# ----------------------------------------------------------------------------
cols = ["vegetarian_friendly", "vegan_options", "gluten_free"]

df_trip_clean[cols] = df_trip_clean[cols].replace({"Y": True, "N": False}).astype(bool)


# ----------------------------------------------------------------------------
# T.3.7) Construire une colonne "url" à partir de "restaurant_link" (robuste)
# ----------------------------------------------------------------------------
df_trip_clean["url"] = (
    "https://www.tripadvisor.fr/Restaurant_Review-"
    + df_trip_clean["restaurant_link"].astype(str)
    + ".html"
)

# vérifier
print(df_trip_clean[["restaurant_link", "url"]].head())


# ----------------------------------------------------------------------------
# T.3.8) price_level -> price_category (1 petit, 2 moyen, 3 confort) (robuste)
# ----------------------------------------------------------------------------

df_trip_clean["price_level"] = df_trip_clean["price_level"].fillna("Unknown")

# mapper les valeurs
price_mapping = {"€": 1, "€€-€€€": 2, "€€€€": 3}
df_trip_clean["price_category"] = (
    df_trip_clean["price_level"].map(price_mapping).fillna(1)
)

# vérifier
print(df_trip_clean["price_category"].value_counts())


# ----------------------------------------------------------------------------
# T.3.9) cuisines -> tags normalisés + catégories + (option) colonnes booléennes
# ----------------------------------------------------------------------------


## DEF : Normaliser + tokenizer meals
def parse_cuisines(x):
    if pd.isna(x) or str(x).strip() == "" or str(x).strip().lower() == "none":
        return []
    items = [t.strip() for t in str(x).split(",")]
    items = [t for t in items if t]
    # Normalisation simple (optionnel)
    # ex: uniformiser "Fast food" vs "Fast food " etc.
    return items


df_trip_clean["cuisines_list"] = df_trip_clean["cuisines"].apply(parse_cuisines)


CUISINE_BY_CONTINENT = {
    "Européen": {
        "European",
        "French",
        "Italian",
        "Southern-Italian",
        "Central-Italian",
        "Northern-Italian",
        "Neapolitan",
        "Sicilian",
        "Sardinian",
        "Tuscan",
        "Campania",
        "Lazio",
        "Romana",
        "Apulian",
        "Calabrian",
        "Romagna",
        "Spanish",
        "Catalan",
        "Greek",
        "Portuguese",
        "Belgian",
        "British",
        "Irish",
        "Scottish",
        "Welsh",
        "German",
        "Swiss",
        "Austrian",
        "Dutch",
        "Scandinavian",
        "Swedish",
        "Norwegian",
        "Danish",
        "Polish",
        "Hungarian",
        "Romanian",
        "Croatian",
        "Albanian",
        "Ukrainian",
        "Russian",
        "Central European",
        "Eastern European",
    },
    "Asiatique": {
        "Asian",
        "Japanese",
        "Japanese Fusion",
        "Sushi",
        "Chinese",
        "Beijing cuisine",
        "Fujian",
        "Xinjiang",
        "Yunnan",
        "Korean",
        "Vietnamese",
        "Thai",
        "Indian",
        "Pakistani",
        "Bangladeshi",
        "Sri Lankan",
        "Nepalese",
        "Tibetan",
        "Indonesian",
        "Malaysian",
        "Singaporean",
        "Taiwanese",
        "Philippine",
        "Mongolian",
        "Central Asian",
        "Uzbek",
        "Afghani",
    },
    "Moyen-Orient": {
        "Middle Eastern",
        "Lebanese",
        "Israeli",
        "Persian",
        "Arabic",
        "Armenian",
        "Assyrian",
    },
    "Africain": {
        "African",
        "Moroccan",
        "Tunisian",
        "Algerian",
        "Egyptian",
        "Ethiopian",
        "Nigerian",
    },
    "Américain": {
        "American",
        "Mexican",
        "South American",
        "Brazilian",
        "Argentinian",
        "Peruvian",
        "Colombian",
        "Chilean",
        "Venezuelan",
        "Canadian",
        "Native American",
        "Southwestern",
    },
    "Caraïbes": {"Caribbean", "Jamaican", "Cuban", "Puerto Rican", "Bahamian"},
    "International": {
        "International",
        "Fusion",
        "Contemporary",
        "Medicinal foods",
        "Caucasian",
    },
    "Autres": {"Medicinal foods", "Caucasian"},
}

CUISINE_TYPE_MAP = {
    "Fast food": "Fast-Food",
    "Street Food": "Street-Food",
    "Pizza": "Pizza",
    "Diner": "Diner",
    "Bar": "Bar",
    "Pub": "Bar",
    "Wine Bar": "Bar",
    "Brew Pub": "Bar",
    "Dining bars": "Bar",
    "Beer restaurants": "Bar",
    "Cafe": "Café",
    "Deli": "Deli",
    "Healthy": "Healthy",
    "Soups": "Healthy",
    "Medicinal foods": "Healthy",
    "Grill": "Grill",
    "Barbecue": "Grill",
    "Steakhouse": "Steakhouse",
    "Seafood": "Seafood",
    "Sushi": "Sushi",
    "Japanese sweets parlour": "Sweets",
    "Fruit parlours": "Sweets",
    "Gastropub": "Gastropub",
    "Contemporary": "Gastronomic",
    "Fusion": "Fusion",
}

# 1) mapping inverse cuisine -> continent
CUISINE_CONTINENT_MAP = {
    cuisine: continent
    for continent, cuisines in CUISINE_BY_CONTINENT.items()
    for cuisine in cuisines
}


# 2) fonctions correctes
def get_cuisine_continent(cuisines_list):
    if not isinstance(cuisines_list, list):
        return "Autres"
    for c in cuisines_list:
        continent = CUISINE_CONTINENT_MAP.get(c)
        if continent:
            return continent
    return "Autres"


def get_cuisine_type(cuisines_list):
    if not isinstance(cuisines_list, list):
        return "Other"
    for c in cuisines_list:
        cuisine_type = CUISINE_TYPE_MAP.get(c)
        if cuisine_type:
            return cuisine_type
    return "Autres"


# 3) apply (PAS map)
df_trip_clean["cuisine_continent"] = df_trip_clean["cuisines_list"].apply(
    get_cuisine_continent
)
df_trip_clean["cuisine_type"] = df_trip_clean["cuisines_list"].apply(get_cuisine_type)

print(df_trip_clean[["cuisines_list", "cuisine_continent", "cuisine_type"]])


# ----------------------------------------------------------------------------
# T.3.10) Calcul du RecoScore (robuste)
# ----------------------------------------------------------------------------

# 1) Normalisations
# Rating global
df_trip_clean["Q_rating"] = df_trip_clean["avg_rating_filled"] / 5

# Qualité fine (food / service / value)
df_trip_clean["Q_partial"] = (
    df_trip_clean["partial_quality_score_weighted"].fillna(
        df_trip_clean["partial_quality_score_weighted"].median()
    )
    / 5
)

# Excellence
df_trip_clean["E"] = df_trip_clean["excellent_pct"].fillna(0) / 100

# Fiabilité
df_trip_clean["R"] = np.log1p(df_trip_clean["total_reviews_count_filled"]) / np.log1p(
    df_trip_clean["total_reviews_count_filled"].max()
)

# Accessibilité
df_trip_clean["A"] = (
    df_trip_clean["accessibility_score_5"].fillna(
        df_trip_clean["accessibility_score_5"].median()
    )
    / 5
)

# 2) Qualité globale (Q combinée) : combiner rating global (70%) + qualité fine (30%) :
df_trip_clean["Q"] = 0.7 * df_trip_clean["Q_rating"] + 0.3 * df_trip_clean["Q_partial"]

# 3) Score final de recommandation streamlit
df_trip_clean["reco_score"] = (
    100
    * (
        0.40 * df_trip_clean["Q"]
        + 0.25 * df_trip_clean["E"]
        + 0.20 * df_trip_clean["R"]
        + 0.15 * df_trip_clean["A"]
    )
    + (df_trip_clean["awards_binary"] * 3)
).round(2)

print(df_trip_clean[["Q_rating", "Q_partial", "Q", "E", "R", "reco_score"]])


# ----------------------------------------------------------------------------
# T.3.11) Extraction postal_code depuis address (robuste)
# ----------------------------------------------------------------------------
df_trip_clean["postal_code"] = (
    df_trip_clean["address"].astype(str).str.extract(r"\b(\d{5})\b", expand=False)
)
print(df_trip_clean[["address", "postal_code"]])


# ===================================================================================
# T.4) Transformer (robuste)
# ===================================================================================

# Garder uniquement les colonnes utiles
cols_keep = [
    "restaurant_link",
    "restaurant_name",
    "country",
    "region",
    "city",
    "address",
    "latitude",
    "longitude",
    "postal_code",
    "avg_rating_filled",
    "total_reviews_count_filled",
    "url",
    "price_category",
    "reco_score",
    "vegetarian_friendly",
    "vegan_options",
    "gluten_free",
    "is_halal",
    "is_kosher",
    "cuisine_continent",
    "cuisine_type",
]

df_trip_clean_norm = df_trip_clean[cols_keep].copy()

# Colonnes standard (type + distance_km placeholder)
df_trip_clean_norm["source"] = "tripadvisor_european_restaurants"
df_trip_clean_norm["type"] = "Restaurant"
df_trip_clean_norm["distance_km"] = np.nan
df_trip_clean_norm["max_people"] = np.nan

# Normaliser / renommer

rename_map = {
    "restaurant_link": "source_id",
    "restaurant_name": "name",
    "latitude": "lat",
    "longitude": "lon",
    "cuisines_list": " snippet",
    "price_category": "price_level",
    "avg_rating_filled": "rating",
    "total_reviews_count_filled": "review_count",
}
df_trip_clean_normux = df_trip_clean_norm.rename(columns=rename_map, errors="ignore")

# réordonner les colonnes UX
UX_COLUMNS = [
    "source_id",
    "source",
    "type",
    "url",
    "lat",
    "lon",
    "name",
    "address",
    "postal_code",
    "city",
    "region",
    "country",
    "snippet",
    "rating",
    "review_count",
    "price_level",
    "reco_score",
    "cuisine_continent",
    "cuisine_type",
    "vegetarian_friendly",
    "vegan_options",
    "gluten_free",
    "is_halal",
    "is_kosher",
    "max_people",
    "distance_km",
]

for col in UX_COLUMNS:
    if col not in df_trip_clean_normux.columns:
        df_trip_clean_normux[col] = None

df_trip_clean_normux_ = df_trip_clean_normux[UX_COLUMNS]

# sauvegarder les changements
df_tripadvisor_France = df_trip_clean_normux_.copy()

# Export parquet final (un seul fichier)
df_tripadvisor_France.to_parquet("df_tripadvisor_France.parquet", index=False)


# 7) Vérifications
print(df_tripadvisor_France.shape)
print(df_tripadvisor_France.head(2))
print(df_tripadvisor_France.info())
