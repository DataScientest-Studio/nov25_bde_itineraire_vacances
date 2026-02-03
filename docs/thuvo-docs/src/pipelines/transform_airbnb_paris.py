import json

import numpy as np
import pandas as pd
import pyarrow

# ===================================================================================
# T.1) Charger les données Airbnb_paris
# ===================================================================================

# lire CSV_PATH
CSV_PATH = "df_airBnb_paris.csv"
df_listings = pd.read_csv(CSV_PATH, sep=",", encoding="utf-8", low_memory=False)

# vérifier
print(df_listings.shape)
print(df_listings.info())


# ===================================================================================
# T.2) Nettoyer + compléter + transformer les valeurs
# ===================================================================================

# 1) transformer en booléen les colonnes require_guest_phone_verification (0% missing), require_guest_profile_picture (0% missing),
BOOL_COLS = [
    "require_guest_phone_verification",
    "require_guest_profile_picture",
    "host_is_superhost",
    "host_identity_verified",
]
for col in BOOL_COLS:
    if col in df_listings.columns:
        df_listings[col] = (
            df_listings[col].astype(str).str.lower().isin(["t", "true", "1", "yes"])
        )


# 2) convertir review_scores_rating sur 5
df_listings["review_scores_rating"] = (
    pd.to_numeric(df_listings["review_scores_rating"], errors="coerce") / 20
).round(2)


# 3) Nettoyer + transformer en float : price (0% missing), cleaning_fee (27% missing), security_deposit (31% missing),
# extra_people (0% missing), weekly_price (79% missing), monthly_price (87% missing)
PRICE_COLS = [
    "price",
    "cleaning_fee",
    "security_deposit",
    "extra_people",
    "weekly_price",
    "monthly_price",
]
for col in PRICE_COLS:
    if col in df_listings.columns:
        df_listings[col] = (
            df_listings[col].astype(str).str.replace(r"[^0-9.]", "", regex=True)
        )
        df_listings[col] = pd.to_numeric(df_listings[col], errors="coerce")
        df_listings[col] = df_listings[col].astype("float").fillna(0)


# 4) créer et calculer la colonne "price_level" en mode quantiles basant sur colonne "price" : 1=petit <= q1 ; q1 < 2=moyen <= q2 ; 3=confort > q2

# regarder la distribution
df_listings["price"].describe(percentiles=[0.33, 0.66])

# calculer les seuils
q1 = df_listings["price"].quantile(0.33)
q2 = df_listings["price"].quantile(0.66)
q1, q2

# créer price_level
df_listings["price_level"] = np.nan

df_listings.loc[df_listings["price"] <= q1, "price_level"] = 1
df_listings.loc[
    (df_listings["price"] > q1) & (df_listings["price"] <= q2), "price_level"
] = 2
df_listings.loc[df_listings["price"] > q2, "price_level"] = 3

# vérifier
print(df_listings[["price_level", "price"]].head(5))


# ===================================================================================
# T.3) DataFrame Final
# ===================================================================================

# Garder uniquement les colonnes utiles
cols_keep = [
    "id",
    "listing_url",
    "name",
    "city",
    "zipcode",
    "state",
    "country",
    "summary",
    "latitude",
    "longitude",
    "review_scores_rating",
    "number_of_reviews",
    "accommodates",
    "price",
    "price_level",
]

df_airbnb_clean = df_listings[cols_keep].copy()

# Créer les colonnes standards et les colonnes manquantes pour UX design
df_airbnb_clean["source"] = "AirBnb-Paris"
df_airbnb_clean["type"] = "Airbnb"
df_airbnb_clean["distance_km"] = np.nan
df_airbnb_clean["address"] = np.nan

# Normaliser / renommer
rename_map = {
    "id": "source_id",
    "listing_url": "url",
    "latitude": "lat",
    "longitude": "lon",
    "zipcode": "postal_code",
    "state": "region",
    "review_scores_rating": "rating",
    "number_of_reviews": "review_count",
    "accommodates": "max_people",
    "summary": "snippet",
}
df_airbnb_clean_norm = df_airbnb_clean.rename(columns=rename_map, errors="ignore")

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
    "max_people",
    "distance_km",
    "price",
]

for col in UX_COLUMNS:
    if col not in df_airbnb_clean_norm.columns:
        df_airbnb_clean_norm[col] = None

df_airbnb_clean_normux = df_airbnb_clean_norm[UX_COLUMNS]

# sauvegarder les changements
df_airbnb_Paris = df_airbnb_clean_normux.copy()

# Export parquet final (un seul fichier)
df_airbnb_Paris.to_parquet(
    "df_airbnb_Paris.parquet",
    index=False,
    engine="pyarrow",
    compression="zstd",
    compression_level=9,
)

# vérifier
print(df_airbnb_Paris.head(3))
print(df_airbnb_Paris.info())
