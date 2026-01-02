import polars as pl
import math


# ---------------------------------------------------------
# Distance Haversine (km)
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    # Vérification des valeurs manquantes
    if (
        lat1 is None or lon1 is None or
        lat2 is None or lon2 is None
    ):
        return None

    # Vérification des types numériques
    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return None

    # Formule Haversine
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )

    return 2 * R * math.asin(math.sqrt(a))

# ---------------------------------------------------------
# Module principal
# ---------------------------------------------------------
def add_proximity(lf: pl.LazyFrame, resolver, level: str = "commune") -> pl.LazyFrame:
    if level not in ("commune", "region"):
        raise ValueError("level doit être 'commune' ou 'region'")

    key = level  # "commune" ou "region"
    centroid_func = (
        resolver.get_city_centroid if level == "commune"
        else resolver.get_region_centroid
    )


    # 1) Extraire les valeurs uniques présentes dans TON dataset
    values = (
        lf.lazy().select(key)
        .unique()
        .collect()[key] 
        .to_list()
    )
    

    # 2) Calculer les centroids valides
    centroids = {}
    for v in values:
        c = centroid_func(v)
        if c is None:
            continue
        centroids[v] = c
    
    # 3) Si aucun centroid valide → colonne vide
    if len(centroids) == 0:
        return lf.with_columns(
            pl.lit(None).alias(f"proximity_{level}")
        )

    # 4) Construire un DataFrame Polars propre
    df_centroids = (
        pl.DataFrame({
            key: list(centroids.keys()),
            "centroid_lat": [c[0] for c in centroids.values()],
            "centroid_lon": [c[1] for c in centroids.values()],
        })
        .with_columns(pl.col(key).cast(pl.Utf8))  # IMPORTANT
    )

    # 5) Join
    lf = lf.join(df_centroids, on=key, how="left")

    # 6) Distance Haversine vectorisée
    return lf.with_columns(
        pl.when(
            pl.col("latitude").is_not_null()
            & pl.col("longitude").is_not_null()
            & pl.col("centroid_lat").is_not_null()
            & pl.col("centroid_lon").is_not_null()
        )
        .then(
            pl.struct(["latitude", "longitude", "centroid_lat", "centroid_lon"])
            .map_elements(lambda r: haversine(
                r["latitude"],
                r["longitude"],
                r["centroid_lat"],
                r["centroid_lon"]
            ))
        )
        .otherwise(None)
        .alias(f"proximity_{level}")
    )
