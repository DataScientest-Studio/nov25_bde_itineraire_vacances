import polars as pl
import math
from etl.utils.bounding_box import BoundingBoxResolver

resolver = BoundingBoxResolver()

# ---------------------------------------------------------
# Distance Haversine (km)
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    if (
        lat1 is None or lon1 is None or
        lat2 is None or lon2 is None
    ):
        return None

    try:
        lat1 = float(lat1)
        lon1 = float(lon1)
        lat2 = float(lat2)
        lon2 = float(lon2)
    except (TypeError, ValueError):
        return None

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
def add_proximity(lf: pl.LazyFrame, resolver, level: str = "commune", tau: float = 5.0) -> pl.LazyFrame:
    """
    Ajoute :
    - proximity_{level} : distance brute (km)
    - proximity_{level}_norm : score normalisé (0–1), 1 = très proche

    Normalisation : exp(-distance / tau)
    tau = distance caractéristique (5 km par défaut)
    """

    if level not in ("commune", "region"):
        raise ValueError("level doit être 'commune' ou 'region'")

    key = level
    centroid_func = (
        resolver.get_city_centroid if level == "commune"
        else resolver.get_region_centroid
    )

    # 1) Extraire les valeurs uniques
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
        if c is not None:
            centroids[v] = c

    # 3) Aucun centroid → colonnes vides
    if len(centroids) == 0:
        return lf.with_columns([
            pl.lit(None).alias(f"proximity_{level}"),
            pl.lit(None).alias(f"proximity_{level}_norm"),
        ])

    # 4) DataFrame des centroids
    df_centroids = (
        pl.DataFrame({
            key: list(centroids.keys()),
            "centroid_lat": [c[0] for c in centroids.values()],
            "centroid_lon": [c[1] for c in centroids.values()],
        })
        .with_columns(pl.col(key).cast(pl.Utf8))
    )

    # 5) Join
    lf = lf.join(df_centroids, on=key, how="left")

    # 6) Distance Haversine brute
    dist_col = f"proximity_{level}"
    lf = lf.with_columns(
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
        .alias(dist_col)
    )


    # 7) Log-scaling pour lisser les outliers
    lf = lf.with_columns([
        pl.col(dist_col).log1p().alias(f"{dist_col}_log")
    ])

    # 8) Normalisation exponentielle sur la distance lissée
    lf = lf.with_columns([
        pl.when(pl.col(f"{dist_col}_log").is_not_null())
        .then(pl.col(f"{dist_col}_log").map_elements(lambda d: math.exp(-d / tau)))
        .otherwise(None)
        .alias(f"{dist_col}_norm")
    ])


    return lf