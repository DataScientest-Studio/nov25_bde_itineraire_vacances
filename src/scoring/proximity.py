import math
import polars as pl
from scoring.utils.bounding_box import BoundingBoxResolver

resolver = BoundingBoxResolver()


# ---------------------------------------------------------
# Distance Haversine (km)
# ---------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1, lon1, lat2, lon2 = map(float, (lat1, lon1, lat2, lon2))
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
def add_proximity(
    lf: pl.LazyFrame,
    resolver: BoundingBoxResolver,
    level: str = "commune",
    tau: float = 5.0,
    key_col: str = "addr_commune",  
) -> pl.LazyFrame:
    """
    Ajoute :
    - proximity_{level} : distance brute (km)
    - proximity_{level}_norm : score normalisé (0–1)

    Normalisation : exp(-distance / tau)
    """

    if level not in ("commune", "region"):
        raise ValueError("level doit être 'commune' ou 'region'")

    centroid_func = (
        resolver.get_city_centroid if level == "commune" else resolver.get_region_centroid
    )

    # 1) Extraire les valeurs uniques
    values = lf.select(key_col).unique().collect().get_column(key_col).to_list()

    # 2) Construire un DataFrame des centroids valides
    centroids = [
        (v, *centroid_func(v)) for v in values if centroid_func(v) is not None
    ]

    if not centroids:
        return lf.with_columns(
            pl.lit(None).alias(f"proximity_{level}"),
            pl.lit(None).alias(f"proximity_{level}_norm"),
        )

    df_centroids = pl.DataFrame(
        {
            key_col: [c[0] for c in centroids],
            "centroid_lat": [c[1] for c in centroids],
            "centroid_lon": [c[2] for c in centroids],
        }
    )

    # 3) Join
    lf = lf.join(df_centroids.lazy(), on=key_col, how="left")

    dist_col = f"proximity_{level}"

    # 4) Distance Haversine
    lf = lf.with_columns(
        pl.struct(["latitude", "longitude", "centroid_lat", "centroid_lon"])
        .map_elements(lambda r: haversine(r["latitude"], r["longitude"], r["centroid_lat"], r["centroid_lon"]))
        .alias(dist_col)
    )

    # 5) Normalisation exponentielle
    lf = lf.with_columns(
        pl.col(dist_col)
        .log1p()
        .map_elements(lambda d: math.exp(-d / tau) if d is not None else None)
        .alias(f"{dist_col}_norm")
    )

    return lf