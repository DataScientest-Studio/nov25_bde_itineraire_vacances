from __future__ import annotations

import math
from typing import Dict, Literal

import numpy as np
import polars as pl

from app.pipeline.features.osrm import OSRMClientAsync
import logging
logger = logging.getLogger("uvicorn.error")

TransportMode = Literal["walk", "bike", "car"]

TRANSPORT_MAX_RADIUS_KM: Dict[TransportMode, float] = {
    "walk": 14.0,
    "bike": 27.0,
    "car": 40.0,
}

# Pour limiter le nombre de POIs avant OSRM
DEFAULT_MAX_POIS_PER_CLUSTER = 100


# ------------------------------------------
# haversine en Polars
# ------------------------------------------
def haversine_single(
    latitude1: float, longitude1: float, latitude2: float, longitude2: float
) -> float:
    R = 6371.0  # km
    latitude1, longitude1, latitude2, longitude2 = map(
        math.radians, [latitude1, longitude1, latitude2, longitude2]
    )
    dlat = latitude2 - latitude1
    dlongitude = longitude2 - longitude1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(latitude1) * math.cos(latitude2) * math.sin(dlongitude / 2) ** 2
    )
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def haversine_expr(
    latitude1_col: str, longitude1_col: str, latitude2_col: str, longitude2_col: str
) -> pl.Expr:
    return pl.struct(
        [latitude1_col, longitude1_col, latitude2_col, longitude2_col]
    ).map_elements(
        lambda s: haversine_single(
            s[latitude1_col],
            s[longitude1_col],
            s[latitude2_col],
            s[longitude2_col],
        ),
        return_dtype=pl.Float64,
    )


# ------------------------------------------
# Score filtering
# ------------------------------------------
def filter_by_final_score(
    df: pl.DataFrame,
    max_pois_per_cluster: int = DEFAULT_MAX_POIS_PER_CLUSTER,
    min_score: float | None = None,
) -> pl.DataFrame:
    """
    Garder les POIs avec les meilleurs final_score par cluster.
    - Optionnel : seuil min_score
    - Toujours : limite max_pois_per_cluster par cluster
    """
    if df.is_empty():
        return df

    # filter poi que pour itinéraire
    if "itineraire" in df.columns:
        df = df.filter(pl.col("itineraire") == True)

    # Optionnel : filtre par seuil
    if min_score is not None:
        df = df.filter(pl.col("final_score") >= min_score)

    if df.is_empty():
        return df

    # Tri par cluster
    # 1. Score diversité
    df_sorted = df.with_columns(
        (pl.col("final_score") * 0.2 + pl.col("diversity_commune_norm") * 0.8).alias(
            "score_diversity"
        )
    )

    # 2. Tri par cluster
    df_sorted = df_sorted.sort(
        ["cluster_id", "score_diversity"], descending=[False, True]
    )

    # Rang dans le cluster
    df_ranked = df_sorted.with_columns(
        pl.col("poi_id").cum_count().over("cluster_id").alias("rank_in_cluster")
    )

    # Filtrer sur le rang
    df_filtered = df_ranked.filter(pl.col("rank_in_cluster") < max_pois_per_cluster)

    # On peut drop la Colonne de ranking si pas utile ensuite
    return df_filtered.drop("rank_in_cluster")


# ------------------------------------------
# Restaurant filtering
# ------------------------------------------
def split_restaurants_and_others(
    df: pl.DataFrame,
    restaurant_subcategories: list[str] = [
        "Restaurants",
        "Restauration rapide",
        "Bars & cafés",
    ],
    k_restos: int = 2,
) -> pl.DataFrame:
    if df.is_empty():
        return df

    # Détection fine des restaurants
    restos = df.filter(pl.col("sub_category").is_in(restaurant_subcategories))
    others = df.filter(~pl.col("sub_category").is_in(restaurant_subcategories))

    # Trier par cluster puis score
    restos_sorted = restos.sort(
        ["cluster_id", "final_score"], descending=[False, True]
    )

    restos_ranked = restos_sorted.with_columns(
        pl.col("poi_id").cum_count().over("cluster_id").alias("rank_resto")
    )

    restos_top_k = restos_ranked.filter(pl.col("rank_resto") < k_restos).drop("rank_resto")
    print("NB TOTAL POI :", df.height)
    print("NB RESTOS DETECTES :", restos.height)
    print("NB OTHERS :", others.height)
    print("NB RESTOS TOP K :", restos_top_k.height)

    print("SUB_CATEGORY RESTOS :", restos["sub_category"].unique())

    return pl.concat([others, restos_top_k]).unique(subset=["poi_id"])


RESTAURANT_SUBCATEGORIES = [
    "Restaurants",
    "Restauration rapide",
    "Bars & cafés",
]

def smart_restaurant_sampling(df: pl.DataFrame,
                              max_per_subcat_per_cell: int = 2) -> pl.DataFrame:
    restos = (
        df.filter(pl.col("sub_category").is_in(RESTAURANT_SUBCATEGORIES))
          .sort("final_score", descending=True)
          .group_by(["sub_category", "h3_r7"])
          .head(max_per_subcat_per_cell)
    )

    others = df.filter(~pl.col("sub_category").is_in(RESTAURANT_SUBCATEGORIES))
    cols = restos.columns
    others = others.select(cols)

    logger.info(f"[rebalance] restos kept: {restos.height}, others: {others.height}")
    return pl.concat([restos, others])


def ensure_minimum_per_category(df: pl.DataFrame,
                                max_per_category: int = 10) -> pl.DataFrame:
    non_resto = df.filter(~pl.col("sub_category").is_in(RESTAURANT_SUBCATEGORIES))

    grouped = (
        non_resto.sort("final_score", descending=True)
                 .group_by("main_category")
                 .head(max_per_category)
    )

    restos = df.filter(pl.col("sub_category").is_in(RESTAURANT_SUBCATEGORIES))
    cols = grouped.columns
    restos = restos.select(cols)


    logger.info(f"[rebalance] non-resto kept: {grouped.height}, restos kept: {restos.height}")
    return pl.concat([grouped, restos])


def limit_density(df: pl.DataFrame,
                  max_per_cell: int = 10) -> pl.DataFrame:
    limited = (
        df.sort("final_score", descending=True)
          .group_by("h3_r7")
          .head(max_per_cell)
    )

    logger.info(f"[rebalance] after density limit: {limited.height} POIs")
    return limited


def rebalance_pois(df: pl.DataFrame) -> pl.DataFrame:
    logger.info(f"[rebalance] initial POIs: {df.height}")

    df1 = smart_restaurant_sampling(df, max_per_subcat_per_cell=3)
    df2 = ensure_minimum_per_category(df1, max_per_category=40)
    df3 = limit_density(df2, max_per_cell=15)

    df3 = df3.unique(subset=["poi_id"])
    logger.info(f"[rebalance] final POIs: {df3.height}")

    return df3

# ------------------------------------------
# Transport filtering
# ------------------------------------------
def filter_by_transport_mode(
    df: pl.DataFrame,
    mode: TransportMode,
    radius_override_km: float | None = None,
) -> pl.DataFrame:
    """
    Filtrer les POIs par compatibilité avec le mode de transport
    en fonction d'un rayon max autour du centroïde du cluster.
    """
    if df.is_empty():
        return df

    max_radius_km = radius_override_km or TRANSPORT_MAX_RADIUS_KM[mode]

    # Centroïde par cluster
    centroids = df.group_by("cluster_id").agg(
        [
            pl.mean("latitude").alias("cluster_latitude"),
            pl.mean("longitude").alias("cluster_longitude"),
        ]
    )

    # Join centroids
    df_with_centroid = df.join(centroids, on="cluster_id", how="left")

    # Calcul distance POI -> centroïde
    df_with_dist = df_with_centroid.with_columns(
        haversine_expr(
            "latitude", "longitude", "cluster_latitude", "cluster_longitude"
        ).alias("dist_to_cluster_center_km")
    )

    # Filtre sur le rayon
    df_filtered = df_with_dist.filter(
        pl.col("dist_to_cluster_center_km") <= max_radius_km
    )

    # Optionnel : tu peux drop les colonnes de centroid/distance si pas utiles après
    return df_filtered.drop(["cluster_latitude", "cluster_longitude"])


# ------------------------------------------
# Préparation OSRM
# ------------------------------------------
def prepare_osrm_nodes(df: pl.DataFrame) -> pl.DataFrame:
    """
    Prépare un df minimal pour OSRM (nodes à passer à la requête).
    On impose un ordre stable par cluster puis par final_score.
    """

    if df.is_empty():
        return df

    df_nodes = (
        df.sort(["cluster_id", "final_score"], descending=[False, True])
        .with_row_index(name="osrm_index")  # index stable pour matrice
        .select(
            [
                "osrm_index",
                "poi_id",
                "cluster_id",
                "latitude",
                "longitude",
                "main_category",
                "sub_category",
                "final_score",
            ]
        )
    )

    return df_nodes


# ------------------------------------------
# Pipeline complet
# ------------------------------------------
def build_osrm_ready_pois(
    df: pl.DataFrame,
    mode: TransportMode,
    max_pois_per_cluster: int = DEFAULT_MAX_POIS_PER_CLUSTER,
    min_score: float | None = None,
    radius_override_km: float | None = None,
) -> pl.DataFrame:
    """
    Pipeline post_clustering simplifié :
    1) filtre par final_score
    2) enforce 2 restos par cluster
    3) filtre par mode de transport
    4) prépare un df compact pour OSRM
    """
    if isinstance(df, pl.LazyFrame):
        df = df.collect()

    if df.is_empty():
        return df

    logger.info("=== 1. Chargement initial ===")
    logger.info(f"Total POIs : {df.height}")

    # 1) Rééquilibrage intelligent AVANT tout filtrage
    df = rebalance_pois(df)

    logger.info("=== 2. Après rééquilibrage ===")
    logger.info(f"Total POIs : {df.height}")

    # 2) Split restos / non-restos (top K par cluster)
    df_filtered = split_restaurants_and_others(df, k_restos=3)

    logger.info("=== 3. Après split restos ===")
    logger.info(f"Total POIs : {df_filtered.height}")

    # 3) Filtre transport
    df_transport_filtered = filter_by_transport_mode(
        df_filtered,
        mode=mode,
        radius_override_km=radius_override_km,
    )

    logger.info("=== 4. Après filtre transport ===")
    logger.info(f"Total POIs : {df_transport_filtered.height}")


    # 4) Filtre par score
    df_score_filtered = filter_by_final_score(
        df_transport_filtered,
        max_pois_per_cluster=max_pois_per_cluster,
        min_score=min_score,
    )

    logger.info("=== 5. Après filtre par score ===")
    logger.info(f"Total POIs : {df_score_filtered.height}")

    # 5) Préparation pour OSRM
    df_osrm = prepare_osrm_nodes(df_score_filtered)
    logger.info("=== 6. Formatage final ===")
    logger.info(f"Total POIs : {df_osrm.height}")

    return df_osrm


##############################
# OSRM ASYNC
##############################


async def build_osrm_matrices_async(
    df_clustered: pl.DataFrame, osrm: OSRMClientAsync, profile: str = "foot"
):
    """
    Construit les matrices OSRM (distance + durée) en mode async.
    profile = "foot" | "bike" | "driving"
    """

    # 1) Extraire coords (lon, lat)
    coords = df_clustered.select(["longitude", "latitude"]).to_numpy().tolist()
    coords = [tuple(row) for row in coords]

    # 2) Ajouter osrm_index
    if "osrm_index" not in df_clustered.columns:
        df_clustered = df_clustered.with_columns(
            pl.Series("osrm_index", list(range(len(df_clustered))))
        )

    # 3) Appel OSRM
    result = await osrm.table(coords, annotations="duration,distance", profile=profile)

    # 4) Matrices numpy
    dist_matrix = np.array(result["distances"])
    dur_matrix = np.array(result["durations"])

    # 5) Conversion Polars
    df_osrm_dist = pl.DataFrame(dist_matrix).with_row_index("osrm_index")
    df_osrm_dur = pl.DataFrame(dur_matrix).with_row_index("osrm_index")

    return df_clustered, df_osrm_dist, df_osrm_dur
