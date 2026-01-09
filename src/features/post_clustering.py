from __future__ import annotations
import polars as pl
import math
import numpy as np
from typing import Literal, Dict

from src.features.osrm import OSRMClientAsync

TransportMode = Literal["walk", "bike", "car"]

TRANSPORT_MAX_RADIUS_KM: Dict[TransportMode, float] = {
    "walk": 2.0,
    "bike": 5.0,
    "car": 20.0,
}

# Pour limiter le nombre de POIs avant OSRM (optionnel)
DEFAULT_MAX_POIS_PER_CLUSTER = 40

# Nombre cible de restaurants par jour/cluster
TARGET_RESTAURANTS_PER_CLUSTER = 2

# ------------------------------------------
# haversine en Polars
# ------------------------------------------
def haversine_single(latitude1: float, longitude1: float, latitude2: float, longitude2: float) -> float:
    R = 6371.0  # km
    latitude1, longitude1, latitude2, longitude2 = map(math.radians, [latitude1, longitude1, latitude2, longitude2])
    dlat = latitude2 - latitude1
    dlongitude = longitude2 - longitude1
    a = math.sin(dlat / 2) ** 2 + math.cos(latitude1) * math.cos(latitude2) * math.sin(dlongitude / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))
  
def haversine_expr(latitude1_col: str, longitude1_col: str, latitude2_col: str, longitude2_col: str) -> pl.Expr:
    return (
        pl.struct([latitude1_col, longitude1_col, latitude2_col, longitude2_col])
        .map_elements(
            lambda s: haversine_single(
                s[latitude1_col],
                s[longitude1_col],
                s[latitude2_col],
                s[longitude2_col],
            ),
            return_dtype=pl.Float64,
        )
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
        (
            pl.col("final_score") * 0.6
            + pl.col("diversity_commune_norm") * 0.4
        ).alias("score_diversity")
    )

    # 2. Tri par cluster
    df_sorted = (
        df_sorted
        .sort(["cluster_id", "score_diversity"], descending=[False, True])
    )


    # Rang dans le cluster
    df_ranked = df_sorted.with_columns(
        pl.col("poi_id")
        .cum_count()
        .over("cluster_id")
        .alias("rank_in_cluster")
    )

    # Filtrer sur le rang
    df_filtered = df_ranked.filter(pl.col("rank_in_cluster") < max_pois_per_cluster)

    # On peut drop la Colonne de ranking si pas utile ensuite
    return df_filtered.drop("rank_in_cluster")

# ------------------------------------------
# Restaurant filtering          
# ------------------------------------------
def enforce_restaurant_constraint(
    df_filtered: pl.DataFrame,
    df_full: pl.DataFrame,
    target_restaurants: int = TARGET_RESTAURANTS_PER_CLUSTER,
    restaurant_category: str = "Gastronomie & Restauration",
) -> pl.DataFrame:
    """
    Garantit qu'on a jusqu'à target_restaurants par cluster.
    - df_filtered : résultat du ScoreFilter (déjà réduit)
    - df_full : dataframe complet avant filtrage (même clusters)
    """
    if df_filtered.is_empty():
        return df_filtered

    # Restaurants déjà présents dans df_filtered
    filtered_restos = df_filtered.filter(pl.col("main_category") == restaurant_category)

    # Nombre de restos par cluster après filtrage
    resto_counts = (
        filtered_restos
        .group_by("cluster_id")
        .agg(pl.len().alias("n_restos_filtered"))
    )

    # Clusters où il manque des restos
    clusters_needing_restos = resto_counts.filter(
        pl.col("n_restos_filtered") < target_restaurants
    )

    # Si tous les clusters ont déjà assez de restos → rien à faire
    if clusters_needing_restos.is_empty():
        return df_filtered

    # On complète cluster par cluster
    # 1. Préparer un df des restos candidats dans df_full (pas encore retenus)
    full_restos = df_full.filter(pl.col("main_category") == restaurant_category)

    # Exclure ceux déjà dans df_filtered (sur poi_id)
    existing_ids = df_filtered.select("poi_id").to_series()
    full_restos_candidates = full_restos.filter(~pl.col("poi_id").is_in(existing_ids))

    if full_restos_candidates.is_empty():
        # Aucun resto en plus disponible
        return df_filtered

    # Trier les restos candidats par final_score décroissant dans chaque cluster
    full_restos_candidates = full_restos_candidates.sort(
        ["cluster_id", "final_score"], descending=[False, True]
    )

    # Joindre le nombre de restos déjà présents
    full_restos_candidates = full_restos_candidates.join(
        clusters_needing_restos,
        on="cluster_id",
        how="inner",
    )

    # Calculer combien il en manque par cluster
    full_restos_candidates = full_restos_candidates.with_columns(
        (target_restaurants - pl.col("n_restos_filtered")).alias("missing_restos")
    )

    # Rang des restos dans chaque cluster (candidats)
    full_restos_candidates = full_restos_candidates.with_columns(
        pl.col("poi_id").cumcount().over("cluster_id").alias("rank_resto_candidate")
    )

    # Ne garder que les missing_restos premiers par cluster
    full_restos_candidates = full_restos_candidates.filter(
        pl.col("rank_resto_candidate") < pl.col("missing_restos")
    )

    # On n'a plus besoin des colonnes techniques
    full_restos_candidates = full_restos_candidates.drop(
        ["n_restos_filtered", "missing_restos", "rank_resto_candidate"]
    )

    # Union des POIs déjà filtrés + restos ajoutés
    df_with_restos = (
        pl.concat([df_filtered, full_restos_candidates])
        .unique(subset=["poi_id"])  # sécurité contre doublons
    )

    return df_with_restos

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
    centroids = (
        df.group_by("cluster_id")
        .agg(
            [
                pl.mean("latitude").alias("cluster_latitude"),
                pl.mean("longitude").alias("cluster_longitude"),
            ]
        )
    )

    # Join centroids
    df_with_centroid = df.join(centroids, on="cluster_id", how="left")

    # Calcul distance POI -> centroïde
    df_with_dist = df_with_centroid.with_columns(
        haversine_expr("latitude", "longitude", "cluster_latitude", "cluster_longitude").alias(
            "dist_to_cluster_center_km"
        )
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
        df
        .sort(["cluster_id", "final_score"], descending=[False, True])
        .with_row_index(name="osrm_index")  # index stable pour matrice
        .select(
            [
                "osrm_index",   # identifiant interne pour la matrice
                "poi_id",
                "cluster_id",
                "latitude",
                "longitude",
                "main_category",
                "final_score",
                # tu peux garder d'autres colonnes si utile
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
    target_restaurants: int = TARGET_RESTAURANTS_PER_CLUSTER,
    restaurant_category: str = "Gastronomie & Restauration",
    radius_override_km: float | None = None,
) -> pl.DataFrame:
    """
    Pipeline post_clustering simplifié :
    1) filtre par final_score
    2) enforce 2 restos par cluster
    3) filtre par mode de transport
    4) prépare un df compact pour OSRM
    """
    if df.is_empty():
        return df

    # 1) Filtre par score
    df_score_filtered = filter_by_final_score(
        df,
        max_pois_per_cluster=max_pois_per_cluster,
        min_score=min_score,
    )

    # 2) Contrainte restos
    df_with_restos = enforce_restaurant_constraint(
        df_filtered=df_score_filtered,
        df_full=df,
        target_restaurants=target_restaurants,
        restaurant_category=restaurant_category,
    )

    # 3) Filtre transport
    df_transport_filtered = filter_by_transport_mode(
        df_with_restos,
        mode=mode,
        radius_override_km=radius_override_km,
    )

    # 4) Préparation pour OSRM
    df_osrm = prepare_osrm_nodes(df_transport_filtered)

    return df_osrm

##############################
# OSRM ASYNC
##############################

async def build_osrm_matrices_async(
    df_clustered: pl.DataFrame,
    osrm: OSRMClientAsync,
):
    # 1) Extraire coords (latitude, longitude)
    coords = df_clustered.select(["latitude", "longitude"]).to_numpy().tolist()
    coords = [tuple(row) for row in coords]

    # 2) Ajouter osrm_index
    df_clustered = df_clustered.with_columns(
        pl.Series("osrm_index", list(range(len(df_clustered))))
    )

    # 3) Appel OSRM asynchrone
    result = await osrm.table(coords, annotations="duration,distance")

    # 4) Extraire matrices
    dist_matrix = np.array(result["distances"])
    dur_matrix = np.array(result["durations"])

    # 5) Convertir en DataFrames Polars
    df_osrm_dist = pl.DataFrame(dist_matrix).with_row_index("osrm_index")
    df_osrm_dur = pl.DataFrame(dur_matrix).with_row_index("osrm_index")

    return df_clustered, df_osrm_dist, df_osrm_dur
