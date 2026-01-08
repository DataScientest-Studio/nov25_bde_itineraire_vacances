import time
from pathlib import Path
from typing import Dict

import polars as pl

from src.features.poi_filter import POIFilter
from src.features.spatial_clustering import SpatialClusterer
from src.features.post_clustering import build_osrm_ready_pois
from src.features.itinerary_optimizer import ItineraryOptimizer
from src.features.osrm import OSRMClientAsync


DEFAULT_VISIT_TIME = 45 * 60  # 45 minutes en secondes

class ItineraryPipeline:
    """
    Pipeline complet :
        1. Filtrage
        2. Clustering spatial
        3. Préparation OSRM (build_osrm_ready_pois)
        4. OSRM (matrices durée/distance)
        5. TSP par jour
        6. Enrichissement / assemblage multi-jour
    """

    def __init__(self, pois_path: Path):
        self.pois_path = pois_path
        self.pois_lf = pl.scan_parquet(self.pois_path)

    # ---------------------------------------------------------
    # FILTRAGE
    # ---------------------------------------------------------
    def _filter_pois(self, commune, main_categories, min_score):
        return (
            POIFilter(self.pois_lf)
            .set_commune(commune)
            .set_categories(main_categories=main_categories)
            .set_min_score(min_score)
            .apply()
        )

    # ---------------------------------------------------------
    # CLUSTERING SPATIAL
    # ---------------------------------------------------------
    def _cluster_pois(self, filtered_lf, nb_days, anchor_latitude, anchor_longitude):
        return (
            SpatialClusterer(filtered_lf)
            .set_nb_days(nb_days)
            .set_anchor(anchor_latitude, anchor_longitude)
            .apply()
        )

    # ---------------------------------------------------------
    # PRÉPARATION OSRM (post-clustering + sélection POIs)
    # ---------------------------------------------------------
    def _build_osrm_ready_pois(
        self,
        df_prepared: pl.DataFrame,
        mode: str = "walk",
        max_pois_per_cluster: int = 40,
        min_score: float = 0.2,
        target_restaurants: int = 2,
        restaurant_category: str = "Gastronomie & Restauration",
    ) -> pl.DataFrame:
        """
        Encapsule l'appel à build_osrm_ready_pois pour garder le pipeline lisible.
        df_prepared = sortie de _cluster_pois (éventuellement déjà "balancée").
        """

        df_prepared = (
            df_prepared
            .rename({
                "main_category": "main_category",
                "day": "cluster_id",
            })
        )

        df_clustered = build_osrm_ready_pois(
            df=df_prepared,
            mode=mode,
            max_pois_per_cluster=max_pois_per_cluster,
            min_score=min_score,
            target_restaurants=target_restaurants,
            restaurant_category=restaurant_category,
        )

        return df_clustered

    # ---------------------------------------------------------
    # OSRM MATRICES (async → sync via asyncio.run)
    # ---------------------------------------------------------
    def _compute_osrm_matrices(
        self,
        df_clustered: pl.DataFrame,
        osrm: OSRMClientAsync,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Appelle build_osrm_matrices_async en masquant asyncio.run
        pour garder une API synchrone dans le pipeline.
        """
        df_clustered, df_osrm_dist, df_osrm_dur = asyncio.run(
            build_osrm_matrices_async(df_clustered, osrm)
        )
        return df_clustered, df_osrm_dist, df_osrm_dur

    # ---------------------------------------------------------
    # TSP / ITINÉRAIRE
    # ---------------------------------------------------------
    def _compute_itinerary(
        self,
        df_clustered: pl.DataFrame,
        df_osrm_dur: pl.DataFrame,
    ):
        optimizer = ItineraryOptimizer.from_list_matrix(
            df_pois=df_clustered,
            matrix=df_osrm_dur.to_numpy(),
            metric="duration",
        )
        df_itinerary = optimizer.solve_all_days()
        return optimizer, df_itinerary

    # ---------------------------------------------------------
    # ENRICHISSEMENT (inchangé)
    # ---------------------------------------------------------
    def enrich_itinerary(self, df_day, matrix_durations, matrix_distances, order):
        poi_order = [i - 1 for i in order if i != 0]

        durations = []
        distances = []
        visit_times = []

        for i in range(1, len(order)):
            prev_idx = order[i - 1]
            curr_idx = order[i]

            durations.append(float(matrix_durations[prev_idx][curr_idx]))
            distances.append(float(matrix_distances[prev_idx][curr_idx]))
            visit_times.append(int(DEFAULT_VISIT_TIME))

        step_total = [d + v for d, v in zip(durations, visit_times)]

        cum_durations = []
        cum_distances = []
        cum_total = []

        total_d = 0.0
        total_t = 0.0
        total_all = 0.0

        for dist, dur, step in zip(distances, durations, step_total):
            total_d += dist
            total_t += dur
            total_all += step

            cum_distances.append(total_d)
            cum_durations.append(total_t)
            cum_total.append(total_all)

        periods = ["morning" if t < 4 * 3600 else "afternoon" for t in cum_total]

        derived = pl.DataFrame({
            "order": pl.Series("order", list(range(1, len(poi_order) + 1)), dtype=pl.Int64),
            "distance_from_prev": pl.Series("distance_from_prev", distances, dtype=pl.Float64),
            "duration_from_prev": pl.Series("duration_from_prev", durations, dtype=pl.Float64),
            "visit_time": pl.Series("visit_time", visit_times, dtype=pl.Int64),
            "step_total_duration": pl.Series("step_total_duration", step_total, dtype=pl.Float64),
            "cum_distance": pl.Series("cum_distance", cum_distances, dtype=pl.Float64),
            "cum_duration": pl.Series("cum_duration", cum_durations, dtype=pl.Float64),
            "cum_total_duration": pl.Series("cum_total_duration", cum_total, dtype=pl.Float64),
            "day_total_distance": pl.Series("day_total_distance", [total_d] * len(poi_order), dtype=pl.Float64),
            "day_total_duration": pl.Series("day_total_duration", [total_all] * len(poi_order), dtype=pl.Float64),
            "period": pl.Series("period", periods, dtype=pl.Utf8),
        })

        return df_day[poi_order].with_columns(derived)

    # ---------------------------------------------------------
    # PIPELINE COMPLET
    # ---------------------------------------------------------
    def run(
        self,
        commune,
        main_categories,
        min_score,
        nb_days,
        anchor_latitude,
        anchor_longitude,
        osrm: OSRMClientAsync,
        osrm_mode: str = "walk",
        max_pois_per_cluster: int = 40,
        osrm_min_score: float = 0.2,
        target_restaurants: int = 2,
        restaurant_category: str = "Gastronomie & Restauration",
    ):
        """
        Pipeline synchrone de bout en bout.
        Retourne :
            - df_clustered prêt OSRM
            - df_osrm_dist, df_osrm_dur
            - df_itinerary
            - optimizer (pour routes GeoJSON, etc.)
        """

        # 1. Filtrage
        filtered_lf = self._filter_pois(commune, main_categories, min_score)

        # 2. Clustering : df_prepared (jour, etc.)
        df_prepared = self._cluster_pois(filtered_lf, nb_days, anchor_latitude, anchor_longitude)

        # 3. Préparation OSRM (build_osrm_ready_pois)
        df_clustered = self._build_osrm_ready_pois(
            df_prepared=df_prepared,
            mode=osrm_mode,
            max_pois_per_cluster=max_pois_per_cluster,
            min_score=osrm_min_score,
            target_restaurants=target_restaurants,
            restaurant_category=restaurant_category,
        )

        # 4. OSRM matrices (async → sync)
        df_clustered, df_osrm_dist, df_osrm_dur = self._compute_osrm_matrices(
            df_clustered=df_clustered,
            osrm=osrm,
        )

        # 5. Itinéraire optimisé
        optimizer, df_itinerary = self._compute_itinerary(df_clustered, df_osrm_dur)

        return df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer