import time
from pathlib import Path
from typing import Dict
import asyncio

import polars as pl

from features.poi_filter import POIFilter
from features.spatial_clustering import SpatialClusterer
from features.post_clustering import build_osrm_ready_pois, build_osrm_matrices_async
from features.itinerary_optimizer import ItineraryOptimizer
from features.optimizer_ga import GeneticAlgo
from features.osrm import OSRMClientAsync



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
    def _filter_pois(self, commune, main_categories, sub_categories, min_score):
        return (
            POIFilter(self.pois_lf)
            .set_commune(commune)
            .set_categories(main_categories=main_categories, sub_categories=sub_categories)
            .set_min_score(min_score)
            .apply()
        )

    # ---------------------------------------------------------
    # CLUSTERING SPATIAL
    # ---------------------------------------------------------
    def _cluster_pois(self, filtered_lf, nb_days, anchor_lat, anchor_lon):
        return (
            SpatialClusterer(filtered_lf)
            .set_nb_days(nb_days)
            .set_anchor(anchor_lat, anchor_lon)
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


        if "day" in df_prepared.columns and "cluster_id" not in df_prepared.columns:
            df_prepared = df_prepared.rename({"day": "cluster_id"})

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
    # TSP par GA / ITINÉRAIRE
    # ---------------------------------------------------------
    def _compute_itinerary_ga(self, df_clustered, df_osrm_dur, df_osrm_dist):
        results = []

        for cluster_id in df_clustered["cluster_id"].unique():
            df_day = df_clustered.filter(pl.col("cluster_id") == cluster_id)

            # Cluster vide → on ignore
            if df_day.height == 0:
                print(f"[WARN] Cluster {cluster_id} vide — ignoré")
                continue

            # Garantir sub_category
            # if "sub_category" not in df_day.columns:
            #     df_day = df_day.with_columns(pl.lit("Unknown").alias("sub_category"))
            # else:
            #     df_day = df_day.with_columns(pl.col("sub_category").fill_null("Unknown"))

            # Conversion en pandas pour le GA
            df_day_pd = df_day.to_pandas()

            # Lancer le GA
            ga = GeneticAlgo(
                poi_df=df_day_pd,
                duration_matrix=df_osrm_dur.to_numpy()
            )
            ga.setup_toolbox(itin_min_poi=5, itin_max_poi=15)

            best_route, fitness = ga.run_ga(
                pop_size=50,
                ngen=50,
                cxpb=0.75,
                mutpb=0.3
            )

            # Route vide → on ignore
            if len(best_route) == 0:
                print(f"[WARN] GA n'a pas trouvé de route pour cluster {cluster_id}")
                continue

            # Filtrer les POIs valides
            valid_pois = set(df_day["poi_id"].to_list())
            best_route_filtered = [pid for pid in best_route if pid in valid_pois]

            # Supprimer les doublons en conservant l'ordre
            seen = set()
            best_route_unique = []
            for pid in best_route_filtered:
                if pid not in seen:
                    best_route_unique.append(pid)
                    seen.add(pid)

            # Construire df_route
            df_route = df_day.filter(pl.col("poi_id").is_in(best_route_unique))

            if df_route.height == 0:
                print(f"[WARN] df_route vide pour cluster {cluster_id}")
                continue

            # Ajouter la colonne order
            df_route = df_route.with_columns(
                pl.Series("order", list(range(1, df_route.height + 1)))
            )

            # Enrichissement
            df_enriched = self.enrich_itinerary(
                df_day=df_route,
                matrix_durations=df_osrm_dur.to_numpy(),
                matrix_distances=df_osrm_dist.to_numpy(),
                order=[ga.poi_to_index[pid] for pid in best_route_unique]
            )

            # Ajouter cluster_id
            df_enriched = df_enriched.with_columns(
                pl.lit(cluster_id).alias("cluster_id")
            )

            results.append(df_enriched)

        if len(results) == 0:
            return pl.DataFrame()

        optimizer = "ga"
        df_itinerary = pl.concat(results)

        return df_itinerary, optimizer


    # ---------------------------------------------------------
    # ENRICHISSEMENT
    # ---------------------------------------------------------
    def enrich_itinerary(self, df_day, matrix_durations, matrix_distances, order):
        n = len(order)

        # 1. Distances et durées depuis le précédent
        distance_from_prev = [0.0]
        duration_from_prev = [0.0]

        for i in range(n - 1):
            d = float(matrix_distances[order[i], order[i + 1]])
            t = float(matrix_durations[order[i], order[i + 1]])
            distance_from_prev.append(d)
            duration_from_prev.append(t)

        # 2. Cumul
        cumulative_distance = []
        cumulative_duration = []

        cum_d = 0.0
        cum_t = 0.0

        for d, t in zip(distance_from_prev, duration_from_prev):
            cum_d += d
            cum_t += t
            cumulative_distance.append(cum_d)
            cumulative_duration.append(cum_t)

        # 3. Ajout des colonnes
        df_enriched = df_day.with_columns([
            pl.Series("distance_from_prev", distance_from_prev).cast(pl.Float64),
            pl.Series("duration_from_prev", duration_from_prev).cast(pl.Float64),
            pl.Series("cumulative_distance", cumulative_distance).cast(pl.Float64),
            pl.Series("cumulative_duration", cumulative_duration).cast(pl.Float64),
        ])

        # 4. Totaux
        df_enriched = df_enriched.with_columns([
            pl.lit(float(cumulative_distance[-1])).alias("day_total_distance"),
            pl.lit(float(cumulative_duration[-1])).alias("day_total_duration"),
        ])

        return df_enriched
    # ---------------------------------------------------------
    # PIPELINE COMPLET
    # ---------------------------------------------------------
    def run(
        self,
        commune,
        main_categories,
        sub_categories,
        min_score,
        nb_days,
        anchor_lat,
        anchor_lon,
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
        filtered_lf = self._filter_pois(commune, main_categories, sub_categories, min_score)

        # 2. Clustering : df_prepared (jour, etc.)
        # df_prepared = self._cluster_pois(filtered_lf, nb_days, anchor_lat, anchor_lon)
        df_prepared = (
        self._cluster_pois(filtered_lf, nb_days, anchor_lat, anchor_lon)
        .collect()
        )

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
        optimizer, df_itinerary = self._compute_itinerary(df_clustered, df_osrm_dur) # TSP NN2O
        
        # df_itinerary = self._compute_itinerary_ga(
        #     df_clustered=df_clustered,
        #     df_osrm_dur=df_osrm_dur,
        #     df_osrm_dist=df_osrm_dist
        # )# GA algorithm
        
        
        # ENRICHISSEMENT PAR JOUR
        df_enriched_days = []

        for day in df_itinerary["cluster_id"].unique():
            df_day = df_itinerary.filter(pl.col("cluster_id") == day)

            order = df_day["order"].to_list()
            matrix_dur = df_osrm_dur.to_numpy()
            matrix_dist = df_osrm_dist.to_numpy()

            df_enriched = self.enrich_itinerary(
                df_day=df_day,
                matrix_durations=matrix_dur,
                matrix_distances=matrix_dist,
                order=order
            )

            df_enriched_days.append(df_enriched)

        df_itinerary = pl.concat(df_enriched_days)

        return df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer # pour TSP NN2O
        #return df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary # pour TSP avec GA