import time
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
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
        3. Préparation OSRM
        4. OSRM matrices
        5. Solveur (NN2O / GA / AUTO)
        6. Enrichissement
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
    # PRÉPARATION OSRM
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
    # OSRM MATRICES
    # ---------------------------------------------------------
    def _get_osrm_profile(self, transport_mode: str) -> str:
        if transport_mode == "walk":
            return "foot"
        if transport_mode == "bike":
            return "bike"
        if transport_mode == "car":
            return "driving"
        return "foot"

    def _compute_osrm_matrices(
        self,
        df_clustered: pl.DataFrame,
        osrm: OSRMClientAsync,
        transport_mode: str,
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:

        profile = self._get_osrm_profile(transport_mode)

        df_clustered, df_osrm_dist, df_osrm_dur = asyncio.run(
            build_osrm_matrices_async(df_clustered, osrm, profile=profile)
        )
        return df_clustered, df_osrm_dist, df_osrm_dur

    # ---------------------------------------------------------
    # SOLVEUR NN2O
    # ---------------------------------------------------------
    def _compute_itinerary_nn2o(
        self,
        df_clustered: pl.DataFrame,
        df_osrm_dur: pl.DataFrame,
    ) -> Tuple[str, pl.DataFrame]:

        # NN2O attend un DataFrame POLARS complet avec cluster_id
        optimizer = ItineraryOptimizer.from_list_matrix(
            df_pois=df_clustered,                 
            matrix=df_osrm_dur.to_numpy(),        
            metric="duration",
        )

        df_itinerary = optimizer.solve_all_days()  

        return "nn2o", df_itinerary
    # ---------------------------------------------------------
    # SOLVEUR GA (sans enrichissement)
    # ---------------------------------------------------------

    def _compute_itinerary_ga(
        self,
        df_clustered: pl.DataFrame,
        df_osrm_dur: pl.DataFrame,
    ) -> Tuple[str, pl.DataFrame]:

        results = []

        for cluster_id in df_clustered["cluster_id"].unique():
            df_day = df_clustered.filter(pl.col("cluster_id") == cluster_id)

            # Pas d’itinéraire possible avec < 2 POIs
            if df_day.height < 2:
                continue

            # 1. Indices OSRM globaux du cluster
            indices = df_day["osrm_index"].to_list()

            # 2. Matrice OSRM locale (réduite au cluster)
            global_matrix = df_osrm_dur.to_numpy()
            local_matrix = global_matrix[np.ix_(indices, indices)]

            # 3. GA travaille sur la matrice locale
            df_day_pd = df_day.to_pandas()

            ga = GeneticAlgo(
                poi_df=df_day_pd,
                duration_matrix=local_matrix
            )
            ga.setup_toolbox(itin_min_poi=5, itin_max_poi=15)

            best_route_local, fitness = ga.run_ga(
                pop_size=50,
                ngen=50,
                cxpb=0.75,
                mutpb=0.3
            )

            # Si GA n’a rien trouvé → on ignore ce cluster
            if len(best_route_local) == 0:
                continue

            # 4. Remapping indices locaux → indices OSRM globaux
            try:
                best_route_global = [indices[i] for i in best_route_local]
            except IndexError:
                print("IndexError: best_route_local contient un indice hors limites")
                print("indices:", indices)
                print("best_route_local:", best_route_local)
                continue

            # 5. Filtrer les POIs du cluster
            df_route = df_day.filter(pl.col("osrm_index").is_in(best_route_global))

            if df_route.is_empty():
                continue

            # 6. Trier df_route dans l’ordre de best_route_global
            order_map = {v: i for i, v in enumerate(best_route_global)}
            df_route = df_route.sort(
                pl.col("osrm_index").replace(order_map)
            )

            # 7. Ajouter une colonne order cohérente (0..n-1)
            df_route = df_route.with_columns(
                pl.Series("order", list(range(df_route.height)))
            )

            # 8. Réinjecter cluster_id
            df_route = df_route.with_columns(
                pl.lit(cluster_id).alias("cluster_id")
            )

            results.append(df_route)

        # Aucun cluster n’a produit d’itinéraire
        if len(results) == 0:
            return "ga", pl.DataFrame({"cluster_id": []})

        # Concaténation finale
        df_itinerary = pl.concat(results)
        return "ga", df_itinerary

    # ---------------------------------------------------------
    # ENRICHISSEMENT
    # ---------------------------------------------------------
    def enrich_itinerary(self, df_day, matrix_durations, matrix_distances, order):
        n = len(order)

        distance_from_prev = [0.0]
        duration_from_prev = [0.0]

        for i in range(n - 1):
            d = float(matrix_distances[order[i], order[i + 1]])
            t = float(matrix_durations[order[i], order[i + 1]])
            distance_from_prev.append(d)
            duration_from_prev.append(t)

        cumulative_distance = []
        cumulative_duration = []

        cum_d = 0.0
        cum_t = 0.0

        for d, t in zip(distance_from_prev, duration_from_prev):
            cum_d += d
            cum_t += t
            cumulative_distance.append(cum_d)
            cumulative_duration.append(cum_t)

        df_enriched = df_day.with_columns([
            pl.Series("distance_from_prev", distance_from_prev).cast(pl.Float64),
            pl.Series("duration_from_prev", duration_from_prev).cast(pl.Float64),
            pl.Series("cumulative_distance", cumulative_distance).cast(pl.Float64),
            pl.Series("cumulative_duration", cumulative_duration).cast(pl.Float64),
        ])

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
        transport_mode: str = "walk",
        solver: str = "nn2o",   # <--- AJOUT ICI
        max_pois_per_cluster: int = 40,
        osrm_min_score: float = 0.2,
        target_restaurants: int = 2,
        restaurant_category: str = "Gastronomie & Restauration",
    ):

        # 1. Filtrage
        filtered_lf = self._filter_pois(commune, main_categories, sub_categories, min_score)

        # 2. Clustering
        df_prepared = (
            self._cluster_pois(filtered_lf, nb_days, anchor_lat, anchor_lon)
            .collect()
        )

        # 3. Préparation OSRM
        df_clustered = self._build_osrm_ready_pois(
            df_prepared=df_prepared,
            mode=transport_mode,
            max_pois_per_cluster=max_pois_per_cluster,
            min_score=osrm_min_score,
            target_restaurants=target_restaurants,
            restaurant_category=restaurant_category,
        )

        # 4. OSRM matrices
        df_clustered, df_osrm_dist, df_osrm_dur = self._compute_osrm_matrices(
            df_clustered=df_clustered,
            osrm=osrm,
            transport_mode=transport_mode,
        )
        print("DIST MATRIX SHAPE:", df_osrm_dist.shape)
        print("DUR MATRIX SHAPE:", df_osrm_dur.shape)
        print("DIST SAMPLE:", df_osrm_dist.to_numpy()[0][:10])
        print("DUR SAMPLE:", df_osrm_dur.to_numpy()[0][:10])

        print(df_clustered.height)
        print(df_clustered.head())

        # 5. Solveur
        if solver == "nn2o":
            optimizer, df_itinerary = self._compute_itinerary_nn2o(df_clustered, df_osrm_dur)

        elif solver == "ga":
            optimizer, df_itinerary = self._compute_itinerary_ga(df_clustered, df_osrm_dur)

        elif solver == "auto":
            # GA → fallback NN2O
            optimizer, df_itinerary = self._compute_itinerary_ga(df_clustered, df_osrm_dur)
            if df_itinerary.is_empty():
                optimizer, df_itinerary = self._compute_itinerary_nn2o(df_clustered, df_osrm_dur)
                optimizer = "nn2o_auto_fallback"
            else:
                optimizer = "ga_auto"
        else:
            raise ValueError(f"Solver inconnu : {solver}")


        if df_itinerary.is_empty() or "cluster_id" not in df_itinerary.columns:
            print("⚠️ df_itinerary vide ou sans cluster_id")
            return df_clustered, df_osrm_dist, df_osrm_dur, pl.DataFrame(), optimizer

        # 6. Enrichissement (une seule fois)
        df_enriched_days = []

        for day in df_itinerary["cluster_id"].unique():
            df_day = df_itinerary.filter(pl.col("cluster_id") == day)

            # 1. Toujours trier par osrm_index
            df_day = df_day.sort("osrm_index")

            # 2. L’ordre doit être la liste des osrm_index
            order = df_day["osrm_index"].to_list()

            # 3. Matrices OSRM
            matrix_dur = df_osrm_dur.to_numpy()
            matrix_dist = df_osrm_dist.to_numpy()

            # 4. Enrichissement correct
            df_enriched = self.enrich_itinerary(
                df_day=df_day,
                matrix_durations=matrix_dur,
                matrix_distances=matrix_dist,
                order=order
            )

            df_enriched_days.append(df_enriched)

        df_itinerary = pl.concat(df_enriched_days)

        return df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer