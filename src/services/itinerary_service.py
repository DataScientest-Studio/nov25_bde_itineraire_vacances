from pathlib import Path

import polars as pl

from features.osrm import OSRMClientAsync
from pipeline.itinerary_pipeline import ItineraryPipeline


class ItineraryService:
    def __init__(self, osrm_url: str):
        self.osrm = OSRMClientAsync(osrm_url)
        self.pipeline = ItineraryPipeline()

    def compute(self, req, pois_df: pl.DataFrame):
        """
        Compute itinerary from:
        - request parameters
        - POIs already filtered from the database
        """

        # 1. CLUSTERING
        df_clustered = self.pipeline.cluster_pois(
            pois_df=pois_df,
            nb_days=req.nb_days,
            anchor_lat=req.start.lat,
            anchor_lon=req.start.lon,
        )

        # 2. OSRM MATRICES
        df_osrm_dist, df_osrm_dur = self.pipeline.compute_osrm_matrices(
            df_clustered=df_clustered, osrm=self.osrm, transport_mode=req.transport_mode
        )

        # 3. SOLVEUR (NN2O / GA / AUTO)
        df_itinerary, optimizer = self.pipeline.solve_itinerary(
            df_clustered=df_clustered,
            df_osrm_dist=df_osrm_dist,
            df_osrm_dur=df_osrm_dur,
            solver=req.solver,
        )

        # 4. ENRICHISSEMENT FINAL
        df_itinerary = self.pipeline.enrich_itinerary(df_itinerary)

        # 5. FORMATAGE API
        result = []
        for cluster_id in df_itinerary["cluster_id"].unique():
            df_day = df_itinerary.filter(pl.col("cluster_id") == cluster_id)

            result.append(
                {
                    "day": int(cluster_id),
                    "pois": df_day.to_dicts(),
                    "total_distance_km": float(df_day["day_total_distance"][0]),
                    "total_duration_min": float(df_day["day_total_duration"][0] / 60),
                }
            )

        trip_total_distance = sum(day["total_distance_km"] for day in result)
        trip_total_duration = sum(day["total_duration_min"] for day in result)

        return {
            "itinerary": result,
            "trip_total_distance_km": trip_total_distance,
            "trip_total_duration_min": trip_total_duration,
            "optimizer": optimizer,
        }
