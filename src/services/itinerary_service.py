from pathlib import Path
import polars as pl

from pipeline.itinerary_pipeline import ItineraryPipeline
from features.osrm import OSRMClientAsync


class ItineraryService:

    def __init__(self, pois_path: Path, osrm_url: str):
        self.pipeline = ItineraryPipeline(pois_path)
        self.osrm = OSRMClientAsync(osrm_url)

    def compute(self, req):

        # Appel pipeline avec solveur paramétrable
        df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer = (
            self.pipeline.run(
                commune=req.commune,
                main_categories=req.main_categories,
                sub_categories=req.sub_categories,
                min_score=req.min_score,
                nb_days=req.nb_days,
                anchor_lat=req.start.lat,
                anchor_lon=req.start.lon,
                osrm=self.osrm,
                osrm_mode=req.osrm_mode,
                solver=req.solver,        # <--- AJOUT ICI
            )
        )

        # Formatage final pour l’API
        result = []

        for cluster_id in df_itinerary["cluster_id"].unique():
            df_day = df_itinerary.filter(pl.col("cluster_id") == cluster_id)

            result.append({
                "day": int(cluster_id),
                "optimizer": optimizer,  # <--- OPTIONNEL MAIS UTILE
                "pois": df_day.to_dicts(),
                "total_distance_km": float(df_day["day_total_distance"][0]),
                "total_duration_min": float(df_day["day_total_duration"][0] / 60),
            })

        return result