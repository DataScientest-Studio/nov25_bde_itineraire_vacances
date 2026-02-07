import polars as pl
from typing import List, Dict, Any
import logging
from app.pipeline.features.osrm import OSRMClientAsync
from app.pipeline.itinerary_pipeline import ItineraryPipeline

logger = logging.getLogger("uvicorn")


class ItineraryService:
    """
    Service orchestrant :
    - conversion des POIs en DataFrame
    - appel du pipeline (clustering, OSRM, solveur, enrichissement)
    - formatage final pour l'API
    """

    def __init__(self, osrm_url: str):
        self.osrm = OSRMClientAsync(osrm_url)
        self.pipeline = ItineraryPipeline()
    
    def debug_step(self,df, step_name):
        logger.info(f"=== {step_name} ===")
        logger.info(f"Total POIs : {df.shape[0]}")

    def compute_itinerary(
        self,
        pois: List[Dict[str, Any]],
        days: int,
        transport_mode: str,
        solver: str,
        start_lat: float,
        start_lon: float,
    ):
        """
        Compute itinerary from:
        - list of POIs (déjà filtrés)
        - request parameters
        - start point (lat/lon) fourni par l’utilisateur
        """

        # 1. Convertir la liste de POIs en DataFrame Polars
        pois_df = pl.DataFrame(pois)
        self.debug_step(pois_df, "1. Chargement initial")


        # 2. Pipeline complet
        df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer = (
            self.pipeline.run_from_pois_df(
                pois_df=pois_df,
                nb_days=days,
                anchor_lat=start_lat,  
                anchor_lon=start_lon,  
                osrm=self.osrm,
                transport_mode=transport_mode,
                solver=solver,
            )
        )
        self.debug_step(df_clustered, "2. Après clustering")
        self.debug_step(df_osrm_dist, "3. Après OSRM distance")
        self.debug_step(df_osrm_dur, "4. Après OSRM durée")
        self.debug_step(df_itinerary, "5. Après solveur")

        # 3. Aucun itinéraire trouvé
        if df_itinerary.is_empty():
            return {
                "itinerary": [],
                "trip_total_distance_km": 0.0,
                "trip_total_duration_min": 0.0,
                "optimizer": optimizer,
            }
        
        self.debug_step(df_itinerary, "6. Formatage final")

        # 4. Formatage final pour l’API
        result = []
        for cluster_id in df_itinerary["cluster_id"].unique():
            df_day = df_itinerary.filter(pl.col("cluster_id") == cluster_id)

            result.append(
                {
                    "day": int(cluster_id),
                    "pois": df_day.to_dicts(),
                    "total_distance_km": float(df_day["day_total_distance_km"][0]),
                    "total_duration_min": float(df_day["day_total_duration_min"][0]),
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
