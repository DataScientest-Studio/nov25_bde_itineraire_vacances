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

    def __init__(self, osrm_client: OSRMClientAsync):
        self.osrm = osrm_client
        self.pipeline = ItineraryPipeline()

    def debug_step(self, df, step_name):
        logger.info(f"=== {step_name} ===")
        logger.info(f"Total POIs : {df.shape[0]}")

    async def compute_itinerary(
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

        # 0. META
        meta = {
            poi.poi_id: {
                "nom_du_poi": poi.nom_du_poi,
                "description": poi.description,
                "adresse": poi.adresse,
                "contact_phone": poi.contact_phone,
                "contact_mail": poi.contact_mail,
                "contact_website": poi.contact_website,
                "itineraire": poi.itineraire,
                "h3_r7": poi.h3_r7,
                "diversity_commune_norm": poi.diversity_commune_norm,
            }
            for poi in pois
        }

        # 1. DataFrame Polars
        pois_df = pl.DataFrame(
            [
                {
                    "poi_id": poi.poi_id,
                    "nom_du_poi": poi.nom_du_poi,
                    "latitude": poi.latitude,
                    "longitude": poi.longitude,
                    "main_category": poi.main_category,
                    "sub_category": poi.sub_category,
                    "h3_r7": poi.h3_r7,
                    "diversity_commune_norm": poi.diversity_commune_norm,
                    "final_score": poi.final_score,
                }
                for poi in pois
            ]
        )

        self.debug_step(pois_df, "1. Chargement initial")

        # 2. Pipeline complet
        df_clustered, df_osrm_dist, df_osrm_dur, df_itinerary, optimizer = (
            await self.pipeline.run_from_pois_df(
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

        # 4. Formatage final
        result_days: List[Dict[str, Any]] = []

        for cluster_id in df_itinerary["cluster_id"].unique():
            df_day = df_itinerary.filter(pl.col("cluster_id") == cluster_id).sort("order")

            pois_for_day: List[Dict[str, Any]] = []
            day_total_distance_km = float(df_day["day_total_distance_km"][0])
            day_total_duration_min = float(df_day["day_total_duration_min"][0])

            # appel OSRM pour la géométrie complète du jour
            coords_day = [
                (row["longitude"], row["latitude"])
                for row in df_day.to_dicts()
            ]

            osrm_route = await self.osrm.route_full(coords_day, profile=transport_mode)

            for row in df_day.to_dicts():
                m = meta.get(row["poi_id"], {})

                poi_payload = {
                    # OSRM / pipeline fields
                    "osrm_index": row["osrm_index"],
                    "cluster_id": row["cluster_id"],
                    "poi_id": row["poi_id"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                    "main_category": row["main_category"],
                    "sub_category": row.get("sub_category"),
                    "final_score": row["final_score"],
                    "order": row["order"],
                    "solver_used": row["solver_used"],
                    "distance_from_prev_km": row["distance_from_prev_km"],
                    "duration_from_prev_min": row["duration_from_prev_min"],
                    "cumulative_distance_km": row["cumulative_distance_km"],
                    "cumulative_duration_min": row["cumulative_duration_min"],
                    "day_total_distance_km": row["day_total_distance_km"],
                    "day_total_duration_min": row["day_total_duration_min"],
                    # META fields
                    "nom_du_poi": m.get("nom_du_poi"),
                    "description": m.get("description"),
                    "adresse": m.get("adresse"),
                    "contact_phone": m.get("contact_phone"),
                    "contact_mail": m.get("contact_mail"),
                    "contact_website": m.get("contact_website"),
                    "itineraire": m.get("itineraire"),
                }

                pois_for_day.append(poi_payload)

            result_days.append(
                {
                    "day": int(cluster_id),
                    "pois": pois_for_day,
                    "total_distance_km": day_total_distance_km,
                    "total_duration_min": day_total_duration_min,
                    "geometry": osrm_route["geometry"],
                }
            )

        trip_total_distance = sum(day["total_distance_km"] for day in result_days)
        trip_total_duration = sum(day["total_duration_min"] for day in result_days)

        return {
            "itinerary": result_days,
            "trip_total_distance_km": trip_total_distance,
            "trip_total_duration_min": trip_total_duration,
            "optimizer": optimizer,
        }