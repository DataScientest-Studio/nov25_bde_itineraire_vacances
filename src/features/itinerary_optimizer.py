from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Sequence, Literal, Optional
import polars as pl
import math
import numpy as np
import asyncio

from src.features.osrm import OSRMClientAsync


@dataclass
class ItineraryOptimizer:
    """
    Optimise les itinéraires par jour à partir d'une matrice OSRM.
    - travaille par 'day'
    - utilise osrm_index pour mapper les lignes/colonnes de la matrice aux POIs
    - heuristique : nearest neighbor + 2-opt
    """
    df_pois: pl.DataFrame                     # df_clustered
    dist_matrix: np.ndarray                   # matrice distances/durations NxN
    metric: Literal["distance", "duration"] = "duration"

    @classmethod
    def from_list_matrix(
        cls,
        df_pois: pl.DataFrame,
        matrix: Sequence[Sequence[float]],
        metric: Literal["distance", "duration"] = "duration",
    ) -> "ItineraryOptimizer":
        return cls(
            df_pois=df_pois,
            dist_matrix=np.array(matrix, dtype=float),
            metric=metric,
        )

    # ---------- Heuristique TSP : nearest neighbor ----------

    def _nearest_neighbor(self, indices: List[int], start_index: Optional[int] = None) -> List[int]:
        """
        Construit un tour initial avec nearest neighbor.
        'indices' sont des osrm_index (subset pour un jour donné).
        """
        if not indices:
            return []

        remaining = set(indices)

        if start_index is None:
            current = indices[0]
        else:
            current = start_index
            if current not in remaining:
                remaining.add(current)

        tour = [current]
        remaining.remove(current)

        while remaining:
            best_next = None
            best_cost = math.inf
            for j in remaining:
                cost = self.dist_matrix[current, j]
                if cost < best_cost:
                    best_cost = cost
                    best_next = j
            tour.append(best_next)
            remaining.remove(best_next)
            current = best_next

        return tour

    # ---------- Amélioration : 2-opt ----------

    def _tour_cost(self, tour: List[int]) -> float:
        if len(tour) < 2:
            return 0.0
        total = 0.0
        for i in range(len(tour) - 1):
            total += self.dist_matrix[tour[i], tour[i + 1]]
        return total

    def _two_opt(self, tour: List[int], max_iters: int = 50) -> List[int]:
        """
        2-opt simple : essaie d'améliorer le tour en supprimant les croisements.
        """
        if len(tour) < 4:
            return tour

        best = tour[:]
        best_cost = self._tour_cost(best)
        improved = True
        iter_count = 0

        while improved and iter_count < max_iters:
            improved = False
            iter_count += 1
            for i in range(1, len(best) - 2):
                for k in range(i + 1, len(best) - 1):
                    new_tour = best[:]
                    new_tour[i:k+1] = reversed(best[i:k+1])
                    new_cost = self._tour_cost(new_tour)
                    if new_cost < best_cost:
                        best = new_tour
                        best_cost = new_cost
                        improved = True
            # si aucune amélioration sur cette itération, on sort

        return best

    # ---------- Solveur par jour ----------

    def solve_day(
        self,
        day: int | str,
        start_poi_id: Optional[int] = None,
    ) -> pl.DataFrame:
        """
        Calcule un ordre de visite optimisé pour un 'day' donné.
        Retourne un DataFrame avec l'ordre, le coût cumulé, etc.
        """
        df_day = self.df_pois.filter(pl.col("cluster_id") == day)

        if df_day.height == 0:
            return df_day.with_columns(
                pl.lit(None).alias("visit_order"),
                pl.lit(None).alias("cum_cost"),
            )

        # osrm_index pour ce jour
        osrm_indices = df_day.select("osrm_index").to_series().to_list()

        # déterminer le osrm_index de départ si start_poi_id fourni
        start_osrm = None
        if start_poi_id is not None:
            matched = df_day.filter(pl.col("poi_id") == start_poi_id)
            if matched.height > 0:
                start_osrm = matched.select("osrm_index").item()

        # 1. nearest neighbor
        nn_tour = self._nearest_neighbor(osrm_indices, start_index=start_osrm)

        # 2. 2-opt
        tour = self._two_opt(nn_tour)

        # construire un mapping osrm_index -> ordre
        osrm_to_order = {idx: order for order, idx in enumerate(tour)}

        # coût cumulé le longitudeg du tour
        cum_cost = [0.0]
        for i in range(len(tour) - 1):
            step_cost = self.dist_matrix[tour[i], tour[i + 1]]
            cum_cost.append(cum_cost[-1] + step_cost)

        osrm_to_cumcost = {idx: c for idx, c in zip(tour, cum_cost)}

        # enrichir df_day
        df_day = df_day.with_columns([
            pl.col("osrm_index").map_elements(
                lambda x: osrm_to_order.get(x, None),
                return_dtype=pl.Int64
            ).alias("visit_order"),
            pl.col("osrm_index").map_elements(
                lambda x: osrm_to_cumcost.get(x, None),
                return_dtype=pl.Float64
            ).alias("cum_cost"),
        ])

        return df_day.sort("visit_order")

    # ---------- Solveur pour tous les jours ----------

    def solve_all_days(self) -> pl.DataFrame:
        """
        Applique l’optimisation à tous les jours présents dans df_pois.
        Retourne un DataFrame concaténé avec l'ordre par day.
        """
        days = self.df_pois.select("cluster_id").unique().to_series().to_list()

        solved_list = []
        for d in days:
            solved_day = self.solve_day(d)
            solved_list.append(solved_day)

        return pl.concat(solved_list).sort(["cluster_id", "visit_order"])

    # ---------- - Calcule l’itinéraire optimisé pour ce jour ---
    
    async def build_geojson_for_day_async(self, day, osrm: OSRMClientAsync):
        """
        Génère la route OSRM (GeoJSON) pour un jour donné, en mode async.
        """
        df_day = self.solve_day(day)
        return await build_day_route_geojson_async(df_day, osrm)



    async def build_geojson_all_days_async(self, df_itinerary: pl.DataFrame, osrm: OSRMClientAsync):
        """
        Génère les routes OSRM (GeoJSON) pour tous les jours.
        Retourne un dict {day: GeoJSON}.
        """

        days = (
            df_itinerary
            .select("cluster_id")
            .unique()
            .to_series()
            .to_list()
        )

        tasks = {}

        for day in days:
            df_day = df_itinerary.filter(pl.col("cluster_id") == day)
            tasks[day] = self.build_day_route_geojson_async(df_day, osrm)

        results = await asyncio.gather(*tasks.values())

        return {day: geo for day, geo in zip(tasks.keys(), results)}


    async def build_day_route_geojson_async(self, df_day: pl.DataFrame, osrm: OSRMClientAsync):
        df_day = df_day.sort("visit_order")

        coords = df_day.select(["latitude", "longitude"]).to_numpy().tolist()
        coords = [tuple(row) for row in coords]

        tasks = []
        for i in range(len(coords) - 1):
            start = coords[i]
            end = coords[i + 1]
            tasks.append(osrm.route_geojson(start, end))

        segments = await asyncio.gather(*tasks)

        full_coords = []
        for i, seg in enumerate(segments):
            if i == 0:
                full_coords.extend(seg["coordinates"])
            else:
                full_coords.extend(seg["coordinates"][1:])

        return {
            "type": "LineString",
            "coordinates": full_coords
        }

