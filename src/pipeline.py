# pipeline.py

import time
from pathlib import Path
from typing import Dict

import polars as pl

from features.poi_filter import POIFilter
from features.spatial_clustering import SpatialClusterer
from features.poi_balancer import POIBalancer
from features.osrm import OSRMClient
from features.tsp_solver import TSPSolver


DEFAULT_VISIT_TIME = 45 * 60  # 45 minutes en secondes


class ItineraryPipeline:
    """
    Pipeline complet :
        1. Filtrage
        2. Clustering spatial
        3. Balancing
        4. OSRM + TSP par jour
        5. Enrichissement (ordre, distances, durées, cumul, matin/après-midi)
        6. Assemblage multi-jour
    """

    def __init__(self, pois_path: Path):
        self.pois_path = pois_path
        self.pois_lf = pl.scan_parquet(self.pois_path)

    # ---------------------------------------------------------
    # ENRICHISSEMENT POUR LE FRONT
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
    # CLUSTERING
    # ---------------------------------------------------------

    def _cluster_pois(self, filtered_lf, nb_days, anchor_lat, anchor_lon):
        return (
            SpatialClusterer(filtered_lf)
            .set_nb_days(nb_days)
            .set_anchor(anchor_lat, anchor_lon)
            .apply()
        )

    # ---------------------------------------------------------
    # BALANCING
    # ---------------------------------------------------------

    def _balance_pois(self, clustered_lf, nb_days, mode):
        return (
            POIBalancer(clustered_lf)
            .set_mode(mode)
            .set_nb_days(nb_days)
            .apply()
        )

    # ---------------------------------------------------------
    # OSRM + TSP + ENRICHISSEMENT
    # ---------------------------------------------------------
    
    def choose_lunch_poi(df_day: pl.DataFrame) -> pl.DataFrame:
        """
        Première version naïve :
        - filtre les restaurants
        - prend le premier (par score, ou au hasard, ou par proximité plus tard)
        """
        if "main_category" not in df_day.columns:
            return pl.DataFrame()

        df_restos = df_day.filter(pl.col("main_category") == "restaurant")

        if df_restos.height == 0:
            return pl.DataFrame()

        # Ici on prend juste le premier, tu pourras raffiner plus tard
        return df_restos.head(1)


    def _build_itineraries(self, balanced_lf, nb_days, anchor_lat, anchor_lon, start_time="09:00"):
        client = OSRMClient()
        itineraries = {}

        # Convertir start_time en minutes
        h, m = map(int, start_time.split(":"))
        start_minutes = h * 60 + m

        for day in range(nb_days):
            df_day = balanced_lf.filter(pl.col("day") == day).collect()

            if df_day.height == 0:
                itineraries[f"day_{day + 1}"] = pl.DataFrame()
                continue

            # --- 1. Coordonnées ---
            coords = [(anchor_lat, anchor_lon)] + list(
                zip(df_day["latitude"].to_list(), df_day["longitude"].to_list())
            )

            # --- 2. OSRM ---
            data = client.table(coords, annotations="duration,distance")
            matrix_durations = data["durations"]
            matrix_distances = data["distances"]

            # --- 3. TSP ---
            solver = TSPSolver(matrix_durations)
            order = solver.solve()  # inclut l’ancre (index 0)

            # --- 4. Réordonner les POIs ---
            poi_order = [i - 1 for i in order if i != 0]
            df_ordered = df_day[poi_order]  # 100% compatible toutes versions Polars

            # --- 5. Calcul segment + cumul ---
            segment_durations = []
            segment_distances = []
            cumulative_durations = []
            cumulative_distances = []

            total_d = 0
            total_s = 0

            for i in range(1, len(order)):
                prev_idx = order[i - 1]
                curr_idx = order[i]

                d = matrix_durations[prev_idx][curr_idx]
                s = matrix_distances[prev_idx][curr_idx]

                segment_durations.append(d)
                segment_distances.append(s)

                total_d += d
                total_s += s

                cumulative_durations.append(total_d)
                cumulative_distances.append(total_s)

            # --- 6. Arrival times ---
            arrival_times = [start_minutes + cum for cum in cumulative_durations]
            arrival_times_fmt = []
            for t in arrival_times:
                minutes = int(round(t))
                h = minutes // 60
                m = minutes % 60
                arrival_times_fmt.append(f"{h:02d}:{m:02d}")
               

            # --- 7. Ajout des colonnes ---
            df_ordered = df_ordered.with_columns([
                pl.Series("visit_order", list(range(1, len(poi_order) + 1))),
                pl.Series("segment_duration", segment_durations),
                pl.Series("segment_distance", segment_distances),
                pl.Series("cumulative_duration", cumulative_durations),
                pl.Series("cumulative_distance", cumulative_distances),
                pl.Series("arrival_time", arrival_times_fmt),
                pl.lit(total_d).alias("total_duration"),
                pl.lit(total_s).alias("total_distance"),
            ])

            itineraries[f"day_{day + 1}"] = df_ordered

        return itineraries



    # ---------------------------------------------------------
    # PIPELINE PUBLIC
    # ---------------------------------------------------------

    def run(
        self,
        commune,
        main_categories,
        min_score,
        nb_days,
        mode,
        anchor_lat,
        anchor_lon,
        verbose=True,
    ):
        start_total = time.perf_counter()

        if verbose:
            print("=== Étape 1 : Filtrage ===")
        filtered_lf = self._filter_pois(commune, main_categories, min_score)

        if verbose:
            print("=== Étape 2 : Clustering ===")
        clustered_lf = self._cluster_pois(filtered_lf, nb_days, anchor_lat, anchor_lon)

        if verbose:
            print("=== Étape 3 : Balancing ===")
        balanced_lf = self._balance_pois(clustered_lf, nb_days, mode)

        if verbose:
            print("=== Étape 4 : OSRM + TSP + Enrichissement ===")
        itineraries = self._build_itineraries(
            balanced_lf, nb_days, anchor_lat, anchor_lon
        )

        end_total = time.perf_counter()
        if verbose:
            print(f"\n=== Temps total : {end_total - start_total:.2f} sec ===")

        return itineraries

# ---------------------------------------------------------
# Exemple d'utilisation standalone (équivalent à ton main.py)
# ---------------------------------------------------------

if __name__ == "__main__":
    DATA_DIR = Path("../data")
    POIS_PATH = DATA_DIR / "processed" / "merged_20260101_234939.parquet"

    pipeline = ItineraryPipeline(POIS_PATH)

    commune = "Paris"
    main_categories =             main_categories=[
                "Culture & Musées",
                "Patrimoine & Monuments",
                "Gastronomie & Restauration",
                "Bien-être & Santé", 
                "Hébergement",
                "Sports & Loisirs",
                "Santé & Urgences",
                "Famille & Enfants",
                "Culture & Musées",
                "Transports",
                "Shopping & Artisanat",
                "Nature & Paysages",
                "Patrimoine & Monuments",
                "Services & Mobilité",
                "Commerce & Shopping",
                "Camping & Plein Air",
                "Commodités",
                "Transports touristiques",
                "Loisirs & Clubs",
                "Événements & Traditions",
                "Information Touristique"
            ],
    min_score = 0.9
    nb_days = 3
    mode = "walk"
    anchor_lat = 48.8566
    anchor_lon = 2.3522

    itineraries = pipeline.run(
        commune=commune,
        main_categories=main_categories,
        min_score=min_score,
        nb_days=nb_days,
        mode=mode,
        anchor_lat=anchor_lat,
        anchor_lon=anchor_lon,
        verbose=True,
    )

    for day, df in itineraries.items():
        print(f"\n--- {day} ---")
        print(df)