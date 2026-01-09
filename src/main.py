import time
from pathlib import Path
import polars as pl
import asyncio

from src.features.poi_filter import POIFilter
from src.features.spatial_clustering import SpatialClusterer
from src.features.poi_selector import POISelector
from src.features.post_clustering import build_osrm_matrices_async
#from src.features.itinerary_optimizer import ItineraryOptimizer
from src.features.osrm import OSRMClientAsync
from src.data.etl.save import save_parquet


import pydeck as pdk


PROJECT_ROOT = Path(__file__).resolve().parent.parent
POIS_PATH = PROJECT_ROOT / "data" / "processed" / "merged_20260108_174125.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed"


MAIN_CATEGORY = ["Culture & Musées","Patrimoine & Monuments",
                "Gastronomie & Restauration","Bien-être & Santé", "Hébergement","Sports & Loisirs","Santé & Urgences",
                "Famille & Enfants","Culture & Musées","Transports",
                "Shopping & Artisanat","Nature & Paysages","Patrimoine & Monuments",
                "Services & Mobilité","Commerce & Shopping","Camping & Plein Air","Commodités",
                "Transports touristiques","Loisirs & Clubs","Événements & Traditions","Information Touristique"
            ]


def main():
    start_total = time.perf_counter()

    ###############################
    # Filtrage des POIs
    ###############################
    print(POIS_PATH)
    print(POIS_PATH.exists())

    pois_lf= pl.scan_parquet(POIS_PATH)
    print(len(pois_lf.collect()))

    filter_pois = POIFilter(pois_lf)
    filtered_pois = (
        filter_pois
        .set_commune("Paris")
        .set_categories(
            main_categories=MAIN_CATEGORY
        )
        .set_min_score(0.3)
        .apply()
    )
    print(len(filtered_pois.collect()))

    ###############################
    # Clustering des POIs
    ###############################
    user_nb_days = 1
    anchor_lat = 48.8566
    anchor_lon = 2.3522

    clustered = (
        SpatialClusterer(filtered_pois)  # LazyFrame avec score_final
        .set_nb_days(user_nb_days)
        .set_anchor(anchor_lat, anchor_lon)
        .apply()
    )

    df_clustered = (
        clustered
        .select([
            "poi_id",
            "main_category",
            "sub_category",
            "longitude",
            "latitude",
            "final_score",
            "h3_r8",
            "cluster_id",
            "diversity_commune_norm",
            "itineraire"
        ])
        .collect()
    )
    save_parquet(df_clustered, OUTPUT_PATH)
    
    ###############################
    # POIs Selection
    ###############################
    selector = POISelector(
        transport_mode="walk",
        min_restaurants_per_cluster=2,
        max_pois_per_subcategorie=3,
        w_final_score=0.7,
        w_diversity=0.3,
        diversity_col="diversity_subcat_norm",
    )

    lf_osrm_ready = selector.select(df_clustered)

    save_parquet(lf_osrm_ready, OUTPUT_PATH / "lf_osrm_ready")

    #print(selector.profiling)  # Durées cumulées par étape

    """
    # 2. Harmonisation des colonnes
    df_prepared = (
        df_clustered
        .rename({
            "main_category": "main_category",
            "day": "cluster_id",
        })
    )
    # 3. Post-clustering simplifié
    df_clustered = build_osrm_ready_pois(
        df=df_prepared,
        mode="walk",
        max_pois_per_cluster=40,
        min_score=0.2,
        target_restaurants=2,
        restaurant_category="Gastronomie & Restauration",
    )

    """
    ################################
    # ITINERARY OPTIMIZATION
    ################################
    
    # 4. construction des matrices
    osrm = OSRMClientAsync()
    df_clustered, df_osrm_dist, df_osrm_dur = asyncio.run(build_osrm_matrices_async(lf_osrm_ready, osrm))

    print(df_clustered)
    print(df_osrm_dist)

    print(df_osrm_dur)

    save_parquet(df_osrm_dur, OUTPUT_PATH / "df_osrm_dur")
    save_parquet(df_osrm_dist, OUTPUT_PATH / "df_osrm_dist")
    save_parquet(df_clustered, OUTPUT_PATH / "df_clustered")

    """
    ################################
    # ITINERARY OPTIMIZATION
    ################################

    optimizer = ItineraryOptimizer.from_list_matrix(
            df_pois=df_clustered,
            matrix=df_osrm_dur.to_numpy(),          # ou dist_matrix
            metric="duration",          # indicatif, tu peux en faire qqch plus tard
            )

    df_itinerary = optimizer.solve_all_days()

    print(df_itinerary)
    """
    end_total = time.perf_counter()
    print(f"\n=== Temps total du process : {end_total - start_total:.2f} sec ===")
    
if __name__ == "__main__":
    main()