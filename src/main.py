import time
from pathlib import Path
import polars as pl
import asyncio

from features.poi_filter import POIFilter
from features.spatial_clustering import SpatialClusterer
from features.post_clustering import build_osrm_ready_pois, build_osrm_matrices_async
from features.itinerary_optimizer import ItineraryOptimizer
from features.osrm import OSRMClientAsync


import pydeck as pdk


DATA_DIR = Path("data")
print(DATA_DIR)
POIS_PATH = DATA_DIR / "processed" / "merged_20260106_135958.parquet"

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
    user_nb_days = 3
    anchor_lat = 48.8566
    anchor_lon = 2.3522

    clustered = (
        SpatialClusterer(filtered_pois)  # LazyFrame avec score_final
        .set_nb_days(user_nb_days)
        .set_anchor(anchor_lat, anchor_lon)
        .apply()
    )

    # df_clustered = (
    #     clustered
    #     .with_row_index(name="poi_id")      # crée 0,1,2,3,...
    #     .with_columns(
    #         (pl.col("poi_id") + 1).alias("poi_id")   # décale pour commencer à 1
    #     )
    #     .select([
    #         "poi_id",
    #         "main_category",
    #         "longitude",
    #         "latitude",
    #         "final_score",
    #         "h3_r8",
    #         "day",
    #         "diversity_commune_norm",
    #         "itineraire"
    #     ])
    #     .collect()
    # )
    df_clustered = (
        clustered
        .select([
            "poi_id",
            "main_category",
            "longitude",
            "latitude",
            "final_score",
            "h3_r8",
            "day",
            "diversity_commune_norm",
            "itineraire"
        ])
        .collect()
    )

    layer = pdk.Layer(
    "ScatterplotLayer",
    df_clustered.to_dicts(),
    get_position=["longitude", "latitude"],
    get_fill_color=[255, 0, 0],
    get_radius=40,
        pickable=True,
    )

    view = pdk.ViewState(
        longitude=df_clustered["longitude"].mean(),
        latitude=df_clustered["latitude"].mean(),
        zoom=11,
    )

    deck = pdk.Deck(layers=[layer], initial_view_state=view)

    # print(clustered.filter(pl.col("day") == 0).collect())  # jour 1
    # print(clustered.filter(pl.col("day") == 1).collect())  # jour 2
    # print(clustered.filter(pl.col("day") == 2).collect())  # jour 2
    
    ###############################
    # Post clustering
    ###############################

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
    
    # 4. construction des matrices
    osrm = OSRMClientAsync()
    df_clustered, df_osrm_dist, df_osrm_dur = asyncio.run(build_osrm_matrices_async(df_clustered, osrm))


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
        
    end_total = time.perf_counter()
    print(f"\n=== Temps total du process : {end_total - start_total:.2f} sec ===")
    
if __name__ == "__main__":
    main()