import time
from pathlib import Path
import polars as pl

from features.poi_filter import POIFilter
from features.spatial_clustering import SpatialClusterer
from features.poi_balancer import POIBalancer
from features.osrm import OSRMClient
from features.tsp_solver import TSPSolver

import pydeck as pdk


DATA_DIR = Path("../data")
print(DATA_DIR)
POIS_PATH = DATA_DIR / "processed" / "merged_20260106_135958.parquet"

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
            main_categories=["Culture & Musées", "Patrimoine & Monuments", "Gastronomie & Restauration", "Bien-être & Santé"]
        )
        .set_min_score(0.4)
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

    df = (
    clustered
    .select([
        "longitude",
        "latitude",
        "final_score",        # optionnel
        "h3_r8",      # optionnel
    ])
        .collect()
    )

    layer = pdk.Layer(
    "ScatterplotLayer",
    df.to_dicts(),
    get_position=["longitude", "latitude"],
    get_fill_color=[255, 0, 0],
    get_radius=40,
        pickable=True,
    )

    view = pdk.ViewState(
        longitude=df["longitude"].mean(),
        latitude=df["latitude"].mean(),
        zoom=11,
    )

    deck = pdk.Deck(layers=[layer], initial_view_state=view)

    # print(clustered.filter(pl.col("day") == 0).collect())  # jour 1
    # print(clustered.filter(pl.col("day") == 1).collect())  # jour 2
    # print(clustered.filter(pl.col("day") == 2).collect())  # jour 2
    
    ###############################
    # Post clustering
    ###############################
    
    # post clustering pour équilibrer le nombre de POIs par jour
    mode = "walk"
    
    balanced = POIBalancer(clustered) \
                .set_mode(mode) \
                .set_nb_days(user_nb_days) \
                .apply()

    #print(balanced.filter(pl.col("day") == 0).collect())  # jour 1
    #print(balanced.filter(pl.col("day") == 1).collect())  # jour 2
    #print(balanced.filter(pl.col("day") == 2).collect())  # jour 3
    #print(balanced.filter(pl.col("day") == 3).collect())  # jour 4

    ###############################
    # OSRM pour chaque jour
    ###############################

    client = OSRMClient()

    for day in range(user_nb_days):
        df_day = balanced.filter(pl.col("day") == day).collect()
        # 5.1 Ajouter l’ancrage en tête
        coords = [(anchor_lat, anchor_lon)] + list(zip(df_day["latitude"], df_day["longitude"]))
        # 5.2 Matrice OSRM
        matrix = client.table(coords, annotations="duration")["durations"]

        ###############################
        # TSP pour chaque jour
        ###############################
        
        solver = TSPSolver(matrix)
        order = solver.solve()

        # enlever l’ancrage (index 0)
        poi_order = [i - 1 for i in order if i != 0]

        itinerary = df_day[poi_order]

        print(f'itinerary - {day}', itinerary)
        
    end_total = time.perf_counter()
    print(f"\n=== Temps total du process : {end_total - start_total:.2f} sec ===")
    
if __name__ == "__main__":
    main()