from pipeline.itinerary_pipeline import ItineraryPipeline
from features.osrm import OSRMClientAsync
from pathlib import Path

path_input=Path("../data/processed/merged_20260108_174125.parquet")

pipeline = ItineraryPipeline(path_input)
print(path_input)
osrm = OSRMClientAsync()

modes = ["walk", "bike", "car"]

for mode in modes:
    print(f"\n=== TEST MODE: {mode} ===")

    df_clustered, df_dist, df_dur, df_itinerary, optimizer = pipeline.run(
        commune="Paris",
        main_categories=["Patrimoine & Monuments", "Gastronomie & Restauration", "Shopping & Artisanat"],
        sub_categories=["Restaurants","Bibliothèques & médiation","Restauration rapide","Bars & cafés","Religieux"],
        min_score=0.15,
        nb_days=1,
        anchor_lat=48.86666,
        anchor_lon=2.33333,
        osrm=osrm,
        transport_mode=mode,
        solver="auto",   # "nn2o" ou "ga" ou "auto"
    )


    print("Durée totale :", df_itinerary["day_total_duration"].sum())
    print("Distance totale :", df_itinerary["day_total_distance"].sum())
    print("Optimizer :", optimizer)