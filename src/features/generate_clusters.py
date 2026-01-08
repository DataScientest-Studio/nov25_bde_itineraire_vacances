import time
from pathlib import Path
import polars as pl

from poi_filter import POIFilter
from spatial_clustering import SpatialClusterer



DATA_DIR = Path("../../data")
POIS_PATH = DATA_DIR / "processed" / "merged_20260106_135958.parquet"
OUTPUT_DIR = DATA_DIR / "processed" 

MAIN_CATEGORY = ["Culture & Musées","Patrimoine & Monuments",
                "Gastronomie & Restauration","Bien-être & Santé", "Hébergement","Sports & Loisirs","Santé & Urgences",
                "Famille & Enfants","Culture & Musées","Transports",
                "Shopping & Artisanat","Nature & Paysages","Patrimoine & Monuments",
                "Services & Mobilité","Commerce & Shopping","Camping & Plein Air","Commodités",
                "Transports touristiques","Loisirs & Clubs","Événements & Traditions","Information Touristique"
            ]

def save_clustered_by_day(clustered_lf: pl.LazyFrame, OUTPUT_DIR: str):
    output = Path(OUTPUT_DIR)
    output.mkdir(parents=True, exist_ok=True)

    # On matérialise une seule fois
    df = clustered_lf.collect()

    # Vérification
    if "day" not in df.columns:
        raise ValueError("La colonne 'day' est absente du DataFrame.")

    # Sauvegarde jour par jour
    for day, subdf in df.group_by("day"):
        file_path = output / f"day_{day}.parquet"
        subdf.write_parquet(file_path)
        print(f"✔️ Sauvegardé : {file_path}")


def main():
    start_total = time.perf_counter()

    commune = 'Quiberon'

    ###############################
    # Filtrage des POIs
    ###############################

    pois_lf= pl.scan_parquet(POIS_PATH)
    print(len(pois_lf.collect()))

    filter_pois = POIFilter(pois_lf)
    filtered_pois = (
        filter_pois
        .set_commune(commune)
        .set_categories(
            main_categories=MAIN_CATEGORY
        )
        .set_min_score(0.2)
        .apply()
    )
    print(len(filtered_pois.collect()))

    ###############################
    # Clustering des POIs
    ###############################
    user_nb_days = 2
    anchor_lat = 48.8566
    anchor_lon = 2.3522

    clustered = (
        SpatialClusterer(filtered_pois)  # LazyFrame avec score_final
        .set_nb_days(user_nb_days)
        .set_anchor(anchor_lat, anchor_lon)
        .apply()
    )

    print(clustered.filter(pl.col("day") == 0).collect())  # jour 1
    print(clustered.filter(pl.col("day") == 1).collect())  # jour 2
    print(clustered.filter(pl.col("day") == 2).collect())  # jour 2

    save_clustered_by_day(clustered, f"clustered_days_{commune}")
    
if __name__ == "__main__":
    main()