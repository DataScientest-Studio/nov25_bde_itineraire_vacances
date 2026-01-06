import polars as pl
from typing import Optional, List


class POIFilter:
    """
    Classe responsable du filtrage des POIs en LazyFrame.
    Elle applique les filtres utilisateur (région, commune, catégories)
    ainsi que les filtres internes (score final).
    """

    def __init__(self, pois_lf: pl.LazyFrame):
        """
        Initialise le filtre avec un LazyFrame de POIs.
        """
        self.lf = pois_lf

        # paramètres utilisateur
        self.region: Optional[str] = None
        self.commune: Optional[str] = None
        self.main_categories: Optional[List[str]] = None
        self.sub_categories: Optional[List[str]] = None

        # paramètre interne
        self.min_score: Optional[float] = None

    # -----------------------------
    # SETTERS
    # -----------------------------

    def set_region(self, region: str):
        self.region = region
        return self

    def set_commune(self, commune: str):
        self.commune = commune
        return self

    def set_categories(
        self,
        main_categories: Optional[List[str]] = None,
        sub_categories: Optional[List[str]] = None,
    ):
        self.main_categories = main_categories
        self.sub_categories = sub_categories
        return self

    def set_min_score(self, min_score: float):
        self.min_score = min_score
        return self

    # -----------------------------
    # APPLY FILTERS
    # -----------------------------

    def apply(self) -> pl.LazyFrame:
        """
        Applique tous les filtres configurés et retourne un LazyFrame filtré.
        Aucun collect() n'est effectué ici.
        """

        lf = self.lf

        # Filtre région
        if self.region:
            lf = lf.filter(pl.col("region") == self.region)

        # Filtre commune
        if self.commune:
            lf = lf.filter(pl.col("commune") == self.commune)

        # Filtre main_category
        if self.main_categories:
            lf = lf.filter(pl.col("main_category").is_in(self.main_categories))

        # Filtre sub_category
        if self.sub_categories:
            lf = lf.filter(pl.col("sub_category").is_in(self.sub_categories))

        # Filtre score final
        if self.min_score is not None:
            lf = lf.filter(pl.col("final_score") >= self.min_score)

        return lf


if __name__ == "__main__":
    import time
    from pathlib import Path
    import json
    
    DATA_DIR = Path("../../data/processed").absolute()
    with open(DATA_DIR /"admin_hexes.json", "r", encoding="utf-8") as f:
        admin_hexes = json.load(f)
    commune = "Annecy"
    region = "Auvergne Rhone Alpes"

    pois_lf= pl.scan_parquet(DATA_DIR / "merged_20260101_234939.parquet")
    print(f"Total rows: {len(pois_lf.collect())}")
    
    print("=== POI Filter ===")
    print("Commune")
    start_time_commune = time.perf_counter()
    filter_pois_commune = POIFilter(pois_lf)
    filtered_pois_commune = (
        filter_pois_commune
        .set_commune(commune)
        .apply()
    )
    print(f"Filtered rows: {len(filtered_pois_commune.collect())}")
    end_time_commune = time.perf_counter()
    print(f"Total time: {end_time_commune - start_time_commune:.2f} seconds")

    print("Région")
    start_time_region = time.perf_counter()
    filter_pois_region = POIFilter(pois_lf)
    filtered_pois_region = (
        filter_pois_region
        .set_region(region)
        .apply()
    )
    print(f"Filtered rows: {len(filtered_pois_region.collect())}")
    end_time_region = time.perf_counter()
    print(f"Total time: {end_time_region - start_time_region:.2f} seconds")   
    print()

    print("=== POI Filter with h3 ===")
    print("Commune")
    start_time_h3_commune = time.perf_counter()
    hexes_paris = admin_hexes["commune"][commune]
    pois_paris = pois_lf.filter(pl.col("h3_r8").is_in(hexes_paris))
    print(f"Filtered rows: {len(pois_paris.collect())}")
    end_time_h3_commune = time.perf_counter()
    print(f"Total time: {end_time_h3_commune - start_time_h3_commune:.2f} seconds")
    
    print("Région")
    start_time_h3_region = time.perf_counter()
    hexes_region = admin_hexes["region"]["Auvergne-Rhône-Alpes"]
    print(f"Hexes region: {len(hexes_region)}")

    pois_region = pois_lf.filter(pl.col("h3_r8").is_in(hexes_region))
    print(f"Filtered rows: {len(pois_region.collect())}")
    end_time_h3_region = time.perf_counter()
    print(f"Total time: {end_time_h3_region - start_time_h3_region:.2f} seconds")