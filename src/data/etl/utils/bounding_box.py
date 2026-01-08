import geopandas as gpd
from shapely.geometry import Point
from pathlib import Path

DATA_DIR = Path("../../data")
print('data_dir',DATA_DIR)
REGIONS_PATH = DATA_DIR / "processed" / "regions.parquet"
DEPARTEMENTS_PATH = DATA_DIR / "processed" / "departements.parquet"
COMMUNES_PATH = DATA_DIR / "processed" / "communes.parquet"

class BoundingBoxResolver:
    def __init__(self, 
                 regions_path=REGIONS_PATH,
                 departements_path=DEPARTEMENTS_PATH,
                 communes_path=COMMUNES_PATH):
        
        self.regions = gpd.read_parquet(Path(regions_path))
        self.departements = gpd.read_parquet(Path(departements_path))
        self.communes = gpd.read_parquet(Path(communes_path))


    # ---------------------------
    # REGION
    # ---------------------------

    def get_region_bbox(self, region_name):
        row = self.regions[self.regions["nom"] == region_name]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "latitude_min": r.latitude_min,
            "latitude_max": r.latitude_max,
            "longitude_min": r.longitude_min,
            "longitude_max": r.longitude_max
        }

    def get_region_centroid(self, region_name):
        row = self.regions[self.regions["nom"] == region_name]
        if row.empty:
            return None
        r = row.iloc[0]
        return (r.centroid_latitude, r.centroid_longitude)

    def poi_in_region(self, latitude, longitude, region_name):
        row = self.regions[self.regions["nom"] == region_name]
        if row.empty:
            return False
        polygon = row.iloc[0].geometry
        return polygon.contains(Point(longitude, latitude))

    # ---------------------------
    # CITY
    # ---------------------------

    def get_city_bbox(self, city_name):
        row = self.communes[self.communes["nom"] == city_name]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "latitude_min": r.latitude_min,
            "latitude_max": r.latitude_max,
            "longitude_min": r.longitude_min,
            "longitude_max": r.longitude_max
        }

    def get_city_centroid(self, city_name):
        row = self.communes[self.communes["nom"] == city_name]
        if row.empty:
            return None
        r = row.iloc[0]
        return (r.centroid_latitude, r.centroid_longitude)

    def poi_in_city(self, latitude, longitude, city_name):
        row = self.communes[self.communes["nom"] == city_name]
        if row.empty:
            return False
        polygon = row.iloc[0].geometry
        return polygon.contains(Point(longitude, latitude))
    

# test
if __name__ == "__main__":
    resolver = BoundingBoxResolver()
    print(resolver.get_city_centroid("Hermival-les-Vaux"))
    print(resolver.poi_in_city(45.75, 4.85, "Lyon"))
    print(resolver.get_region_bbox("Île-de-France"))