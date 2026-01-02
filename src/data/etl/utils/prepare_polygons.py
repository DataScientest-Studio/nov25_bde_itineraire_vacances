import geopandas as gpd
from pathlib import Path


DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
OUTPUT_DIR = DATA_DIR / "processed"

def process_layer(name, filename):
    print(f"📌 Traitement : {name}")

    gdf = gpd.read_file(filename)

    # Calcul bounding box
    gdf["lon_min"] = gdf.geometry.bounds["minx"]
    gdf["lat_min"] = gdf.geometry.bounds["miny"]
    gdf["lon_max"] = gdf.geometry.bounds["maxx"]
    gdf["lat_max"] = gdf.geometry.bounds["maxy"]

    # Calcul centroid
    gdf["centroid"] = gdf.geometry.centroid
    gdf["centroid_lat"] = gdf["centroid"].y
    gdf["centroid_lon"] = gdf["centroid"].x

    # Sauvegarde Parquet
    out_path = OUTPUT_DIR / f"{name}.parquet"
    out_path.parent.mkdir(exist_ok=True)
    gdf.to_parquet(out_path, index=False)

    print(f"✔️ Sauvegardé : {out_path}")

def main():
    raw_dir = Path("input/ign_polygons")

    layers = {
        "regions": RAW_DIR / "regions-100m.geojson",
        "departements": RAW_DIR / "departements-100m.geojson",
        "communes": RAW_DIR / "communes-100m.geojson"
    }

    for name, path in layers.items():
        process_layer(name, path)

if __name__ == "__main__":
    main()