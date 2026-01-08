#!/usr/bin/env python3
import json
from pathlib import Path

from bounding_box import BoundingBoxResolver
from h3_utils import polygon_to_cells

DATA_DIR = Path("data")
GEOJSON_COMMUNES = DATA_DIR / "raw" / "communes-100m.geojson"
ADMIN_HEXES_PATH = DATA_DIR / "processed" / "admin_hexes.json"

# ---------------------------------------------------------
# Communes → H3
# ---------------------------------------------------------

def build_communes_hex(resolver, res=8):
    results = {}

    for _, row in resolver.communes.iterrows():
        name = row["nom"]

        # Récupérer le polygone bounding box
        bbox = resolver.get_city_bbox(name)
        if bbox is None:
            continue

        # Convertir dict bbox → liste de tuples (latitude, longitude)
        polygon = [
            (bbox["lat_min"], bbox["lon_min"]),
            (bbox["lat_min"], bbox["lon_max"]),
            (bbox["lat_max"], bbox["lon_max"]),
            (bbox["lat_max"], bbox["lon_min"]),
            (bbox["lat_min"], bbox["lon_min"]),
        ]

        # Convertir en H3
        hexes = polygon_to_cells(polygon, res)
        results[name] = hexes

    return results


# ---------------------------------------------------------
# Régions → H3
# ---------------------------------------------------------

def build_regions_hex(resolver, res=6):
    results = {}

    for _, row in resolver.regions.iterrows():
        name = row["nom"]

        bbox = resolver.get_region_bbox(name)
        if bbox is None:
            continue

        polygon = [
            (bbox["lat_min"], bbox["lon_min"]),
            (bbox["lat_min"], bbox["lon_max"]),
            (bbox["lat_max"], bbox["lon_max"]),
            (bbox["lat_max"], bbox["lon_min"]),
            (bbox["lat_min"], bbox["lon_min"]),
        ]

        hexes = polygon_to_cells(polygon, res)
        results[name] = hexes

    return results


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("Chargement des parquets...")
    resolver = BoundingBoxResolver()

    print("Génération des hexagones communes...")
    communes_hex = build_communes_hex(resolver, res=8)

    print("Génération des hexagones régions...")
    regions_hex = build_regions_hex(resolver, res=6)

    print("Sauvegarde dans admin_hexes.json...")
    out = {
        "commune": communes_hex,
        "region": regions_hex
    }

    with open(ADMIN_HEXES_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print("Terminé !")


if __name__ == "__main__":
    main()