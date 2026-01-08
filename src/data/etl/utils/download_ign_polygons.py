import requests
from pathlib import Path

DATA_DIR = Path("data")
BASE_URL = "https://etalab-datasets.geo.data.gouv.fr/contours-administratifs/latitudeest/geojson"

FILES = {
    "regions": "regions-100m.geojson",
    "departements": "departements-100m.geojson",
    "communes": "communes-100m.geojson"
}

IGN_GEOJSON = DATA_DIR / "raw"


def download_file(url, dest):
    print(f"Téléchargement : {url}")
    r = requests.get(url)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"✔️ Fichier sauvegardé : {dest}")

def main():
    raw_dir = IGN_GEOJSON
    raw_dir.mkdir(parents=True, exist_ok=True)

    for name, filename in FILES.items():
        url = f"{BASE_URL}/{filename}"
        dest = raw_dir / filename
        download_file(url, dest)

if __name__ == "__main__":
    main()