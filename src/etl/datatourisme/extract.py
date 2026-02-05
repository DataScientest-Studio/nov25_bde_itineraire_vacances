import gzip
import shutil
from datetime import datetime
from pathlib import Path

import requests
import urllib3

from src.config.settings import RAW_DIR

# 🔧 CONFIGURATION
# Lien Open Data DataTourisme (Format CSV standard).
# Si tu as ton propre flux IDF, remplace ce lien :
DATATOURISME_URL = (
    "https://www.data.gouv.fr/fr/datasets/r/5eb5139a-5b6d-4767-9372-2d244675543c"
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def extract_datatourisme():
    print("📥 Début de l'extraction DataTourisme...")

    # Création du dossier raw
    (RAW_DIR / "datatourisme").mkdir(parents=True, exist_ok=True)

    # Nom du fichier de sortie
    output_path = (
        RAW_DIR
        / "datatourisme"
        / f"datatourisme_{datetime.now().strftime('%Y%m%d')}.csv"
    )

    try:
        print(f"🌍 Téléchargement depuis : {DATATOURISME_URL}")
        with requests.get(DATATOURISME_URL, stream=True, verify=False) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        print(f"✅ Fichier brut téléchargé : {output_path}")
        return str(output_path)

    except Exception as e:
        print(f"❌ Erreur Extraction : {e}")
        raise e


if __name__ == "__main__":
    extract_datatourisme()
