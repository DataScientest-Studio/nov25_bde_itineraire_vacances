
from pathlib import Path
import json
import polars as pl
import requests

from etl.utils.utils import download_with_retry


ROOT = Path(__file__).parent # parent directory of the script
INPUT_DIR = ROOT / "config"
input_path = INPUT_DIR / "index.json"
BASE_URL = "https://object.files.data.gouv.fr/hydra-parquet/hydra-parquet"

def load_index(path: str = input_path) -> dict:
    """Charge le fichier index contenant {uuid: region}."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_url(uuid: str) -> str:
    """Construit l'URL Parquet à partir d'un UUID."""
    return f"{BASE_URL}/{uuid}.parquet"

def extract_parquet_from_url(url: str, retries: int = 3, timeout: int = 30) -> pl.DataFrame:
    """Télécharge un fichier Parquet et le charge en Polars."""
    raw_bytes = download_with_retry(url, retries=retries, timeout=timeout)
    return pl.read_parquet(raw_bytes)

def extract_all() -> list[pl.DataFrame]:
    """Extrait tous les Parquet listés dans l'index et ajoute la Colonne région."""
    index = load_index()

    df_list = []
    for uuid, region in index.items():
        try:
            url = build_url(uuid)
            df = extract_parquet_from_url(url)

            # Ajout de la région (important pour la suite du pipeline)
            df = df.with_columns(pl.lit(region).alias("region"))

            df_list.append(df)
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Impossible de télécharger {uuid}: {e}")

    return df_list