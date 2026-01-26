import pandas as pd

REQUIRED_COLUMNS = [
    "osrm_index",
    "latitude",
    "longitude",
    "main_category",
    "sub_category",
]

def load_poi_parquet(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    # Vérification des colonnes
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required POI columns: {missing}")

    # Normalisation des IDs
    df = df.sort_values("osrm_index").reset_index(drop=True)
    df["osrm_index"] = df.index

    return df