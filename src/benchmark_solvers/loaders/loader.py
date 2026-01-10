import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


def load_osrm_matrix(path: str) -> np.ndarray:
    """Charge une matrice OSRM Parquet et retire la colonne d’index éventuelle."""
    df = pl.read_parquet(path)

    # Supprimer colonne index si présente
    for col in ["osrm_index", "index", "Unnamed: 0"]:
        if col in df.columns:
            df = df.drop(col)

    mat = df.to_numpy()

    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"Matrice non carrée dans {path} : {mat.shape}")

    return mat


def load_pois(path: str) -> pd.DataFrame:
    """Charge un fichier POIs (parquet ou csv)."""
    path = str(path)

    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_all_matrices_and_pois(
    matrix_paths: List[str],
    pois_paths: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, pd.DataFrame]]:
    """Charge toutes les matrices OSRM + POIs correspondants."""
    if len(matrix_paths) != len(pois_paths):
        raise ValueError("matrix_paths et pois_paths doivent avoir la même longueur.")

    matrices = {}
    pois = {}

    for i, (mp, pp) in enumerate(zip(matrix_paths, pois_paths)):
        name = f"M{i+1}"
        matrices[name] = load_osrm_matrix(mp)
        pois[name] = load_pois(pp)

    return matrices, pois


def validate_matrix_pois(matrix: np.ndarray, pois_df: pd.DataFrame, name: str):
    """Valide la cohérence entre une matrice OSRM et ses POIs."""
    n = matrix.shape[0]

    if len(pois_df) != n:
        raise ValueError(
            f"[{name}] Incohérence : matrice {n}×{n} mais {len(pois_df)} POIs."
        )

    if not {"latitude", "longitude"}.issubset(pois_df.columns):
        raise ValueError(f"[{name}] Le fichier POIs doit contenir lat et lon.")

    # Vérification diagonale
    if not np.allclose(np.diag(matrix), 0):
        raise ValueError(f"[{name}] La diagonale de la matrice OSRM doit être nulle.")

    return True


def validate_all(matrices: Dict[str, np.ndarray], pois: Dict[str, pd.DataFrame]):
    """Valide toutes les paires matrice + POIs."""
    for name in matrices:
        validate_matrix_pois(matrices[name], pois[name], name)
    return True