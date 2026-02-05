import numpy as np
import pandas as pd


def load_osrm_matrix_parquet(path: str) -> np.ndarray:
    df = pd.read_parquet(path)

    # # On suppose que la matrice est stockée sous forme carrée
    # matrix = df.to_numpy(dtype=float)

    # # Nettoyage éventuel
    # matrix = np.nan_to_num(matrix, nan=999999, posinf=999999)

    return df
