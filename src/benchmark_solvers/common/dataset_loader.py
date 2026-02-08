import glob
import re

import numpy as np
import pandas as pd

from benchmark_solvers.common.osrm_matrix import load_osrm_matrix_parquet
from benchmark_solvers.common.poi_loader import load_poi_parquet


def load_all_datasets(
    poi_dir="benchmark_solvers/data/", matrix_dir="benchmark_solvers/data/"
):
    datasets = {}

    poi_files = glob.glob(f"{poi_dir}/pois_*.parquet")
    matrix_files = glob.glob(f"{matrix_dir}/matrix_*.parquet")

    # Extraire la taille (ex: 14 dans pois_14.parquet)
    extract = lambda f: int(re.findall(r"(\d+)", f)[-1])

    poi_map = {extract(f): f for f in poi_files}
    matrix_map = {extract(f): f for f in matrix_files}

    common_sizes = sorted(set(poi_map.keys()) & set(matrix_map.keys()))

    for size in common_sizes:
        # ----------------------------------------------------------------------
        # 1) Charger les POIs
        # ----------------------------------------------------------------------
        poi_df = load_poi_parquet(poi_map[size])

        # Reset index pour éviter KeyError
        poi_df = poi_df.reset_index(drop=True)

        # Vérification obligatoire : osrm_index doit exister
        if "osrm_index" not in poi_df.columns:
            raise ValueError(f"Missing osrm_index in {poi_map[size]}")

        # Vérifier que osrm_index = 0..N-1
        expected = list(range(len(poi_df)))
        actual = sorted(int(x) for x in poi_df.osrm_index.unique())

        if expected != actual:
            raise ValueError(
                f"osrm_index mismatch in {poi_map[size]} : "
                f"expected {expected}, got {actual}"
            )

        # ----------------------------------------------------------------------
        # 2) Charger la matrice OSRM
        # ----------------------------------------------------------------------
        matrix_df = load_osrm_matrix_parquet(matrix_map[size])

        # Si la matrice contient une colonne osrm_index → on la met en index
        if "osrm_index" in matrix_df.columns:
            matrix_df = matrix_df.set_index("osrm_index")

        # Vérifier que l’index de la matrice est bien 0..N-1
        matrix_index = sorted(int(x) for x in matrix_df.index.tolist())

        if matrix_index != expected:
            raise ValueError(
                f"Matrix index mismatch for size {size}: "
                f"expected {expected}, got {matrix_index}"
            )

        # Vérifier que la matrice est carrée
        if matrix_df.shape[0] != matrix_df.shape[1]:
            raise ValueError(f"Matrix for size {size} is not square: {matrix_df.shape}")

        # Conversion en numpy
        matrix = matrix_df.to_numpy()

        # ----------------------------------------------------------------------
        # 3) Stockage final
        # ----------------------------------------------------------------------
        datasets[f"{size}_pois"] = {
            "size": size,
            "poi_df": poi_df,
            "matrix": matrix,
        }

    return datasets
