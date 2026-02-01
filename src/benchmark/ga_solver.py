import polars as pl
import numpy as np
from typing import Tuple, Optional
from features.optimizer_ga import GeneticAlgo 


def run_ga_on_cluster(
    df_cluster: pl.DataFrame,
    matrices: dict,
    matrix_id: str,
    pop_size: int,
    ngen: int,
    cxpb: float,
    mutpb: float,
    itin_min_poi: int = 5,
    itin_max_poi: int = 15,
):
    if df_cluster.height < 2:
        return None, None

    indices = df_cluster["osrm_index"].to_list()

    # 🔥 Correction : conversion Polars → NumPy
    global_matrix = matrices[matrix_id]["dur"].to_numpy()

    # 🔥 Maintenant l’indexation fonctionne
    local_matrix = global_matrix[np.ix_(indices, indices)]

    df_day_pd = df_cluster.to_pandas()

    ga = GeneticAlgo(
        poi_df=df_day_pd,
        duration_matrix=local_matrix,
    )
    ga.setup_toolbox(itin_min_poi=itin_min_poi, itin_max_poi=itin_max_poi)

    best_route_local, fitness = ga.run_ga(
        pop_size=pop_size,
        ngen=ngen,
        cxpb=cxpb,
        mutpb=mutpb,
    )

    if not best_route_local:
        return None, None

    return best_route_local, fitness