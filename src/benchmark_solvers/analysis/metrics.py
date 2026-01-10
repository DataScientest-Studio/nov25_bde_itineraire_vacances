from __future__ import annotations
from typing import List, Tuple, Dict, Callable, Any, Optional

import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def compute_best_per_matrix(df: pd.DataFrame) -> pd.Series:
    """
    Meilleur coût (tous solveurs confondus) par matrice.
    """
    return df.groupby("matrix")["cost"].min()


def add_gap_column(df: pd.DataFrame, best_per_matrix: pd.Series) -> pd.DataFrame:
    """
    Ajoute une colonne 'gap' = (cost - best_matrix) / best_matrix
    """
    df = df.copy()
    df = df.join(best_per_matrix.rename("best_matrix"), on="matrix")
    df["gap"] = (df["cost"] - df["best_matrix"]) / df["best_matrix"]
    return df


def stability_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Stabilité : moyenne, médiane, min, max, écart-type du coût et du gap par solver.
    """
    agg = df.groupby("solver").agg(
        cost_mean=("cost", "mean"),
        cost_median=("cost", "median"),
        cost_min=("cost", "min"),
        cost_max=("cost", "max"),
        cost_std=("cost", "std"),
        gap_mean=("gap", "mean"),
        gap_median=("gap", "median"),
        gap_std=("gap", "std"),
        time_mean=("time_sec", "mean"),
        time_std=("time_sec", "std"),
    )
    return agg.reset_index()

def robustness_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robustesse : performance moyenne par (solver, matrix).
    Permet de voir comment un solver se comporte selon les instances.
    """
    agg = df.groupby(["solver", "matrix"]).agg(
        cost_mean=("cost", "mean"),
        cost_std=("cost", "std"),
        gap_mean=("gap", "mean"),
        gap_std=("gap", "std"),
    )
    return agg.reset_index()

def pareto_front(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pareto qualité / temps (sur les moyennes par solver).
    On considère:
      - gap_mean (à minimiser)
      - time_mean (à minimiser)
    Retourne les solveurs non dominés.
    """
    stats = df.groupby("solver").agg(
        gap_mean=("gap", "mean"),
        time_mean=("time_sec", "mean"),
    ).reset_index()

    # Détection Pareto (non dominé)
    is_pareto = []
    for i, row_i in stats.iterrows():
        dominated = False
        for j, row_j in stats.iterrows():
            if j == i:
                continue
            if (
                row_j["gap_mean"] <= row_i["gap_mean"]
                and row_j["time_mean"] <= row_i["time_mean"]
                and (row_j["gap_mean"] < row_i["gap_mean"]
                     or row_j["time_mean"] < row_i["time_mean"])
            ):
                dominated = True
                break
        is_pareto.append(not dominated)
    stats["pareto"] = is_pareto
    return stats

# ============================================================
# 5. Perturbations (robustesse) et sensibilité (hyperparamètres)
# ============================================================

def perturb_matrix(
    D: np.ndarray,
    noise_level: float = 0.05,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Ajoute du bruit multiplicatif sur les distances: D' = D * (1 + eps)
    eps ~ U(-noise_level, noise_level)
    """
    rng = np.random.default_rng(random_state)
    eps = rng.uniform(-noise_level, noise_level, size=D.shape)
    D_pert = D * (1.0 + eps)
    np.fill_diagonal(D_pert, 0.0)
    return D_pert


def generate_sensitivity_grid(
    base_kwargs: Dict[str, Any],
    param_grid: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    """
    Génère une grille de paramètres (pour GA, SA, etc.)
    base_kwargs : paramètres de base
    param_grid : {"mutation_rate": [0.1, 0.2], "pop_size": [50, 100], ...}
    """
    from itertools import product

    keys = list(param_grid.keys())
    values = [param_grid[k] for k in keys]

    configs = []
    for combo in product(*values):
        kwargs = base_kwargs.copy()
        for k, v in zip(keys, combo):
            kwargs[k] = v
        configs.append(kwargs)
    return configs


def solver_ranking_by_distance(df):
    """
    Classement des solveurs par distance totale moyenne (km).
    Plus la distance est faible, meilleur est le solveur.
    """
    ranking = (
        df.groupby("solver")["distance_km"]
        .mean()
        .sort_values()
        .reset_index()
    )
    ranking["rank"] = ranking["distance_km"].rank(method="dense")
    return ranking