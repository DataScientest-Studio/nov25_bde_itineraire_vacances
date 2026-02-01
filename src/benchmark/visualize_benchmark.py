import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path


def _save_fig(fig, save_path: str | None):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Graphe sauvegardé : {path}")


# ---------------------------------------------------------
# 1) Courbes du seuil AUTO
# ---------------------------------------------------------

def plot_size_comparison(df: pl.DataFrame, save_path: str | None = None):
    dfp = df.to_pandas()

    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    sns.lineplot(data=dfp, x="cluster_size", y="total_distance", hue="solver", ax=axes[0], marker="o")
    axes[0].set_title("Distance totale vs Taille du cluster")
    axes[0].grid(True)

    sns.lineplot(data=dfp, x="cluster_size", y="total_duration", hue="solver", ax=axes[1], marker="o")
    axes[1].set_title("Durée totale vs Taille du cluster")
    axes[1].grid(True)

    sns.lineplot(data=dfp, x="cluster_size", y="runtime_ms", hue="solver", ax=axes[2], marker="o")
    axes[2].set_title("Temps d'exécution vs Taille du cluster")
    axes[2].grid(True)

    plt.tight_layout()

    _save_fig(fig, save_path)
    plt.show()


# ---------------------------------------------------------
# 2) Stabilité par matrice
# ---------------------------------------------------------

def plot_stability(df: pl.DataFrame, solver_name: str, save_path: str | None = None):
    dfp = df.to_pandas()

    fig, axes = plt.subplots(3, 1, figsize=(10, 14))

    sns.boxplot(data=dfp, x="matrix_id", y="total_distance", ax=axes[0])
    axes[0].set_title(f"Stabilité {solver_name.upper()} – Distance totale")
    axes[0].grid(True)

    sns.boxplot(data=dfp, x="matrix_id", y="total_duration", ax=axes[1])
    axes[1].set_title(f"Stabilité {solver_name.upper()} – Durée totale")
    axes[1].grid(True)

    sns.boxplot(data=dfp, x="matrix_id", y="runtime_ms", ax=axes[2])
    axes[2].set_title(f"Stabilité {solver_name.upper()} – Runtime")
    axes[2].grid(True)

    plt.tight_layout()

    _save_fig(fig, save_path)
    plt.show()


# ---------------------------------------------------------
# 3) Analyse de la stabilité de l’ordre
# ---------------------------------------------------------

def plot_order_stability(df: pl.DataFrame, solver_name: str, save_path: str | None = None):
    dfp = df.to_pandas()

    order_counts = (
        dfp.groupby(["matrix_id", "order_signature"])
        .size()
        .reset_index(name="count")
    )

    fig = plt.figure(figsize=(10, 6))
    sns.barplot(data=order_counts, x="matrix_id", y="count", hue="order_signature")
    plt.title(f"Stabilité des ordres – {solver_name.upper()}")
    plt.grid(True)

    _save_fig(fig, save_path)
    plt.show()

# ---------------------------------------------------------
# 4) Analyse des configurations GA
# ---------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


def plot_ga_tuning(df_ga_tuning: pl.DataFrame, save_path: str | None = None):
    """
    Visualise les performances des différentes configurations GA :
    - runtime
    - distance
    - ratio gain/coût
    """

    pdf = df_ga_tuning.to_pandas()

    # ---------------------------------------------------------
    # 1. Runtime GA
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=pdf,
        x="matrix_id",
        y="runtime_ga",
        hue="config",
        marker="o"
    )
    plt.title("Runtime GA selon configuration")
    plt.xlabel("Taille du cluster")
    plt.ylabel("Runtime (ms)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path.replace(".png", "_runtime.png"), dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 2. Distance GA
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=pdf,
        x="matrix_id",
        y="dist_ga",
        hue="config",
        marker="o"
    )
    plt.title("Distance GA selon configuration")
    plt.xlabel("Taille du cluster")
    plt.ylabel("Distance totale")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path.replace(".png", "_distance.png"), dpi=150)
    plt.show()

    # ---------------------------------------------------------
    # 3. Ratio gain/coût
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=pdf,
        x="matrix_id",
        y="ratio_gain_cost",
        hue="config",
        marker="o"
    )
    plt.title("Ratio gain/coût (GA vs NN2O) selon configuration")
    plt.xlabel("Taille du cluster")
    plt.ylabel("Ratio gain/coût")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path.replace(".png", "_ratio.png"), dpi=150)
    plt.show()