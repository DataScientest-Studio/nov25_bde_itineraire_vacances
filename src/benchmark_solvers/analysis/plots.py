import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# 1. Boxplots (coûts, gaps)
# ============================================================

def boxplot_costs(df: pd.DataFrame, by: str = "solver", figsize=(8, 5)):
    """
    Boxplot des coûts par solver ou par matrice.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=by, y="cost")
    plt.title(f"Distribution des coûts par {by}")
    plt.tight_layout()
    plt.show()


def boxplot_gaps(df: pd.DataFrame, by: str = "solver", figsize=(8, 5)):
    """
    Boxplot des gaps par solver ou par matrice.
    """
    if "gap" not in df.columns:
        raise ValueError("La colonne 'gap' est manquante. Calcule-la avant.")

    plt.figure(figsize=figsize)
    sns.boxplot(data=df, x=by, y="gap")
    plt.title(f"Distribution des gaps par {by}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 2. Heatmap gap moyen solver × matrice
# ============================================================

def heatmap_gap(df: pd.DataFrame, figsize=(8, 5)):
    """
    Heatmap du gap moyen par (solver, matrix).
    """
    if "gap" not in df.columns:
        raise ValueError("La colonne 'gap' est manquante. Calcule-la avant.")

    pivot = df.groupby(["solver", "matrix"])["gap"].mean().reset_index()
    pivot = pivot.pivot(index="solver", columns="matrix", values="gap")

    plt.figure(figsize=figsize)
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="viridis")
    plt.title("Gap moyen par solver et par matrice")
    plt.tight_layout()
    plt.show()


# ============================================================
# 3. Front de Pareto (qualité vs temps)
# ============================================================

def plot_pareto(stats_pareto: pd.DataFrame, figsize=(7, 5)):
    """
    Scatter qualité/temps avec surlignage du front de Pareto.
    stats_pareto doit contenir:
      - solver
      - gap_mean
      - time_mean
      - pareto (bool)
    """
    plt.figure(figsize=figsize)
    sns.scatterplot(
        data=stats_pareto,
        x="time_mean",
        y="gap_mean",
        hue="pareto",
        style="pareto",
        s=120,
        palette={True: "red", False: "gray"},
    )

    for _, row in stats_pareto.iterrows():
        plt.text(row["time_mean"], row["gap_mean"], row["solver"], fontsize=9)

    plt.xlabel("Temps moyen (s)")
    plt.ylabel("Gap moyen")
    plt.title("Front de Pareto (gap vs temps)")
    plt.tight_layout()
    plt.show()


# ============================================================
# 4. Distribution des coûts (histogrammes)
# ============================================================

def hist_costs(df: pd.DataFrame, solver: str, figsize=(8, 5)):
    """
    Histogramme des coûts pour un solver donné.
    """
    subset = df[df["solver"] == solver]

    plt.figure(figsize=figsize)
    sns.histplot(subset["cost"], kde=True)
    plt.title(f"Distribution des coûts — {solver}")
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. Courbes de stabilité (écart-type par matrice)
# ============================================================

def stability_plot(df: pd.DataFrame, figsize=(8, 5)):
    """
    Affiche l'écart-type du coût par solver et par matrice.
    """
    stab = df.groupby(["solver", "matrix"])["cost"].std().reset_index()

    plt.figure(figsize=figsize)
    sns.barplot(data=stab, x="matrix", y="cost", hue="solver")
    plt.title("Stabilité (écart-type du coût) par solver et par matrice")
    plt.ylabel("Écart-type du coût")
    plt.tight_layout()
    plt.show()

# =====================================================
# 6. Radar chart (distance / gap / temps / stabilité)
# =====================================================

def radar_chart(df):
    """
    Radar chart comparant les solveurs sur :
    - distance_km (normalisée)
    - gap (normalisé)
    - temps (normalisé)
    - stabilité (normalisée)
    """

    metrics = df.groupby("solver").agg({
        "distance_km": "mean",
        "gap": "mean",
        "time_sec": "mean",
        "cost": "std",
    }).rename(columns={"cost": "stability"})

    # Normalisation 0-1
    norm = (metrics - metrics.min()) / (metrics.max() - metrics.min())

    labels = norm.columns.tolist()
    solvers = norm.index.tolist()
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # boucle fermée

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for solver in solvers:
        values = norm.loc[solver].tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=solver)
        ax.fill(angles, values, alpha=0.15)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.set_title("Radar chart solveurs")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    return fig