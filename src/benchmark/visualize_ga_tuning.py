from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl


def _save_fig(fig, save_path: str | None):
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"✓ Graphe sauvegardé : {path}")


def plot_ga_tuning(df_ga_tuning: pl.DataFrame, save_path: str | None = None):
    pdf = df_ga_tuning.to_pandas()

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=pdf,
        x="cluster_size",
        y="runtime_ms",
        hue="config",
        marker="o",
    )
    plt.title("Runtime GA selon configuration")
    plt.grid(True)
    _save_fig(plt.gcf(), save_path)
    plt.show()


def plot_runtime(df_ga_tuning, save_path: str | None = None):
    pdf = df_ga_tuning.to_pandas()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pdf, x="config", y="runtime_ms", palette="Set2")
    plt.title("Runtime GA par configuration")
    plt.ylabel("Runtime (ms)")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_fig(plt.gcf(), save_path)
    plt.show()


def plot_fitness(df_ga_tuning, save_path: str | None = None):
    pdf = df_ga_tuning.to_pandas()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=pdf, x="config", y="fitness", palette="Set3")
    plt.title("Fitness GA par configuration")
    plt.ylabel("Fitness")
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    _save_fig(plt.gcf(), save_path)
    plt.show()


def plot_tradeoff(df_ga_tuning, save_path: str | None = None):
    pdf = df_ga_tuning.to_pandas()

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=pdf, x="runtime_ms", y="fitness", hue="config", style="config", s=120
    )
    plt.title("Tradeoff Runtime vs Fitness")
    plt.xlabel("Runtime (ms)")
    plt.ylabel("Fitness")
    plt.grid(True, linestyle="--", alpha=0.5)
    _save_fig(plt.gcf(), save_path)
    plt.show()
