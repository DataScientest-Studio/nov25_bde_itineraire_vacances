import matplotlib.pyplot as plt
import numpy as np


def radar_chart(df, solver_name):
    metrics = ["efficiency", "robustness", "edge_ratio", "pareto_score"]
    values = df[df["solver"] == solver_name][metrics].iloc[0].values

    # Normalisation (0-1)
    values = (values - values.min()) / (values.max() - values.min() + 1e-9)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title(f"Radar Chart – {solver_name}")
    return fig


def pareto_plot(df, matrix_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    subset = df[df["matrix"] == matrix_name]

    for _, row in subset.iterrows():
        ax.scatter(row["mean_cost"], row["mean_time"], label=row["solver"], s=100)

    ax.set_xlabel("Distance moyenne")
    ax.set_ylabel("Temps moyen (s)")
    ax.set_title(f"Pareto Distance/Temps – {matrix_name}")
    ax.legend()
    return fig


def boxplot_costs(runner_results, matrix_name):
    fig, ax = plt.subplots(figsize=(7, 5))

    data = {}
    for r in runner_results:
        if r["matrix"] == matrix_name:
            data.setdefault(r["solver"], []).append(r["mean_cost"])

    ax.boxplot(data.values(), labels=data.keys())
    ax.set_title(f"Distribution des coûts – {matrix_name}")
    ax.set_ylabel("Distance")
    return fig
