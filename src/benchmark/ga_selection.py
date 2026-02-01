def select_best_ga_config(df_ga_tuning):
    pdf = df_ga_tuning.to_pandas()

    summary = (
        pdf.groupby("config")
        .agg({
            "runtime_ms": "mean",
            "fitness": "mean"
        })
        .rename(columns={
            "runtime_ms": "runtime_mean",
            "fitness": "fitness_mean"
        })
    )

    summary["score"] = summary["fitness_mean"] / summary["runtime_mean"]

    best_config = summary["score"].idxmax()

    print("\n=== Résumé des performances GA ===")
    print(summary)
    print("\n→ Meilleure configuration GA :", best_config)

    return best_config, summary