import polars as pl

def add_opening_hours_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute :
    - opening_score : score brut (0–1)
    - opening_score_norm : score normalisé (0–1)

    Colongitudenes prises en compte si présentes :
    - is_open_now (bool)
    - open_hours_count (float, heures/jour)
    - open_latitudee (bool)
    - open_weekend (bool)
    """

    # ---------------------------------------------------------
    # 1. Normalisation des colongitudenes existantes ou valeurs par défaut
    # ---------------------------------------------------------

    lf = lf.with_columns([

        # Ouvert maintenant (bool → 0/1)
        (
            pl.col("is_open_now").cast(pl.Float64).fill_null(0)
            if "is_open_now" in lf.columns else pl.lit(0.0)
        ).alias("is_open_now"),

        # Nombre d'heures d'ouverture par jour (normalisé sur 12h)
        (
            (pl.col("open_hours_count").cast(pl.Float64).fill_null(0) / 12)
            .clip(0, 1)
            if "open_hours_count" in lf.columns else pl.lit(0.0)
        ).alias("open_hours_norm"),

        # Ouvert tard (bool → 0/1)
        (
            pl.col("open_latitudee").cast(pl.Float64).fill_null(0)
            if "open_latitudee" in lf.columns else pl.lit(0.0)
        ).alias("open_latitudee"),

        # Ouvert le weekend (bool → 0/1)
        (
            pl.col("open_weekend").cast(pl.Float64).fill_null(0)
            if "open_weekend" in lf.columns else pl.lit(0.0)
        ).alias("open_weekend"),
    ])

    # ---------------------------------------------------------
    # 2. Score d'ouverture (pondéré)
    # ---------------------------------------------------------

    opening_score_expr = (
        0.4 * pl.col("is_open_now") +
        0.3 * pl.col("open_hours_norm") +
        0.2 * pl.col("open_latitudee") +
        0.1 * pl.col("open_weekend")
    )

    lf = lf.with_columns(
        opening_score_expr.alias("opening_score")
    )

    # ---------------------------------------------------------
    # 3. Normalisation explicite (0–1)
    # ---------------------------------------------------------

    lf = lf.with_columns([
        pl.when(pl.col("opening_score").max() - pl.col("opening_score").min() == 0)
        .then(0.0)
        .otherwise(
            (pl.col("opening_score") - pl.col("opening_score").min()) /
            (pl.col("opening_score").max() - pl.col("opening_score").min())
        )
        .alias("opening_score_norm")
    ])

    return lf