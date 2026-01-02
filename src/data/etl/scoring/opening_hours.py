import polars as pl


def add_opening_hours_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute un score d'ouverture basé sur les colonnes disponibles.
    Fonctionne même si les colonnes n'existent pas encore.

    Colonnes prises en compte si présentes :
    - is_open_now (bool)
    - open_hours_count (nombre d'heures d'ouverture par jour)
    - open_late (bool)
    - open_weekend (bool)
    """

    # ---------------------------------------------------------
    # 1. Normalisation des colonnes existantes ou valeurs par défaut
    # ---------------------------------------------------------

    lf = lf.with_columns([

        # Ouvert maintenant (bool → 0/1)
        (
            pl.col("is_open_now").cast(pl.Int8).fill_null(0)
            if "is_open_now" in lf.columns else pl.lit(0)
        ).alias("is_open_now"),

        # Nombre d'heures d'ouverture par jour (0 si absent)
        (
            pl.col("open_hours_count").cast(pl.Float64).fill_null(0)
            if "open_hours_count" in lf.columns else pl.lit(0)
        ).alias("open_hours_count"),

        # Ouvert tard (bool → 0/1)
        (
            pl.col("open_late").cast(pl.Int8).fill_null(0)
            if "open_late" in lf.columns else pl.lit(0)
        ).alias("open_late"),

        # Ouvert le weekend (bool → 0/1)
        (
            pl.col("open_weekend").cast(pl.Int8).fill_null(0)
            if "open_weekend" in lf.columns else pl.lit(0)
        ).alias("open_weekend"),
    ])

    # ---------------------------------------------------------
    # 2. Score d'ouverture (pondéré)
    # ---------------------------------------------------------

    opening_score_expr = (
        0.4 * pl.col("is_open_now") +
        0.3 * (pl.col("open_hours_count") / 12).clip(0, 1) +  # normalisé sur 12h
        0.2 * pl.col("open_late") +
        0.1 * pl.col("open_weekend")
    )

    return lf.with_columns(
        opening_score_expr.alias("opening_score")
    )
