import polars as pl


def add_popularity(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute un score de popularité basé sur :
    - rating (note moyenne)
    - review_count (nombre d'avis)
    
    Fonctionne même si les colonnes n'existent pas encore :
    - rating manquant → 0
    - review_count manquant → 0
    """

    # Colonnes par défaut si absentes
    lf = lf.with_columns([
        pl.col("rating").fill_null(0).alias("rating").cast(pl.Float64)
        if "rating" in lf.columns else pl.lit(0).alias("rating"),

        pl.col("review_count").fill_null(0).alias("review_count").cast(pl.Float64)
        if "review_count" in lf.columns else pl.lit(0).alias("review_count"),
    ])

    # Score de popularité
    # Formule évolutive :
    #   rating * log(1 + review_count)
    # → stable, robuste, évite les explosions
    popularity_expr = (
        pl.col("rating") * (pl.col("review_count") + 1).log()
    )

    return lf.with_columns(
        popularity_expr.alias("popularity_score")
    )
