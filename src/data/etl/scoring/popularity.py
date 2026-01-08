import polars as pl
import math

def add_popularity(lf: pl.LazyFrame, k: int = 50) -> pl.LazyFrame:
    """
    Ajoute :
    - rating_norm : note normalisée (0–1)
    - reviews_norm : confiance basée sur le volume d'avis (0–1)
    - popularity_norm : score final normalisé (0–1)

    rating peut ne pas exister
    review_count peut ne pas exister
    """

    # ---------------------------------------------------------
    # 1. colonnes par défaut si absentes
    # ---------------------------------------------------------
    lf = lf.with_columns([
        (
            pl.col("rating").cast(pl.Float64).fill_null(0)
            if "rating" in lf.columns else pl.lit(0.0)
        ).alias("rating"),

        (
            pl.col("review_count").cast(pl.Float64).fill_null(0)
            if "review_count" in lf.columns else pl.lit(0.0)
        ).alias("review_count"),
    ])

    # ---------------------------------------------------------
    # 2. Normalisation de la note (0–1)
    # ---------------------------------------------------------
    lf = lf.with_columns([
        (pl.col("rating") / 5).clip(0, 1).alias("rating_norm")
    ])

    # ---------------------------------------------------------
    # 3. Normalisation du volume d'avis (exponentielle)
    #    reviews_norm = 1 - exp(-reviews / k)
    # ---------------------------------------------------------
    lf = lf.with_columns([
        pl.when(pl.col("review_count") > 0)
        .then(
            pl.col("review_count").map_elements(
                lambda r: 1 - math.exp(-r / k)
            )
        )
        .otherwise(0.0)
        .alias("reviews_norm")
    ])

    # ---------------------------------------------------------
    # 4. Score final normalisé
    #    popularity_norm = rating_norm * reviews_norm
    # ---------------------------------------------------------
    lf = lf.with_columns([
        (pl.col("rating_norm") * pl.col("reviews_norm"))
        .alias("popularity_norm")
    ])

    return lf