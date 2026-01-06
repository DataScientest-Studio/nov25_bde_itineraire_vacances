import polars as pl

# Poids configurables (normalisés)
WEIGHTS = {
    "density_commune_norm": 0.20,
    "diversity_commune_norm": 0.25,
    "popularity_norm": 0.30,
    "proximity_commune_norm": 0.10,
    "category_weight_norm": 0.15,
    # Optionnel :
    # "opening_score_norm": 0.10,
}

def add_final_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Combine toutes les métriques normalisées en un score final pondéré.
    Seules les colonnes présentes sont utilisées.
    """

    score_expr = None

    # Construction du score brut
    for col, weight in WEIGHTS.items():
        if col in lf.columns:
            expr = weight * pl.col(col)
            score_expr = expr if score_expr is None else score_expr + expr

    # Si aucune métrique n'est disponible → score = 0
    if score_expr is None:
        return lf.with_columns(pl.lit(0).alias("final_score"))

    # Ajout du score brut
    lf = lf.with_columns(score_expr.alias("final_score_raw"))

    # Normalisation min-max
    lf = lf.with_columns(
        (
            (pl.col("final_score_raw") - pl.col("final_score_raw").min()) /
            (pl.col("final_score_raw").max() - pl.col("final_score_raw").min() + 1e-9)
        ).alias("final_score")
    )

    return lf