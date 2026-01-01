import polars as pl


# Poids configurables (tu peux les modifier à volonté)
WEIGHTS = {
    "density_h3_r8": 0.30,        # densité locale (commune)
    "diversity_h3_r8": 0.20,      # diversité locale
    "popularity_score": 0.25,     # avis / reviews
    "proximity_commune": -0.10,   # plus proche = meilleur
    "category_weight": 0.15,      # importance de la catégorie
}

def add_final_score(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Combine toutes les métriques disponibles dans le LazyFrame
    en un score final pondéré.
    Seules les colonnes présentes sont utilisées.
    """

    score_expr = None

    for col, weight in WEIGHTS.items():
        if col in lf.columns:
            expr = weight * pl.col(col)
            score_expr = expr if score_expr is None else score_expr + expr

    # Si aucune métrique n'est disponible → score = 0
    if score_expr is None:
        return lf.with_columns(pl.lit(0).alias("final_score"))

    # Ajout du score brut
    lf = lf.with_columns(score_expr.alias("final_score_raw"))

    # Normalisation simple (min-max) pour avoir un score lisible
    lf = lf.with_columns(
        (
            (pl.col("final_score_raw") - pl.col("final_score_raw").min())
            / (pl.col("final_score_raw").max() - pl.col("final_score_raw").min() + 1e-9)
        ).alias("final_score")
    )

    return lf