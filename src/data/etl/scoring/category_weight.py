import polars as pl

# Pondération configurable (brute)
CATEGORY_WEIGHTS = {
    "Patrimoine & Monuments": 0.9,
    "Nature & Paysages": 3.5,
    "Culture & Musées": 1.0,
    "Sports & Loisirs": 1.5,
    "Gastronomie & Restauration": 0.8,
    "Shopping & Artisanat": 0.3,
    "Bien-être & Santé": 0.2,
    "Famille & Enfants": 0.5,
    "Transports touristiques": 0.7,
    "Événements & Traditions": 0.6,
}

DEFAULT_WEIGHT = 1.0  # si la catégorie n'est pas connue


def add_category_weight(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute :
    - category_weight : poids brut
    - category_weight_norm : poids normalisé (0–1)
    """

    # 1) Poids brut
    lf = lf.with_columns(
        pl.col("main_category")
        .replace(CATEGORY_WEIGHTS, default=DEFAULT_WEIGHT)
        .alias("category_weight")
    )

    # 2) Normalisation min–max
    lf = lf.with_columns([
        (
            (pl.col("category_weight") - pl.col("category_weight").min()) /
            (pl.col("category_weight").max() - pl.col("category_weight").min())
        ).alias("category_weight_norm")
    ])

    return lf