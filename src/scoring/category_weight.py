import polars as pl

# Pondération configurable (brute)
CATEGORY_WEIGHTS = {
    18: 0.9, #Patrimoine & Monuments
    0: 3.5, #Nature & Paysages
    10: 1.0, #Culture & Musées
    13: 1.5, #Sports & Loisirs
    8: 0.8, #Gastronomie & Restauration
    9: 0.3, #Shopping & Artisanat
    2: 0.2, #Bien-être & Santé
    3: 0.5, #Famille & Enfants
    15: 0.7, #Transports touristiques
    6: 0.6, #Événements & Traditions
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
        pl.col("main_category_id")
        .replace(CATEGORY_WEIGHTS, default=DEFAULT_WEIGHT)
        .alias("category_weight")
    )

    # 2) Normalisation min–max
    lf = lf.with_columns(
        [
            (
                (pl.col("category_weight") - pl.col("category_weight").min())
                / (pl.col("category_weight").max() - pl.col("category_weight").min())
            ).alias("category_weight_norm")
        ]
    )

    return lf
