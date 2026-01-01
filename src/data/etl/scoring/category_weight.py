import polars as pl

# Pondération configurable
CATEGORY_WEIGHTS = {
                    "Patrimoine & Monuments":3.0,
                    "Nature & Paysages":3.5,
                    "Culture & Musées":2.0,
                    "Sports & Loisirs":1.5,
                    "Gastronomie & Restauration":2.5,
                    "Shopping & Artisanat":0.5,
                    "Bien-être & Santé":2.2,
                    "Famille & Enfants":3.1,
                    "Transports touristiques":1.5,
                    "Événements & Traditions":2.1
                    
                    }

DEFAULT_WEIGHT = 1.0  # si la catégorie n'est pas connue


def add_category_weight(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Ajoute une colonne category_weight basée sur un dictionnaire de poids.
    Si la catégorie n'est pas dans le dictionnaire, DEFAULT_WEIGHT est utilisé.
    """
    return lf.with_columns(
        pl.col("main_category")
        .replace(CATEGORY_WEIGHTS, default=DEFAULT_WEIGHT)
        .alias("category_weight")
    )
